"""Monkeypatches that make TransUNetWithTracks exportable via torch.export + AOTInductor
on GPUs where flash-attn has no torch.export-compatible custom-op registration
(e.g. flash-attn 2.5.6 on A100/sm_80).

Two flash-attn call sites are opaque to the dynamo/export tracer:
  1. The Triton rotary-embedding kernel (flash_attn.layers.rotary.apply_rotary_emb_func /
     apply_rotary_emb_kv_) - silently produces an incorrect (uninitialized) graph node
     under non-strict export instead of erroring.
  2. The flash-attention CUDA kernel itself (flash_attn_kvpacked_func) - same failure mode.

Fix:
  1. Rotary: this model always calls RotaryEmbedding with interleaved=False, scale_base=None,
     seqlen_offset=0, no cu_seqlens -- under those conditions apply_rotary_emb_torch (pure
     ATen ops already shipped in flash_attn.layers.rotary) is numerically equivalent to the
     Triton kernel path, and IS traceable/compilable. We monkeypatch the two call sites used
     by RotaryEmbedding.forward.
  2. Attention: flash_attn_kvpacked_func is always the code path used (this model is GQA
     everywhere, num_heads_kv != num_heads), never flash_attn_qkvpacked_func /
     flash_attn_func. We register it as a torch custom op (aido::flash_attn_kv) with a
     fake/meta kernel so export treats it as an atomic, correctly-shaped node while still
     dispatching to the real CUDA kernel at runtime -- i.e. exact same kernel, same numerics,
     just export-visible.

Call `apply()` once before constructing/tracing the model.
"""
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_kvpacked_func
from flash_attn.layers.rotary import apply_rotary_emb_torch
import flash_attn.layers.rotary as _rotary_mod

_APPLIED = False


def _apply_rotary_emb_func_patched(
    x, cos, sin, interleaved=False, inplace=False,
    seqlen_offsets=0, cu_seqlens=None, max_seqlen=None,
):
    assert cu_seqlens is None, "cu_seqlens (varlen) not supported by the export patch"
    assert seqlen_offsets == 0, "seqlen_offsets != 0 (KV-cache decoding) not used by predict_tracks"
    return apply_rotary_emb_torch(x, cos, sin, interleaved=interleaved)


def _apply_rotary_emb_kv_patched(kv, cos, sin, interleaved=False, seqlen_offsets=0):
    assert seqlen_offsets == 0, "seqlen_offsets != 0 (KV-cache decoding) not used by predict_tracks"
    k = apply_rotary_emb_torch(kv[:, :, 0], cos, sin, interleaved=interleaved)
    v = kv[:, :, 1]
    return torch.stack([k, v], dim=2)


def _rotary_forward_patched(self, qkv, kv=None, seqlen_offset=0, max_seqlen=None):
    """Replaces RotaryEmbedding.forward. The original caches cos/sin on
    `self._cos_cached` behind a Python-level `if seqlen > self._seq_len_cached`
    check -- under torch.export that cache gets lifted as a trace-time
    constant tied to the traced sequence length, which both (a) prevents a
    dynamic-L export (the length gets specialized to a constant) and (b) is
    unnecessary for this model, which always calls with kv is not None,
    scale_base=None (no xpos), seqlen_offset=0 (no KV-cache decoding). Under
    those conditions cos/sin depend only on L, so recomputing them fresh from
    `torch.arange(L)` each call is correct, cheap, and fully traceable with a
    symbolic L.
    """
    assert kv is not None, "predict_tracks always calls rotary_emb with separate q, kv (GQA)"
    assert self.scale is None, "xpos (scale_base) not used by this model"
    assert seqlen_offset == 0, "KV-cache decoding not used by predict_tracks"
    L = qkv.shape[1]
    t = torch.arange(L, device=qkv.device, dtype=torch.float32)
    freqs = torch.outer(t, self.inv_freq)
    cos = torch.cos(freqs).to(qkv.dtype)
    sin = torch.sin(freqs).to(qkv.dtype)
    q = apply_rotary_emb_torch(qkv, cos, sin, interleaved=self.interleaved)
    k = apply_rotary_emb_torch(kv[:, :, 0], cos, sin, interleaved=self.interleaved)
    kv_out = torch.stack([k, kv[:, :, 1]], dim=2)
    return q, kv_out


def _register_flash_attn_kv_op():
    if hasattr(torch.ops, "aido") and hasattr(torch.ops.aido, "flash_attn_kv"):
        return

    torch.library.define(
        "aido::flash_attn_kv",
        "(Tensor q, Tensor kv, float softmax_scale, int window_left, int window_right) -> Tensor",
    )

    @torch.library.impl("aido::flash_attn_kv", "cuda")
    def _impl(q, kv, softmax_scale, window_left, window_right):
        return flash_attn_kvpacked_func(
            q, kv, dropout_p=0.0, causal=False, softmax_scale=softmax_scale,
            window_size=(window_left, window_right),
        )

    @torch.library.register_fake("aido::flash_attn_kv")
    def _meta(q, kv, softmax_scale, window_left, window_right):
        b, s, h, d = q.shape
        return q.new_empty(b, s, h, d)


def _patched_flash_attn_kvpacked_func(
    q, kv, dropout_p=0.0, softmax_scale=None, causal=False,
    alibi_slopes=None, window_size=(-1, -1), deterministic=False, return_attn_probs=False,
):
    assert not causal, "predict_tracks never uses causal attention"
    assert alibi_slopes is None, "predict_tracks never uses ALiBi"
    assert not return_attn_probs
    wl, wr = window_size
    return torch.ops.aido.flash_attn_kv(q, kv, float(softmax_scale), int(wl), int(wr))


def _get_embs_patched(self, input_ids, up_to=None, output_hidden_states=False):
    """Same as TransUNetModel.get_embs, but avoids slicing an nn.ModuleList
    (`self.up_sampling_layers[:up_to]`) when up_to is None (always true for
    predict_tracks) -- torch.export's proxy tracer chokes on ModuleList
    __getitem__ with a slice (AttrProxy missing 'path' TypeError)."""
    hidden_states = []
    x = self.conv_dna(input_ids)
    if output_hidden_states:
        hidden_states.append(x)

    horizontal_xs = []
    for i, layer in enumerate(self.down_sampling_layers):
        x_horizontal = self.horizontal_layers[i](x)
        x = layer(x)
        horizontal_xs.append(x_horizontal)
        if output_hidden_states:
            hidden_states.append(x)

    x = self.transformer(x)
    if output_hidden_states:
        hidden_states.append(x)

    up_layers = self.up_sampling_layers if up_to is None else self.up_sampling_layers[:up_to]
    for i, layer in enumerate(up_layers):
        x = layer(x) + horizontal_xs[-1]
        del horizontal_xs[-1]
        if output_hidden_states:
            hidden_states.append(x)

    if up_to is None:
        x = self.final_conv(x)

    return x, hidden_states


def _apply_track_specific_activation_patched(self, logits, logits_mask, sigmoid_mask, softplus_mask):
    """Same math as BorzoiSimpleAdapter._apply_track_specific_activation, but
    fully vectorized (no `if mask.any():` Python-bool branch on tensor data).
    The three masks are mutually exclusive and exhaustive over the channel
    dim (logits_mask=SPLICE_SITES, sigmoid_mask=SPLICE_SITE_USAGE,
    softplus_mask=everything else), so
        logits*(1 - sm - pm) + sigmoid(logits)*sm + softplus(logits)*pm
    reduces to raw logits / sigmoid / softplus exactly where the original
    branch-based clone-and-overwrite would have -- but as one data-independent
    elementwise expression, which torch.export can trace without needing to
    guard on the (unbacked, data-dependent) boolean value of `.any()`.
    """
    sm = sigmoid_mask.to(logits.dtype).view(1, -1, 1)
    pm = softplus_mask.to(logits.dtype).view(1, -1, 1)
    lm = 1.0 - sm - pm
    return logits * lm + torch.sigmoid(logits) * sm + F.softplus(logits) * pm


def apply():
    global _APPLIED
    if _APPLIED:
        return
    _register_flash_attn_kv_op()
    _rotary_mod.apply_rotary_emb_func = _apply_rotary_emb_func_patched
    _rotary_mod.apply_rotary_emb_kv_ = _apply_rotary_emb_kv_patched
    _rotary_mod.RotaryEmbedding.forward = _rotary_forward_patched
    import mha as _mha_mod
    _mha_mod.flash_attn_kvpacked_func = _patched_flash_attn_kvpacked_func
    import modeling_transunet as _modeling_mod
    _modeling_mod.TransUNetModel.get_embs = _get_embs_patched
    import modeling_transunet_with_tracks as _tracks_mod
    _tracks_mod.BorzoiSimpleAdapter._apply_track_specific_activation = (
        _apply_track_specific_activation_patched
    )
    _APPLIED = True

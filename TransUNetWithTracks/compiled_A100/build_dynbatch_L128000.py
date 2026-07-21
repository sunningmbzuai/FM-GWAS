import sys, time
MODEL_DIR = "/lustre/scratch/users/Ning.Sun/ning/FM-GWAS/TransUNetWithTracks"
PATCH_DIR = MODEL_DIR + "/compiled_A100"
OUT_DIR = MODEL_DIR + "/compiled_A100"
sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, PATCH_DIR)
import export_patches
export_patches.apply()

import torch
from torch.export import Dim
from configuration_transunet_with_tracks import TransUNetWithTracksConfig
from modeling_transunet_with_tracks import TransUNetWithTracksModel

t0 = time.time()
config = TransUNetWithTracksConfig.from_pretrained(MODEL_DIR)
model = TransUNetWithTracksModel.from_pretrained(MODEL_DIR, config=config)
model = model.cuda().to(torch.bfloat16).eval()
print(f"loaded in {time.time()-t0:.1f}s")

class TrackWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):
        out = self.model.predict_tracks(input_ids, is_human=True)
        return torch.cat([out.cage.values, out.rna_seq.values, out.atac.values, out.dnase.values], dim=-1)

wrapper = TrackWrapper(model).cuda().eval()

L = 128000
assert L % 128 == 0
torch.manual_seed(0)
input_ids = torch.randint(0, 6, (2, L), device="cuda", dtype=torch.long)
with torch.no_grad():
    out_eager0 = wrapper(input_ids)
print("eager shape", out_eager0.shape)

batch_dim = Dim("batch", min=1, max=8)
dynamic_shapes = {"input_ids": {0: batch_dim}}

print("Exporting with dynamic BATCH only (L fixed=128000)...")
t0 = time.time()
with torch.no_grad():
    ep = torch.export.export(wrapper, (input_ids,), dynamic_shapes=dynamic_shapes, strict=False)
print(f"export done in {time.time()-t0:.1f}s")

pkg_path = OUT_DIR + "/aido_dna3ag_1bp4_L128000_dynbatch_a100.pt2"
t0 = time.time()
with torch.no_grad():
    torch._inductor.aoti_compile_and_package(ep, package_path=pkg_path)
print(f"compiled in {time.time()-t0:.1f}s -> {pkg_path}")

runner = torch._inductor.aoti_load_package(pkg_path)
for B in [1, 2, 4, 8]:
    ii = torch.randint(0, 6, (B, L), device="cuda", dtype=torch.long)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_e = wrapper(ii)
        out_c = runner(ii)
    torch.cuda.synchronize()
    diff = (out_e.float() - out_c.float()).abs()
    corr = torch.corrcoef(torch.stack([out_e.float().flatten(), out_c.float().flatten()]))[0, 1]
    print(f"B={B}: max abs diff={diff.max().item():.5f} corr={corr.item():.6f} (roundtrip {time.time()-t0:.2f}s)")
print("DONE")

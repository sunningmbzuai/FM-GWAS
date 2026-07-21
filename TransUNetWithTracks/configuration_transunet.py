from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Optional, Union, List

logger = logging.get_logger(__name__)

TRANSUNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "": "https://huggingface.co//resolve/main/config.json",
}

class TransUNetConfig(PretrainedConfig):
    model_type = "transunet"

    def __init__(
        self,
        dimensions: List[int] = [512, 768, 1024, 1280, 1536, 1792, 2048, 2048],
        depth: int = 8,
        heads: List[int] = [8, 8, 8, 8, 8, 16, 16, 16],
        window_size: int = 128,
        num_pooling_layers: int = 7,
        num_layers_per_downsampling_block: int = 1,
        num_layers_per_upsampling_block: int = 1,
        dropout_rate: float = 0.2,
        attn_dropout: float = 0.05,
        pos_dropout: float = 0.01,
        rotary_embedding_base: int = 100_000,
        middle_ropebase: int = 100_000,
        post_norm: bool = False,
        use_horiz_id: bool = False,
        checkpointing: bool = False, # << Pan: activation recomputation
        tp_degree: Optional[int] = 1, # << Pan: add tensor parallelism degree for FlashAttention
        use_swiglu: bool = False,
        linear_dimension_conversion: bool = True,
        **kwargs,
    ):

        assert len(dimensions) == num_pooling_layers + 1, f"Expected {num_pooling_layers + 1} dimensions, got {len(dimensions)}"
        assert len(heads) == len(dimensions), f"Expected {len(dimensions)} heads, got {len(heads)}"
        assert all(h%4==0 for h in heads), "All heads must be divisible by 4"
        assert all(d%h==0 for d,h in zip(dimensions, heads)), "All dimensions must be divisible by their corresponding number of heads"
        assert all(d//h%8==0 for d,h in zip(dimensions, heads)), f"Head size must be divisible by 8"

        self.dimensions = dimensions
        self.depth = depth
        self.heads = heads
        self.window_size = window_size
        self.num_pooling_layers = num_pooling_layers
        self.num_layers_per_downsampling_block = num_layers_per_downsampling_block
        self.num_layers_per_upsampling_block = num_layers_per_upsampling_block
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.rotary_embedding_base = rotary_embedding_base
        self.middle_ropebase = middle_ropebase
        self.post_norm = post_norm
        self.use_horiz_id = use_horiz_id
        self.checkpointing = checkpointing
        self.tp_degree = tp_degree # << Pan: add tensor parallelism degree for FlashAttention
        self.use_swiglu = use_swiglu
        self.linear_dimension_conversion = linear_dimension_conversion
        super().__init__(**kwargs)


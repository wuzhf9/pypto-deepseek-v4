"""Static DeepSeek V4 Flash configuration for the bf16 PyPTO path.

The model parameters in this file mirror the official DeepSeek V4 Flash
``config.json``. Runtime-only choices for this repository should be added as
separate compile options instead of changing these model hyperparameters.
"""

from dataclasses import dataclass
from typing import Literal


CompressRatio = Literal[0, 4, 128]
ScoreFunc = Literal["softmax", "sigmoid", "sqrtsoftplus"]
TorchDType = Literal["bfloat16"]
RopeScalingType = Literal["yarn"]


OFFICIAL_COMPRESS_RATIOS: tuple[CompressRatio, ...] = (
    0,
    0,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    0,
)


@dataclass(frozen=True)
class DeepSeekV4FlashConfig:
    """Official DeepSeek V4 Flash model hyperparameters.

    Field names use the inference-side naming from
    ``../deepseek_v4_flash/inference/model.py`` where practical, with the
    original HuggingFace names retained for metadata that is not part of
    ``ModelArgs``.
    """

    architectures: tuple[str, ...] = ("DeepseekV4ForCausalLM",)
    model_type: str = "deepseek_v4"
    transformers_version: str = "4.57.1"

    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 0
    eos_token_id: int = 1
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    torch_dtype: TorchDType = "bfloat16"
    use_cache: bool = True
    vocab_size: int = 129280

    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_mtp_layers: int = 1
    n_hash_layers: int = 3

    n_heads: int = 64
    n_kv_heads: int = 1
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    window_size: int = 128
    max_position_embeddings: int = 1048576

    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    score_func: ScoreFunc = "sqrtsoftplus"
    swiglu_limit: float = 10.0
    topk_method: str = "noaux_tc"

    rms_norm_eps: float = 1e-6
    hc_eps: float = 1e-6
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20

    rope_scaling_type: RopeScalingType = "yarn"
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    compress_rope_theta: float = 160000.0
    compress_ratios: tuple[CompressRatio, ...] = OFFICIAL_COMPRESS_RATIOS

    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512

    @property
    def hidden_size(self) -> int:
        return self.dim

    @property
    def moe_intermediate_size(self) -> int:
        return self.moe_inter_dim

    @property
    def num_hidden_layers(self) -> int:
        return self.n_layers

    @property
    def num_nextn_predict_layers(self) -> int:
        return self.n_mtp_layers

    @property
    def num_hash_layers(self) -> int:
        return self.n_hash_layers

    @property
    def num_attention_heads(self) -> int:
        return self.n_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.n_kv_heads

    @property
    def num_experts_per_tok(self) -> int:
        return self.n_activated_experts

    @property
    def qk_rope_head_dim(self) -> int:
        return self.rope_head_dim

    @property
    def sliding_window(self) -> int:
        return self.window_size

    @property
    def routed_scaling_factor(self) -> float:
        return self.route_scale

    @property
    def scoring_func(self) -> ScoreFunc:
        return self.score_func

    @property
    def rope_scaling(self) -> dict[str, int | float | str]:
        return {
            "beta_fast": self.beta_fast,
            "beta_slow": self.beta_slow,
            "factor": self.rope_factor,
            "original_max_position_embeddings": self.original_seq_len,
            "type": self.rope_scaling_type,
        }

    @property
    def norm_eps(self) -> float:
        """Alias used by the official inference ``ModelArgs``."""
        return self.rms_norm_eps

    @property
    def n_total_layers(self) -> int:
        return self.n_layers + self.n_mtp_layers

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def index_nope_head_dim(self) -> int:
        return self.index_head_dim - self.rope_head_dim

    @property
    def heads_per_o_group(self) -> int:
        return self.n_heads // self.o_groups

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.dim

    @property
    def mix_hc_dim(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    @property
    def index_weights_scale(self) -> float:
        return (self.index_head_dim**-0.5) * (self.index_n_heads**-0.5)

    @property
    def normal_layer_compress_ratios(self) -> tuple[CompressRatio, ...]:
        return self.compress_ratios[: self.n_layers]

    @property
    def mtp_layer_compress_ratios(self) -> tuple[CompressRatio, ...]:
        return self.compress_ratios[self.n_layers :]

    def validate(self) -> None:
        if len(self.compress_ratios) != self.n_total_layers:
            raise ValueError(
                "compress_ratios must contain one entry per normal layer plus "
                f"MTP layer, got {len(self.compress_ratios)} for {self.n_total_layers}"
            )
        if any(ratio not in (0, 4, 128) for ratio in self.compress_ratios):
            raise ValueError("compress_ratios may only contain 0, 4, or 128")
        if self.n_heads % self.o_groups != 0:
            raise ValueError("n_heads must be divisible by o_groups")
        if self.head_dim <= self.rope_head_dim:
            raise ValueError("head_dim must be greater than rope_head_dim")
        if self.index_head_dim <= self.rope_head_dim:
            raise ValueError("index_head_dim must be greater than rope_head_dim")
        if self.n_kv_heads != 1:
            raise ValueError("DeepSeek V4 Flash bf16 path expects MQA with one KV head")
        if self.torch_dtype != "bfloat16":
            raise ValueError("This PyPTO path only targets bf16 inference")


FLASH_CONFIG = DeepSeekV4FlashConfig()
FLASH_CONFIG.validate()


# Compatibility aliases for kernel modules that prefer constant-style names.
VOCAB_SIZE = FLASH_CONFIG.vocab_size
HIDDEN_SIZE = FLASH_CONFIG.dim
MOE_INTERMEDIATE_SIZE = FLASH_CONFIG.moe_inter_dim
NUM_HIDDEN_LAYERS = FLASH_CONFIG.n_layers
NUM_MTP_LAYERS = FLASH_CONFIG.n_mtp_layers
NUM_TOTAL_LAYERS = FLASH_CONFIG.n_total_layers
NUM_HASH_LAYERS = FLASH_CONFIG.n_hash_layers
NUM_ATTENTION_HEADS = FLASH_CONFIG.n_heads
NUM_KEY_VALUE_HEADS = FLASH_CONFIG.n_kv_heads
HEAD_DIM = FLASH_CONFIG.head_dim
ROPE_HEAD_DIM = FLASH_CONFIG.rope_head_dim
NOPE_HEAD_DIM = FLASH_CONFIG.nope_head_dim
Q_LORA_RANK = FLASH_CONFIG.q_lora_rank
O_LORA_RANK = FLASH_CONFIG.o_lora_rank
O_GROUPS = FLASH_CONFIG.o_groups
HEADS_PER_O_GROUP = FLASH_CONFIG.heads_per_o_group
WINDOW_SIZE = FLASH_CONFIG.window_size
MAX_POSITION_EMBEDDINGS = FLASH_CONFIG.max_position_embeddings
RMS_NORM_EPS = FLASH_CONFIG.rms_norm_eps
HC_EPS = FLASH_CONFIG.hc_eps
HC_MULT = FLASH_CONFIG.hc_mult
HC_DIM = FLASH_CONFIG.hc_dim
HC_SINKHORN_ITERS = FLASH_CONFIG.hc_sinkhorn_iters
N_ROUTED_EXPERTS = FLASH_CONFIG.n_routed_experts
N_SHARED_EXPERTS = FLASH_CONFIG.n_shared_experts
N_ACTIVATED_EXPERTS = FLASH_CONFIG.n_activated_experts
ROUTE_SCALE = FLASH_CONFIG.route_scale
SWIGLU_LIMIT = FLASH_CONFIG.swiglu_limit
ORIGINAL_SEQ_LEN = FLASH_CONFIG.original_seq_len
ROPE_THETA = FLASH_CONFIG.rope_theta
ROPE_FACTOR = FLASH_CONFIG.rope_factor
BETA_FAST = FLASH_CONFIG.beta_fast
BETA_SLOW = FLASH_CONFIG.beta_slow
COMPRESS_ROPE_THETA = FLASH_CONFIG.compress_rope_theta
COMPRESS_RATIOS = FLASH_CONFIG.compress_ratios
INDEX_N_HEADS = FLASH_CONFIG.index_n_heads
INDEX_HEAD_DIM = FLASH_CONFIG.index_head_dim
INDEX_NOPE_HEAD_DIM = FLASH_CONFIG.index_nope_head_dim
INDEX_TOPK = FLASH_CONFIG.index_topk


__all__ = [
    "FLASH_CONFIG",
    "DeepSeekV4FlashConfig",
    "CompressRatio",
    "ScoreFunc",
    "TorchDType",
    "RopeScalingType",
    "OFFICIAL_COMPRESS_RATIOS",
    "VOCAB_SIZE",
    "HIDDEN_SIZE",
    "MOE_INTERMEDIATE_SIZE",
    "NUM_HIDDEN_LAYERS",
    "NUM_MTP_LAYERS",
    "NUM_TOTAL_LAYERS",
    "NUM_HASH_LAYERS",
    "NUM_ATTENTION_HEADS",
    "NUM_KEY_VALUE_HEADS",
    "HEAD_DIM",
    "ROPE_HEAD_DIM",
    "NOPE_HEAD_DIM",
    "Q_LORA_RANK",
    "O_LORA_RANK",
    "O_GROUPS",
    "HEADS_PER_O_GROUP",
    "WINDOW_SIZE",
    "MAX_POSITION_EMBEDDINGS",
    "RMS_NORM_EPS",
    "HC_EPS",
    "HC_MULT",
    "HC_DIM",
    "HC_SINKHORN_ITERS",
    "N_ROUTED_EXPERTS",
    "N_SHARED_EXPERTS",
    "N_ACTIVATED_EXPERTS",
    "ROUTE_SCALE",
    "SWIGLU_LIMIT",
    "ORIGINAL_SEQ_LEN",
    "ROPE_THETA",
    "ROPE_FACTOR",
    "BETA_FAST",
    "BETA_SLOW",
    "COMPRESS_ROPE_THETA",
    "COMPRESS_RATIOS",
    "INDEX_N_HEADS",
    "INDEX_HEAD_DIM",
    "INDEX_NOPE_HEAD_DIM",
    "INDEX_TOPK",
]

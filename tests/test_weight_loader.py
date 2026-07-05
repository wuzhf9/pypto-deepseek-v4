from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from models.config import FLASH_CONFIG
from serving.weight_loader import (
    DeepSeekV4WeightLoader,
    dequant_fp4_weight_to_bf16,
    dequant_fp8_weight_to_bf16,
    normalize_param_name,
)


def _save_checkpoint(tmp_path, tensors):
    save_file(tensors, tmp_path / "model.safetensors")
    weight_map = {name: "model.safetensors" for name in tensors}
    return {"weight_map": weight_map}


def _small_config():
    return replace(
        FLASH_CONFIG,
        dim=4,
        hc_mult=2,
        n_layers=1,
        n_routed_experts=2,
        moe_inter_dim=3,
        vocab_size=8,
    )


def _official_checkpoint_path():
    path = Path(__file__).resolve().parents[2] / "deepseek_v4_flash"
    if not (path / "model.safetensors.index.json").exists():
        pytest.skip(f"Official checkpoint is not available at {path}")
    return path


def test_normalize_param_name_maps_hf_names():
    assert normalize_param_name("model.embed_tokens.weight") == "embed.weight"
    assert normalize_param_name("model.layers.2.self_attn.q_a_proj.weight") == "layers.2.attn.wq_a.weight"
    assert normalize_param_name("model.layers.2.mlp.experts.7.gate_proj.weight") == "layers.2.ffn.experts.7.w1.weight"
    assert normalize_param_name("lm_head.weight") == "head.weight"
    assert normalize_param_name("model.layers.3.mlp.gate.e_score_correction_bias") == "layers.3.ffn.gate.bias"


def test_dequant_fp8_matches_block_scale():
    weight = torch.ones(128, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1), 2.0, dtype=torch.float32)
    out = dequant_fp8_weight_to_bf16(weight, scale)
    assert out.dtype is torch.bfloat16
    assert torch.equal(out, torch.full((128, 128), 2.0, dtype=torch.bfloat16))


def test_dequant_fp4_unpacks_low_then_high_nibbles():
    packed = torch.tensor([[0x21, 0x43] + [0] * 14], dtype=torch.uint8).view(torch.int8)
    scale = torch.ones(1, 1, dtype=torch.float32)
    out = dequant_fp4_weight_to_bf16(packed, scale)
    assert out.dtype is torch.bfloat16
    assert torch.equal(out[0, :4], torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16))


def test_loader_global_and_layer_layouts(tmp_path):
    cfg = _small_config()
    tensors = {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4).bfloat16(),
        "model.norm.weight": torch.arange(4, dtype=torch.float32).bfloat16(),
        "lm_head.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "model.hc_head_fn": torch.arange(16, dtype=torch.float32).reshape(2, 8),
        "model.hc_head_scale": torch.tensor([1.25], dtype=torch.float32),
        "model.hc_head_base": torch.tensor([0.5, -0.5], dtype=torch.float32),
        "model.layers.0.hc_attn_fn": torch.arange(48, dtype=torch.float32).reshape(6, 8),
        "model.layers.0.hc_attn_scale": torch.arange(3, dtype=torch.float32),
        "model.layers.0.hc_attn_base": torch.arange(6, dtype=torch.float32),
        "model.layers.0.hc_ffn_fn": torch.arange(100, 148, dtype=torch.float32).reshape(6, 8),
        "model.layers.0.hc_ffn_scale": torch.arange(3, 6, dtype=torch.float32),
        "model.layers.0.hc_ffn_base": torch.arange(6, 12, dtype=torch.float32),
        "model.layers.0.input_layernorm.weight": torch.arange(4, dtype=torch.float32).bfloat16(),
        "model.layers.0.self_attn.q_a_proj.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4).bfloat16(),
        "model.layers.0.self_attn.q_a_layernorm.weight": torch.arange(2, dtype=torch.float32).bfloat16(),
        "model.layers.0.self_attn.q_b_proj.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2).bfloat16(),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).bfloat16(),
        "model.layers.0.self_attn.kv_a_layernorm.weight": torch.arange(3, dtype=torch.float32).bfloat16(),
        "model.layers.0.self_attn.attn_sink": torch.arange(2, dtype=torch.float32),
        "model.layers.0.self_attn.wo_a.weight": torch.arange(20, dtype=torch.float32).reshape(5, 4).bfloat16(),
        "model.layers.0.self_attn.wo_b.weight": torch.arange(20, dtype=torch.float32).reshape(4, 5).bfloat16(),
    }
    index = _save_checkpoint(tmp_path, tensors)

    loader = DeepSeekV4WeightLoader(tmp_path, index, config=cfg)
    assert torch.equal(loader.get_embedding_weight(), tensors["model.embed_tokens.weight"])

    head = loader.get_head_weights()
    assert head.hc_fn_t.shape == (8, 16)
    assert torch.equal(head.hc_fn_t[:, :2], tensors["model.hc_head_fn"].t())
    assert torch.count_nonzero(head.hc_fn_t[:, 2:]) == 0
    assert torch.equal(head.hc_base[:2], tensors["model.hc_head_base"])
    assert torch.count_nonzero(head.hc_base[2:]) == 0
    assert torch.equal(head.head_w, tensors["lm_head.weight"])

    hc = loader.get_layer_hc(0)
    assert torch.equal(hc.attn_hc_fn_t, tensors["model.layers.0.hc_attn_fn"].t())
    assert torch.equal(hc.ffn_hc_fn_t, tensors["model.layers.0.hc_ffn_fn"].t())

    attn = loader.get_layer_attention_common(0)
    assert torch.equal(attn.wq_a_t, tensors["model.layers.0.self_attn.q_a_proj.weight"].t())
    assert torch.equal(attn.wq_b_t, tensors["model.layers.0.self_attn.q_b_proj.weight"].t())
    assert torch.equal(attn.wkv_t, tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"].t())
    assert torch.equal(attn.wo_a_t, tensors["model.layers.0.self_attn.wo_a.weight"].t())
    assert torch.equal(attn.wo_b_t, tensors["model.layers.0.self_attn.wo_b.weight"].t())


def test_loader_compressor_indexer_and_moe_layouts(tmp_path):
    cfg = _small_config()
    tensors = {
        "model.layers.0.self_attn.compressor.wkv.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4).bfloat16(),
        "model.layers.0.self_attn.compressor.wgate.weight": torch.arange(8, 16, dtype=torch.float32).reshape(2, 4).bfloat16(),
        "model.layers.0.self_attn.compressor.ape": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "model.layers.0.self_attn.compressor.norm.weight": torch.arange(2, dtype=torch.float32).bfloat16(),
        "model.layers.0.self_attn.indexer.q_b_proj.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2).bfloat16(),
        "model.layers.0.self_attn.indexer.weights_proj.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4).bfloat16(),
        "model.layers.0.self_attn.indexer.compressor.wkv.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4).bfloat16(),
        "model.layers.0.self_attn.indexer.compressor.wgate.weight": torch.arange(8, 16, dtype=torch.float32).reshape(2, 4).bfloat16(),
        "model.layers.0.self_attn.indexer.compressor.ape": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "model.layers.0.self_attn.indexer.compressor.norm.weight": torch.arange(2, dtype=torch.float32).bfloat16(),
        "model.layers.0.mlp.gate.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4).bfloat16(),
        "model.layers.0.mlp.gate.e_score_correction_bias": torch.arange(2, dtype=torch.float32),
        "model.layers.0.mlp.gate.tid2eid": torch.tensor([[1, 0], [0, 1]], dtype=torch.int64),
        "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).bfloat16(),
        "model.layers.0.mlp.shared_experts.down_proj.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3).bfloat16(),
        "model.layers.0.mlp.shared_experts.up_proj.weight": torch.arange(12, 24, dtype=torch.float32).reshape(3, 4).bfloat16(),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.full((3, 4), 1.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.full((4, 3), 2.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.full((3, 4), 3.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.full((3, 4), 4.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.full((4, 3), 5.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.full((3, 4), 6.0, dtype=torch.bfloat16),
    }
    index = _save_checkpoint(tmp_path, tensors)
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=cfg)

    compressor = loader.get_layer_compressor_ratio4_attention(0)
    assert torch.equal(compressor.wkv_t, tensors["model.layers.0.self_attn.compressor.wkv.weight"].t())
    assert torch.equal(compressor.wgate_t, tensors["model.layers.0.self_attn.compressor.wgate.weight"].t())
    assert compressor.ape.dtype is torch.float32

    indexer = loader.get_layer_indexer(0)
    assert torch.equal(indexer.idx_wq_b_t, tensors["model.layers.0.self_attn.indexer.q_b_proj.weight"].t())
    assert torch.equal(indexer.idx_weights_proj_t, tensors["model.layers.0.self_attn.indexer.weights_proj.weight"].t())

    hash_gate = loader.get_layer_moe_gate(0, hash_route=True)
    assert torch.equal(hash_gate.gate_w_t, tensors["model.layers.0.mlp.gate.weight"].t())
    assert hash_gate.tid2eid.dtype is torch.int32
    topk_gate = loader.get_layer_moe_gate(0, hash_route=False)
    assert torch.equal(topk_gate.gate_bias, tensors["model.layers.0.mlp.gate.e_score_correction_bias"])

    shared = loader.get_layer_moe_shared(0)
    assert torch.equal(shared.shared_w1_t, tensors["model.layers.0.mlp.shared_experts.gate_proj.weight"].t())
    assert torch.equal(shared.shared_w2_t, tensors["model.layers.0.mlp.shared_experts.down_proj.weight"].t())

    expert = loader.get_moe_routed_expert(0, 1)
    assert torch.equal(expert.w1_t, tensors["model.layers.0.mlp.experts.1.gate_proj.weight"].t())
    packed = loader.get_layer_moe_routed_pack(0)
    assert packed.routed_w1_t.shape == (2, 4, 3)
    assert packed.routed_w2_t.shape == (2, 3, 4)
    assert torch.equal(packed.routed_w3_t[0], tensors["model.layers.0.mlp.experts.0.up_proj.weight"].t())


def test_loader_dequantizes_quantized_weight_from_weight_map(tmp_path):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8_e4m3fn is unavailable in this PyTorch build")

    tensors = {
        "model.layers.0.self_attn.q_a_proj.weight": torch.ones(128, 128, dtype=torch.float32).to(torch.float8_e4m3fn),
        "model.layers.0.self_attn.q_a_proj.weight_scale_inv": torch.full((1, 1), 2.0, dtype=torch.float32),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(1, 16, dtype=torch.int8),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
    }
    index = _save_checkpoint(tmp_path, tensors)
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=_small_config())

    fp8 = loader.get_linear_weight("layers.0.attn.wq_a.weight")
    assert fp8.dtype is torch.bfloat16
    assert torch.equal(fp8, torch.full((128, 128), 2.0, dtype=torch.bfloat16))

    fp4 = loader.get_linear_weight("layers.0.ffn.experts.0.w1.weight")
    assert fp4.shape == (1, 32)
    assert fp4.dtype is torch.bfloat16
    assert torch.count_nonzero(fp4) == 0


def test_cache_release_and_release_prefix(tmp_path):
    tensors = {
        "model.embed_tokens.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    index = _save_checkpoint(tmp_path, tensors)
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=_small_config())

    embed = loader.get_tensor("embed.weight")
    assert loader.cache_bytes >= embed.numel() * embed.element_size()
    loader.release("embed.weight")
    assert ("embed.weight" not in {key[0] for key in loader._cache})

    loader.get_tensor("layers.0.attn_norm.weight")
    assert loader.cache_bytes > 0
    loader.release_prefix("layers.0.")
    assert loader.cache_bytes == 0


def test_loader_reuses_safetensors_file_handle(tmp_path):
    tensors = {
        "model.embed_tokens.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    index = _save_checkpoint(tmp_path, tensors)
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=_small_config())

    loader.get_tensor("embed.weight", cache=False)
    loader.get_tensor("norm.weight", cache=False)
    assert len(loader._file_handles) == 1

    loader.release("embed.weight")
    assert len(loader._file_handles) == 1

    loader.close()
    assert len(loader._file_handles) == 0


def test_loader_uses_routed_pack_cache_when_available(tmp_path):
    cfg = _small_config()
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.full((3, 4), -1.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.full((4, 3), -1.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.full((3, 4), -1.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.full((3, 4), -1.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.full((4, 3), -1.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.full((3, 4), -1.0, dtype=torch.bfloat16),
    }
    index = _save_checkpoint(checkpoint, tensors)
    save_file(
        {
            "routed_w1_t": torch.full((2, 4, 3), 1.0, dtype=torch.bfloat16),
            "routed_w2_t": torch.full((2, 3, 4), 2.0, dtype=torch.bfloat16),
            "routed_w3_t": torch.full((2, 4, 3), 3.0, dtype=torch.bfloat16),
        },
        cache_dir / "layer_000_routed_pack.safetensors",
    )

    loader = DeepSeekV4WeightLoader(checkpoint, index, config=cfg, routed_pack_cache_dir=cache_dir)
    packed = loader.get_layer_moe_routed_pack(0)
    assert torch.equal(packed.routed_w1_t, torch.full((2, 4, 3), 1.0, dtype=torch.bfloat16))
    assert torch.equal(packed.routed_w2_t, torch.full((2, 3, 4), 2.0, dtype=torch.bfloat16))
    assert torch.equal(packed.routed_w3_t, torch.full((2, 4, 3), 3.0, dtype=torch.bfloat16))


def test_official_lowvram_weight_index_smoke_loads_representative_weights():
    checkpoint = _official_checkpoint_path()
    weight_index = checkpoint / "bf16_lowvram_cache" / "weight_index.json"
    if not weight_index.exists():
        pytest.skip(f"Official low-vram weight index is not available at {weight_index}")

    loader = DeepSeekV4WeightLoader(checkpoint, weight_index)

    hc = loader.get_layer_hc(0)
    assert hc.attn_hc_fn_t.shape == (FLASH_CONFIG.hc_dim, FLASH_CONFIG.mix_hc_dim)
    assert hc.attn_hc_fn_t.dtype is torch.float32
    assert hc.ffn_hc_fn_t.is_contiguous()

    q_a_t = loader.get_linear_t("layers.0.attn.wq_a.weight", cache=False)
    assert q_a_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.q_lora_rank)
    assert q_a_t.dtype is torch.bfloat16
    assert q_a_t.is_contiguous()

    wkv_t = loader.get_linear_t("layers.0.attn.wkv.weight", cache=False)
    assert wkv_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.head_dim)
    assert wkv_t.dtype is torch.bfloat16

    hca_comp = loader.get_layer_compressor_ratio128(3)
    assert hca_comp.wkv_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.head_dim)
    assert hca_comp.wgate_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.head_dim)
    assert hca_comp.ape.shape == (128, FLASH_CONFIG.head_dim)
    assert hca_comp.norm_w.shape == (FLASH_CONFIG.head_dim,)

    indexer = loader.get_layer_indexer(2)
    assert indexer.idx_wq_b_t.shape == (FLASH_CONFIG.q_lora_rank, FLASH_CONFIG.index_n_heads * FLASH_CONFIG.index_head_dim)
    assert indexer.idx_weights_proj_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.index_n_heads)
    assert indexer.idx_comp_wkv_t.shape == (FLASH_CONFIG.dim, 2 * FLASH_CONFIG.index_head_dim)
    assert indexer.idx_comp_norm_w.shape == (FLASH_CONFIG.index_head_dim,)

    hash_gate = loader.get_layer_moe_gate(0, hash_route=True)
    assert hash_gate.gate_w_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.n_routed_experts)
    assert hash_gate.tid2eid.shape == (FLASH_CONFIG.vocab_size, FLASH_CONFIG.n_activated_experts)
    assert hash_gate.tid2eid.dtype is torch.int32

    topk_gate = loader.get_layer_moe_gate(3, hash_route=False)
    assert topk_gate.gate_w_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.n_routed_experts)
    assert topk_gate.gate_bias.shape == (FLASH_CONFIG.n_routed_experts,)
    assert topk_gate.gate_bias.dtype is torch.float32

    expert = loader.get_moe_routed_expert(0, 0)
    assert expert.w1_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.moe_inter_dim)
    assert expert.w2_t.shape == (FLASH_CONFIG.moe_inter_dim, FLASH_CONFIG.dim)
    assert expert.w3_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.moe_inter_dim)
    assert expert.w1_t.dtype is torch.bfloat16


def test_official_raw_safetensors_index_infers_quantized_weight_kind():
    checkpoint = _official_checkpoint_path()
    loader = DeepSeekV4WeightLoader(checkpoint, checkpoint / "model.safetensors.index.json")

    q_a_t = loader.get_linear_t("layers.0.attn.wq_a.weight", cache=False)
    assert q_a_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.q_lora_rank)
    assert q_a_t.dtype is torch.bfloat16

    expert_w1_t = loader.get_linear_t("layers.0.ffn.experts.0.w1.weight", cache=False)
    assert expert_w1_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.moe_inter_dim)
    assert expert_w1_t.dtype is torch.bfloat16

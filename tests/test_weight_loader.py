from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from models.config import FLASH_CONFIG
from serving.runtime_types import HostStagingTensor, RuntimeWeight, StagingKind
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
        n_activated_experts=2,
        moe_inter_dim=3,
        vocab_size=8,
    )


def _official_checkpoint_path():
    path = Path(__file__).resolve().parents[2] / "deepseek_v4_flash"
    if not (path / "model.safetensors.index.json").exists():
        pytest.skip(f"Official checkpoint is not available at {path}")
    return path


def _host(value: RuntimeWeight | HostStagingTensor) -> torch.Tensor:
    return value.host_tensor


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
    embedding = loader.get_embedding_weight()
    assert isinstance(embedding, RuntimeWeight)
    assert embedding.key.name == "embed.weight"
    assert torch.equal(_host(embedding), tensors["model.embed_tokens.weight"])

    head = loader.get_head_weights()
    assert _host(head.hc_fn_t).shape == (8, 16)
    assert torch.equal(_host(head.hc_fn_t)[:, :2], tensors["model.hc_head_fn"].t())
    assert torch.count_nonzero(_host(head.hc_fn_t)[:, 2:]) == 0
    assert torch.equal(_host(head.hc_base)[:2], tensors["model.hc_head_base"])
    assert torch.count_nonzero(_host(head.hc_base)[2:]) == 0
    assert torch.equal(_host(head.head_w), tensors["lm_head.weight"])
    head_again = loader.get_head_weights()
    assert head_again.hc_fn_t is head.hc_fn_t
    assert head_again.hc_base is head.hc_base
    assert head_again.head_w is head.head_w

    hc = loader.get_layer_hc(0)
    assert torch.equal(_host(hc.attn_hc_fn_t), tensors["model.layers.0.hc_attn_fn"].t())
    assert torch.equal(_host(hc.ffn_hc_fn_t), tensors["model.layers.0.hc_ffn_fn"].t())
    hc_again = loader.get_layer_hc(0)
    assert hc_again.attn_hc_fn_t is hc.attn_hc_fn_t
    assert hc_again.ffn_hc_fn_t is hc.ffn_hc_fn_t

    attn = loader.get_layer_attention_common(0)
    assert torch.equal(_host(attn.wq_a_t), tensors["model.layers.0.self_attn.q_a_proj.weight"].t())
    assert torch.equal(_host(attn.wq_b_t), tensors["model.layers.0.self_attn.q_b_proj.weight"].t())
    assert torch.equal(_host(attn.wkv_t), tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"].t())
    assert torch.equal(_host(attn.wo_a_t), tensors["model.layers.0.self_attn.wo_a.weight"].t())
    assert torch.equal(_host(attn.wo_b_t), tensors["model.layers.0.self_attn.wo_b.weight"].t())
    attn_again = loader.get_layer_attention_common(0)
    assert attn_again.attn_norm_w is attn.attn_norm_w
    assert attn_again.attn_sink is attn.attn_sink


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
    assert torch.equal(_host(compressor.wkv_t), tensors["model.layers.0.self_attn.compressor.wkv.weight"].t())
    assert torch.equal(_host(compressor.wgate_t), tensors["model.layers.0.self_attn.compressor.wgate.weight"].t())
    assert _host(compressor.ape).dtype is torch.float32

    indexer = loader.get_layer_indexer(0)
    assert torch.equal(_host(indexer.idx_wq_b_t), tensors["model.layers.0.self_attn.indexer.q_b_proj.weight"].t())
    assert torch.equal(
        _host(indexer.idx_weights_proj_t), tensors["model.layers.0.self_attn.indexer.weights_proj.weight"].t()
    )

    hash_gate = loader.get_layer_moe_gate(0, hash_route=True)
    assert torch.equal(_host(hash_gate.gate_w_t), tensors["model.layers.0.mlp.gate.weight"].t())
    assert hash_gate.tid2eid is not None
    assert _host(hash_gate.tid2eid).dtype is torch.int32
    topk_gate = loader.get_layer_moe_gate(0, hash_route=False)
    assert topk_gate.gate_bias is not None
    assert torch.equal(_host(topk_gate.gate_bias), tensors["model.layers.0.mlp.gate.e_score_correction_bias"])

    shared = loader.get_layer_moe_shared(0)
    assert torch.equal(_host(shared.shared_w1_t), tensors["model.layers.0.mlp.shared_experts.gate_proj.weight"].t())
    assert torch.equal(_host(shared.shared_w2_t), tensors["model.layers.0.mlp.shared_experts.down_proj.weight"].t())
    shared_cache_bytes = loader.layout_cache_bytes
    shared_again = loader.get_layer_moe_shared(0)
    assert shared_again.shared_w1_t is shared.shared_w1_t
    assert shared_again.shared_w2_t is shared.shared_w2_t
    assert shared_again.shared_w3_t is shared.shared_w3_t
    assert loader.layout_cache_bytes == shared_cache_bytes

    expert = loader.get_moe_routed_expert(0, 1)
    assert torch.equal(expert.w1_t, tensors["model.layers.0.mlp.experts.1.gate_proj.weight"].t())
    assert loader.layout_cache_bytes == shared_cache_bytes
    packed = loader.get_layer_moe_routed_pack(0)
    assert _host(packed.routed_w1_t).shape == (2, 4, 3)
    assert _host(packed.routed_w2_t).shape == (2, 3, 4)
    assert packed.routed_w1_t.kind is StagingKind.PREFILL_ROUTED
    assert packed.routed_w1_t.slot == "w1_t"
    assert torch.equal(_host(packed.routed_w3_t)[0], tensors["model.layers.0.mlp.experts.0.up_proj.weight"].t())
    assert loader.layout_cache_bytes == shared_cache_bytes

    selected = loader.get_layer_moe_selected_experts(0, torch.tensor([[1, 0]], dtype=torch.int32))
    assert _host(selected.selected_w1_t).shape == (2, 4, 3)
    assert _host(selected.selected_w2_t).shape == (2, 3, 4)
    assert _host(selected.selected_w3_t).shape == (2, 4, 3)
    assert selected.selected_w1_t.kind is StagingKind.DECODE_SELECTED
    assert selected.selected_w1_t.slot == "w1_t"
    assert torch.equal(_host(selected.selected_w1_t)[0], tensors["model.layers.0.mlp.experts.1.gate_proj.weight"].t())
    assert torch.equal(_host(selected.selected_w2_t)[0], tensors["model.layers.0.mlp.experts.1.down_proj.weight"].t())
    assert torch.equal(_host(selected.selected_w3_t)[0], tensors["model.layers.0.mlp.experts.1.up_proj.weight"].t())
    assert torch.equal(_host(selected.selected_w1_t)[1], tensors["model.layers.0.mlp.experts.0.gate_proj.weight"].t())
    assert torch.equal(_host(selected.selected_w2_t)[1], tensors["model.layers.0.mlp.experts.0.down_proj.weight"].t())
    assert torch.equal(_host(selected.selected_w3_t)[1], tensors["model.layers.0.mlp.experts.0.up_proj.weight"].t())
    assert loader.layout_cache_bytes == shared_cache_bytes


def test_loader_selected_experts_validate_ids(tmp_path):
    cfg = _small_config()
    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.full((3, 4), 1.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.full((4, 3), 2.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.full((3, 4), 3.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.full((3, 4), 4.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.full((4, 3), 5.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.full((3, 4), 6.0, dtype=torch.bfloat16),
    }
    index = _save_checkpoint(tmp_path, tensors)
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=cfg)

    with pytest.raises(ValueError, match="selected expert ids"):
        loader.get_layer_moe_selected_experts(0, [0])
    with pytest.raises(ValueError, match="expert_id"):
        loader.get_layer_moe_selected_experts(0, [0, cfg.n_routed_experts])


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

    fp8 = _host(loader._get_transposed_weight("layers.0.attn.wq_a.weight"))
    assert fp8.dtype is torch.bfloat16
    assert torch.equal(fp8, torch.full((128, 128), 2.0, dtype=torch.bfloat16))

    fp4 = _host(loader._get_transposed_weight("layers.0.ffn.experts.0.w1.weight", cache=False))
    assert fp4.shape == (32, 1)
    assert fp4.dtype is torch.bfloat16
    assert torch.count_nonzero(fp4) == 0


def test_layout_cache_release_and_release_prefix(tmp_path):
    tensors = {
        "model.embed_tokens.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(2, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_a_proj.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3).bfloat16(),
    }
    index = _save_checkpoint(tmp_path, tensors)
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=_small_config())

    embed = loader.get_embedding_weight()
    assert loader.layout_cache_bytes == _host(embed).numel() * _host(embed).element_size()
    loader.release("embed.weight")
    assert "embed.weight" not in {key[0].name for key in loader._layout_cache}

    loader.get_layer_ffn_norm(0)
    loader._get_transposed_weight("layers.0.attn.wq_a.weight")
    assert loader.layout_cache_bytes > 0
    loader.release_prefix("layers.0.")
    assert loader.layout_cache_bytes == 0


def test_runtime_layout_cache_reuses_final_tensor_until_release(tmp_path):
    tensors = {
        "model.layers.0.self_attn.q_a_proj.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).bfloat16(),
    }
    index = _save_checkpoint(tmp_path, tensors)
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=_small_config(), profile=True)

    first = loader._get_transposed_weight("layers.0.attn.wq_a.weight")
    second = loader._get_transposed_weight("layers.0.attn.wq_a.weight")
    assert second is first
    assert loader.layout_cache_bytes == _host(first).numel() * _host(first).element_size()

    stats = {name: count for name, count, _ in loader.profile_summary()}
    assert stats["transpose.linear_t"] == 1
    assert stats["cache.layout.miss"] == 1
    assert stats["cache.layout.hit"] == 1

    loader.release_prefix("layers.0.")
    assert loader.layout_cache_bytes == 0
    loader._get_transposed_weight("layers.0.attn.wq_a.weight")
    stats = {name: count for name, count, _ in loader.profile_summary()}
    assert stats["transpose.linear_t"] == 2
    assert stats["cache.layout.miss"] == 2


def test_identity_runtime_layout_uses_fixed_cache(tmp_path):
    tensors = {
        "model.layers.0.post_attention_layernorm.weight": torch.arange(4, dtype=torch.float32).bfloat16(),
    }
    index = _save_checkpoint(tmp_path, tensors)
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=_small_config(), profile=True)

    first = loader.get_layer_ffn_norm(0)
    second = loader.get_layer_ffn_norm(0)
    assert second is first
    assert loader.layout_cache_bytes == _host(first).numel() * _host(first).element_size()

    stats = {name: count for name, count, _ in loader.profile_summary()}
    assert stats["cache.layout.miss"] == 1
    assert stats["cache.layout.hit"] == 1


def test_non_identity_runtime_layout_requires_builder(tmp_path):
    index = _save_checkpoint(tmp_path, {"model.norm.weight": torch.ones(2, dtype=torch.bfloat16)})
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=_small_config())

    with pytest.raises(ValueError, match="requires an explicit builder"):
        loader._get_runtime_weight("norm.weight", dtype=torch.bfloat16, layout="linear_t")


def test_runtime_layout_cache_key_separates_dtype_layout_and_padding(tmp_path):
    tensors = {
        "model.layers.0.self_attn.q_a_proj.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3).bfloat16(),
    }
    index = _save_checkpoint(tmp_path, tensors)
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=_small_config())

    bf16 = loader._get_transposed_weight("layers.0.attn.wq_a.weight")
    fp32 = loader._get_transposed_weight("layers.0.attn.wq_a.weight", dtype=torch.float32)
    assert _host(bf16).dtype is torch.bfloat16
    assert _host(fp32).dtype is torch.float32
    assert bf16 is not fp32

    padded_a = loader._get_runtime_weight(
        "synthetic",
        dtype=torch.float32,
        layout="padded",
        padding_profile="width=8",
        build=lambda: torch.ones(8, dtype=torch.float32),
    )
    padded_b = loader._get_runtime_weight(
        "synthetic",
        dtype=torch.float32,
        layout="padded",
        padding_profile="width=16",
        build=lambda: torch.ones(16, dtype=torch.float32),
    )
    assert _host(padded_a).shape == (8,)
    assert _host(padded_b).shape == (16,)


def test_runtime_layout_cache_keeps_all_entries_and_honors_cache_false(tmp_path):
    tensors = {
        "model.layers.0.self_attn.q_a_proj.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2).bfloat16(),
        "model.layers.0.self_attn.q_b_proj.weight": torch.arange(4, 8, dtype=torch.float32).reshape(2, 2).bfloat16(),
    }
    index = _save_checkpoint(tmp_path, tensors)
    layout_nbytes = 4 * torch.tensor([], dtype=torch.bfloat16).element_size()
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=_small_config(), profile=True)

    first = loader._get_transposed_weight("layers.0.attn.wq_a.weight")
    loader._get_transposed_weight("layers.0.attn.wq_b.weight")
    assert loader.layout_cache_bytes == 2 * layout_nbytes
    assert {key[0].name for key in loader._layout_cache} == {
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_b.weight",
    }

    reused = loader._get_transposed_weight("layers.0.attn.wq_a.weight")
    assert reused is first

    loader.release()
    uncached = loader._get_transposed_weight("layers.0.attn.wq_a.weight", cache=False)
    assert torch.equal(_host(uncached), tensors["model.layers.0.self_attn.q_a_proj.weight"].t())
    assert loader.layout_cache_bytes == 0

    stats = {name: count for name, count, _ in loader.profile_summary()}
    assert "cache.layout.evict" not in stats


def test_loader_reuses_safetensors_file_handle(tmp_path):
    tensors = {
        "model.embed_tokens.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    index = _save_checkpoint(tmp_path, tensors)
    loader = DeepSeekV4WeightLoader(tmp_path, index, config=_small_config())

    loader._get_runtime_weight("embed.weight", dtype=torch.bfloat16, cache=False)
    loader._get_runtime_weight("norm.weight", dtype=torch.bfloat16, cache=False)
    assert len(loader._file_handles) == 1

    loader.release("embed.weight")
    assert len(loader._file_handles) == 1

    loader.close()
    assert len(loader._file_handles) == 0


def test_loader_uses_layer_expert_cache_for_expert_selected_and_pack(tmp_path):
    cfg = _small_config()
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    expert_cache_dir = tmp_path / "expert_cache"
    expert_cache_dir.mkdir()

    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.full((3, 4), -1.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.full((4, 3), -2.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.full((3, 4), -3.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.full((3, 4), -4.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.full((4, 3), -5.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.full((3, 4), -6.0, dtype=torch.bfloat16),
    }
    index = _save_checkpoint(checkpoint, tensors)
    save_file(
        {
            "expert_000.w1_t": torch.full((4, 3), 1.0, dtype=torch.bfloat16),
            "expert_000.w2_t": torch.full((3, 4), 2.0, dtype=torch.bfloat16),
            "expert_000.w3_t": torch.full((4, 3), 3.0, dtype=torch.bfloat16),
            "expert_001.w1_t": torch.full((4, 3), 4.0, dtype=torch.bfloat16),
            "expert_001.w2_t": torch.full((3, 4), 5.0, dtype=torch.bfloat16),
            "expert_001.w3_t": torch.full((4, 3), 6.0, dtype=torch.bfloat16),
        },
        expert_cache_dir / "layer_000_experts.safetensors",
    )

    loader = DeepSeekV4WeightLoader(checkpoint, index, config=cfg, expert_cache_dir=expert_cache_dir)
    expert = loader.get_moe_routed_expert(0, 1)
    assert torch.equal(expert.w1_t, torch.full((4, 3), 4.0, dtype=torch.bfloat16))
    assert torch.equal(expert.w2_t, torch.full((3, 4), 5.0, dtype=torch.bfloat16))
    assert torch.equal(expert.w3_t, torch.full((4, 3), 6.0, dtype=torch.bfloat16))

    selected = loader.get_layer_moe_selected_experts(0, [1, 0])
    assert torch.equal(_host(selected.selected_w1_t)[0], torch.full((4, 3), 4.0, dtype=torch.bfloat16))
    assert torch.equal(_host(selected.selected_w2_t)[0], torch.full((3, 4), 5.0, dtype=torch.bfloat16))
    assert torch.equal(_host(selected.selected_w3_t)[0], torch.full((4, 3), 6.0, dtype=torch.bfloat16))
    assert torch.equal(_host(selected.selected_w1_t)[1], torch.full((4, 3), 1.0, dtype=torch.bfloat16))
    assert torch.equal(_host(selected.selected_w2_t)[1], torch.full((3, 4), 2.0, dtype=torch.bfloat16))
    assert torch.equal(_host(selected.selected_w3_t)[1], torch.full((4, 3), 3.0, dtype=torch.bfloat16))

    packed = loader.get_layer_moe_routed_pack(0)
    assert torch.equal(_host(packed.routed_w1_t)[0], torch.full((4, 3), 1.0, dtype=torch.bfloat16))
    assert torch.equal(_host(packed.routed_w1_t)[1], torch.full((4, 3), 4.0, dtype=torch.bfloat16))
    assert torch.equal(_host(packed.routed_w2_t)[0], torch.full((3, 4), 2.0, dtype=torch.bfloat16))
    assert torch.equal(_host(packed.routed_w2_t)[1], torch.full((3, 4), 5.0, dtype=torch.bfloat16))
    assert torch.equal(_host(packed.routed_w3_t)[0], torch.full((4, 3), 3.0, dtype=torch.bfloat16))
    assert torch.equal(_host(packed.routed_w3_t)[1], torch.full((4, 3), 6.0, dtype=torch.bfloat16))
    assert loader._expert_cache.open_handle_count == 1

    loader.close()
    assert loader._expert_cache.open_handle_count == 0


def test_loader_uses_v2_lazy_slices_for_selected_experts(tmp_path, monkeypatch):
    cfg = _small_config()
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    expert_cache_dir = tmp_path / "expert_cache"
    expert_cache_dir.mkdir()
    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.full((3, 4), -1.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.full((4, 3), -2.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.full((3, 4), -3.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.full((3, 4), -4.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.full((4, 3), -5.0, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.full((3, 4), -6.0, dtype=torch.bfloat16),
    }
    index = _save_checkpoint(checkpoint, tensors)
    packed_w1 = torch.stack(
        [torch.full((4, 3), 1.0, dtype=torch.bfloat16), torch.full((4, 3), 4.0, dtype=torch.bfloat16)]
    )
    packed_w2 = torch.stack(
        [torch.full((3, 4), 2.0, dtype=torch.bfloat16), torch.full((3, 4), 5.0, dtype=torch.bfloat16)]
    )
    packed_w3 = torch.stack(
        [torch.full((4, 3), 3.0, dtype=torch.bfloat16), torch.full((4, 3), 6.0, dtype=torch.bfloat16)]
    )
    save_file(
        {
            "routed_w1_t": packed_w1,
            "routed_w2_t": packed_w2,
            "routed_w3_t": packed_w3,
        },
        expert_cache_dir / "layer_000_experts.safetensors",
    )
    loader = DeepSeekV4WeightLoader(
        checkpoint,
        index,
        config=cfg,
        expert_cache_dir=expert_cache_dir,
        profile=True,
    )

    def fail_single_expert(*args, **kwargs):
        del args, kwargs
        raise AssertionError("v2 selected path must not call load_expert")

    monkeypatch.setattr(loader._expert_cache, "load_expert", fail_single_expert)
    selected = loader.get_layer_moe_selected_experts(0, [1, 0])

    assert torch.equal(_host(selected.selected_w1_t), packed_w1[[1, 0]])
    assert torch.equal(_host(selected.selected_w2_t), packed_w2[[1, 0]])
    assert torch.equal(_host(selected.selected_w3_t), packed_w3[[1, 0]])
    assert selected.selected_w1_t.kind is StagingKind.DECODE_SELECTED
    assert selected.selected_w1_t.slot == "w1_t"
    profile = {name: (count, elapsed_ms) for name, count, elapsed_ms in loader.profile_summary()}
    assert profile["expert_cache.v2.selected_slice_copy"][0] == 1
    assert profile["selected_experts.build"][0] == 1
    assert "expert_cache.load" not in profile


def test_loader_uses_v2_full_clones_for_routed_pack(tmp_path, monkeypatch):
    cfg = _small_config()
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    expert_cache_dir = tmp_path / "expert_cache"
    expert_cache_dir.mkdir()
    index = _save_checkpoint(checkpoint, {})
    packed_w1 = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).reshape(2, 4, 3)
    packed_w2 = torch.arange(2 * 3 * 4, dtype=torch.bfloat16).reshape(2, 3, 4) + 100
    packed_w3 = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).reshape(2, 4, 3) + 200
    save_file(
        {
            "routed_w1_t": packed_w1,
            "routed_w2_t": packed_w2,
            "routed_w3_t": packed_w3,
        },
        expert_cache_dir / "layer_000_experts.safetensors",
    )
    loader = DeepSeekV4WeightLoader(
        checkpoint,
        index,
        config=cfg,
        expert_cache_dir=expert_cache_dir,
        profile=True,
    )

    def fail_single_expert(*args, **kwargs):
        del args, kwargs
        raise AssertionError("v2 routed pack path must not call load_expert")

    monkeypatch.setattr(loader._expert_cache, "load_expert", fail_single_expert)
    packed = loader.get_layer_moe_routed_pack(0)

    assert torch.equal(_host(packed.routed_w1_t), packed_w1)
    assert torch.equal(_host(packed.routed_w2_t), packed_w2)
    assert torch.equal(_host(packed.routed_w3_t), packed_w3)
    assert packed.routed_w1_t.kind is StagingKind.PREFILL_ROUTED
    assert packed.routed_w1_t.slot == "w1_t"
    profile = {name: (count, elapsed_ms) for name, count, elapsed_ms in loader.profile_summary()}
    assert profile["expert_cache.v2.packed_clone"][0] == 1
    assert "expert_cache.v2.prefill_mmap_view" not in profile
    assert "expert_cache.v2.expert_slice" not in profile
    assert "expert_cache.load" not in profile


def test_official_lowvram_weight_index_smoke_loads_representative_weights():
    checkpoint = _official_checkpoint_path()
    weight_index = checkpoint / "bf16_lowvram_cache" / "weight_index.json"
    if not weight_index.exists():
        pytest.skip(f"Official low-vram weight index is not available at {weight_index}")

    loader = DeepSeekV4WeightLoader(checkpoint, weight_index)

    hc = loader.get_layer_hc(0)
    assert _host(hc.attn_hc_fn_t).shape == (FLASH_CONFIG.hc_dim, FLASH_CONFIG.mix_hc_dim)
    assert _host(hc.attn_hc_fn_t).dtype is torch.float32
    assert _host(hc.ffn_hc_fn_t).is_contiguous()

    q_a_t = _host(loader._get_transposed_weight("layers.0.attn.wq_a.weight", cache=False))
    assert q_a_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.q_lora_rank)
    assert q_a_t.dtype is torch.bfloat16
    assert q_a_t.is_contiguous()

    wkv_t = _host(loader._get_transposed_weight("layers.0.attn.wkv.weight", cache=False))
    assert wkv_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.head_dim)
    assert wkv_t.dtype is torch.bfloat16

    hca_comp = loader.get_layer_compressor_ratio128(3)
    assert _host(hca_comp.wkv_t).shape == (FLASH_CONFIG.dim, FLASH_CONFIG.head_dim)
    assert _host(hca_comp.wgate_t).shape == (FLASH_CONFIG.dim, FLASH_CONFIG.head_dim)
    assert _host(hca_comp.ape).shape == (128, FLASH_CONFIG.head_dim)
    assert _host(hca_comp.norm_w).shape == (FLASH_CONFIG.head_dim,)

    indexer = loader.get_layer_indexer(2)
    assert _host(indexer.idx_wq_b_t).shape == (
        FLASH_CONFIG.q_lora_rank,
        FLASH_CONFIG.index_n_heads * FLASH_CONFIG.index_head_dim,
    )
    assert _host(indexer.idx_weights_proj_t).shape == (FLASH_CONFIG.dim, FLASH_CONFIG.index_n_heads)
    assert _host(indexer.idx_comp_wkv_t).shape == (FLASH_CONFIG.dim, 2 * FLASH_CONFIG.index_head_dim)
    assert _host(indexer.idx_comp_norm_w).shape == (FLASH_CONFIG.index_head_dim,)

    hash_gate = loader.get_layer_moe_gate(0, hash_route=True)
    assert _host(hash_gate.gate_w_t).shape == (FLASH_CONFIG.dim, FLASH_CONFIG.n_routed_experts)
    assert hash_gate.tid2eid is not None
    assert _host(hash_gate.tid2eid).shape == (FLASH_CONFIG.vocab_size, FLASH_CONFIG.n_activated_experts)
    assert _host(hash_gate.tid2eid).dtype is torch.int32

    topk_gate = loader.get_layer_moe_gate(3, hash_route=False)
    assert _host(topk_gate.gate_w_t).shape == (FLASH_CONFIG.dim, FLASH_CONFIG.n_routed_experts)
    assert topk_gate.gate_bias is not None
    assert _host(topk_gate.gate_bias).shape == (FLASH_CONFIG.n_routed_experts,)
    assert _host(topk_gate.gate_bias).dtype is torch.float32

    expert = loader.get_moe_routed_expert(0, 0)
    assert expert.w1_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.moe_inter_dim)
    assert expert.w2_t.shape == (FLASH_CONFIG.moe_inter_dim, FLASH_CONFIG.dim)
    assert expert.w3_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.moe_inter_dim)
    assert expert.w1_t.dtype is torch.bfloat16

    selected = loader.get_layer_moe_selected_experts(0, torch.arange(FLASH_CONFIG.n_activated_experts, dtype=torch.int32))
    assert _host(selected.selected_w1_t).shape == (
        FLASH_CONFIG.n_activated_experts,
        FLASH_CONFIG.dim,
        FLASH_CONFIG.moe_inter_dim,
    )
    assert _host(selected.selected_w2_t).shape == (
        FLASH_CONFIG.n_activated_experts,
        FLASH_CONFIG.moe_inter_dim,
        FLASH_CONFIG.dim,
    )
    assert _host(selected.selected_w3_t).shape == (
        FLASH_CONFIG.n_activated_experts,
        FLASH_CONFIG.dim,
        FLASH_CONFIG.moe_inter_dim,
    )
    assert _host(selected.selected_w1_t).dtype is torch.bfloat16


def test_official_raw_safetensors_index_infers_quantized_weight_kind():
    checkpoint = _official_checkpoint_path()
    loader = DeepSeekV4WeightLoader(checkpoint, checkpoint / "model.safetensors.index.json")

    q_a_t = _host(loader._get_transposed_weight("layers.0.attn.wq_a.weight", cache=False))
    assert q_a_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.q_lora_rank)
    assert q_a_t.dtype is torch.bfloat16

    expert_w1_t = _host(loader._get_transposed_weight("layers.0.ffn.experts.0.w1.weight", cache=False))
    assert expert_w1_t.shape == (FLASH_CONFIG.dim, FLASH_CONFIG.moe_inter_dim)
    assert expert_w1_t.dtype is torch.bfloat16

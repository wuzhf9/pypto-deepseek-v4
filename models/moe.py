"""DeepSeek V4 Flash expert-major prefill and selected-expert decode MoE kernels."""

import torch
import torch.nn.functional as F

import pypto.language as pl

from models.common import assert_divisible
from models.config import FLASH_CONFIG as M
from models.expert import expert_shared_fwd
from models.gate import gate_hash_fwd, gate_topk_fwd
from models.linear import T_TILE


B = 1
S_DYN = pl.dynamic("S_DYN")

HIDDEN = M.dim
MOE_INTER_DIM = M.moe_inter_dim
N_EXPERTS = M.n_routed_experts
TOPK = M.n_activated_experts
VOCAB = M.vocab_size
ROUTE_SCALE = M.route_scale
SWIGLU_LIMIT = M.swiglu_limit
DEFAULT_SEQ_LEN = 8
S_DECODE = 1
MOE_K_TILE = 256
MOE_O_TILE = 64
MOE_OUT_GROUP = 2
ROUTE_META_PAD = 8
MAX_SEQ_LEN = 4096
MAX_ROUTES = MAX_SEQ_LEN * TOPK
PACKED_CAPACITY = MAX_ROUTES + N_EXPERTS * (T_TILE - 1)

assert_divisible(HIDDEN, MOE_K_TILE, "routed expert hidden input size")
assert_divisible(MOE_INTER_DIM, MOE_K_TILE, "routed expert intermediate input size")
assert_divisible(MOE_INTER_DIM, MOE_O_TILE, "routed expert intermediate output size")
assert_divisible(HIDDEN, MOE_O_TILE, "routed expert hidden output size")
assert_divisible(MOE_INTER_DIM // MOE_O_TILE, MOE_OUT_GROUP, "routed expert intermediate output blocks")
assert_divisible(HIDDEN // MOE_O_TILE, MOE_OUT_GROUP, "routed expert hidden output blocks")

MOE_HIDDEN_K_BLOCKS = HIDDEN // MOE_K_TILE
MOE_INTER_K_BLOCKS = MOE_INTER_DIM // MOE_K_TILE
MOE_INTER_O_BLOCKS = MOE_INTER_DIM // MOE_O_TILE
MOE_HIDDEN_O_BLOCKS = HIDDEN // MOE_O_TILE


@pl.jit.inline
def _pack_routes_by_expert(
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    expert_counts: pl.Tensor[[N_EXPERTS, 1], pl.INT32],
    expert_bases: pl.Tensor[[N_EXPERTS, 1], pl.INT32],
    packed_token_ids: pl.Tensor[[PACKED_CAPACITY, ROUTE_META_PAD], pl.INT32],
    packed_weights: pl.Tensor[[1, PACKED_CAPACITY], pl.FP32],
    route_to_packed: pl.Tensor[[MAX_ROUTES, ROUTE_META_PAD], pl.INT32],
):
    """Pack route metadata into 16-row-aligned expert-major segments."""
    indices.bind_dynamic(1, S_DYN)
    weights.bind_dynamic(1, S_DYN)
    tokens = pl.tensor.dim(indices, 1)
    indices_flat = pl.reshape(indices, [tokens, TOPK])
    weights_flat = pl.reshape(weights, [tokens, TOPK])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_route_pack"):
        counts = pl.array.create(N_EXPERTS, pl.INT32)
        cursors = pl.array.create(N_EXPERTS, pl.INT32)
        for e in pl.range(N_EXPERTS):
            counts[e] = 0

        for t in pl.range(tokens):
            for k in pl.range(TOPK):
                expert_id = pl.cast(pl.read(indices_flat, [t, k]), pl.INDEX)
                counts[expert_id] = counts[expert_id] + 1

        running = pl.const(0, pl.INT32)
        for e in pl.range(N_EXPERTS):
            count = counts[e]
            pl.write(expert_counts, [e, 0], count)
            pl.write(expert_bases, [e, 0], running)
            cursors[e] = running
            count_index = pl.cast(count, pl.INDEX)
            padded_count = pl.cast(((count_index + T_TILE - 1) // T_TILE) * T_TILE, pl.INT32)
            running = running + padded_count

        for t in pl.range(tokens):
            for k in pl.range(TOPK):
                expert_id = pl.cast(pl.read(indices_flat, [t, k]), pl.INDEX)
                packed_row_i32 = cursors[expert_id]
                cursors[expert_id] = packed_row_i32 + 1
                packed_row = pl.cast(packed_row_i32, pl.INDEX)
                route_id = t * TOPK + k
                pl.write(packed_token_ids, [packed_row, 0], pl.cast(t, pl.INT32))
                pl.write(packed_weights, [0, packed_row], pl.read(weights_flat, [t, k]))
                pl.write(route_to_packed, [route_id, 0], packed_row_i32)

    return expert_counts, expert_bases, packed_token_ids, packed_weights, route_to_packed


@pl.jit.inline
def _run_expert_major_routed_experts(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    expert_counts: pl.Tensor[[N_EXPERTS, 1], pl.INT32],
    expert_bases: pl.Tensor[[N_EXPERTS, 1], pl.INT32],
    packed_token_ids: pl.Tensor[[PACKED_CAPACITY, ROUTE_META_PAD], pl.INT32],
    packed_weights: pl.Tensor[[1, PACKED_CAPACITY], pl.FP32],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    packed_y: pl.Tensor[[PACKED_CAPACITY, HIDDEN], pl.BF16],
):
    """Run each routed expert on compact tiles containing up to 16 routes."""
    x.bind_dynamic(1, S_DYN)
    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, HIDDEN])
    routed_w1_flat = pl.reshape(routed_w1_t, [N_EXPERTS * HIDDEN, MOE_INTER_DIM])
    routed_w2_flat = pl.reshape(routed_w2_t, [N_EXPERTS * MOE_INTER_DIM, HIDDEN])
    routed_w3_flat = pl.reshape(routed_w3_t, [N_EXPERTS * HIDDEN, MOE_INTER_DIM])

    for expert_id in pl.parallel(N_EXPERTS):
        expert_count = pl.cast(pl.read(expert_counts, [expert_id, 0]), pl.INDEX)
        expert_base = pl.cast(pl.read(expert_bases, [expert_id, 0]), pl.INDEX)
        expert_tiles = (expert_count + T_TILE - 1) // T_TILE
        w1_start = expert_id * HIDDEN
        w2_start = expert_id * MOE_INTER_DIM
        w3_start = expert_id * HIDDEN

        for tile_id in pl.parallel(expert_tiles):
            packed_row0 = expert_base + tile_id * T_TILE
            valid_rows = pl.min(T_TILE, expert_count - tile_id * T_TILE)
            x_tile = pl.create_tensor([T_TILE, HIDDEN], dtype=pl.BF16)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_expert_gather"):
                for row in pl.range(valid_rows):
                    token_id = pl.cast(pl.read(packed_token_ids, [packed_row0 + row, 0]), pl.INDEX)
                    x_row = pl.slice(x_flat, [1, HIDDEN], [token_id, 0])
                    x_tile = pl.assemble(x_tile, x_row, [row, 0])

            gate_tile_fp32 = pl.create_tensor([T_TILE, MOE_INTER_DIM], dtype=pl.FP32)
            up_tile_fp32 = pl.create_tensor([T_TILE, MOE_INTER_DIM], dtype=pl.FP32)

            for og_idx in pl.spmd(MOE_INTER_O_BLOCKS // MOE_OUT_GROUP, name_hint="moe_expert_gate_up"):
                og = og_idx * MOE_OUT_GROUP
                for o_inner in pl.pipeline(MOE_OUT_GROUP, stage=2):
                    o0 = (og + o_inner) * MOE_O_TILE
                    x0 = pl.slice(x_tile, [T_TILE, MOE_K_TILE], [0, 0], valid_shape=[valid_rows, MOE_K_TILE])
                    w10 = pl.slice(routed_w1_flat, [MOE_K_TILE, MOE_O_TILE], [w1_start, o0])
                    w30 = pl.slice(routed_w3_flat, [MOE_K_TILE, MOE_O_TILE], [w3_start, o0])
                    gate_acc = pl.matmul(x0, w10, out_dtype=pl.FP32)
                    up_acc = pl.matmul(x0, w30, out_dtype=pl.FP32)
                    # The 256x64 weight tiles do not fit with double-buffered K reduction.
                    for kb in pl.range(1, MOE_HIDDEN_K_BLOCKS):
                        k0 = kb * MOE_K_TILE
                        xk = pl.slice(x_tile, [T_TILE, MOE_K_TILE], [0, k0], valid_shape=[valid_rows, MOE_K_TILE])
                        w1k = pl.slice(routed_w1_flat, [MOE_K_TILE, MOE_O_TILE], [w1_start + k0, o0])
                        w3k = pl.slice(routed_w3_flat, [MOE_K_TILE, MOE_O_TILE], [w3_start + k0, o0])
                        gate_acc = pl.matmul_acc(gate_acc, xk, w1k)
                        up_acc = pl.matmul_acc(up_acc, xk, w3k)
                    gate_tile_fp32[:, o0 : o0 + MOE_O_TILE] = gate_acc
                    up_tile_fp32[:, o0 : o0 + MOE_O_TILE] = up_acc

            hidden_tile_full = pl.create_tensor([T_TILE, MOE_INTER_DIM], dtype=pl.BF16)
            for ob in pl.spmd(MOE_INTER_O_BLOCKS, name_hint="moe_expert_swiglu"):
                o0 = ob * MOE_O_TILE
                gate_bf16 = pl.cast(gate_tile_fp32[:, o0 : o0 + MOE_O_TILE], target_type=pl.BF16, mode="rint")
                up_bf16 = pl.cast(up_tile_fp32[:, o0 : o0 + MOE_O_TILE], target_type=pl.BF16, mode="rint")
                gate_tile = pl.cast(gate_bf16, target_type=pl.FP32)
                up_tile = pl.cast(up_bf16, target_type=pl.FP32)
                limit = pl.full([T_TILE, MOE_O_TILE], dtype=pl.FP32, value=SWIGLU_LIMIT)
                neg_limit = pl.full([T_TILE, MOE_O_TILE], dtype=pl.FP32, value=-SWIGLU_LIMIT)
                gate_clamped = pl.minimum(gate_tile, limit)
                up_clamped = pl.minimum(pl.maximum(up_tile, neg_limit), limit)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_clamped)), 1.0))
                hidden_fp32 = pl.mul(pl.mul(gate_clamped, sigmoid), up_clamped)
                route_weight_row = pl.slice(
                    packed_weights,
                    [1, T_TILE],
                    [0, packed_row0],
                    valid_shape=[1, valid_rows],
                )
                route_weight_row = pl.fillpad(route_weight_row, pad_value=pl.PadValue.zero)
                route_weight_col = pl.reshape(route_weight_row, [T_TILE, 1])
                hidden_fp32 = pl.row_expand_mul(hidden_fp32, route_weight_col)
                hidden_tile = pl.cast(hidden_fp32, target_type=pl.BF16, mode="rint")
                hidden_tile_full[:, o0 : o0 + MOE_O_TILE] = pl.fillpad(hidden_tile, pad_value=pl.PadValue.zero)

            packed_y_tile = pl.create_tensor([T_TILE, HIDDEN], dtype=pl.BF16)
            for og_idx in pl.spmd(MOE_HIDDEN_O_BLOCKS // MOE_OUT_GROUP, name_hint="moe_expert_w2"):
                og = og_idx * MOE_OUT_GROUP
                for o_inner in pl.pipeline(MOE_OUT_GROUP, stage=2):
                    o0 = (og + o_inner) * MOE_O_TILE
                    h0 = pl.slice(
                        hidden_tile_full,
                        [T_TILE, MOE_K_TILE],
                        [0, 0],
                        valid_shape=[valid_rows, MOE_K_TILE],
                    )
                    w20 = pl.slice(routed_w2_flat, [MOE_K_TILE, MOE_O_TILE], [w2_start, o0])
                    acc = pl.matmul(h0, w20, out_dtype=pl.FP32)
                    for kb in pl.range(1, MOE_INTER_K_BLOCKS):
                        k0 = kb * MOE_K_TILE
                        hk = pl.slice(
                            hidden_tile_full,
                            [T_TILE, MOE_K_TILE],
                            [0, k0],
                            valid_shape=[valid_rows, MOE_K_TILE],
                        )
                        w2k = pl.slice(routed_w2_flat, [MOE_K_TILE, MOE_O_TILE], [w2_start + k0, o0])
                        acc = pl.matmul_acc(acc, hk, w2k)
                    acc_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    packed_y_tile[:, o0 : o0 + MOE_O_TILE] = pl.fillpad(
                        acc_bf16,
                        pad_value=pl.PadValue.zero,
                    )

            packed_y = pl.assemble(packed_y, packed_y_tile, [packed_row0, 0])

    return packed_y


@pl.jit.inline
def _run_selected_experts_decode(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    selected_w1_t: pl.Tensor[[TOPK, HIDDEN, MOE_INTER_DIM], pl.BF16],
    selected_w2_t: pl.Tensor[[TOPK, MOE_INTER_DIM, HIDDEN], pl.BF16],
    selected_w3_t: pl.Tensor[[TOPK, HIDDEN, MOE_INTER_DIM], pl.BF16],
    route_y: pl.Tensor[[B, S_DYN, TOPK, HIDDEN], pl.BF16],
):
    """Compute decode routed experts from top-k ordered selected weights."""
    x.bind_dynamic(1, S_DYN)
    weights.bind_dynamic(1, S_DYN)
    route_y.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, HIDDEN])
    weights_flat = pl.reshape(weights, [tokens, TOPK])
    route_y_flat = pl.reshape(route_y, [tokens * TOPK, HIDDEN])
    selected_w1_flat = pl.reshape(selected_w1_t, [TOPK * HIDDEN, MOE_INTER_DIM])
    selected_w2_flat = pl.reshape(selected_w2_t, [TOPK * MOE_INTER_DIM, HIDDEN])
    selected_w3_flat = pl.reshape(selected_w3_t, [TOPK * HIDDEN, MOE_INTER_DIM])

    for k in pl.range(TOPK):
        w1_start = k * HIDDEN
        w2_start = k * MOE_INTER_DIM
        w3_start = k * HIDDEN
        for t in pl.range(tokens):
            route_weight_scalar = pl.read(weights_flat, [t, k])
            gate_tile_fp32 = pl.create_tensor([T_TILE, MOE_INTER_DIM], dtype=pl.FP32)
            up_tile_fp32 = pl.create_tensor([T_TILE, MOE_INTER_DIM], dtype=pl.FP32)

            for og_idx in pl.spmd(MOE_INTER_O_BLOCKS // MOE_OUT_GROUP, name_hint="moe_selected_gate_up"):
                og = og_idx * MOE_OUT_GROUP
                for o_inner in pl.pipeline(MOE_OUT_GROUP, stage=2):
                    o0 = (og + o_inner) * MOE_O_TILE
                    x0 = pl.slice(x_flat, [T_TILE, MOE_K_TILE], [t, 0], valid_shape=[1, MOE_K_TILE])
                    w10 = pl.slice(selected_w1_flat, [MOE_K_TILE, MOE_O_TILE], [w1_start, o0])
                    w30 = pl.slice(selected_w3_flat, [MOE_K_TILE, MOE_O_TILE], [w3_start, o0])
                    gate_acc = pl.matmul(x0, w10, out_dtype=pl.FP32)
                    up_acc = pl.matmul(x0, w30, out_dtype=pl.FP32)
                    # The 256x64 weight tiles do not fit with double-buffered K reduction.
                    for kb in pl.range(1, MOE_HIDDEN_K_BLOCKS):
                        k0 = kb * MOE_K_TILE
                        xk = pl.slice(x_flat, [T_TILE, MOE_K_TILE], [t, k0], valid_shape=[1, MOE_K_TILE])
                        w1k = pl.slice(selected_w1_flat, [MOE_K_TILE, MOE_O_TILE], [w1_start + k0, o0])
                        w3k = pl.slice(selected_w3_flat, [MOE_K_TILE, MOE_O_TILE], [w3_start + k0, o0])
                        gate_acc = pl.matmul_acc(gate_acc, xk, w1k)
                        up_acc = pl.matmul_acc(up_acc, xk, w3k)
                    gate_tile_fp32[:, o0 : o0 + MOE_O_TILE] = gate_acc
                    up_tile_fp32[:, o0 : o0 + MOE_O_TILE] = up_acc

            hidden_tile_full = pl.create_tensor([T_TILE, MOE_INTER_DIM], dtype=pl.BF16)
            for ob in pl.spmd(MOE_INTER_O_BLOCKS, name_hint="moe_selected_swiglu"):
                o0 = ob * MOE_O_TILE
                gate_bf16 = pl.cast(gate_tile_fp32[:, o0 : o0 + MOE_O_TILE], target_type=pl.BF16, mode="rint")
                up_bf16 = pl.cast(up_tile_fp32[:, o0 : o0 + MOE_O_TILE], target_type=pl.BF16, mode="rint")
                gate_tile = pl.cast(gate_bf16, target_type=pl.FP32)
                up_tile = pl.cast(up_bf16, target_type=pl.FP32)
                limit = pl.full([T_TILE, MOE_O_TILE], dtype=pl.FP32, value=SWIGLU_LIMIT)
                neg_limit = pl.full([T_TILE, MOE_O_TILE], dtype=pl.FP32, value=-SWIGLU_LIMIT)
                gate_clamped = pl.minimum(gate_tile, limit)
                up_clamped = pl.minimum(pl.maximum(up_tile, neg_limit), limit)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_clamped)), 1.0))
                hidden_fp32 = pl.mul(pl.mul(pl.mul(gate_clamped, sigmoid), up_clamped), route_weight_scalar)
                hidden_tile = pl.cast(hidden_fp32, target_type=pl.BF16, mode="rint")
                hidden_tile_full[:, o0 : o0 + MOE_O_TILE] = pl.fillpad(hidden_tile, pad_value=pl.PadValue.zero)

            for og_idx in pl.spmd(MOE_HIDDEN_O_BLOCKS // MOE_OUT_GROUP, name_hint="moe_selected_w2"):
                og = og_idx * MOE_OUT_GROUP
                for o_inner in pl.pipeline(MOE_OUT_GROUP, stage=2):
                    o0 = (og + o_inner) * MOE_O_TILE
                    h0 = pl.slice(hidden_tile_full, [T_TILE, MOE_K_TILE], [0, 0], valid_shape=[1, MOE_K_TILE])
                    w20 = pl.slice(selected_w2_flat, [MOE_K_TILE, MOE_O_TILE], [w2_start, o0])
                    acc = pl.matmul(h0, w20, out_dtype=pl.FP32)
                    for kb in pl.range(1, MOE_INTER_K_BLOCKS):
                        k0 = kb * MOE_K_TILE
                        hk = pl.slice(hidden_tile_full, [T_TILE, MOE_K_TILE], [0, k0], valid_shape=[1, MOE_K_TILE])
                        w2k = pl.slice(selected_w2_flat, [MOE_K_TILE, MOE_O_TILE], [w2_start + k0, o0])
                        acc = pl.matmul_acc(acc, hk, w2k)
                    acc_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    dst = t * TOPK + k
                    route_y_flat = pl.assemble(route_y_flat, pl.slice(acc_bf16, [1, MOE_O_TILE], [0, 0]), [dst, o0])

    return pl.reshape(route_y_flat, [B, tokens, TOPK, HIDDEN])


@pl.jit.inline
def _combine_route_major(
    route_y: pl.Tensor[[B, S_DYN, TOPK, HIDDEN], pl.BF16],
    shared_y: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Combine shared expert output with TOPK routed outputs."""
    route_y.bind_dynamic(1, S_DYN)
    shared_y.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(shared_y, 1)
    route_y_flat = pl.reshape(route_y, [tokens * TOPK, HIDDEN])
    shared_y_flat = pl.reshape(shared_y, [tokens, HIDDEN])
    out_flat = pl.reshape(out, [tokens, HIDDEN])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_route_combine"):
        for t in pl.range(tokens):
            acc = pl.cast(shared_y_flat[t : t + 1, 0:HIDDEN], target_type=pl.FP32)
            for k in pl.range(TOPK):
                src = t * TOPK + k
                route_row = pl.cast(route_y_flat[src : src + 1, 0:HIDDEN], target_type=pl.FP32)
                acc = pl.add(acc, route_row)
            out_flat[t : t + 1, 0:HIDDEN] = pl.cast(acc, target_type=pl.BF16, mode="rint")

    return pl.reshape(out_flat, [B, tokens, HIDDEN])


@pl.jit.inline
def _combine_expert_major(
    packed_y: pl.Tensor[[PACKED_CAPACITY, HIDDEN], pl.BF16],
    route_to_packed: pl.Tensor[[MAX_ROUTES, ROUTE_META_PAD], pl.INT32],
    shared_y: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Combine shared output with expert-major routed rows via the inverse route map."""
    shared_y.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(shared_y, 1)
    shared_y_flat = pl.reshape(shared_y, [tokens, HIDDEN])
    out_flat = pl.reshape(out, [tokens, HIDDEN])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_expert_combine"):
        for t in pl.range(tokens):
            acc = pl.cast(shared_y_flat[t : t + 1, 0:HIDDEN], target_type=pl.FP32)
            for k in pl.range(TOPK):
                route_id = t * TOPK + k
                packed_row = pl.cast(pl.read(route_to_packed, [route_id, 0]), pl.INDEX)
                route_row = pl.cast(pl.slice(packed_y, [1, HIDDEN], [packed_row, 0]), target_type=pl.FP32)
                acc = pl.add(acc, route_row)
            out_flat[t : t + 1, 0:HIDDEN] = pl.cast(acc, target_type=pl.BF16, mode="rint")

    return pl.reshape(out_flat, [B, tokens, HIDDEN])


@pl.jit.inline
def moe_hash_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Interface for ``MoE.forward`` when ``Gate.hash`` is true."""
    x.bind_dynamic(1, S_DYN)
    tokens = pl.tensor.dim(x, 1)
    indices = pl.create_tensor([B, tokens, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([B, tokens, TOPK], dtype=pl.FP32)
    expert_counts = pl.create_tensor([N_EXPERTS, 1], dtype=pl.INT32)
    expert_bases = pl.create_tensor([N_EXPERTS, 1], dtype=pl.INT32)
    packed_token_ids = pl.create_tensor([PACKED_CAPACITY, ROUTE_META_PAD], dtype=pl.INT32)
    packed_weights = pl.create_tensor([1, PACKED_CAPACITY], dtype=pl.FP32)
    route_to_packed = pl.create_tensor([MAX_ROUTES, ROUTE_META_PAD], dtype=pl.INT32)
    packed_y = pl.create_tensor([PACKED_CAPACITY, HIDDEN], dtype=pl.BF16)
    shared_y = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)

    gate_hash_fwd(x, gate_w_t, tid2eid, input_ids, indices, weights)
    _pack_routes_by_expert(
        indices,
        weights,
        expert_counts,
        expert_bases,
        packed_token_ids,
        packed_weights,
        route_to_packed,
    )
    _run_expert_major_routed_experts(
        x,
        expert_counts,
        expert_bases,
        packed_token_ids,
        packed_weights,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        packed_y,
    )
    expert_shared_fwd(x, shared_w1_t, shared_w2_t, shared_w3_t, shared_y)
    _combine_expert_major(packed_y, route_to_packed, shared_y, out)
    return out


@pl.jit.inline
def moe_topk_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Interface for ``MoE.forward`` when ``Gate.hash`` is false."""
    x.bind_dynamic(1, S_DYN)
    tokens = pl.tensor.dim(x, 1)
    indices = pl.create_tensor([B, tokens, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([B, tokens, TOPK], dtype=pl.FP32)
    expert_counts = pl.create_tensor([N_EXPERTS, 1], dtype=pl.INT32)
    expert_bases = pl.create_tensor([N_EXPERTS, 1], dtype=pl.INT32)
    packed_token_ids = pl.create_tensor([PACKED_CAPACITY, ROUTE_META_PAD], dtype=pl.INT32)
    packed_weights = pl.create_tensor([1, PACKED_CAPACITY], dtype=pl.FP32)
    route_to_packed = pl.create_tensor([MAX_ROUTES, ROUTE_META_PAD], dtype=pl.INT32)
    packed_y = pl.create_tensor([PACKED_CAPACITY, HIDDEN], dtype=pl.BF16)
    shared_y = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)

    gate_topk_fwd(x, gate_w_t, gate_bias, indices, weights)
    _pack_routes_by_expert(
        indices,
        weights,
        expert_counts,
        expert_bases,
        packed_token_ids,
        packed_weights,
        route_to_packed,
    )
    _run_expert_major_routed_experts(
        x,
        expert_counts,
        expert_bases,
        packed_token_ids,
        packed_weights,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        packed_y,
    )
    expert_shared_fwd(x, shared_w1_t, shared_w2_t, shared_w3_t, shared_y)
    _combine_expert_major(packed_y, route_to_packed, shared_y, out)
    return out


@pl.jit.inline
def moe_selected_decode_experts_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    selected_w1_t: pl.Tensor[[TOPK, HIDDEN, MOE_INTER_DIM], pl.BF16],
    selected_w2_t: pl.Tensor[[TOPK, MOE_INTER_DIM, HIDDEN], pl.BF16],
    selected_w3_t: pl.Tensor[[TOPK, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run selected decode experts after gate indices/weights are known."""
    x.bind_dynamic(1, S_DYN)
    tokens = pl.tensor.dim(x, 1)
    route_y = pl.create_tensor([B, tokens, TOPK, HIDDEN], dtype=pl.BF16)
    shared_y = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)

    _run_selected_experts_decode(
        x,
        weights,
        selected_w1_t,
        selected_w2_t,
        selected_w3_t,
        route_y,
    )
    expert_shared_fwd(x, shared_w1_t, shared_w2_t, shared_w3_t, shared_y)
    _combine_route_major(route_y, shared_y, out)
    return out


@pl.jit
def moe_hash_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return moe_hash_fwd(
        x,
        gate_w_t,
        tid2eid,
        input_ids,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        out,
    )


@pl.jit
def moe_topk_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return moe_topk_fwd(
        x,
        gate_w_t,
        gate_bias,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        out,
    )


@pl.jit
def moe_selected_decode_experts_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    selected_w1_t: pl.Tensor[[TOPK, HIDDEN, MOE_INTER_DIM], pl.BF16],
    selected_w2_t: pl.Tensor[[TOPK, MOE_INTER_DIM, HIDDEN], pl.BF16],
    selected_w3_t: pl.Tensor[[TOPK, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return moe_selected_decode_experts_fwd(
        x,
        weights,
        selected_w1_t,
        selected_w2_t,
        selected_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        out,
    )


def _expert_forward_golden(
    x: torch.Tensor,
    w1_t: torch.Tensor,
    w2_t: torch.Tensor,
    w3_t: torch.Tensor,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = torch.matmul(x.float(), w1_t.float()).to(torch.bfloat16).float()
    up = torch.matmul(x.float(), w3_t.float()).to(torch.bfloat16).float()
    if SWIGLU_LIMIT > 0:
        up = torch.clamp(up, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    hidden = F.silu(gate) * up
    if weights is not None:
        hidden = hidden * weights.float()
    hidden_bf16 = hidden.to(torch.bfloat16)
    out = torch.matmul(hidden_bf16.float(), w2_t.float()).to(torch.bfloat16)
    return gate.to(torch.bfloat16), up.to(torch.bfloat16), hidden_bf16, out


def golden_moe_forward(tensors, *, hash_route: bool):
    """Route-major golden for official ``MoE.forward`` single-card bf16 logic."""
    x = tensors["x"]
    bsz, seq_len, _ = x.shape
    tokens = bsz * seq_len
    x_flat = x.reshape(tokens, HIDDEN)

    logits = torch.matmul(x.float(), tensors["gate_w_t"].float())
    scores = torch.sqrt(F.softplus(logits))
    if hash_route:
        indices = tensors["tid2eid"][tensors["input_ids"].long()]
    else:
        biased_scores = scores + tensors["gate_bias"].view(1, 1, N_EXPERTS)
        indices = biased_scores.topk(TOPK, dim=-1)[1]

    weights = scores.gather(-1, indices.long())
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ROUTE_SCALE

    route_y = torch.zeros(bsz, seq_len, TOPK, HIDDEN, dtype=torch.bfloat16)
    routed_acc = torch.zeros(tokens, HIDDEN, dtype=torch.float32)
    indices_flat = indices.reshape(tokens, TOPK).long()
    weights_flat = weights.reshape(tokens, TOPK)

    for t in range(tokens):
        b = t // seq_len
        s = t - b * seq_len
        x_row = x_flat[t : t + 1].view(1, 1, HIDDEN)
        for k in range(TOPK):
            expert_id = int(indices_flat[t, k].item())
            route_weight = weights_flat[t : t + 1, k : k + 1].view(1, 1, 1)
            _, _, _, route_out = _expert_forward_golden(
                x_row,
                tensors["routed_w1_t"][expert_id],
                tensors["routed_w2_t"][expert_id],
                tensors["routed_w3_t"][expert_id],
                route_weight,
            )
            route_y[b, s, k, :] = route_out.view(HIDDEN)

    # Match official expert-major accumulation order.
    route_y_flat = route_y.reshape(tokens, TOPK, HIDDEN)
    for expert_id in range(N_EXPERTS):
        for t in range(tokens):
            for k in range(TOPK):
                if int(indices_flat[t, k].item()) == expert_id:
                    routed_acc[t] += route_y_flat[t, k].float()

    shared_gate, shared_up, shared_hidden, shared_y = _expert_forward_golden(
        x,
        tensors["shared_w1_t"],
        tensors["shared_w2_t"],
        tensors["shared_w3_t"],
        weights=None,
    )
    out = (routed_acc + shared_y.reshape(tokens, HIDDEN).float()).to(torch.bfloat16).reshape(bsz, seq_len, HIDDEN)

    tensors["out"][:] = out


def golden_moe_selected_decode_experts_forward(tensors):
    """Golden for selected decode experts after gate weights are known."""
    x = tensors["x"]
    bsz, seq_len, _ = x.shape
    if bsz != B:
        raise ValueError(f"selected decode expects batch {B}, got {bsz}")
    if seq_len != S_DECODE:
        raise ValueError(f"selected decode expects sequence length {S_DECODE}, got {seq_len}")

    route_y = torch.zeros(bsz, seq_len, TOPK, HIDDEN, dtype=torch.bfloat16)
    weights = tensors["weights"]
    for k in range(TOPK):
        route_weight = weights[:, :, k : k + 1]
        _, _, _, route_out = _expert_forward_golden(
            x,
            tensors["selected_w1_t"][k],
            tensors["selected_w2_t"][k],
            tensors["selected_w3_t"][k],
            route_weight,
        )
        route_y[:, :, k, :] = route_out

    _, _, _, shared_y = _expert_forward_golden(
        x,
        tensors["shared_w1_t"],
        tensors["shared_w2_t"],
        tensors["shared_w3_t"],
        weights=None,
    )
    out = (route_y.float().sum(dim=2) + shared_y.float()).to(torch.bfloat16)
    tensors["out"][:] = out


def golden_moe_hash(tensors):
    golden_moe_forward(tensors, hash_route=True)


def golden_moe_topk(tensors):
    golden_moe_forward(tensors, hash_route=False)


def _build_tensor_specs(seq_len: int, *, hash_route: bool, profile_inputs: bool = False):
    from models.golden import TensorSpec

    if not 1 <= seq_len <= MAX_SEQ_LEN:
        raise ValueError(f"MoE seq_len must be in [1, {MAX_SEQ_LEN}], got {seq_len}")

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.2

    def init_gate_w_t():
        return torch.randn(HIDDEN, N_EXPERTS) * 0.02

    def init_gate_bias():
        return torch.randn(N_EXPERTS) * 0.02

    def init_input_ids():
        return torch.randint(0, VOCAB, (B, seq_len), dtype=torch.int64)

    def init_tid2eid():
        base = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
        token_offsets = torch.arange(VOCAB, dtype=torch.int32).view(VOCAB, 1)
        return (base + token_offsets) % N_EXPERTS

    def init_routed_w1_t():
        return torch.randn(N_EXPERTS, HIDDEN, MOE_INTER_DIM) * 0.02

    def init_routed_w2_t():
        return torch.randn(N_EXPERTS, MOE_INTER_DIM, HIDDEN) * 0.02

    def init_routed_w3_t():
        return torch.randn(N_EXPERTS, HIDDEN, MOE_INTER_DIM) * 0.02

    def init_shared_w1_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM) * 0.02

    def init_shared_w2_t():
        return torch.randn(MOE_INTER_DIM, HIDDEN) * 0.02

    def init_shared_w3_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM) * 0.02

    routed_w1_init = torch.zeros if profile_inputs else init_routed_w1_t
    routed_w2_init = torch.zeros if profile_inputs else init_routed_w2_t
    routed_w3_init = torch.zeros if profile_inputs else init_routed_w3_t
    shared_w1_init = torch.zeros if profile_inputs else init_shared_w1_t
    shared_w2_init = torch.zeros if profile_inputs else init_shared_w2_t
    shared_w3_init = torch.zeros if profile_inputs else init_shared_w3_t

    specs = [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("gate_w_t", [HIDDEN, N_EXPERTS], torch.bfloat16, init_value=init_gate_w_t),
    ]
    if hash_route:
        specs.extend(
            [
                TensorSpec("tid2eid", [VOCAB, TOPK], torch.int32, init_value=init_tid2eid),
                TensorSpec("input_ids", [B, seq_len], torch.int64, init_value=init_input_ids),
            ]
        )
    else:
        specs.append(TensorSpec("gate_bias", [N_EXPERTS], torch.float32, init_value=init_gate_bias))

    specs.extend(
        [
            TensorSpec("routed_w1_t", [N_EXPERTS, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=routed_w1_init),
            TensorSpec("routed_w2_t", [N_EXPERTS, MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=routed_w2_init),
            TensorSpec("routed_w3_t", [N_EXPERTS, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=routed_w3_init),
            TensorSpec("shared_w1_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=shared_w1_init),
            TensorSpec("shared_w2_t", [MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=shared_w2_init),
            TensorSpec("shared_w3_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=shared_w3_init),
            TensorSpec("out", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
        ]
    )
    return specs

def build_moe_hash_specs(seq_len: int = DEFAULT_SEQ_LEN, *, profile_inputs: bool = False):
    return _build_tensor_specs(seq_len, hash_route=True, profile_inputs=profile_inputs)


def build_moe_topk_specs(seq_len: int = DEFAULT_SEQ_LEN, *, profile_inputs: bool = False):
    return _build_tensor_specs(seq_len, hash_route=False, profile_inputs=profile_inputs)


def build_moe_selected_decode_specs(seq_len: int = S_DECODE, *, profile_inputs: bool = False):
    from models.golden import TensorSpec

    if seq_len != S_DECODE:
        raise ValueError(f"selected decode specs require seq_len={S_DECODE}, got {seq_len}")

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.2

    def init_weights():
        weights = torch.rand(B, seq_len, TOPK, dtype=torch.float32)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights * ROUTE_SCALE

    def init_selected_w1_t():
        return torch.randn(TOPK, HIDDEN, MOE_INTER_DIM) * 0.02

    def init_selected_w2_t():
        return torch.randn(TOPK, MOE_INTER_DIM, HIDDEN) * 0.02

    def init_selected_w3_t():
        return torch.randn(TOPK, HIDDEN, MOE_INTER_DIM) * 0.02

    def init_shared_w1_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM) * 0.02

    def init_shared_w2_t():
        return torch.randn(MOE_INTER_DIM, HIDDEN) * 0.02

    def init_shared_w3_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM) * 0.02

    selected_w1_init = torch.zeros if profile_inputs else init_selected_w1_t
    selected_w2_init = torch.zeros if profile_inputs else init_selected_w2_t
    selected_w3_init = torch.zeros if profile_inputs else init_selected_w3_t
    shared_w1_init = torch.zeros if profile_inputs else init_shared_w1_t
    shared_w2_init = torch.zeros if profile_inputs else init_shared_w2_t
    shared_w3_init = torch.zeros if profile_inputs else init_shared_w3_t

    return [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("weights", [B, seq_len, TOPK], torch.float32, init_value=init_weights),
        TensorSpec("selected_w1_t", [TOPK, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=selected_w1_init),
        TensorSpec("selected_w2_t", [TOPK, MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=selected_w2_init),
        TensorSpec("selected_w3_t", [TOPK, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=selected_w3_init),
        TensorSpec("shared_w1_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=shared_w1_init),
        TensorSpec("shared_w2_t", [MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=shared_w2_init),
        TensorSpec("shared_w3_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=shared_w3_init),
        TensorSpec("out", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
    ]


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash MoE validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--case", choices=["all", "hash", "topk", "selected-decode"], default="all")
    parser.add_argument(
        "--profile-inputs",
        action="store_true",
        help="Use lightweight expert weights and skip numerical validation; tensor shapes and dtypes are unchanged.",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    compare_fn = {
        "out": ratio_allclose(atol=1e-3, rtol=2.0 / 128, max_error_ratio=0.005),
    }
    all_cases = {
        "hash": ("moe-hash", moe_hash_test, build_moe_hash_specs, golden_moe_hash),
        "topk": ("moe-topk", moe_topk_test, build_moe_topk_specs, golden_moe_topk),
        "selected-decode": (
            "moe-selected-decode",
            moe_selected_decode_experts_test,
            build_moe_selected_decode_specs,
            golden_moe_selected_decode_experts_forward,
        ),
    }
    if args.case == "all":
        cases = [all_cases["hash"], all_cases["topk"], all_cases["selected-decode"]]
    else:
        cases = [all_cases[args.case]]

    if args.profile_inputs:
        print("[PROFILE] lightweight expert inputs enabled; numerical validation skipped", flush=True)

    failed = False
    for name, fn, build_specs, golden_fn in cases:
        print(f"[CASE] {name}", flush=True)
        seq_len = S_DECODE if name == "moe-selected-decode" else args.seq_len
        result = run_jit(
            fn=fn,
            specs=build_specs(seq_len, profile_inputs=args.profile_inputs),
            golden_fn=None if args.profile_inputs else golden_fn,
            runtime_cfg=runtime_cfg,
            compile_only=args.compile_only,
            compare_fn=compare_fn,
        )
        if not result.passed:
            failed = True
            if result.error:
                print(result.error)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "B",
    "S_DYN",
    "HIDDEN",
    "MOE_INTER_DIM",
    "N_EXPERTS",
    "TOPK",
    "VOCAB",
    "ROUTE_SCALE",
    "SWIGLU_LIMIT",
    "DEFAULT_SEQ_LEN",
    "S_DECODE",
    "MOE_K_TILE",
    "MOE_O_TILE",
    "MOE_OUT_GROUP",
    "ROUTE_META_PAD",
    "MAX_SEQ_LEN",
    "MAX_ROUTES",
    "PACKED_CAPACITY",
    "_pack_routes_by_expert",
    "_run_expert_major_routed_experts",
    "_run_selected_experts_decode",
    "_combine_route_major",
    "_combine_expert_major",
    "moe_hash_fwd",
    "moe_topk_fwd",
    "moe_selected_decode_experts_fwd",
    "moe_hash_test",
    "moe_topk_test",
    "moe_selected_decode_experts_test",
    "golden_moe_forward",
    "golden_moe_selected_decode_experts_forward",
    "golden_moe_hash",
    "golden_moe_topk",
    "build_moe_hash_specs",
    "build_moe_topk_specs",
    "build_moe_selected_decode_specs",
]

"""Executable split-projection gate; run only after the parent's GPU preflight.

    python -m rtp_llm.test.kimi_k31_natural_projection_cuda_test

The exported check_projection_contract also accepts a loaded real DSA layer.
BF16 GEMM tolerances are the existing bounded_projection_jit_gpu.py contract
(rtol=0.015625, atol=0.03125). Same-input norms, suffix copy, FP8 encoding and
IEEE FP32 head weights require exact equality. Integrated output accuracy is
an independent gate; this script does not claim it from synthetic projections.
"""

import json
from types import SimpleNamespace

import torch


def check_attention_history_contract(layer, hidden, fmha, cache, metadata):
    """Instrument one real prepared eager layer; caller owns collective order.

    Call on every rank after common plan preparation. Run again with another
    collectively prepared H chunk size to prove history visits stay KV-sized.
    No attention, quantization, cache writer or transport is replaced.
    """
    from contextlib import ExitStack
    from unittest.mock import patch
    from rtp_llm.models_py.mla_routes import resolve_mla_routes

    sfa, _, index, index_cache = resolve_mla_routes(fmha, cache, sparse=True, attention_tag="full")
    if index_cache is None or not index.is_sparse():
        raise ValueError("history contract requires a real sparse Prefill cache")
    owner = sfa.workspace_owner
    counts = dict(indexer=0, sfa=0, v_up=0, key_write=0, sfa_write=0, h_chunks=0)

    def counted(name, function):
        def invoke(*args, **kwargs):
            counts[name] += 1
            return function(*args, **kwargs)
        return invoke

    bmm, consume = torch.bmm, owner.row_exchange.consume_chunks
    def observed_bmm(left, right, *args, **kwargs):
        if right is owner.value_weights[layer.layer_idx]:
            counts["v_up"] += 1
        return bmm(left, right, *args, **kwargs)
    def observed_consume(*args, **kwargs):
        kwargs["consumer"] = counted("h_chunks", kwargs["consumer"])
        return consume(*args, **kwargs)

    with ExitStack() as stack:
        for name, params in (("indexer", index.fmha_params), ("sfa", sfa.page_rr_params)):
            adapter = params.cache_adapter
            stack.enter_context(patch.object(adapter, "read_prefix_chunk", counted(name, adapter.read_prefix_chunk)))
        stack.enter_context(patch.object(layer.indexer.indexer_op, "write_key",
            counted("key_write", layer.indexer.indexer_op.write_key)))
        stack.enter_context(patch.object(sfa, "_write_kv_cache", counted("sfa_write", sfa._write_kv_cache)))
        stack.enter_context(patch.object(owner.row_exchange, "consume_chunks", observed_consume))
        stack.enter_context(patch.object(torch, "bmm", observed_bmm))
        output = layer.forward(hidden, fmha, cache, metadata=metadata)
    torch.cuda.synchronize(hidden.device)
    assert counts["indexer"] == len(index.fmha_params.indexer_execution.blocks), counts
    assert counts["sfa"] == len(sfa.page_rr_params.sfa_execution.blocks), counts
    assert counts["h_chunks"] == (hidden.shape[0] + owner.route_chunk_rows - 1) // owner.route_chunk_rows, counts
    assert counts["v_up"] == counts["key_write"] == counts["sfa_write"] == 1, counts
    return output, counts


def check_projection_contract(layer, hidden, *, chunk_rows=1024):
    from rtp_llm.models_py.modules.kimi_k3.natural_dsa import Fp8NaturalProjection
    from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
        grouped_q_kv_rmsnorm, independent_q_kv_rmsnorm,
    )

    rows = hidden.shape[0]
    qr, kr, sr = layer.q_rank, layer.kv_rank, layer.suffix_dim
    fused = layer.input(hidden)
    q_raw, kv_raw, suffix, _, gate = fused.split(
        [qr, kr, sr, layer._input_padding, layer.heads * layer.v_dim], -1)
    expected_q, expected_kv = grouped_q_kv_rmsnorm(
        q_raw, kv_raw, layer.q_norm.weight, layer.kv_norm.weight,
        layer.q_norm.variance_epsilon, kv_suffix=suffix,
    )
    diagnostics = {"rows": rows, "chunk_rows": chunk_rows, "checks": 0}

    def check(name, actual, expected, *, exact=False):
        if actual.dtype == torch.float8_e4m3fn:
            actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
        torch.testing.assert_close(actual, expected, rtol=0 if exact else 0.015625,
                                   atol=0 if exact else 0.03125, msg=name)
        diagnostics["checks"] += 1
        if actual.numel() and actual.is_floating_point():
            diagnostics[name + "_max_abs"] = float((actual.float() - expected.float()).abs().max())

    for name in ("natural_q_input", "natural_kv_input", "natural_gate_input"):
        weight = getattr(layer, name).weight
        assert weight.untyped_storage().data_ptr() == layer.input.weight.untyped_storage().data_ptr()
    diagnostics["shared_weight_storage"] = True
    diagnostics["weight_strides"] = {name: list(getattr(layer, name).weight.stride())
                                     for name in ("input", "natural_q_input", "natural_kv_input", "natural_gate_input")}
    packed = layer.natural_kv_input(hidden)
    check("kv_projection", packed, fused[:, qr:qr + kr + sr])
    independent_kv = independent_q_kv_rmsnorm(
        kv_raw, layer.kv_norm.weight, layer.kv_norm.variance_epsilon,
        q_width=qr, kv_width=kr, full_rows=rows, suffix=suffix,
    )
    check("same_input_kv_norm", independent_kv, expected_kv, exact=True)
    packed_suffix = independent_kv.as_strided((rows, sr), (independent_kv.stride(0), 1),
                                            independent_kv.storage_offset() + kr)
    check("suffix_copy", packed_suffix, suffix, exact=True)
    full_index_query = layer.indexer.project_query(expected_q)
    legacy_index_query, legacy_key = layer.indexer._get_q_k_bf16(expected_q, hidden, None, None)
    check("index_query_composition", full_index_query, legacy_index_query, exact=True)
    check("index_key_composition", layer.indexer.project_key(hidden), legacy_key, exact=True)
    full_fp8, full_scale = layer.indexer.indexer_op.encode_query(full_index_query)
    full_weights = layer.indexer.project_head_weights(hidden, full_scale, is_prefill=True)
    full_query = layer.q_b(expected_q)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        part = hidden[start:end].contiguous()
        merged_q, merged_gate = layer.natural_q_gate_input(part)
        check("merged_q_projection", merged_q, q_raw[start:end])
        check("merged_gate_projection", merged_gate, gate[start:end])
        if isinstance(layer.natural_q_input, Fp8NaturalProjection):
            encoded_input = layer.natural_q_input.quantize_input(part)
            raw = layer.natural_q_input.forward_quantized(*encoded_input)
            projected_gate = layer.natural_gate_input.forward_quantized(*encoded_input)
            check("shared_encoding_q", raw, layer.natural_q_input(part), exact=True)
            check("shared_encoding_gate", projected_gate, layer.natural_gate_input(part), exact=True)
        else:
            raw = layer.natural_q_input(part)
            projected_gate = layer.natural_gate_input(part)
        check("q_projection", raw, q_raw[start:end])
        same_q = independent_q_kv_rmsnorm(
            q_raw[start:end], layer.q_norm.weight, layer.q_norm.variance_epsilon,
            q_width=qr, kv_width=kr, full_rows=rows,
        )
        check("same_input_q_norm", same_q, expected_q[start:end], exact=True)
        actual_q = independent_q_kv_rmsnorm(
            raw, layer.q_norm.weight, layer.q_norm.variance_epsilon,
            q_width=qr, kv_width=kr, full_rows=rows,
        )
        check("split_q_norm", actual_q, expected_q[start:end])
        check("query_up", layer.q_b(actual_q), full_query[start:end])
        check("gate", projected_gate, gate[start:end])
        query = layer.indexer.project_query(expected_q[start:end].contiguous(), original_rows=rows)
        check("index_query", query, full_index_query[start:end])
        # Quantization is tested with identical BF16 input, independently of
        # permitted GEMM rounding. No FP8 equality is inferred across GEMMs.
        encoded, scale = layer.indexer.indexer_op.encode_query(full_index_query[start:end])
        check("fp8_encoding", encoded, full_fp8[start:end], exact=True)
        check("fp32_q_scale", scale, full_scale[start:end], exact=True)
        weights = layer.indexer.project_head_weights(part, scale, is_prefill=True, original_rows=rows)
        check("ieee_head_weights", weights, full_weights[start:end], exact=True)
        assert weights.dtype == torch.float32
    torch.cuda.synchronize(hidden.device)
    return diagnostics


def make_layer(*, fp8=False):
    """Real CUDA modules at trained architecture dimensions; synthetic weights."""
    from rtp_llm.config.dsa_config import DsaConfig
    from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
    from rtp_llm.models.kimi_k3.fp8_weight import KimiK31LoadIndexerFp8Weight
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
    from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import _transform_scale_ue8m0
    from rtp_llm.models_py.modules.kimi_k3.dsa import KimiK31DSA
    from rtp_llm.ops import RoleType
    from rtp_llm.utils.model_weight import W

    device = torch.device("cuda")
    torch.manual_seed(31031)
    def matrix(i, o):
        return (torch.randn(i, o, device=device) / i**0.5).bfloat16()
    weights = {
        W.mla_fusedqkrope_w: matrix(7168, 1536 + 512 + 64 + 96 * 128),
        W.mla_q_b_w: matrix(1536, 96 * 192),
        W.attn_o_w: matrix(96 * 128, 7168),
        W.mla_q_a_ln_gamma: torch.ones(1536, device=device, dtype=torch.bfloat16),
        W.mla_kv_a_ln_gamma: torch.ones(512, device=device, dtype=torch.bfloat16),
        W.mla_indexer_k_norm_w: torch.ones(128, device=device, dtype=torch.bfloat16),
        W.mla_indexer_k_norm_b: torch.zeros(128, device=device, dtype=torch.bfloat16),
        W.mla_indexer_weights_proj_w: torch.randn(7168, 32, device=device, dtype=torch.float32) / 7168**0.5,
    }
    e8m0 = is_deep_gemm_e8m0_used()
    if fp8:
        fused = weights[W.mla_fusedqkrope_w]
        qkv = 1536 + 512 + 64
        weights[W.mla_fusedqkrope_w] = torch.cat(
            (fused[:, :qkv], fused.new_zeros((7168, (-qkv) % 128)), fused[:, qkv:]), -1)
        for key, scale in ((W.mla_fusedqkrope_w, W.mla_fusedqkrope_s),
                           (W.mla_q_b_w, W.mla_q_b_s), (W.attn_o_w, W.attn_o_s)):
            i, o = weights[key].shape
            weight, scales = KimiK31LoadIndexerFp8Weight._quantize_matrix(
                weights[key].T.contiguous(), use_ue8m0=e8m0)
            if e8m0:
                scales = _transform_scale_ue8m0(scales, mn=o)
            weights[key] = weight if e8m0 else weight.reshape(i, o)
            weights[scale] = scales if e8m0 else scales.reshape((i + 127) // 128, (o + 127) // 128)
    for key, scale, i, o in ((W.mla_indexer_qb_w, W.mla_indexer_qb_s, 1536, 4096),
                            (W.mla_indexer_k_w, W.mla_indexer_k_s, 7168, 128)):
        weight, scales = KimiK31LoadIndexerFp8Weight._quantize_matrix(matrix(i, o).T.contiguous(), use_ue8m0=e8m0)
        if e8m0:
            scales = _transform_scale_ue8m0(scales, mn=o)
        # Match the existing loader's pre-DeepGEMM logical shape contract.
        weights[key] = weight if e8m0 else weight.reshape(i, o)
        weights[scale] = scales if e8m0 else scales.reshape((i + 127) // 128, (o + 127) // 128)
    policy = DsaConfig((0,), 32, 128, 2048, 0, "layernorm", True, 1, 1)
    attn = SimpleNamespace(head_num=96, q_lora_rank=1536, kv_lora_rank=512,
        rope_head_dim=64, nope_head_dim=128, v_head_dim=128, indexer_head_num=32,
        indexer_head_dim=128, indexer_topk=2048, kernel_tokens_per_block=64,
        rope_config=SimpleNamespace(indexer_is_neox_style=False))
    config = SimpleNamespace(attn_config=attn, hidden_size=7168, is_mtp=False,
        layernorm_eps=1e-5, indexer_projection_quant_config=Fp8BlockWiseQuantConfig(),
        k3_runtime_config=SimpleNamespace(dsa=policy, mla_use_nope=True, mla_use_output_gate=True))
    parallel = SimpleNamespace(tp_size=4, tp_rank=0, role_type=RoleType.PREFILL,
        prefill_cp_config=SimpleNamespace(is_enabled=lambda: True, preserve_tp_layout=True))
    return KimiK31DSA(config, parallel, weights, 0)


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA numerical gate must run on the parent-selected GPU host")
    for fp8 in (False, True):
        layer = make_layer(fp8=fp8)
        for rows in (1, 2, 31, 32, 33, 127, 128, 129, 1025, 1026, 1055, 1057, 8192, 8193):
            hidden = torch.randn(rows, 7168, device="cuda", dtype=torch.bfloat16)
            print(json.dumps(dict(fp8=fp8, **check_projection_contract(
                layer, hidden, chunk_rows=8192 if rows >= 8192 else 1024))), flush=True)


if __name__ == "__main__":
    main()

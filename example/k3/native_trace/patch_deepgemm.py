"""Add observation-only stores to an explicitly audited DeepGEMM source tree."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

KERNEL = "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
RUNTIME = "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
API = "csrc/apis/mega.hpp"
PYTHON = "deep_gemm/mega/__init__.py"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected one source anchor: {before!r}")
    return text.replace(before, after, 1)


def routed_store(body, row_expression):
    return """
                        if constexpr (kK3Trace) {
                            DG_STATIC_ASSERT(L1_OUT_BLOCK_N == 64 and ATOM_M == 8, "Unsupported K3 trace layout");
                            if (not task_info.is_shared()) {
                                const K3MegaMoETrace trace(k3_trace,
                                    size_t(kNumRanks) * k3_trace_capacity * kNumTopk,
                                    kIntermediateHidden);
                                #pragma unroll
                                for (uint32_t r = 0; r < 2; ++ r) {
                                    const uint32_t trace_row = ROW_EXPRESSION + r;
                                    if (trace_row < valid_m) {
                                        const auto src = *workspace.get_token_src_metadata_ptr(pool_m_idx + trace_row);
                                        if (src.token_idx >= k3_trace_capacity) {
                                            atomicExch(trace.overflow, 1);
                                            continue;
                                        }
                                        const size_t slot = (size_t(src.rank_idx) * k3_trace_capacity + src.token_idx)
                                            * kNumTopk + src.topk_idx;
                                        BODY
                                    }
                                }
                            }
                        }
""".replace(
        "ROW_EXPRESSION", row_expression
    ).replace(
        "BODY", body
    )


def patch_kernel(text):
    text = replace_once(
        text,
        "#include <cstdint>",
        "#include <cstdint>\n#include <deep_gemm/layout/k3_mega_moe_trace.cuh>",
    )
    text = replace_once(
        text,
        "template <\n    uint32_t kNumMaxTokensPerRank,",
        "template <\n    bool kK3Trace,\n    uint32_t kNumMaxTokensPerRank,",
    )
    text = replace_once(
        text,
        "sm100_fp8_fp4_mega_moe_impl(void* y,",
        "sm100_fp8_fp4_mega_moe_impl(void* y,\n                            void* k3_trace,\n                            uint32_t k3_trace_capacity,",
    )
    row = "epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + (lane_idx % 4) * 2"
    fc1 = routed_store(
        """const uint32_t channel = n_block_idx * 64 + warp_idx_in_wg * 16 + lane_idx / 4 + k * 8;
                                        const size_t gate_idx = (slot * 2) * kIntermediateHidden + channel;
                                        const size_t up_idx = gate_idx + kIntermediateHidden;
                                        trace.fc1[gate_idx] = reinterpret_cast<const float*>(&fp32_values[k * 2])[r];
                                        trace.fc1[up_idx] = reinterpret_cast<const float*>(&fp32_values[k * 2 + 1])[r];
                                        trace.rounded[gate_idx] = reinterpret_cast<const uint16_t*>(&bf16_gate)[r];
                                        trace.rounded[up_idx] = reinterpret_cast<const uint16_t*>(&bf16_up)[r];""",
        row,
    )
    anchor = "auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + 1]);"
    text = replace_once(text, anchor, anchor + fc1)
    quantized = routed_store(
        """#pragma unroll
                                        for (uint32_t k = 0; k < 2; ++ k) {
                                            const uint32_t channel = n_block_idx * 64 + warp_idx_in_wg * 16 + lane_idx / 4 + k * 8;
                                            const size_t idx = slot * kIntermediateHidden + channel;
                                            trace.activation[idx] = reinterpret_cast<const float*>(&activation_values[i][k])[r];
                                            trace.fp8[idx] = reinterpret_cast<const uint8_t*>(&fp8x4_values)[k * 2 + r];
                                            if (channel % 32 == 0) {
                                                trace.scales[slot * (kIntermediateHidden / 32) + channel / 32] =
                                                    reinterpret_cast<const uint32_t*>(&sf)[r] >> 23;
                                            }
                                            if (channel == 0) {
                                                trace.expert_ids[slot] = sym_buffer.rank_idx * kNumExpertsPerRank
                                                    + task_info.local_expert_idx;
                                            }
                                        }""",
        row,
    )
    anchor = "const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));"
    return replace_once(text, anchor, anchor + quantized)


def patch_runtime(text):
    text = replace_once(
        text,
        "#include <deep_gemm/layout/mega_moe.cuh>",
        "#include <deep_gemm/layout/mega_moe.cuh>\n#include <deep_gemm/layout/k3_mega_moe_trace.cuh>",
    )
    text = replace_once(
        text,
        "        void* y;",
        "        void* y;\n        void* k3_trace;\n        uint32_t k3_trace_capacity;",
    )
    text = replace_once(
        text,
        "#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>",
        "// K3 native trace ABI 2\n#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>",
    )
    text = replace_once(
        text,
        "&sm100_fp8_fp4_mega_moe_impl<\n",
        "&sm100_fp8_fp4_mega_moe_impl<\n        {},\n",
    )
    # RTP has a preamble format argument before the template arguments.
    if "args.nvlink_barrier_timeout_secs,\n    args.num_max_tokens_per_rank" in text:
        text = replace_once(
            text,
            "args.nvlink_barrier_timeout_secs,\n    args.num_max_tokens_per_rank",
            'args.nvlink_barrier_timeout_secs,\n    args.k3_trace ? "true" : "false",\n    args.num_max_tokens_per_rank',
        )
    else:
        text = replace_once(
            text,
            ')", args.num_max_tokens_per_rank,',
            ')", args.k3_trace ? "true" : "false", args.num_max_tokens_per_rank,',
        )
    text = replace_once(
        text,
        "            args.y,",
        "            args.y,\n            args.k3_trace,\n            args.k3_trace_capacity,",
    )
    text = replace_once(
        text,
        "static void sm100_fp8_fp4_mega_moe(\n    const torch::Tensor& y,",
        "static void sm100_fp8_fp4_mega_moe(\n    const torch::Tensor& y,\n    const std::optional<torch::Tensor>& k3_trace_opt,",
    )
    anchor = "    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());"
    validation = """
    void* k3_trace_ptr = nullptr;
    uint32_t k3_trace_capacity = 0;
    if (k3_trace_opt.has_value()) {
        const auto& trace = k3_trace_opt.value();
        DG_HOST_ASSERT(intermediate_hidden > 0 and intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(trace.is_cuda() and trace.device() == y.device());
        DG_HOST_ASSERT(trace.scalar_type() == torch::kUInt8 and trace.dim() == 1 and trace.is_contiguous());
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(trace.data_ptr()) % alignof(float) == 0);
        DG_HOST_ASSERT(trace.nbytes() >= sizeof(int32_t));
        const size_t bytes_per_token = size_t(num_ranks) * num_topk
            * (17 * intermediate_hidden + intermediate_hidden / 32 + sizeof(int32_t));
        const size_t capacity = (trace.nbytes() - sizeof(int32_t)) / bytes_per_token;
        DG_HOST_ASSERT(capacity > 0 and capacity <= size_t(num_max_tokens_per_rank));
        DG_HOST_ASSERT(capacity >= size_t(num_tokens));
        DG_HOST_ASSERT(trace.nbytes() == K3MegaMoETrace::bytes(
            size_t(num_ranks) * capacity * num_topk, intermediate_hidden));
        k3_trace_capacity = static_cast<uint32_t>(capacity);
        k3_trace_ptr = trace.data_ptr();
    }
"""
    text = replace_once(text, anchor, anchor + validation)
    return replace_once(
        text,
        "        .y = y.data_ptr(),",
        "        .y = y.data_ptr(),\n        .k3_trace = k3_trace_ptr,\n        .k3_trace_capacity = k3_trace_capacity,",
    )


def patch_api(text):
    text = replace_once(
        text,
        'm.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe);',
        'm.def("k3_trace_abi", []() { return 2; });\n    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe);',
    )
    text = replace_once(
        text,
        "static void fp8_fp4_mega_moe(\n    const torch::Tensor& y,",
        "static void fp8_fp4_mega_moe(\n    const torch::Tensor& y,\n    const std::optional<torch::Tensor>& k3_trace_opt,",
    )
    return replace_once(
        text,
        "sm100_fp8_fp4_mega_moe(y,",
        "sm100_fp8_fp4_mega_moe(y, k3_trace_opt,",
    )


def patch_python(text):
    start = text.index("def fp8_fp4_mega_moe(")
    end = text.index("\ndef bf16_mega_moe(", start)
    part = text[start:end]
    part = replace_once(
        part,
        "):\n    _C.fp8_fp4_mega_moe(",
        ",\n                     k3_trace: Optional[torch.Tensor] = None):\n    _C.fp8_fp4_mega_moe(",
    )
    part = replace_once(
        part,
        "_C.fp8_fp4_mega_moe(\n        y,",
        "_C.fp8_fp4_mega_moe(\n        y, k3_trace,",
    )
    result = text[:start] + part + text[end:]
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--shared-expert",
        action="store_true",
        help="Observe the RTP fused BF16 shared expert as well",
    )
    args = parser.parse_args()
    expected = json.loads(args.manifest.read_text())
    transforms = {
        KERNEL: patch_kernel,
        RUNTIME: patch_runtime,
        API: patch_api,
        PYTHON: patch_python,
    }
    outputs = {}
    for name, transform in transforms.items():
        data = (args.source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected["files"][name]:
            raise ValueError(f"Unrecognized source contents: {name}")
        outputs[name] = transform(data.decode()).encode()
    if args.shared_expert:
        import patch_deepgemm_shared as shared

        for name, transform in (
            (KERNEL, shared.patch_kernel),
            (RUNTIME, shared.patch_runtime),
            (API, shared.patch_api),
            (PYTHON, shared.patch_python),
        ):
            outputs[name] = transform(outputs[name].decode()).encode()
        ast.parse(outputs[PYTHON].decode())
    header = Path(__file__).with_name("k3_mega_moe_trace.cuh").read_bytes()
    header_path = "deep_gemm/include/deep_gemm/layout/k3_mega_moe_trace.cuh"
    if (args.source / header_path).exists():
        raise ValueError("Native trace header already exists")
    outputs[header_path] = header
    helper_path = "deep_gemm/mega/k3_trace.py"
    if (args.source / helper_path).exists():
        raise ValueError("Native trace buffer helper already exists")
    outputs[helper_path] = Path(__file__).with_name("k3_trace_buffers.py").read_bytes()
    # Validate every source anchor before changing any file.
    for name, data in outputs.items():
        (args.source / name).write_bytes(data)
    print(
        json.dumps(
            {
                "source_commit": expected["commit"],
                "files": {
                    name: hashlib.sha256(data).hexdigest()
                    for name, data in outputs.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

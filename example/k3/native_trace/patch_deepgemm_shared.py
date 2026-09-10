"""Optional fused shared-expert observer for the pinned RTP DeepGEMM source.

Applied after the routed ABI 2 patch. Its separate ABI leaves the existing
routed-only vLLM observer unchanged. Stores observe the original TMEM values.
"""

from patch_deepgemm import replace_once


def patch_kernel(text):
    text = replace_once(
        text, "bool kK3Trace,", "bool kK3Trace,\n    bool kK3SharedTrace,"
    )
    text = replace_once(
        text,
        "uint32_t k3_trace_capacity,",
        "uint32_t k3_trace_capacity,\n                            void* k3_shared_trace,\n                            uint32_t k3_shared_trace_capacity,",
    )
    anchor = "auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + 1]);"
    store = """
                            if constexpr (kK3SharedTrace) {
                                DG_STATIC_ASSERT(L1_OUT_BLOCK_N == 64 and ATOM_M == 8, "Unsupported shared trace layout");
                                if (task_info.is_shared()) {
                                    const K3SharedMoETrace trace(k3_shared_trace,
                                        k3_shared_trace_capacity, kSharedIntermediateHidden);
                                    #pragma unroll
                                    for (uint32_t r = 0; r < 2; ++r) {
                                        const uint32_t row = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M
                                            + i * ATOM_M + (lane_idx % 4) * 2 + r;
                                        if (row < valid_m) {
                                            // Shared tasks use full rank-local token offsets, not routed rings.
                                            const uint32_t token = m_idx + row;
                                            const uint32_t channel = n_block_idx * 64 + warp_idx_in_wg * 16 + lane_idx / 4 + k * 8;
                                            if (token >= k3_shared_trace_capacity or channel >= kSharedIntermediateHidden) {
                                                atomicExch(trace.overflow, 1);
                                                continue;
                                            }
                                            const size_t gate = size_t(token) * 2 * kSharedIntermediateHidden + channel;
                                            const size_t up = gate + kSharedIntermediateHidden;
                                            trace.fc1[gate] = reinterpret_cast<const float*>(&fp32_values[k * 2])[r];
                                            trace.fc1[up] = reinterpret_cast<const float*>(&fp32_values[k * 2 + 1])[r];
                                            trace.rounded[gate] = reinterpret_cast<const uint16_t*>(&bf16_gate)[r];
                                            trace.rounded[up] = reinterpret_cast<const uint16_t*>(&bf16_up)[r];
                                            if (channel == 0) trace.valid[token] = 1;
                                        }
                                    }
                                }
                            }
"""
    return replace_once(text, anchor, anchor + store)


def patch_runtime(text):
    if "const int& shared_hidden, const int& shared_intermediate_hidden," not in text:
        raise ValueError(
            "Shared observer requires the audited RTP fused shared-expert API"
        )
    text = replace_once(
        text,
        "uint32_t k3_trace_capacity;",
        "uint32_t k3_trace_capacity;\n        void* k3_shared_trace;\n        uint32_t k3_shared_trace_capacity;",
    )
    text = replace_once(
        text,
        "&sm100_fp8_fp4_mega_moe_impl<\n        {},",
        "&sm100_fp8_fp4_mega_moe_impl<\n        {}, {},",
    )
    text = replace_once(
        text,
        'args.k3_trace ? "true" : "false",',
        'args.k3_trace ? "true" : "false", args.k3_shared_trace ? "true" : "false",',
    )
    text = replace_once(
        text,
        "            args.k3_trace_capacity,",
        "            args.k3_trace_capacity,\n            args.k3_shared_trace,\n            args.k3_shared_trace_capacity,",
    )
    text = replace_once(
        text,
        "const std::optional<torch::Tensor>& k3_trace_opt,",
        "const std::optional<torch::Tensor>& k3_trace_opt,\n    const std::optional<torch::Tensor>& k3_shared_trace_opt,",
    )
    validation = """
    void* k3_shared_trace_ptr = nullptr;
    uint32_t k3_shared_trace_capacity = 0;
    if (k3_shared_trace_opt.has_value()) {
        const auto& trace = k3_shared_trace_opt.value();
        DG_HOST_ASSERT(shared_hidden > 0 and shared_intermediate_hidden > 0 and shared_intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(trace.is_cuda() and trace.device() == y.device());
        DG_HOST_ASSERT(trace.scalar_type() == torch::kUInt8 and trace.dim() == 1 and trace.is_contiguous());
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(trace.data_ptr()) % alignof(float) == 0);
        DG_HOST_ASSERT(trace.nbytes() >= sizeof(int32_t));
        const size_t per_token = 12 * size_t(shared_intermediate_hidden) + sizeof(int32_t);
        const size_t capacity = (trace.nbytes() - sizeof(int32_t)) / per_token;
        DG_HOST_ASSERT(capacity > 0 and capacity <= size_t(num_max_tokens_per_rank));
        DG_HOST_ASSERT(capacity >= size_t(num_tokens));
        DG_HOST_ASSERT(trace.nbytes() == K3SharedMoETrace::bytes(capacity, shared_intermediate_hidden));
        k3_shared_trace_capacity = static_cast<uint32_t>(capacity);
        k3_shared_trace_ptr = trace.data_ptr();
    }
"""
    text = replace_once(
        text,
        "    const auto num_experts = num_experts_per_rank * num_ranks;",
        validation + "\n    const auto num_experts = num_experts_per_rank * num_ranks;",
    )
    return replace_once(
        text,
        "        .k3_trace_capacity = k3_trace_capacity,",
        "        .k3_trace_capacity = k3_trace_capacity,\n        .k3_shared_trace = k3_shared_trace_ptr,\n        .k3_shared_trace_capacity = k3_shared_trace_capacity,",
    )


def patch_api(text):
    text = replace_once(
        text,
        'm.def("k3_trace_abi", []() { return 2; });',
        'm.def("k3_trace_abi", []() { return 2; });\n    m.def("k3_shared_trace_abi", []() { return 1; });',
    )
    text = replace_once(
        text,
        "const std::optional<torch::Tensor>& k3_trace_opt,",
        "const std::optional<torch::Tensor>& k3_trace_opt,\n    const std::optional<torch::Tensor>& k3_shared_trace_opt,",
    )
    return replace_once(
        text,
        "sm100_fp8_fp4_mega_moe(y, k3_trace_opt,",
        "sm100_fp8_fp4_mega_moe(y, k3_trace_opt, k3_shared_trace_opt,",
    )


def patch_python(text):
    text = replace_once(
        text,
        "k3_trace: Optional[torch.Tensor] = None):",
        "k3_trace: Optional[torch.Tensor] = None,\n                     k3_shared_trace: Optional[torch.Tensor] = None):",
    )
    return replace_once(
        text, "        y, k3_trace,", "        y, k3_trace, k3_shared_trace,"
    )

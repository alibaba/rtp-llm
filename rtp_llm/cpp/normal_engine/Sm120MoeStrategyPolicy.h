#pragma once

#include "rtp_llm/cpp/config/RoleTypes.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace rtp_llm {

inline std::string_view safeView(const char* s) noexcept {
    return s ? std::string_view(s) : std::string_view{};
}

// Python startup normalization handles any legacy DSv4 environment aliases; the
// engine only needs whitespace trimming and an empty-to-"auto" fallback.
inline std::string normalizeMoeStrategy(std::string_view requested) {
    const auto first = requested.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string_view::npos) {
        return "auto";
    }
    const auto trimmed = requested.substr(first, requested.find_last_not_of(" \t\n\r\f\v") - first + 1);
    return std::string(trimmed);
}

// Engine-side admission for SM120 MoE strategies and DeepSeek-V4 worker topologies.
// Keep in sync with `validate_sm120_dsv4_worker_profile` in
// `rtp_llm/config/sm120_dsv4_policy.py` and the strategy table in
// `rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/strategy/fp8_fp4.py`.
// Reject unsupported topologies before NCCL collectives can hang.

// DP>1 sm120_a2a_decode PDFUSION requires a phase-padding speculative mode
// (`NormalEngine::isMTPEagle`: MTP, EAGLE, DSPARK, DFLASH) to keep EP peers collective-aligned.
inline bool sm120DecodeNonMtpFusedProfileUnsupported(std::string_view model_type,
                                                     const char*      moe_strategy,
                                                     RoleType         role,
                                                     int64_t          pp_size,
                                                     int64_t          tp_size,
                                                     int64_t          dp_size,
                                                     int64_t          ep_size,
                                                     int64_t          world_size,
                                                     bool             pads_missing_phases,
                                                     bool             local_cp) {
    return model_type == "deepseek_v4" && safeView(moe_strategy) == "sm120_a2a_decode" && role == RoleType::PDFUSION
           && pp_size == 1 && tp_size == 1 && dp_size > 1 && ep_size == dp_size && world_size == dp_size
           && !pads_missing_phases && !local_cp;
}

// sm120_a2a_decode derives collective extents dynamically per round and cannot run in CUDA graphs.
inline bool sm120DecodeRequiresEagerExecution(const char* moe_strategy, bool enable_cuda_graph) {
    return safeView(moe_strategy) == "sm120_a2a_decode" && enable_cuda_graph;
}

// Reject explicit sm120_fused_moe with DP>1 across backends: its WORLD collectives require ranks
// to stay phase-aligned. Unresolved "auto" is filtered in Python strategy selection (physical_dp_size / is_pure_cp).
inline bool sm120FusedMoeDpDecodeMisaligned(const char* moe_strategy, int64_t dp_size) {
    return dp_size > 1 && safeView(moe_strategy) == "sm120_fused_moe";
}

// Pure-CP sm120_fused_moe (TP == EP > 1 on non-DECODE roles; mirrors `is_pure_cp`)
// cannot be captured in CUDA graphs, including unresolved `auto` when the
// worker runs DeepSeek-V4 on SM120.
inline bool sm120CpPrefillRequiresEagerExecution(const char*      moe_strategy,
                                                 std::string_view model_type,
                                                 RoleType         role,
                                                 int64_t          pp_size,
                                                 int64_t          tp_size,
                                                 int64_t          ep_size,
                                                 bool             prefill_cp,
                                                 bool             enable_cuda_graph,
                                                 bool             is_sm120_device = false) {
    const auto view         = safeView(moe_strategy);
    const bool matches_auto = view == "auto" && is_sm120_device && model_type == "deepseek_v4";
    return (view == "sm120_fused_moe" || matches_auto) && enable_cuda_graph && prefill_cp && role != RoleType::DECODE
           && pp_size == 1 && tp_size > 1 && tp_size == ep_size;
}

inline bool sm120CpPrefillRequiresEagerExecution(const char* moe_strategy,
                                                 RoleType    role,
                                                 int64_t     pp_size,
                                                 int64_t     tp_size,
                                                 int64_t     ep_size,
                                                 bool        prefill_cp,
                                                 bool        enable_cuda_graph) {
    return sm120CpPrefillRequiresEagerExecution(
        moe_strategy, "", role, pp_size, tp_size, ep_size, prefill_cp, enable_cuda_graph, false);
}

}  // namespace rtp_llm

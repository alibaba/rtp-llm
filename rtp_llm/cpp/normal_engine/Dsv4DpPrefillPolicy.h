#pragma once

#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/model_utils/Dsv4MoeStrategy.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace rtp_llm {

// Engine-side admission for exact-SM120 DSv4 worker topologies. These rules
// intentionally overlap with the Python startup gate
// `_validate_sm120_dsv4_worker_profile` in rtp_llm/config/server_config_setup.py
// and the auto-selection table in
// rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/strategy/fp8_fp4.py:
// a wrong topology can hang inside NCCL before Python error handling runs, so
// the engine must refuse it independently. Keep all three sites in sync.

// The non-speculative fused sm120_a2a_decode PDFUSION profile is withdrawn: its
// per-step prefill/decode phase vote could not keep DP peers inside the same
// collective order. Startup rejects exactly this profile before warmup or
// graph capture; the same topology with active speculation (MTP) or
// separate PREFILL/DECODE roles stays supported.
inline bool sm120DecodeNonMtpFusedProfileUnsupported(std::string_view model_type,
                                                     const char*      moe_strategy,
                                                     RoleType         role,
                                                     int64_t          pp_size,
                                                     int64_t          tp_size,
                                                     int64_t          dp_size,
                                                     int64_t          ep_size,
                                                     int64_t          world_size,
                                                     bool             speculative,
                                                     bool             local_cp) {
    return model_type == "deepseek_v4" && moe_strategy && std::string_view(moe_strategy) == "sm120_a2a_decode"
           && role == RoleType::PDFUSION && pp_size == 1 && tp_size == 1 && dp_size > 1 && ep_size == dp_size
           && world_size == dp_size && !speculative && !local_cp;
}

// sm120_a2a_decode execution is eager-only: its DP=EP collective protocol
// derives per-round payload extents from exchanged row counts, which captured
// CUDA graphs cannot reproduce. Reject a resolved sm120_a2a_decode profile that
// still enables CUDA graphs at construction, before warmup.
inline bool sm120DecodeRequiresEagerExecution(const char* moe_strategy, bool enable_cuda_graph) {
    return moe_strategy && std::string_view(moe_strategy) == "sm120_a2a_decode" && enable_cuda_graph;
}

// sm120_fused_moe runs its own DP scheduling in Python, but its decode-capable
// WORLD collectives are not batch-aligned across DP ranks. With DP>1 a rank
// still in prefill while a peer starts decode joins a different collective and
// hangs or corrupts. Reject the explicit strategy name. Unresolved "auto" is
// not rejected here: Python strategy selection (is_pure_cp / physical_dp_size)
// decides whether auto may pick this backend, and NormalEngine has no SM120
// detector to pass a hardware flag. PREFILL-only roles never decode, so they
// remain valid.
inline bool sm120FusedMoeDpDecodeMisaligned(const char* moe_strategy, RoleType role, int64_t dp_size) {
    if (!moe_strategy || dp_size <= 1 || role == RoleType::PREFILL) {
        return false;
    }
    return std::string_view(moe_strategy) == "sm120_fused_moe";
}

}  // namespace rtp_llm

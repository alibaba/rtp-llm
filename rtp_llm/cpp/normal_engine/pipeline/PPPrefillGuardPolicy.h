#pragma once

#include <cstdint>

#include "rtp_llm/cpp/config/RoleTypes.h"

namespace rtp_llm {

// Pure policy for the DSV4 PP PREFILL_CP one-token guard in
// PPExecutor::prepareStreams. Extracted so the executable truth table is
// testable without a full engine; production calls THIS function (no copied
// predicate).
//
// The guard bounds the standalone (PDFUSION) qualification recipe. A
// PREFILL-role server may still receive requests that never enter the PD
// path (PrefillRpcServer bypasses PD locally when max_new_tokens==1,
// beams>1, variable beams, num_return_sequences>1 or
// can_use_pd_separation=false), so the multi-token exemption applies only
// when the stream actually took the PD branch: role PREFILL *and* the
// per-request pd_separation flag set by PrefillRpcServer::prepareGenerateInput
// (observed via GenerateStream::queryPdSep()).
inline bool dsv4PrefillCpGuardRejects(bool     compat_enabled,
                                      RoleType role,
                                      bool     stream_pd_separation,
                                      bool     is_fake_stream,
                                      bool     is_perf_test,
                                      int64_t  max_new_tokens) {
    if (!compat_enabled || is_fake_stream || is_perf_test || max_new_tokens == 1) {
        return false;
    }
    const bool actual_pd_prefill = (role == RoleType::PREFILL) && stream_pd_separation;
    return !actual_pd_prefill;
}

}  // namespace rtp_llm

#pragma once

#include <string>

#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

enum class MMProcessorKind { NONE, LOCAL, REMOTE, INVALID };

inline const char* mmProcessorKindName(MMProcessorKind kind) {
    switch (kind) {
        case MMProcessorKind::NONE:
            return "NONE";
        case MMProcessorKind::LOCAL:
            return "LOCAL";
        case MMProcessorKind::REMOTE:
            return "REMOTE";
        case MMProcessorKind::INVALID:
            return "INVALID";
    }
    return "UNKNOWN";
}

inline const char* vitSeparationName(VitSeparation separation) {
    switch (separation) {
        case VitSeparation::VIT_SEPARATION_LOCAL:
            return "LOCAL";
        case VitSeparation::VIT_SEPARATION_ROLE:
            return "ROLE";
        case VitSeparation::VIT_SEPARATION_REMOTE:
            return "REMOTE";
    }
    return "UNKNOWN";
}

// Keep this ownership rule aligned with LanguageCppEngine in rpc_engine.py.
inline bool ownsMultimodalIngress(RoleType role_type, int64_t tp_rank) {
    return tp_rank == 0 && (role_type == RoleType::PDFUSION || role_type == RoleType::PREFILL);
}

inline MMProcessorKind resolveMMProcessorKind(bool          is_multimodal,
                                              VitSeparation vit_separation,
                                              bool          has_local_engine,
                                              RoleType      role_type,
                                              int64_t       tp_rank) {
    if (!is_multimodal || !ownsMultimodalIngress(role_type, tp_rank)) {
        return MMProcessorKind::NONE;
    }
    if (vit_separation == VitSeparation::VIT_SEPARATION_LOCAL) {
        return has_local_engine ? MMProcessorKind::LOCAL : MMProcessorKind::INVALID;
    }
    if (vit_separation == VitSeparation::VIT_SEPARATION_REMOTE) {
        return has_local_engine ? MMProcessorKind::INVALID : MMProcessorKind::REMOTE;
    }
    return MMProcessorKind::INVALID;
}

inline std::string mmProcessorConfigError(VitSeparation vit_separation,
                                          bool          has_local_engine,
                                          RoleType      role_type,
                                          int64_t       tp_rank,
                                          const std::string& model_type) {
    return "invalid multimodal processor config: vit_separation=" + std::string(vitSeparationName(vit_separation))
           + ", has_local_engine=" + (has_local_engine ? "true" : "false") + ", role_type="
           + roleTypeToString(role_type) + ", tp_rank=" + std::to_string(tp_rank) + ", model_type=" + model_type
           + " (most likely cause: LanguageCppEngine did not create mm_process_engine for this"
             " process -- see rtp_llm/async_decoder_engine/rpc_engine.py)";
}

struct MMProcessorDecision {
    MMProcessorKind kind  = MMProcessorKind::NONE;
    std::string     error;

    bool ok() const {
        return error.empty();
    }
};

inline MMProcessorDecision resolveAndLogMMProcessorKind(bool               is_multimodal,
                                                        VitSeparation      vit_separation,
                                                        bool               has_local_engine,
                                                        RoleType           role_type,
                                                        int64_t            tp_rank,
                                                        const std::string& model_type,
                                                        const std::string& entry) {
    MMProcessorDecision decision;
    decision.kind = resolveMMProcessorKind(is_multimodal, vit_separation, has_local_engine, role_type, tp_rank);

    const std::string described = "entry=" + entry + ", vit_separation=" + vitSeparationName(vit_separation)
                                  + ", role_type=" + roleTypeToString(role_type)
                                  + ", tp_rank=" + std::to_string(tp_rank) + ", model_type=" + model_type;
    RTP_LLM_LOG_INFO("multimodal processor decision: %s, has_local_engine=%s, kind=%s",
                     described.c_str(),
                     has_local_engine ? "true" : "false",
                     mmProcessorKindName(decision.kind));
    if (is_multimodal && decision.kind == MMProcessorKind::NONE) {
        RTP_LLM_LOG_WARNING("this process does not own multimodal ingress: %s", described.c_str());
    }
    if (decision.kind == MMProcessorKind::INVALID) {
        decision.error = mmProcessorConfigError(vit_separation, has_local_engine, role_type, tp_rank, model_type);
    }
    return decision;
}

}  // namespace rtp_llm

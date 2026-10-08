#pragma once

#include <cstdint>
#include <string>

#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

struct PDSupportDecision {
    bool        supported;
    const char* reason;
};

PDSupportDecision checkPDSupport(const GenerateInputPB& request);
ErrorInfo         validatePDHandoff(const GenerateInputPB& request);
std::string       masterEnqueuedHandoffUniqueKey(int64_t request_id);

// QueryConverter performs common PB conversion before PD multimodal preprocessing.
ErrorInfo preprocessForPD(std::shared_ptr<GenerateInput>& input, MultimodalProcessor* processor, bool is_mtp_eagle);

}  // namespace rtp_llm

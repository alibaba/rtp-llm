#pragma once

// Multimodal protobuf-to-engine conversions shared by RPC and transport code.

#include <vector>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class MultimodalPbConverter {
public:
    static MultimodalInputsPB inputsToPb(const std::vector<MultimodalInput>& mm_inputs);

    // Decode and validate an inline response.
    static ErrorResult<MultimodalOutput> inlineOutputFromPb(const MultimodalOutputPB& output_pb);

private:
    static void preprocessConfigToPb(MMPreprocessConfigPB* config_pb, const MMPreprocessConfig& config);
};

}  // namespace rtp_llm

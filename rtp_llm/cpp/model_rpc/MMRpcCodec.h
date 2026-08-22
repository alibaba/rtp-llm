#pragma once

// Multimodal wire-to-engine conversions shared by RPC and transport code.

#include <vector>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class MMRpcCodec {
public:
    // The caller supplies the tensor because not every RPC path carries tensor data.
    static MultimodalInput transMMInputElement(const MultimodalInputPB& input_pb, torch::Tensor tensor);

    static MultimodalInputsPB transMMInputsPB(const std::vector<MultimodalInput> mm_inputs);

    // Decode and validate an inline response.
    static ErrorResult<MultimodalOutput> transMMOutput(const MultimodalOutputPB* output_pb);

private:
    static void transMMPreprocessConfig(MMPreprocessConfigPB* config_pb, const MMPreprocessConfig& config);
};

}  // namespace rtp_llm

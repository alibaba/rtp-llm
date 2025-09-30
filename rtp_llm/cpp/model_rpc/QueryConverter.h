#pragma once

#include <unordered_map>
#include <vector>
#include <cstring>
#include <memory>
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {
class QueryConverter {
public:
    static std::shared_ptr<GenerateInput> transQuery(const GenerateInputPB* input);

    static void
    transResponse(GenerateOutputsPB* outputs, const GenerateOutputs* response, const std::string& aux_string);

    static std::vector<MultimodalInput> transMMInput(const MultimodalInputsPB* mm_inputs);

    static MultimodalInputsPB transMMInputsPB(const std::vector<MultimodalInput> mm_inputs);

    static MultimodalOutput transMMOutput(const MultimodalOutputsPB* outputs_pb);

    static std::vector<RoleAddr> getRoleAddrs(const GenerateConfigPB* config_proto);

private:
    static std::shared_ptr<GenerateConfig> transGenerateConfig(const GenerateConfigPB* config_proto);

    static torch::Tensor transTensor(const TensorPB& tensor_pb);

    static void transTensorPB(TensorPB* t, const rtp_llm::Buffer* buffer);

    static void transMMPreprocessConfig(MMPreprocessConfigPB* config_pb, const MMPreprocessConfig config);
};

}  // namespace rtp_llm

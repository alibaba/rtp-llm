#pragma once

#include <unordered_map>
#include <vector>
#include <cstring>
#include <memory>
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {
class QueryConverter {
public:
    static std::shared_ptr<GenerateInput> transQuery(const GenerateInputPB* input);

    static RequestInfo transRequestInfo(const RequestInfoPB& request_info_pb);

    static void transResponse(GenerateOutputsPB*     outputs,
                              const GenerateOutputs* response,
                              bool                   dump_aux_info,
                              const std::string&     aux_string,
                              const int32_t          eos_token_id);

    static std::vector<RoleAddr> getRoleAddrs(const GenerateConfigPB* config_proto);

    static torch::Tensor transTensor(const TensorPB& tensor_pb);

    static void transTensorPB(TensorPB* tensor_pb, const torch::Tensor& tensor);

    static bool useLegacyDenseMtpHandoff(const char* option) {
        return option == nullptr || std::strcmp(option, "0") != 0;
    }

    static void transMtpProposal(GenerateRequestPB* request,
                                 const SpeculativeExecutorStreamOutput& output,
                                 int64_t target_vocab_size,
                                 bool legacy_dense);
    static torch::Tensor transMtpProposalProbs(const GenerateRequestPB& request);

private:
    static std::shared_ptr<GenerateConfig> transGenerateConfig(const GenerateConfigPB* config_proto);

    template<typename Container, typename Accessor>
    static void
    stackBuffersToTensorPB(TensorPB* target_pb, const Container& source_container, Accessor tensor_accessor);

    template<typename T>
    static void
    mergeAndPadTensorsToTensorPB(TensorPB* target_pb, const std::vector<torch::Tensor>& tensors, T pad_value);
};

}  // namespace rtp_llm

#pragma once
#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/engines/NormalEngine.h"
#include "maga_transformer/cpp/proto/model_rpc_service.grpc.pb.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"
#include <iostream>
#include <memory>
#include <string>

namespace rtp_llm {
class ModelRpcServiceImpl: public ModelRpcService::Service {
public:
    ModelRpcServiceImpl(const MagaInitParams&                                                   maga_init_params,
                        const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                        const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights);
    grpc::Status generate_stream(grpc::ServerContext*                  context,
                                 const GenerateInputPB*                request,
                                 grpc::ServerWriter<GenerateOutputPB>* writer) override;
    void addLoRA(const int64_t                                                   lora_id,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights);
    void removeLoRA(const int64_t lora_id);

private:
    std::unique_ptr<NormalEngine> engine_;
};

}  // namespace rtp_llm

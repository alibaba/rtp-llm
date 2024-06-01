#pragma once
#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/proto/model_rpc_service.grpc.pb.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"
#include "kmonitor/client/MetricsReporter.h"
#include <iostream>
#include <memory>
#include <string>

namespace rtp_llm {
class ModelRpcServiceImpl: public ModelRpcService::Service {
public:
    explicit ModelRpcServiceImpl(const EngineInitParams& maga_init_params);
    grpc::Status generate_stream(grpc::ServerContext*                   context,
                                 const GenerateInputPB*                 request,
                                 grpc::ServerWriter<GenerateOutputsPB>* writer) override;
    void addLoRA(const int64_t                                                   lora_id,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                 const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights);
    void removeLoRA(const int64_t lora_id);

private:
    std::unique_ptr<NormalEngine> engine_ = nullptr;
};

}  // namespace rtp_llm

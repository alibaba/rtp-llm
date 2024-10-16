#pragma once
#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/proto/model_rpc_service.grpc.pb.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "kmonitor/client/MetricsReporter.h"
#include "maga_transformer/cpp/multimodal_processor/MultimodalProcessor.h"
#include <iostream>
#include <memory>
#include <string>

namespace rtp_llm {

struct LoraMutex {
    bool alive_;
    std::unique_ptr<std::shared_mutex> mutex_;
};

class ModelRpcServiceImpl: public ModelRpcService::Service {
public:
    ModelRpcServiceImpl() {}
    grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine, std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);
    grpc::Status generate_stream(grpc::ServerContext*                   context,
                                 const GenerateInputPB*                 request,
                                 grpc::ServerWriter<GenerateOutputsPB>* writer) override;

    LoadBalanceInfo getLoadBalanceInfo();

    void addLora(const std::string& adapter_name,
                 const ft::lora::loraLayerWeightsMap& lora_a_weights,
                 const ft::lora::loraLayerWeightsMap& lora_b_weights);

    void removeLora(const std::string& adapter_name);
    std::shared_ptr<EngineBase> getEngine() const { return engine_; }
private:
    std::shared_ptr<EngineBase> engine_ = nullptr;
    std::unique_ptr<MultimodalProcessor> mm_processor_ = nullptr;
};

}  // namespace rtp_llm

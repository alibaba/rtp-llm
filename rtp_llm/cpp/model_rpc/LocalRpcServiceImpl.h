#pragma once

#include <memory>
#include <string>
#include <iostream>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/dataclass/LoadBalance.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"

namespace rtp_llm {

class LocalRpcServiceImpl: public RpcService::Service {
public:
    LocalRpcServiceImpl() {}
    virtual ~LocalRpcServiceImpl() {}
    virtual grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                              std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
        local_server_ = std::make_shared<LocalRpcServer>();
        return local_server_->init(maga_init_params, mm_process_engine, std::move(propose_params));
    }

    grpc::Status GenerateStreamCall(grpc::ServerContext*                context,
                                 const GenerateInputPB*                 request,
                                 grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        return local_server_->GenerateStreamCall(context, request, writer);
    }

    LoadBalanceInfo getLoadBalanceInfo() {
        return local_server_->getLoadBalanceInfo();
    }

    EngineScheduleInfo getEngineScheduleInfo() {
        return local_server_->getEngineScheduleInfo();
    }

    void addLora(const std::string& adapter_name,
                 const rtp_llm::lora::loraLayerWeightsMap& lora_a_weights,
                 const rtp_llm::lora::loraLayerWeightsMap& lora_b_weights) {
        local_server_->addLora(adapter_name, lora_a_weights, lora_b_weights);
    }

    void removeLora(const std::string& adapter_name) {
        local_server_->removeLora(adapter_name);
    }

    std::shared_ptr<EngineBase> getEngine() const { 
        return local_server_->getEngine();
    };

    std::shared_ptr<MultimodalProcessor> getMultimodalProcessor() const { 
        return local_server_->getMultimodalProcessor();
    };

    virtual size_t onflightRequestNum() {
        return local_server_->onflightRequestNum();
    }

    virtual bool ready() {
        return local_server_->ready();
    }

    virtual void stop() {
        if (local_server_) {
            local_server_->stop();
        }
    }

protected:
    std::shared_ptr<LocalRpcServer> local_server_;
};

typedef LocalRpcServiceImpl RpcServiceImpl;

}  // namespace rtp_llm

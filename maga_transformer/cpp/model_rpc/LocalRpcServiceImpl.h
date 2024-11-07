#pragma once

#include <memory>
#include <string>
#include <iostream>
#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/proto/model_rpc_service.grpc.pb.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "maga_transformer/cpp/multimodal_processor/MultimodalProcessor.h"
#include "maga_transformer/cpp/model_rpc/LocalRpcServer.h"

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

    grpc::Status generate_stream(grpc::ServerContext*                   context,
                                 const GenerateInputPB*                 request,
                                 grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        return local_server_->generate_stream(context, request, writer);
    }

    LoadBalanceInfo getLoadBalanceInfo() {
        return local_server_->getLoadBalanceInfo();
    }

    void addLora(const std::string& adapter_name,
                 const ft::lora::loraLayerWeightsMap& lora_a_weights,
                 const ft::lora::loraLayerWeightsMap& lora_b_weights) {
        local_server_->addLora(adapter_name, lora_a_weights, lora_b_weights);
    }

    void removeLora(const std::string& adapter_name) {
        local_server_->removeLora(adapter_name);
    }

    std::shared_ptr<EngineBase> getEngine() const { 
        return local_server_->getEngine();
    };

    virtual size_t onflightRequestNum() {
        return local_server_->onflightRequestNum();
    }

    virtual bool ready() {
        return local_server_->ready();
    }

protected:
    std::shared_ptr<LocalRpcServer> local_server_;
};

typedef LocalRpcServiceImpl RpcServiceImpl;

}  // namespace rtp_llm

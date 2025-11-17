#pragma once

#include <memory>
#include <string>
#include <iostream>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"

namespace rtp_llm {

class LocalRpcServiceImpl: public RpcService::Service {
public:
    LocalRpcServiceImpl() {}
    virtual ~LocalRpcServiceImpl() {}
    virtual grpc::Status init(const EngineInitParams&                                maga_init_params,
                              py::object                                             mm_process_engine,
                              std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
        local_server_ = std::make_shared<LocalRpcServer>();
        return local_server_->init(maga_init_params, mm_process_engine, std::move(propose_params));
    }

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        return local_server_->GenerateStreamCall(context, request, writer);
    }

    ::grpc::Status DistKvCache(::grpc::ServerContext*        context,
                               const ::DistKvCacheRequestPB* request,
                               ::DistKvCacheResponsePB*      response) override {
        return local_server_->DistKvCache(context, request, response);
    }

    ::grpc::Status
    GetWorkerStatus(::grpc::ServerContext* context, const StatusVersionPB* request, WorkerStatusPB* response) override {
        return local_server_->GetWorkerStatus(context, request, response);
    }

    ::grpc::Status
    GetCacheStatus(::grpc::ServerContext* context, const CacheVersionPB* request, CacheStatusPB* response) override {
        return local_server_->GetCacheStatus(context, request, response);
    }

    ::grpc::Status UpdateSchedulerInfo(::grpc::ServerContext*              context,
                                       const UpdateSchedulerInfoRequestPB* request,
                                       EmptyPB*                            response) override {
        return local_server_->UpdateSchedulerInfo(context, request, response);
    }

    ::grpc::Status
    SetLogLevel(::grpc::ServerContext* context, const SetLogLevelRequestPB* request, EmptyPB* response) override {
        return local_server_->SetLogLevel(context, request, response);
    }

    ::grpc::Status
    CheckHealth(::grpc::ServerContext* context, const EmptyPB* request, CheckHealthResponsePB* response) override {
        return local_server_->CheckHealth(context, request, response);
    }

    ::grpc::Status UpdateEplbConfig(::grpc::ServerContext*           context,
                                    const UpdateEplbConfigRequestPB* request,
                                    EmptyPB*                         response) override {
        return local_server_->UpdateEplbConfig(context, request, response);
    }

    ::grpc::Status SetPause(::grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) override {
        return local_server_->SetPause(context, request, response);
    }

    ::grpc::Status SetRestart(::grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) override {
        return local_server_->SetRestart(context, request, response);
    }

    WorkerStatusInfo getWorkerStatusInfo(int64_t latest_finished_version) {
        return local_server_->getWorkerStatusInfo(latest_finished_version);
    }

    KVCacheInfo getCacheStatusInfo(int64_t latest_cache_version, bool need_cache_keys) {
        return local_server_->getCacheStatusInfo(latest_cache_version, need_cache_keys);
    }

    EngineScheduleInfo getEngineScheduleInfo(int64_t latest_finised_version) {
        return local_server_->getEngineScheduleInfo(latest_finised_version);
    }

    void addLora(const std::string&                        adapter_name,
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

    virtual void stop() {
        if (local_server_) {
            local_server_->stop();
        }
    }

    ::grpc::Status MemoryBlockCache(::grpc::ServerContext*             context,
                                    const ::MemoryBlockCacheRequestPB* request,
                                    ::MemoryBlockCacheResponsePB*      response) override {
        return local_server_->MemoryBlockCache(context, request, response);
    }

protected:
    std::shared_ptr<LocalRpcServer> local_server_;
};

typedef LocalRpcServiceImpl RpcServiceImpl;

}  // namespace rtp_llm

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
    void prepareLocalServer() {
        if (!local_server_) {
            local_server_ = std::make_shared<LocalRpcServer>();
        }
    }
    virtual grpc::Status init(const EngineInitParams&                                maga_init_params,
                              std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                              py::object                                             mm_process_engine) {
        prepareLocalServer();
        return local_server_->init(maga_init_params, std::move(propose_params), mm_process_engine);
    }
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             weight_manager,
                      py::object                                             mm_process_engine) {
        (void)weight_manager;
        prepareLocalServer();
        return local_server_->init(maga_init_params, std::move(propose_params), mm_process_engine);
    }

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("GenerateStreamCall");
        }
        return local_server_->GenerateStreamCall(context, request, writer);
    }

    ::grpc::Status
    GetWorkerStatus(::grpc::ServerContext* context, const StatusVersionPB* request, WorkerStatusPB* response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("GetWorkerStatus");
        }
        return local_server_->GetWorkerStatus(context, request, response);
    }

    ::grpc::Status
    UpdateWeights(::grpc::ServerContext* context, const UpdateWeightsRequestPB* request, EmptyPB* response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("UpdateWeights");
        }
        return local_server_->UpdateWeights(context, request, response);
    }

    ::grpc::Status
    GetCacheStatus(::grpc::ServerContext* context, const CacheVersionPB* request, CacheStatusPB* response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("GetCacheStatus");
        }
        return local_server_->GetCacheStatus(context, request, response);
    }

    ::grpc::Status UpdateSchedulerInfo(::grpc::ServerContext*              context,
                                       const UpdateSchedulerInfoRequestPB* request,
                                       EmptyPB*                            response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("UpdateSchedulerInfo");
        }
        return local_server_->UpdateSchedulerInfo(context, request, response);
    }

    ::grpc::Status
    SetLogLevel(::grpc::ServerContext* context, const SetLogLevelRequestPB* request, EmptyPB* response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("SetLogLevel");
        }
        return local_server_->SetLogLevel(context, request, response);
    }

    ::grpc::Status
    StartProfile(::grpc::ServerContext* context, const StartProfileRequestPB* request, EmptyPB* response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("StartProfile");
        }
        return local_server_->StartProfile(context, request, response);
    }

    ::grpc::Status StartProfileInternal(::grpc::ServerContext*               context,
                                        const StartProfileInternalRequestPB* request,
                                        EmptyPB*                             response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("StartProfileInternal");
        }
        return local_server_->StartProfileInternal(context, request, response);
    }

    ::grpc::Status
    CheckHealth(::grpc::ServerContext* context, const EmptyPB* request, CheckHealthResponsePB* response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("CheckHealth");
        }
        return local_server_->CheckHealth(context, request, response);
    }

    ::grpc::Status UpdateEplbConfig(::grpc::ServerContext*           context,
                                    const UpdateEplbConfigRequestPB* request,
                                    EmptyPB*                         response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("UpdateEplbConfig");
        }
        return local_server_->UpdateEplbConfig(context, request, response);
    }

    ::grpc::Status SetPause(::grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("SetPause");
        }
        return local_server_->SetPause(context, request, response);
    }

    ::grpc::Status SetRestart(::grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("SetRestart");
        }
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

    ::grpc::Status ExecuteFunction(::grpc::ServerContext*     context,
                                   const ::FunctionRequestPB* request,
                                   ::FunctionResponsePB*      response) override {
        if (!readyForRegularRpc()) {
            return notReadyStatus("ExecuteFunction");
        }
        return local_server_->ExecuteFunction(context, request, response);
    }

    ::grpc::Status CpuTpBroadcast(::grpc::ServerContext*           context,
                                  const ::CpuTpBroadcastRequestPB* request,
                                  ::CpuTpBroadcastResponsePB*      response) override {
        if (!local_server_) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "local rpc server is initializing");
        }
        return local_server_->CpuTpBroadcast(context, request, response);
    }

protected:
    bool readyForRegularRpc() const {
        return local_server_ && local_server_->getEngine();
    }

    grpc::Status notReadyStatus(const char* method) const {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, std::string(method) + " rejected: engine is initializing");
    }

    std::shared_ptr<LocalRpcServer> local_server_;
};

typedef LocalRpcServiceImpl RpcServiceImpl;

}  // namespace rtp_llm

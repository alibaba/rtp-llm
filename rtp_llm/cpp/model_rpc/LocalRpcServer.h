#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <iostream>
#include "grpc++/grpc++.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/utils/AtomicUtil.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/freeze/AdmissionGate.h"
#include "rtp_llm/cpp/engine_base/freeze/DrainManager.h"
#include "rtp_llm/cpp/cache/KVCachePhysicalMemoryController.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/WorkerStatusInfo.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
#include "rtp_llm/cpp/multimodal_processor/LocalMultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/RemoteMultimodalProcessor.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
class LocalRpcServer {
public:
    LocalRpcServer() {}
    virtual ~LocalRpcServer() {}
    virtual grpc::Status init(const EngineInitParams&                                maga_init_params,
                              std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                              py::object                                             mm_process_engine);

    grpc::Status
    GetWorkerStatus(grpc::ServerContext* context, const ::StatusVersionPB* request, ::WorkerStatusPB* response);

    grpc::Status
    GetCacheStatus(grpc::ServerContext* context, const ::CacheVersionPB* request, ::CacheStatusPB* response);

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status BatchGenerateCall(grpc::ServerContext*        context,
                                   const BatchGenerateInputPB* request,
                                   BatchGenerateOutputsPB*     response);

    grpc::Status CheckHealth(grpc::ServerContext* context, const EmptyPB* request, CheckHealthResponsePB* response);

    grpc::Status UpdateWeights(grpc::ServerContext* context, const UpdateWeightsRequestPB* request, EmptyPB* response);

    grpc::Status
    UpdateEplbConfig(grpc::ServerContext* context, const UpdateEplbConfigRequestPB* request, EmptyPB* response);

    grpc::Status SetPause(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response);

    grpc::Status SetRestart(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response);

    grpc::Status FreezeServing(grpc::ServerContext* context, const FreezeRequestPB* request, EmptyPB* response);

    grpc::Status ResumeServing(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response);

    grpc::Status
    GetFreezeStatus(grpc::ServerContext* context, const EmptyPB* request, FreezeStatusResponsePB* response);

    grpc::Status SetLogLevel(grpc::ServerContext* context, const SetLogLevelRequestPB* request, EmptyPB* response);

    grpc::Status StartProfile(grpc::ServerContext* context, const StartProfileRequestPB* request, EmptyPB* response);

    grpc::Status
    StartProfileInternal(grpc::ServerContext* context, const StartProfileInternalRequestPB* request, EmptyPB* response);

    grpc::Status
    UpdateSchedulerInfo(grpc::ServerContext* context, const UpdateSchedulerInfoRequestPB* request, EmptyPB* response);

    KVCacheInfo getCacheStatusInfo(int64_t latest_cache_version, bool need_cache_keys);

    WorkerStatusInfo getWorkerStatusInfo(int64_t latest_finished_version);

    std::shared_ptr<EngineBase> getEngine() const {
        return engine_;
    }
    std::shared_ptr<MultimodalProcessor> getMultimodalProcessor() const {
        return mm_processor_;
    }

    int64_t tpSize() const {
        return maga_init_params_.parallelism_config.tp_size;
    }

    virtual size_t onflightRequestNum();

    void stop() {
        (void)engine_->stop();
    }

    virtual EngineScheduleInfo getEngineScheduleInfo(int64_t latest_finised_version);

    void reportWorkerStatusTime(int64_t request_begin_time_us, int64_t request_after_lb_time_us);

    void reportCacheStatusTime(int64_t request_begin_time_us);

    ::grpc::Status
    ExecuteFunction(::grpc::ServerContext* context, const ::FunctionRequestPB* request, ::FunctionResponsePB* response);

public:
    typedef grpc::internal::WriterInterface<GenerateOutputsPB> WriterInterface;

protected:
    // M4 unified admission gate (constraint C5): every inference entry calls
    // this first; non-RUNNING states get a retryable ENGINE_UNAVAILABLE.
    grpc::Status checkAdmission() const {
        return admission_gate_ ? admission_gate_->check() : grpc::Status::OK;
    }

    // Wire the freeze/resume FreezeHooks (M3 drain counters, M5 KV memory,
    // M6 weights, engine quiesce) into engine_->freezeController().
    void installFreezeHooks();

    grpc::Status serializeErrorMsg(const std::string& request_key, ErrorInfo error_info);
    grpc::Status pollStreamOutput(grpc::ServerContext*             context,
                                  const std::string&               request_key,
                                  WriterInterface*                 writer,
                                  std::shared_ptr<GenerateStream>& stream);

    // Shared helpers for single and batch paths
    ErrorInfo prepareInput(const GenerateInputPB& input_pb, std::shared_ptr<GenerateInput>& output);
    ErrorInfo collectStreamOutput(grpc::ServerContext*                  context,
                                  std::shared_ptr<GenerateStream>&      stream,
                                  const std::shared_ptr<GenerateInput>& input,
                                  GenerateOutputs&                      last_outputs);

protected:
    std::shared_ptr<EngineBase>           engine_;
    std::shared_ptr<AdmissionGate>        admission_gate_;
    std::shared_ptr<DrainManager>         drain_manager_;
    std::shared_ptr<TmsBackend>           tms_backend_;
    std::shared_ptr<MultimodalProcessor>  mm_processor_;
    EngineInitParams                      maga_init_params_;
    ProposeModelEngineInitParams*         propose_maga_init_params_;
    kmonitor::MetricsReporterPtr          metrics_reporter_;
    std::atomic<size_t>                   onflight_requests_{0};
    std::shared_ptr<RpcServerRuntimeMeta> meta_;
    py::object                            weight_manager_;
    std::shared_ptr<BroadcastManager>     profile_broadcaster_;
};

}  // namespace rtp_llm

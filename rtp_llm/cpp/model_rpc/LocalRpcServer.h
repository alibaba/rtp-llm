#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <iostream>
#include "grpc++/grpc++.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/utils/AtomicUtil.h"
#include "rtp_llm/cpp/engine_base/WorkerStatusInfo.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
#include "rtp_llm/cpp/multimodal_processor/LocalMultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/RemoteMultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/AotMultiModalProcessor.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
class LocalRpcServer {
public:
    LocalRpcServer() {}
    virtual ~LocalRpcServer() {}
    virtual grpc::Status init(const EngineInitParams&                                maga_init_params,
                              py::object                                             mm_process_engine,
                              std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);

    grpc::Status
    GetWorkerStatus(grpc::ServerContext* context, const ::StatusVersionPB* request, ::WorkerStatusPB* response);

    grpc::Status
    GetCacheStatus(grpc::ServerContext* context, const ::CacheVersionPB* request, ::CacheStatusPB* response);

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status CheckHealth(grpc::ServerContext* context, const EmptyPB* request, CheckHealthResponsePB* response);

    grpc::Status UpdateWeights(grpc::ServerContext* context, const UpdateWeightsRequestPB* request, EmptyPB* response);

    grpc::Status
    UpdateEplbConfig(grpc::ServerContext* context, const UpdateEplbConfigRequestPB* request, EmptyPB* response);

    grpc::Status SetPause(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response);

    grpc::Status SetRestart(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response);

    grpc::Status SetLogLevel(grpc::ServerContext* context, const SetLogLevelRequestPB* request, EmptyPB* response);

    grpc::Status
    UpdateSchedulerInfo(grpc::ServerContext* context, const UpdateSchedulerInfoRequestPB* request, EmptyPB* response);

    KVCacheInfo getCacheStatusInfo(int64_t latest_cache_version, bool need_cache_keys);

    WorkerStatusInfo getWorkerStatusInfo(int64_t latest_finished_version);

    void addLora(const std::string&                        adapter_name,
                 const rtp_llm::lora::loraLayerWeightsMap& lora_a_weights,
                 const rtp_llm::lora::loraLayerWeightsMap& lora_b_weights);

    void removeLora(const std::string& adapter_name);

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

    ::grpc::Status BroadcastTp(::grpc::ServerContext*        context,
                               const ::BroadcastTpRequestPB* request,
                               ::BroadcastTpResponsePB*      response);

public:
    typedef grpc::internal::WriterInterface<GenerateOutputsPB> WriterInterface;

protected:
    grpc::Status serializeErrorMsg(const std::string& request_key, ErrorInfo error_info);
    grpc::Status pollStreamOutput(grpc::ServerContext*             context,
                                  const std::string&               request_key,
                                  WriterInterface*                 writer,
                                  std::shared_ptr<GenerateStream>& stream);

protected:
    std::shared_ptr<EngineBase>           engine_;
    std::shared_ptr<MultimodalProcessor>  mm_processor_;
    EngineInitParams                      maga_init_params_;
    ProposeModelEngineInitParams*         propose_maga_init_params_;
    kmonitor::MetricsReporterPtr          metrics_reporter_;
    std::atomic<size_t>                   onflight_requests_{0};
    std::shared_ptr<RpcServerRuntimeMeta> meta_;
    py::object                            weight_manager_;
};

}  // namespace rtp_llm

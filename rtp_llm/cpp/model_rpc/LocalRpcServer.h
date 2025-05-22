#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <iostream>
#include "grpc++/grpc++.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/utils/AtomicUtil.h"
#include "rtp_llm/cpp/dataclass/LoadBalance.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/utils/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/dataclass/EngineScheduleInfo.h"
#include "rtp_llm/cpp/multimodal_processor/LocalMultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/RemoteMultimodalProcessor.h"
namespace rtp_llm {
class LocalRpcServer {
public:
    LocalRpcServer() {}
    virtual ~LocalRpcServer() {}
    virtual grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                              std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);
    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    LoadBalanceInfo getLoadBalanceInfo();

    void addLora(const std::string& adapter_name,
                 const rtp_llm::lora::loraLayerWeightsMap& lora_a_weights,
                 const rtp_llm::lora::loraLayerWeightsMap& lora_b_weights);

    void removeLora(const std::string& adapter_name);
    
    std::shared_ptr<EngineBase> getEngine() const { return engine_; }
    std::shared_ptr<MultimodalProcessor> getMultimodalProcessor() const { return mm_processor_; }

    int64_t tpSize() const {
        return maga_init_params_.gpt_init_parameter.tp_size_;
    }

    virtual size_t onflightRequestNum();

    bool ready() {
        return true;
    }

    void stop() {
        (void)engine_->stop();
    }

    virtual EngineScheduleInfo getEngineScheduleInfo() {
        return EngineScheduleInfo();
    }

public:
    typedef grpc::internal::WriterInterface<GenerateOutputsPB> WriterInterface;

protected:
    grpc::Status serializeErrorMsg(const std::string& request_key, ErrorInfo error_info);
    grpc::Status pollStreamOutput(grpc::ServerContext*              context,
                                  const std::string&                request_key,
                                  WriterInterface*                  writer,
                                  std::shared_ptr<GenerateStream>&  stream);

protected:
    std::shared_ptr<EngineBase>             engine_;
    std::shared_ptr<MultimodalProcessor>    mm_processor_;
    EngineInitParams                        maga_init_params_;
    kmonitor::MetricsReporterPtr            metrics_reporter_;
    std::atomic<size_t>                     onflight_requests_{0};
    ProposeModelEngineInitParams*           propose_maga_init_params_;
};

}  // namespace rtp_llm

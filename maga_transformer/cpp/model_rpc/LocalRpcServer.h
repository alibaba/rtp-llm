#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <iostream>
#include "grpc++/grpc++.h"
#include "kmonitor/client/MetricsReporter.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/cache/KVCacheBlockAddr.h"
#include "maga_transformer/cpp/utils/ErrorCode.h"
#include "maga_transformer/cpp/proto/model_rpc_service.grpc.pb.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "kmonitor/client/MetricsReporter.h"
#include "maga_transformer/cpp/multimodal_processor/MultimodalProcessor.h"
#include "maga_transformer/cpp/disaggregate/cache_store/NormalCacheStore.h"

namespace rtp_llm {

struct LoraMutex {
    bool alive_;
    std::unique_ptr<std::shared_mutex> mutex_;
};

class AtomicGuard {
public:
    AtomicGuard(std::atomic<size_t>& atomic_var)
        : atomic_var_(atomic_var) {
        atomic_var_++;
    }

    ~AtomicGuard() {
        atomic_var_--;
    }

private:
    std::atomic<size_t>& atomic_var_;
};

class GenerateContext {
public:
    GenerateContext(int64_t request_id) : request_id(request_id) {}
    virtual ~GenerateContext() {
        if (stream && !stream->finished() && !stream->stopped()) {
            stream->cancel();
        }
    }

public:
    int64_t request_id;
    std::shared_ptr<GenerateStream> stream;
};

class LocalRpcServer {
public:
    LocalRpcServer() {}
    virtual ~LocalRpcServer() {}
    virtual grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                              std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);
    grpc::Status generate_stream(grpc::ServerContext*                   context,
                                 const GenerateInputPB*                 request,
                                 grpc::ServerWriter<GenerateOutputsPB>* writer);

    LoadBalanceInfo getLoadBalanceInfo();

    void addLora(const std::string& adapter_name,
                 const ft::lora::loraLayerWeightsMap& lora_a_weights,
                 const ft::lora::loraLayerWeightsMap& lora_b_weights);

    void removeLora(const std::string& adapter_name);
    
    std::shared_ptr<EngineBase> getEngine() const { return engine_; }

    virtual size_t onflightRequestNum();

    bool ready() {
        return true;
    }

protected:
    grpc::Status serializeErrorMsg(int64_t request_id, ErrorCode error_code, const std::string& error_msg);
    grpc::Status pollStreamOutput(grpc::ServerContext*                   context,
                                  int64_t                                request_id,
                                  grpc::internal::WriterInterface<GenerateOutputsPB>* writer,
                                  std::shared_ptr<GenerateStream>&       stream);
    
    void reportMetrics(RPCMetricsCollector* collector);

protected:
    std::shared_ptr<EngineBase> engine_;
    std::unique_ptr<MultimodalProcessor> mm_processor_;
    EngineInitParams maga_init_params_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::atomic<size_t> onflight_requests_{0};
};

}  // namespace rtp_llm

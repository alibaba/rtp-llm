#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/WorkerStatusInfo.h"
#include "rtp_llm/cpp/cache/KvCacheInfo.h"
#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"

namespace th = torch;

namespace rtp_llm {

class RtpLLMOp: public th::jit::CustomClassHolder {
public:
    RtpLLMOp();
    ~RtpLLMOp();

    void init(py::object model, py::object engine_config, py::object vit_config, py::object mm_process_engine, py::object propose_model, py::object token_processor);
    void stop();
    void startHttpServer(py::object model_weights_loader,
                         py::object lora_infos,
                         py::object gang_info,
                         py::object tokenizer,
                         py::object render);
    void addLora(const std::string& adapter_name, py::object lora_a_weights, py::object lora_b_weights);
    void removeLora(const std::string& adapter_name);
    EngineScheduleInfo getEngineScheduleInfo(int64_t latest_finished_version);
    WorkerStatusInfo   getWorkerStatusInfo(int64_t latest_finished_version);
    KVCacheInfo        getCacheStatusInfo(int64_t latest_cache_version);
    // currently only used in BatchDecodeScheduler
    void updateSchedulerInfo(const std::string& scheduler_info);
    bool updateEplbConfig(const EPLBConfig& config);
    void pause();
    void restart();
    void detachPhysicalMemory();
    void attachPhysicalMemory();

private:
    void                                                   _init(int64_t                                                model_rpc_port,
                                                                 int64_t                                                http_port,
                                                                 const EngineInitParams                        maga_init_params,
                                                                 py::object                                             mm_process_engine,
                                                                 std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                                                                 py::object                                             token_processor);
    EngineInitParams                              initModel(py::object model, py::object engine_config, py::object vit_config);
    std::unique_ptr<ProposeModelEngineInitParams> initProposeModel(py::object propose_model, const EngineInitParams& base_params);
    void initRPCServer(const EngineInitParams                        maga_init_params,
                       py::object                                             mm_process_engine,
                       std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                       py::object                                             token_processor);

private:
    std::unique_ptr<RpcServiceImpl> model_rpc_service_;
    std::shared_ptr<HttpApiServer>  http_server_;
    std::unique_ptr<grpc::Server>            grpc_server_;
    std::thread                              grpc_server_thread_;
    std::atomic<bool>                        is_server_ready_{false};
    std::atomic<bool>                        is_server_shutdown_{false};
    size_t                                   model_id_ = 0;
};

void registerRtpLLMOp(const py::module& m);

}

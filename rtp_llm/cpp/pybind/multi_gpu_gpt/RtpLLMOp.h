#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/dataclass/EngineInitParameter.h"
#include "rtp_llm/cpp/dataclass/WorkerStatusInfo.h"
#include "rtp_llm/cpp/dataclass/KvCacheInfo.h"
#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"

namespace th = torch;

namespace torch_ext {

class RtpLLMOp: public th::jit::CustomClassHolder {
public:
    RtpLLMOp();
    ~RtpLLMOp();

    void init(py::object model, py::object mm_process_engine, py::object propose_model, py::object token_processor);
    void stop();
    void startHttpServer(py::object model_weights_loader,
                         py::object lora_infos,
                         py::object gang_info,
                         py::object tokenizer,
                         py::object render);
    void addLora(const std::string& adapter_name, py::object lora_a_weights, py::object lora_b_weights);
    void removeLora(const std::string& adapter_name);
    rtp_llm::EngineScheduleInfo getEngineScheduleInfo(int64_t latest_finished_version);
    rtp_llm::WorkerStatusInfo   getWorkerStatusInfo(int64_t latest_finished_version);
    rtp_llm::KVCacheInfo        getCacheStatusInfo(int64_t latest_cache_version);
    // currently only used in BatchDecodeScheduler
    void updateSchedulerInfo(const std::string& scheduler_info);
    bool updateEplbConfig(const rtp_llm::EplbConfig& config);
    void pause();
    void restart();

private:
    void                                                   _init(int64_t                                                model_rpc_port,
                                                                 int64_t                                                http_port,
                                                                 const rtp_llm::EngineInitParams                        maga_init_params,
                                                                 py::object                                             mm_process_engine,
                                                                 std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                                                 py::object                                             token_processor);
    rtp_llm::EngineInitParams                              initModel(py::object model);
    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> initProposeModel(py::object propose_model);
    void initRPCServer(const rtp_llm::EngineInitParams                        maga_init_params,
                       py::object                                             py_handler,
                       py::object                                             mm_process_engine,
                       std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                       py::object                                             token_processor);

private:
    std::unique_ptr<rtp_llm::RpcServiceImpl> model_rpc_service_;
    std::shared_ptr<rtp_llm::HttpApiServer>  http_server_;
    std::unique_ptr<grpc::Server>            grpc_server_;
    std::thread                              grpc_server_thread_;
    std::atomic<bool>                        is_server_ready_{false};
    std::atomic<bool>                        is_server_shutdown_{false};
    size_t                                   model_id_ = 0;
};

void registerRtpLLMOp(const py::module& m);

}  // namespace torch_ext

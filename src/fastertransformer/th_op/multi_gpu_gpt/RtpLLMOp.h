#pragma once

#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "maga_transformer/cpp/model_rpc/ModelRpcServer.h"
#include "maga_transformer/cpp/HttpApiServer.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

class RtpLLMOp: public th::jit::CustomClassHolder {
public:
    RtpLLMOp();
    ~RtpLLMOp();
    void init(py::object model, py::object mm_process_engine, py::object propose_model, py::object token_processor);
    void addLora(const std::string& adapter_name, py::object lora_a_weights, py::object lora_b_weights);
    void removeLora(const std::string& adapter_name);
    void stop();
    void _init(int64_t model_rpc_port,
               int64_t http_port,
               const rtp_llm::EngineInitParams maga_init_params,
               py::object mm_process_engine,
               std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
               py::object token_processor);
    rtp_llm::LoadBalanceInfo getLoadBalanceInfo();

    // std::shared_ptr<rtp_llm::GenerateStream> forward(std::shared_ptr<rtp_llm::GenerateInput> query);

private:
    rtp_llm::EngineInitParams initModel(py::object model);
    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> initProposeModel(py::object propose_model);

private:
    std::unique_ptr<rtp_llm::HttpApiServer>       http_server_;
    std::unique_ptr<rtp_llm::ModelRpcServiceImpl> model_rpc_server_ = nullptr;
    std::unique_ptr<grpc::Server>                 grpc_server_ = nullptr;
    std::thread                                   grpc_server_thread_;
    std::atomic<bool>                             is_server_ready_{false};
    std::atomic<bool>                             is_server_shutdown_{false};
    kmonitor::MetricsReporterPtr                  metric_reporter_ = nullptr;
};

void registerRtpLLMOp(const py::module& m);

}  // namespace torch_ext

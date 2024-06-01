#pragma once

#include "grpc++/grpc++.h"
// #include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/model_rpc/ModelRpcServer.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {
class RtpLLMOp: public th::jit::CustomClassHolder {
public:
    RtpLLMOp();
    ~RtpLLMOp();
    void init(const ft::GptInitParameter& gpt_init_parameter, py::object layer_weights, py::object weights);

    void addLoRA(const int64_t lora_id, py::object lora_a_weights, py::object lora_b_weights);
    void removeLoRA(const int64_t lora_id);
    void stop();
    void _init(int64_t model_rpc_port, const rtp_llm::EngineInitParams maga_init_params);

    // std::shared_ptr<rtp_llm::GenerateStream> forward(std::shared_ptr<rtp_llm::GenerateInput> query);

private:
    std::unique_ptr<rtp_llm::ModelRpcServiceImpl> model_rpc_server_;
    std::unique_ptr<grpc::Server>                 grpc_server_;
    std::thread                                   grpc_server_thread_;
    std::atomic<bool>                             is_server_ready_{false};
    std::atomic<bool>                             is_server_shutdown_{false};
};

void registerRtpLLMOp(const py::module& m);

}  // namespace torch_ext

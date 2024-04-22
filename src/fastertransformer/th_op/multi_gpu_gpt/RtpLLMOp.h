#pragma once

#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/model_rpc/ModelRpcServer.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {
class RtpLLMOp: public th::jit::CustomClassHolder {
public:
    RtpLLMOp();
    ~RtpLLMOp();
    void init(const c10::intrusive_ptr<GptInitParameter>&                     maga_init_params,
              const std::vector<std::unordered_map<std::string, th::Tensor>>& layer_weights,
              const c10::Dict<std::string, th::Tensor>&                       weights);
    void addLoRA(const int64_t                                                   lora_id,
                 const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                 const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights);
    void removeLoRA(const int64_t lora_id);
    void stop();
    void _init(const rtp_llm::MagaInitParams                                          maga_init_params,
               const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layer_weights,
               const std::unordered_map<std::string, ft::ConstBufferPtr>              weights);
    // std::shared_ptr<rtp_llm::GenerateStream> forward(std::shared_ptr<rtp_llm::GenerateInput> query);

private:
    std::unique_ptr<rtp_llm::ModelRpcServiceImpl> model_rpc_server_;
    std::unique_ptr<grpc::Server> grpc_server_;
    std::thread                   grpc_server_thread_;
    std::atomic<bool>             is_server_shutdown_{false};
};

}  // namespace torch_ext

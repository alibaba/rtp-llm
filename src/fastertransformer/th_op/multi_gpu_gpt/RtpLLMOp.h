#pragma once

#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/engines/NormalEngine.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {
class RtpLLMOp: public th::jit::CustomClassHolder {
public:
    RtpLLMOp();
    void init(const c10::intrusive_ptr<GptInitParameter>&                     maga_init_params,
              const std::vector<std::unordered_map<std::string, th::Tensor>>& layer_weights,
              const c10::Dict<std::string, th::Tensor>&                       weights);
    void _init(const rtp_llm::MagaInitParams                                          maga_init_params,
               const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layer_weights,
               const std::unordered_map<std::string, ft::ConstBufferPtr>              weights);
    ~RtpLLMOp();

    void stop();
    // std::shared_ptr<rtp_llm::GenerateStream> forward(std::shared_ptr<rtp_llm::GenerateInput> query);

private:
    std::unique_ptr<grpc::Server> server_;
    std::thread                   server_thread_;
};

}  // namespace torch_ext

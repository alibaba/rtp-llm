#pragma once

#include <vector>
#include "maga_transformer/cpp/embedding_engine/handlers/HandlerFactory.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingEngine.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

class EmbeddingOpOutput: public th::jit::CustomClassHolder {
public:
    th::Tensor output;
};

class RtpEmbeddingOp: public th::jit::CustomClassHolder {
public:
    RtpEmbeddingOp(const c10::intrusive_ptr<GptInitParameter> gpt_init_params);
    ~RtpEmbeddingOp();
    void init(const c10::intrusive_ptr<GptInitParameter>&                     maga_init_params,
              const std::vector<std::unordered_map<std::string, th::Tensor>>& layer_weights,
              const c10::Dict<std::string, th::Tensor>&                       weights);
    void stop();
    
    th::Tensor handle(th::Tensor token_ids, th::Tensor token_type_ids, th::Tensor input_lengths, int64_t request_id);
    std::vector<std::string> handlerTensorInfo();

private:
    std::unique_ptr<rtp_llm::EmbeddingEngine> embedding_engine_;
    std::unique_ptr<rtp_llm::HandlerBase> handler_;
    std::unordered_map<std::string, ft::ConstBufferPtr>              global_weights_;
    std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layer_weights_;

    std::atomic<bool>             is_server_shutdown_{false};
};

}  // namespace torch_ext
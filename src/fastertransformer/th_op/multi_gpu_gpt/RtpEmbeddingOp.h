#pragma once

#include <pybind11/pytypes.h>
#include <vector>
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingEngine.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#include "maga_transformer/cpp/embedding_engine/arpc/ArpcServiceCreator.h"
#include "maga_transformer/cpp/embedding_engine/arpc/ArpcServerWrapper.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

class EmbeddingOpOutput: public th::jit::CustomClassHolder {
public:
    th::Tensor output;
};

class RtpEmbeddingOp: public th::jit::CustomClassHolder {
public:
    RtpEmbeddingOp();
    ~RtpEmbeddingOp();
    void init(py::object model);
    void stop();
    void startRpcServer(const ft::GptInitParameter& gpt_init_params, py::object py_render, kmonitor::MetricsReporterPtr reporter);

    th::Tensor decode(th::Tensor token_ids, th::Tensor token_type_ids, th::Tensor input_lengths, int64_t request_id);    

private:
    // need to be shared to pass into rpc service
    std::shared_ptr<rtp_llm::EmbeddingEngine> embedding_engine_;
    std::unique_ptr<rtp_llm::ArpcServerWrapper> embedding_rpc_service_;

    std::atomic<bool>            is_server_shutdown_{false};
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;
};

void registerRtpEmbeddingOp(const py::module& m);

}  // namespace torch_ext
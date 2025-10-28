#pragma once

#include <optional>
#include <pybind11/pytypes.h>
#include <vector>
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/embedding_engine/arpc/ArpcServiceCreator.h"
#include "rtp_llm/cpp/embedding_engine/arpc/ArpcServerWrapper.h"
#include "rtp_llm/cpp/multimodal_processor/LocalMultimodalProcessor.h"
#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/model_rpc/EmbeddingRpcServer.h"
namespace th = torch;

namespace rtp_llm {

class EmbeddingOpOutput: public th::jit::CustomClassHolder {
public:
    th::Tensor output;
};

class RtpEmbeddingOp: public th::jit::CustomClassHolder {
public:
    RtpEmbeddingOp();
    ~RtpEmbeddingOp();
    void       init(py::object model, py::object mm_process_engine);
    void       stop();
    py::object decode(th::Tensor                   token_ids,
                      th::Tensor                   token_type_ids,
                      th::Tensor                   input_lengths,
                      int64_t                      request_id,
                      std::vector<MultimodalInput> multimodal_inputs = {});

private:
    void startRpcServer(const GptInitParameter&              gpt_init_params,
                        py::object                           py_render,
                        py::object                           py_tokenizer,
                        kmonitor::MetricsReporterPtr         reporter,
                        std::shared_ptr<MultimodalProcessor> mm_processor);

    void startHttpServer(std::shared_ptr<EmbeddingEngine>     embedding_engine,
                         std::shared_ptr<MultimodalProcessor> mm_processor,
                         const EngineInitParams&              params,
                         py::object                           py_render);

private:
    void initRPCServer(const EngineInitParams               maga_init_params,
                       std::shared_ptr<EmbeddingEngine>     embedding_engine,
                       py::object                           mm_process_engine,
                       std::shared_ptr<MultimodalProcessor> mm_processor);
    // need to be shared to pass into rpc service
    std::shared_ptr<EmbeddingEngine>     embedding_engine_;
    std::shared_ptr<MultimodalProcessor> mm_processor_ = nullptr;
    std::unique_ptr<ArpcServerWrapper>   embedding_rpc_service_;
    std::shared_ptr<HttpApiServer>       http_server_;

    std::atomic<bool>            is_server_shutdown_{false};
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;

    std::unique_ptr<EmbeddingRpcServiceImpl> embedding_grpc_service_;
    std::unique_ptr<grpc::Server>            grpc_server_;
    std::thread                              grpc_server_thread_;

    std::atomic<bool> is_server_ready_{false};
};

void registerRtpEmbeddingOp(const py::module& m);

}  // namespace rtp_llm

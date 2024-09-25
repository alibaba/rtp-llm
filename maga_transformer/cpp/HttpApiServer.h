#pragma once
#include <torch/python.h>
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/http_server/http_server/HttpServer.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"

namespace rtp_llm {

class TokenizerEncodeResponse;

class EmbeddingEndpoint {
public:
    EmbeddingEngine(std::shared_ptr<EmbeddingEngine> embedding_engine, py::object py_render):
        py_render_(py_render), embedding_engine_(embedding_engine) {
    }
    std::pair<EngineOutputs, std::optional<EngineOutputs>> handle(const std::string& body);
private:
    py::object py_render_;
    std::shared_ptr<EmbeddingEngine> embedding_engine_;
};

class Pipeline {
public:
    Pipeline(py::object token_processor): token_processor_(token_processor) {}
    std::string        decode(std::vector<int> token_ids);
    std::vector<int>   encode(std::string prompt);
    static std::string format_response(std::string generate_texts, const GenerateOutputs* generate_outputs);
    std::shared_ptr<TokenizerEncodeResponse> tokenizer(const std::string& prompt);

private:
    py::object token_processor_;
};

class HttpApiServer: public std::enable_shared_from_this<HttpApiServer> {
public:
    // normal engine
    HttpApiServer(std::shared_ptr<EngineBase> engine,
                  const ft::GptInitParameter& params,
                  py::object                  token_processor):
            engine_(engine), params_(params), pipeline_(Pipeline(token_processor)) {

        init_controller(params);
    }

    // embedding engine
    HttpApiServer(std::shared_ptr<EmbeddingEngine> embedding_engine,
                  const ft::GptInitParameter&      gpt_init_params,
                  py::object                       py_render):
            params_(params), embedding_endpoint_(EmbeddingEndpoint(embedding_engine, py_render)) {

        init_controller(params);
    }

    ~HttpApiServer();


    bool start(std::string addrSpec) {
        return http_server_.Start(addrSpec);
    }
    void               stop();
    void               registerResponses();
    static std::string SseResponse(std::string& response) {
        return "data: " + response + "\n\n";
    }
    bool isStopped() const {
        return is_stopped_.load();
    }

private:
    void init_controller(const ft::GptInitParameter& params);

private:
    bool registerRoot();
    bool registerHealth();
    bool registerEmbedding();
    bool registerV1Model();
    bool registerSetLogLevel();
    bool registerTokenizerEncode();
    bool registerInference();
    bool registerInferenceInternal();
    bool registerWorkerStatus();

private:
    http_server::HttpServer http_server_;

    std::shared_ptr<EngineBase> engine_;
    ft::GptInitParameter params_;
    std::optional<EmbeddingEndpoint> embedding_endpoint_;

    Pipeline                               pipeline_;
    std::shared_ptr<ConcurrencyController> controller_;
    std::atomic<bool>                      is_stopped_{false};
    std::atomic<int32_t>                   active_request_count_{0};
};

}  // namespace rtp_llm

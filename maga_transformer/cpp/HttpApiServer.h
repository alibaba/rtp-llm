#pragma once
#include <torch/python.h>
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/http_server/http_server/HttpServer.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"

namespace rtp_llm {

class Pipeline {
public:
    Pipeline(py::object token_processor): token_processor_(token_processor) {}
    std::string decode(std::vector<int> token_ids);
    std::vector<int> encode(std::string prompt);
    static std::string format_response(std::string generate_texts, const GenerateOutputs* generate_outputs);
private:
    py::object token_processor_;
};

class HttpApiServer {
public:
    HttpApiServer(std::shared_ptr<EngineBase> engine, ft::GptInitParameter params, py::object token_processor) :
        engine_(engine), params_(params), pipeline_(Pipeline(token_processor)){}

    bool start(std::string addrSpec) { return http_server_.Start(addrSpec); }
    void stop() { http_server_.Stop(); }
    void registerResponses();
    static std::string SseResponse(std::string& response) {
        return "data: " + response + "\n\n";
    }
private:
    http_server::HttpServer http_server_;
    // attach params and engine to HttpApiServer in RtpLLMOp.cc
    std::shared_ptr<EngineBase> engine_;
    ft::GptInitParameter params_;
    Pipeline pipeline_;
};

} // namespace rtp_llm

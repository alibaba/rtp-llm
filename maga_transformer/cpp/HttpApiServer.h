#pragma once
#include <torch/python.h>
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/http_server/http_server/HttpServer.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/utils/ConcurrencyControllerUtil.h"
#include "autil/EnvUtil.h"

namespace rtp_llm {

class TokenizerEncodeResponse;

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
    HttpApiServer(std::shared_ptr<EngineBase> engine, ft::GptInitParameter params, py::object token_processor):
        engine_(engine), params_(params), pipeline_(Pipeline(token_processor)) {

        bool block = autil::EnvUtil::getEnv("CONCURRENCY_WITH_BLOCK", false);
        if (params.tp_rank_ == 0) {
            int limit = autil::EnvUtil::getEnv("CONCURRENCY_LIMIT", 32);
            FT_LOG_INFO("CONCURRENCY_LIMIT to %d", limit);
            controller_ = std::make_shared<ConcurrencyController>(limit, block);
        } else /* if (params.tp_size_ != 1) */ {
            FT_LOG_INFO("use gang cluster and is worker, set CONCURRENCY_LIMIT to 99");
            controller_ = std::make_shared<ConcurrencyController>(99, block);
        }
    }

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
    bool registerRoot();
    bool registerHealth();
    bool registerV1Model();
    bool registerSetDebugLog();
    bool registerSetDebugPrint();
    bool registerTokenizerEncode();
    bool registerInferenceInternal();
    bool registerWorkerStatus();

private:
    http_server::HttpServer http_server_;

    // attach params and engine to HttpApiServer in RtpLLMOp.cc
    std::shared_ptr<EngineBase> engine_;
    ft::GptInitParameter        params_;

    Pipeline                               pipeline_;
    std::shared_ptr<ConcurrencyController> controller_;
    std::atomic<bool>                      is_stopped_{false};
};

}  // namespace rtp_llm

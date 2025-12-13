#pragma once

#include "autil/AtomicCounter.h"

#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/api_server/openai/OpenaiEndpoint.h"

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"

#include "rtp_llm/cpp/api_server/ConcurrencyControllerUtil.h"
#include "rtp_llm/cpp/api_server/ApiServerMetrics.h"

namespace rtp_llm {

class ChatService {
public:
    ChatService(const std::shared_ptr<EngineBase>&              engine,
                const std::shared_ptr<MultimodalProcessor>&     mm_processor,
                const std::shared_ptr<autil::AtomicCounter>&    request_counter,
                const std::shared_ptr<Tokenizer>&               tokenizer,
                const std::shared_ptr<ChatRender>&              render,
                const ModelConfig&                             model_config,
                const std::shared_ptr<ApiServerMetricReporter>& metric_reporter):
        engine_(engine),
        mm_processor_(mm_processor),
        request_counter_(request_counter),
        openai_endpoint_(new OpenaiEndpoint(tokenizer, render, model_config)),
        metric_reporter_(metric_reporter) {}
    ~ChatService() = default;

public:
    void chatCompletions(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                         const http_server::HttpRequest&                         request,
                         int64_t                                                 request_id);
    void chatRender(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                    const http_server::HttpRequest&                         request);

private:
    std::shared_ptr<GenerateInput>
         fillGenerateInput(int64_t request_id, const ChatCompletionRequest& chat_request, const RenderedInputs& body);
    void generateResponse(const std::shared_ptr<GenerateConfig>&                  config,
                          const GenerateStreamPtr&                                stream,
                          const RenderedInputs&                                   rendered_input,
                          autil::StageTime&                                       iterate_stage_timer,
                          const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                          const ChatCompletionRequest&                            chat_request,
                          const std::string&                                      body,
                          int64_t                                                 request_id,
                          int64_t                                                 start_time_us);
    void generateStreamingResponse(const std::shared_ptr<GenerateConfig>&                  config,
                                   const GenerateStreamPtr&                                stream,
                                   const RenderedInputs&                                   rendered_input,
                                   autil::StageTime&                                       iterate_stage_timer,
                                   const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                   const ChatCompletionRequest&                            chat_request,
                                   const std::string&                                      body,
                                   int64_t                                                 request_id,
                                   int64_t                                                 start_time_us);
    static std::string sseResponse(const std::string& response) {
        return "data: " + response + "\n\n";
    }

private:
    std::shared_ptr<EngineBase>              engine_;
    std::shared_ptr<MultimodalProcessor>     mm_processor_;
    std::shared_ptr<autil::AtomicCounter>    request_counter_;
    std::shared_ptr<OpenaiEndpoint>          openai_endpoint_;
    std::shared_ptr<ApiServerMetricReporter> metric_reporter_;
};

}  // namespace rtp_llm

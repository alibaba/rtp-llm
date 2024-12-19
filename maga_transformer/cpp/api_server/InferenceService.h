#pragma once

#include "autil/AtomicCounter.h"

#include "maga_transformer/cpp/multimodal_processor/MultimodalProcessor.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"

#include "maga_transformer/cpp/http_server/http_server/HttpResponseWriter.h"
#include "maga_transformer/cpp/http_server/http_server/HttpRequest.h"

#include "maga_transformer/cpp/api_server/TokenProcessor.h"
#include "maga_transformer/cpp/api_server/ApiServerMetrics.h"
#include "maga_transformer/cpp/api_server/InferenceDataType.h"
#include "maga_transformer/cpp/api_server/GenerateStreamWrapper.h"
#include "maga_transformer/cpp/api_server/ConcurrencyControllerUtil.h"

namespace rtp_llm {

struct InferenceParsedRequest {
    static InferenceParsedRequest extractRequest(const std::string& body,
            const ft::GptInitParameter& params, const std::shared_ptr<TokenProcessor>& token_processor);
    static void extractRequestTexts(const RawRequest& req, InferenceParsedRequest& pr);
    static void extractRequestUrls(const RawRequest& req, InferenceParsedRequest& pr);
    static void extractRequestGenerateConfigs(RawRequest& req, InferenceParsedRequest& pr,
            const ft::GptInitParameter& params, const std::shared_ptr<TokenProcessor>& token_processor);

    bool batch_infer;
    bool is_streaming;
    bool private_request;
    std::string source;
    std::vector<std::vector<std::string>> input_urls;
    std::vector<std::string> input_texts;
    std::vector<std::shared_ptr<GenerateConfig>> generate_configs;
};

class InferenceService {
public:
    InferenceService(const std::shared_ptr<EngineBase>&              engine,
                     const std::shared_ptr<MultimodalProcessor>&     mm_processor,
                     const std::shared_ptr<autil::AtomicCounter>&    request_counter,
                     const std::shared_ptr<TokenProcessor>&          token_processor,
                     const std::shared_ptr<ConcurrencyController>&   controller,
                     const ft::GptInitParameter&                     params,
                     const std::shared_ptr<ApiServerMetricReporter>& metric_reporter);
    ~InferenceService() = default;

public:
    void inference(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                   const http_server::HttpRequest&                         request,
                   bool                                                    isInternal = false);
private:
    void inferResponse(int64_t                                                 request_id,
                       const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                       const http_server::HttpRequest&                         request);

    std::pair<int, std::vector<std::string>>
    iterateStreams(std::vector<std::shared_ptr<GenerateStreamWrapper>>& streams,
                   const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                   const InferenceParsedRequest& req,
                   autil::StageTime& iterate_stage_timer);

    std::shared_ptr<GenerateInput>
    fillGenerateInput(int64_t request_id,
                      const std::string& text,
                      const std::vector<std::string>& urls,
                      const std::shared_ptr<GenerateConfig>& generate_config);

    std::string doneResponse();
    std::string sseResponse(const std::string& response);
    std::string streamingResponse(const std::any& response);
    std::string completeResponse(const std::any& response);
    std::any    formatResponse(std::vector<MultiSeqsResponse> batch_state,
                               bool batch_infer);

    std::pair<bool, std::string>
    writeResponse(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                  const InferenceParsedRequest& req,
                  const std::any& res);
    void writeDoneResponse(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                           const InferenceParsedRequest& req);
private:
    std::shared_ptr<EngineBase>              engine_;
    std::shared_ptr<MultimodalProcessor>     mm_processor_;
    std::shared_ptr<TokenProcessor>          token_processor_;
    std::shared_ptr<autil::AtomicCounter>    request_counter_;
    std::shared_ptr<ConcurrencyController>   controller_;
    ft::GptInitParameter                     params_;
    std::shared_ptr<ApiServerMetricReporter> metric_reporter_;
};

}  // namespace rtp_llm

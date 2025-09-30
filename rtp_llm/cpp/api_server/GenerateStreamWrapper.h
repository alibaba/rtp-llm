#pragma once

#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

#include "rtp_llm/cpp/api_server/TokenProcessor.h"
#include "rtp_llm/cpp/api_server/ApiServerMetrics.h"
#include "rtp_llm/cpp/api_server/InferenceDataType.h"

namespace rtp_llm {

class GenerateStreamWrapper {
public:
    GenerateStreamWrapper(const std::shared_ptr<ApiServerMetricReporter>& metric_reporter,
                          const std::shared_ptr<TokenProcessor>&          token_processor);
    virtual ~GenerateStreamWrapper() = default;

    void init(const std::shared_ptr<GenerateInput>& input, const std::shared_ptr<EngineBase>& engine);

    void init(GenerateStreamPtr input, const std::shared_ptr<EngineBase>& engine);

    virtual std::pair<MultiSeqsResponse, bool> generateResponse();

private:
    // static for ease of UT
    static MultiSeqsResponse formatResponse(const std::vector<std::string>&        generate_texts,
                                            const GenerateOutputs&                 generate_outputs,
                                            const std::shared_ptr<GenerateConfig>& generate_config,
                                            rtp_llm::BufferPtr                     input_ids);

private:
    GenerateStreamPtr                        stream_;
    std::shared_ptr<GenerateConfig>          generate_config_;
    std::shared_ptr<ApiServerMetricReporter> metric_reporter_;
    std::shared_ptr<TokenProcessor>          token_processor_;
    std::shared_ptr<TokenProcessorPerStream> token_processor_ctx_;
    std::shared_ptr<lora::LoraResourceGuard> lora_guard_;
    std::vector<int>                         output_lens_;
    GenerateOutputs                          outputs_cache_;
    rtp_llm::BufferPtr                       input_ids_;
};

}  // namespace rtp_llm

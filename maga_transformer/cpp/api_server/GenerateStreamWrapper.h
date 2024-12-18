#pragma once

#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/GenerateConfig.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/multimodal_processor/MultimodalProcessor.h"

#include "maga_transformer/cpp/api_server/TokenProcessor.h"
#include "maga_transformer/cpp/api_server/ApiServerMetrics.h"
#include "maga_transformer/cpp/api_server/InferenceDataType.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class GenerateStreamWrapper {
public:
    GenerateStreamWrapper(const std::shared_ptr<ApiServerMetricReporter>& metric_reporter,
                          const std::shared_ptr<TokenProcessor>& token_processor);
    virtual ~GenerateStreamWrapper() = default;

    void init(const std::shared_ptr<GenerateInput>& input,
              const std::shared_ptr<EngineBase>& engine);

    virtual std::pair<MultiSeqsResponse, bool>
    generateResponse();

private:
    // static for ease of UT
    static MultiSeqsResponse
    formatResponse(const std::vector<std::string>&        generate_texts,
                   const GenerateOutputs&                 generate_outputs,
                   const std::shared_ptr<GenerateConfig>& generate_config,
                   ft::BufferPtr                          input_ids);
private:
    GenerateStreamPtr                        stream_;
    std::shared_ptr<GenerateConfig>          generate_config_;
    std::shared_ptr<ApiServerMetricReporter> metric_reporter_;
    std::shared_ptr<TokenProcessor>          token_processor_;
    std::shared_ptr<TokenProcessorPerStream> token_processor_ctx_;
    std::shared_ptr<lora::LoraResourceGuard> lora_guard_;
    std::vector<int>                         output_lens_;
    GenerateOutputs                          outputs_cache_;
    ft::BufferPtr                            input_ids_;
};

}  // namespace rtp_llm

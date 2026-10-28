#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class ModelConfig;
struct GrammarConfig;

struct GrammarPrefetchContext {
    std::unordered_map<std::string, GrammarCompileHandle> handles;

    GrammarCompileHandle* find(const GrammarKeyCpp& key) {
        auto it = handles.find(key.id());
        return it == handles.end() ? nullptr : &it->second;
    }
};

class LogitsProcessorFactory {
public:
    static void init(const ModelConfig&           model_config,
                     const GrammarConfig&         grammar_config,
                     const std::string&           tree_decode_config,
                     kmonitor::MetricsReporterPtr metrics_reporter = nullptr);

    static void shutdown();

    static ErrorResult<std::vector<BaseLogitsProcessorPtr>>
    createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                           int32_t                        init_batch_size,
                           int32_t                        max_batch_size,
                           int64_t                        eos_token_id,
                           GrammarPrefetchContext*        grammar_prefetch = nullptr);

    static GrammarPrefetchContext prefetchGrammars(const std::vector<GrammarKeyCpp>& keys);

    static std::optional<ErrorInfo> validateMtpCompatibility(const std::vector<BaseLogitsProcessorPtr>& processors);

private:
    static std::shared_ptr<XGrammarBackend>& grammarBackend();
};

}  // namespace rtp_llm

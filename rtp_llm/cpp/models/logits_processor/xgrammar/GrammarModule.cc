#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorModules.h"

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarCompiler.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

// Definition of the neutral hook declared in LogitsProcessorModules.h. Keeping it
// here is what confines the concrete grammar types to the xgrammar module: the
// engine sees only registerStructuredOutputLogitsProcessor and never includes
// GrammarCompiler / GrammarLogitsProcessor. The single GrammarLogitsProcessor
// path covers both reasoning and non-reasoning requests — in_think_mode flips
// the matcher's KMP gate at admission rather than dispatching to a separate
// processor type.
void registerStructuredOutputLogitsProcessor(const StructuredOutputConfig& cfg) {
    GrammarCompiler::initialize(cfg);
    LogitsProcessorFactory::registerExtraFactory(
        [](const std::shared_ptr<GenerateInput>& input) -> BaseLogitsProcessorPtr {
            if (!input || !input->generate_config) {
                return nullptr;
            }
            return GrammarLogitsProcessor::tryCreatePending(input);
        });
}

}  // namespace rtp_llm

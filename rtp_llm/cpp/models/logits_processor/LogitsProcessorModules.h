#pragma once

namespace rtp_llm {

struct StructuredOutputConfig;  // config/ConfigModules.h

// Engine-startup hook for module-provided logits processors. Currently registers
// the structured-output (constraint) processor with LogitsProcessorFactory: the
// concrete module owns the definition (see the xgrammar module), while the engine
// calls this one neutral function and stays unaware of any concrete backend type.
//
// Explicitly invoked by the composition root (not self-registering) so the wiring
// cannot be silently dropped by a future link-option change. A disabled config
// makes the registered factory a no-op, so calling it is always safe.
void registerStructuredOutputLogitsProcessor(const StructuredOutputConfig& cfg);

}  // namespace rtp_llm

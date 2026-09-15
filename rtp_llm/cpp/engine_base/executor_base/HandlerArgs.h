#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm {
namespace HandlerArgs {

// Argument names a CustomHandler may declare via extend_forward_args().
// The embedding path (EmbeddingExecutor::postProcess) and the generate path
// (PostLayersProcessor) share this name registry, while each validates its
// supported subset.
enum class Arg : uint32_t {
    // Assembled by EmbeddingExecutor::postProcess.
    INPUT_LENGTHS,
    HIDDEN_STATES,
    INPUT_IDS,
    ATTENTION_MASK,
    MOE_GATING,

    // Assembled by PostLayersProcessor on the generate path.
    LAST_HIDDEN_STATES,
    SELECTED_HIDDEN_STATES,

    // reserve as number marker
    NUM_ARG_TYPES
};

constexpr size_t NUM_ARG_TYPES = static_cast<size_t>(Arg::NUM_ARG_TYPES);

using Flag = std::bitset<NUM_ARG_TYPES>;

bool        set_by_str(Flag& flag, const char* name);
const char* get_name(Arg arg);
bool        has_arg(const Flag& flag, Arg arg);

// Parse extend_forward_args() output into a Flag; unknown names are appended
// to `unknown` (if given) for the caller to handle according to its contract.
Flag parse(const std::vector<std::string>& names, std::vector<std::string>* unknown = nullptr);

}  // namespace HandlerArgs
}  // namespace rtp_llm

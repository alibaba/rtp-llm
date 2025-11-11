#pragma once

#include <bitset>
#include <cstdint>

namespace rtp_llm {
namespace HandlerArgs {

enum class Arg : uint32_t {
    // Args for EmbeddingExecutor
    INPUT_LENGTHS,
    ALL_HIDDEN_STATES,
    INPUT_IDS,
    ATTENTION_MASK,
    ALL_MOE_GATING,

    // Args for NormalExecutor
    LAST_HIDDEN_STATES,
    LAST_MOE_GATING,
    TOKEN_LENGTHS,

    NUM_ARG_TYPES
};

constexpr size_t NUM_ARG_TYPES = static_cast<size_t>(Arg::NUM_ARG_TYPES);

using Flag = std::bitset<NUM_ARG_TYPES>;

bool        set_by_str(Flag& flag, const char* name);
const char* get_name(Arg arg);
bool        has_arg(const Flag& flag, Arg arg);

}  // namespace HandlerArgs
}  // namespace rtp_llm

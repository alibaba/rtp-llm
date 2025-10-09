#include "rtp_llm/cpp/engine_base/ExecutorBase/HandlerArgs.h"
#include <cstring> // For strcmp

namespace rtp_llm {
namespace HandlerArgs {

static const char* names[] = {
    // Args for EmbeddingExecutor
    "input_lengths",
    "all_hidden_states",
    "input_ids",
    "attention_mask",
    "all_moe_gating",

    // Args for NormalExecutor
    "last_hidden_states",
    "last_moe_gating",
    "token_lengths"
};

static_assert(sizeof(names) / sizeof(names[0]) == static_cast<size_t>(Arg::NUM_ARG_TYPES),
              "The number of names must match the number of Arg enums.");

bool set_by_str(Flag& flag, const std::string& name) {
    for (size_t i = 0; i < NUM_ARG_TYPES; ++i) {
        if (std::strcmp(names[i], name.c_str()) == 0) {
            flag.set(i);
            return true;
        }
    }
    return false;
}

const char* get_name(Arg arg) {
    return names[static_cast<size_t>(arg)];
}

bool has_arg(const Flag& flag, Arg arg) {
    return flag.test(static_cast<size_t>(arg));
}

}  // namespace HandlerArgs
}  // namespace rtp_llm

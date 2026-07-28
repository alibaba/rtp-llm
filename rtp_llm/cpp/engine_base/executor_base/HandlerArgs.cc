#include "rtp_llm/cpp/engine_base/executor_base/HandlerArgs.h"

#include <cstring>

namespace rtp_llm {
namespace HandlerArgs {

static const char* names[] = {
    "input_lengths",
    "hidden_states",
    "input_ids",
    "attention_mask",
    "moe_gating",

    "last_hidden_states",
};

static_assert(sizeof(names) / sizeof(names[0]) == NUM_ARG_TYPES,
              "The number of names must match the number of Arg enums.");

bool set_by_str(Flag& flag, const char* name) {
    if (!name) {
        return false;
    }
    for (size_t i = 0; i < NUM_ARG_TYPES; ++i) {
        if (std::strcmp(names[i], name) == 0) {
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

Flag parse(const std::vector<std::string>& arg_names, std::vector<std::string>* unknown) {
    Flag flag;
    for (const auto& name : arg_names) {
        if (!set_by_str(flag, name.c_str()) && unknown) {
            unknown->push_back(name);
        }
    }
    return flag;
}

}  // namespace HandlerArgs
}  // namespace rtp_llm

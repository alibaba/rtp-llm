#pragma once

#include <string.h>

namespace fastertransformer {

#define RUNTIME_ASSERT_OP_ARG(predicate, ...) { \
    if (!(predicate)) { \
        char msg[4096]; \
        sprintf(msg, __VA_ARGS__); \
        throw OpException(OpStatus(OpErrorType::ERROR_INVALID_ARGS, msg)); \
    } \
}

inline bool is_debug_mode() {
    static char* level_name = std::getenv("FT_DEBUG_LEVEL");
    return level_name && (strcmp(level_name, "DEBUG") == 0);
}

} // namespace fastertransformer


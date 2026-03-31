#pragma once

#include <string.h>

namespace rtp_llm {

#define RUNTIME_ASSERT_OP_ARG(predicate, ...)                                                                          \
    {                                                                                                                  \
        if (!(predicate)) {                                                                                            \
            char msg[4096];                                                                                            \
            sprintf(msg, __VA_ARGS__);                                                                                 \
            throw OpException(OpStatus(OpErrorType::ERROR_INVALID_ARGS, msg));                                         \
        }                                                                                                              \
    }

}  // namespace rtp_llm

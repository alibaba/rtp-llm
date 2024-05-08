#pragma once

#include <string.h>

namespace fastertransformer {

#define RUNTIME_ASSERT_OP_ARG(predicate, ...) {                              \
    if (!(predicate)) {                                                      \
        char msg[4096];                                                      \
        sprintf(msg, __VA_ARGS__);                                           \
        throw OpException(OpStatus(OpErrorType::ERROR_INVALID_ARGS, msg));   \
    }                                                                        \
}

#define RUNTIME_ASSERT(predicate) {                                         \
    if (status != OpStatus::SUCCESS) {                                      \
        throw OpException(OpStatus);                                        \
    }                                                                       \

} // namespace fastertransformer


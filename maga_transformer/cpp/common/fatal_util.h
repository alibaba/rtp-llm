#pragma once

#include "src/fastertransformer/utils/logger.h"

#define RAISE_FATAL_ERROR(msg) {                      \
    FT_LOG_ERROR("FATAL ERROR!!! %s", msg.c_str());   \
    fflush(stdout);                                   \
    fflush(stderr);                                   \
    throw std::runtime_error(msg);                    \
    abort();                                          \
}
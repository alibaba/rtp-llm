#pragma once

#include "src/fastertransformer/utils/logger.h"

#define RAISE_FATAL_ERROR(msg) {                                \
        std::string clone_msg(msg);                             \
        FT_LOG_ERROR("FATAL ERROR!!! %s", clone_msg.c_str());   \
        fflush(stdout);                                         \
        fflush(stderr);                                         \
        throw std::runtime_error(clone_msg);                    \
        abort();                                                \
    }

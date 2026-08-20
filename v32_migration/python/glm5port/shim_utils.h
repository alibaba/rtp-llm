#pragma once
#include <cstdio>
#include <stdexcept>
#include <string>
#define RTP_LLM_CHECK_WITH_INFO(cond, ...)                                                                             \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            char _b[512];                                                                                              \
            snprintf(_b, sizeof(_b), __VA_ARGS__);                                                                     \
            throw std::runtime_error(_b);                                                                              \
        }                                                                                                              \
    } while (0)
#define RTP_LLM_CHECK(cond)                                                                                            \
    do {                                                                                                               \
        if (!(cond))                                                                                                   \
            throw std::runtime_error("check failed: " #cond);                                                          \
    } while (0)
#define RTP_LLM_LOG_DEBUG(...)                                                                                         \
    do {                                                                                                               \
    } while (0)
#define RTP_LLM_LOG_INFO(...)                                                                                          \
    do {                                                                                                               \
    } while (0)
#define RTP_LLM_LOG_WARNING(...) fprintf(stderr, "[v32port] warn\n")
#define RTP_LLM_LOG_ERROR(...) fprintf(stderr, "[v32port] error\n")

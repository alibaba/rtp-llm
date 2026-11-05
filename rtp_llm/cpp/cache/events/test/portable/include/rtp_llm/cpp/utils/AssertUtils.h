#pragma once

#include <stdexcept>

// Portable tests retain invariant checks without linking the engine logger.
#define RTP_LLM_CHECK_WITH_INFO(value, ...) \
    do { \
        if (!(value)) { \
            throw std::logic_error("RTP invariant failed: " #value); \
        } \
    } while (false)
#define RTP_LLM_CHECK(value) RTP_LLM_CHECK_WITH_INFO(value)

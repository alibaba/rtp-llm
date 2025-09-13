#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

[[noreturn]] void throwRuntimeError(const char* const file, int const line, std::string const& info = "");

[[noreturn]] void myAssert(const char* const file, int const line, std::string const& info = "");

}  // namespace rtp_llm

#define RTP_LLM_CHECK_WITH_INFO(val, info, ...)                                                                        \
    do {                                                                                                               \
        bool is_valid_val = (val);                                                                                     \
        if (!is_valid_val) {                                                                                           \
            rtp_llm::myAssert(__FILE__, __LINE__, rtp_llm::fmtstr(info, ##__VA_ARGS__));                               \
        }                                                                                                              \
    } while (0)

#define RTP_LLM_CHECK(val) RTP_LLM_CHECK_WITH_INFO(val, "")

#define RTP_LLM_FAIL(info, ...) rtp_llm::myAssert(__FILE__, __LINE__, rtp_llm::fmtstr(info, ##__VA_ARGS__))

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/th_op/ConfigModules.h"
namespace rtp_llm {

[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "") {
    auto error_msg =
        std::string("[FT][ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    fflush(stdout);
    fflush(stderr);
    throw FT_EXCEPTION(error_msg);
}

[[noreturn]] inline void myAssert(const char* const file, int const line, std::string const& info = "") {
    auto error_msg =
        std::string("[FT][ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    RTP_LLM_LOG_ERROR("FATAIL ERROR!!! %s", error_msg.c_str());
    if (StaticConfig::user_ft_core_dump_on_exception) {
        fflush(stdout);
        fflush(stderr);
        abort();
    }
    throwRuntimeError(file, line, info);
}

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

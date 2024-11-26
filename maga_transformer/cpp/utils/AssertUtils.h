#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "maga_transformer/cpp/utils/Exception.h"
#include "maga_transformer/cpp/utils/Logger.h"

namespace rtp_llm {

[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "") {
    auto error_msg = std::string("[FT][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n";
    fflush(stdout);
    fflush(stderr);
    throw FT_EXCEPTION(error_msg);
}

[[noreturn]] inline void myAssert(const char* const file, int const line, std::string const& info = "") {
    auto error_msg = std::string("[FT][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n";
    FT_LOG_ERROR("FATAIL ERROR!!! %s", error_msg.c_str());
    if (std::getenv("FT_CORE_DUMP_ON_EXCEPTION")) {
        fflush(stdout);
        fflush(stderr);
        abort();
    }
    throwRuntimeError(file, line, info);
}

}  // namespace rtp_llm

#define FT_CHECK_WITH_INFO(val, info, ...)                                                  \
    do {                                                                                    \
        bool is_valid_val = (val);                                                          \
        if (!is_valid_val) {						                                        \
            rtp_llm::myAssert(					                                \
                    __FILE__, __LINE__, rtp_llm::fmtstr(info, ##__VA_ARGS__));    \
        }								                                                    \
    } while (0)

#define FT_CHECK(val) FT_CHECK_WITH_INFO(val, "")

#define FT_FAIL(info, ...) rtp_llm::myAssert(__FILE__, __LINE__, rtp_llm::fmtstr(info, ##__VA_ARGS__))

#pragma once

#include "src/fastertransformer/utils/exception.h"
#include "src/fastertransformer/utils/logger.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fastertransformer {

#define PRINT_FUNC_NAME_()                                                                                             \
    do {                                                                                                               \
        std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl;                                                \
    } while (0)

[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "") {
    auto error_msg = std::string("[FT][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n";
    fflush(stdout);
    fflush(stderr);
    throw NEW_FT_EXCEPTION(error_msg);
}

[[noreturn]] inline void myAssert(const char* const file, int const line, std::string const& info = "") {
    if (std::getenv("FT_CORE_DUMP_ON_EXCEPTION")) {
        fflush(stdout);
        fflush(stderr);
        abort();
    }
    throwRuntimeError(file, line, info);
}

#define FT_CHECK_WITH_INFO(val, info, ...)                                                  \
    do {                                                                                    \
        bool is_valid_val = (val);                                                          \
        if (!is_valid_val) {						                                        \
            fastertransformer::myAssert(					                                \
                    __FILE__, __LINE__, fastertransformer::fmtstr(info, ##__VA_ARGS__));    \
        }								                                                    \
    } while (0)

#define FT_CHECK(val) FT_CHECK_WITH_INFO(val, "")

#define FT_FAIL(info, ...) fastertransformer::myAssert(__FILE__, __LINE__, fastertransformer::fmtstr(info, ##__VA_ARGS__))


}  // namespace fastertransformer

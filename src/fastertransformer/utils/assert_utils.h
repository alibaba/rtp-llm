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

inline void myAssert(bool result, const char* const file, int const line, std::string const& info = "") {
    if (!result) {
        if (std::getenv("FT_CORE_DUMP_ON_EXCEPTION")) {
            fflush(stdout);
            fflush(stderr);
            abort();
        }
        throwRuntimeError(file, line, info);
    }
}

#define FT_CHECK(val) fastertransformer::myAssert(val, __FILE__, __LINE__)
#define FT_CHECK_WITH_INFO(val, info, ...)                              \
    do {                                                                \
        bool is_valid_val = (val);                                      \
	if (!is_valid_val) {						\
  	    fastertransformer::myAssert(					\
                is_valid_val, __FILE__, __LINE__, fastertransformer::fmtstr(info, ##__VA_ARGS__)); \
	}								\
    } while (0)

#define FT_THROW(info) throwRuntimeError(__FILE__, __LINE__, info)

[[noreturn]] inline void unreachable() {
    throw NEW_FT_EXCEPTION("unreachable error");
}

}  // namespace fastertransformer

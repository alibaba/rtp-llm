#pragma once

#include <hip/hip_runtime.h>

static inline void check_hip_error(hipError_t result, char const* const func, const char* const file, int const line) {
    if (result != hipSuccess) {
        FT_LOG_ERROR(std::string("[FT][ERROR] HIP runtime error: ") + (hipGetErrorName(result)) + " " + file + ":"
                     + std::to_string(line) + " \n");
        fflush(stdout);
        abort();
    }
}

#define HIP_CHECK(val) check_hip_error((val), #val, __FILE__, __LINE__)

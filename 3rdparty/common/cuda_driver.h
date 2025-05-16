#pragma once

#include <cstdio>
#include <stdexcept>

#include "cuda.h"

inline void checkCu(CUresult err) {
    if (err != CUDA_SUCCESS) {
        char const* str = nullptr;
        if (cuGetErrorName(err, &str) != CUDA_SUCCESS) {
            str = "A cuda driver API error happened, but we failed to query the error name\n";
        }
        printf("%s\n", str);
        throw std::runtime_error(str);
    }
}

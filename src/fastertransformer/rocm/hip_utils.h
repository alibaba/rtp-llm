#pragma once

#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/assert_utils.h"

#include <hip/hip_runtime.h>
#include "cuda_shims.h"  // type and APIs warpper

#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#ifdef SPARSITY_ENABLED
// #include <cusparseLt.h>
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fastertransformer {

#define HIP_CHECK(val) rocm::check((val), #val, __FILE__, __LINE__)
#define check_hip_error(val) rocm::check((val), #val, __FILE__, __LINE__)
#define sync_check_hip_error() rocm::syncAndCheck(__FILE__, __LINE__)

namespace rocm {

/* **************************** debug tools ********************************** */
static const char* _hipGetErrorEnum(hipError_t error) {
    return hipGetErrorString(error);
}
static const char* _hipGetErrorEnum(hipblasStatus_t error) {
    return hipblasStatusToString(error);
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] ROCM runtime error: ") + (_hipGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}
inline void syncAndCheck(const char* const file, int const line) {
    // When FT_DEBUG_LEVEL=DEBUG, must check error
    static char* level_name = std::getenv("FT_DEBUG_LEVEL");
    if (level_name != nullptr) {
        static std::string level = std::string(level_name);
        if (level == "DEBUG") {
            check_hip_error(hipDeviceSynchronize());
            hipError_t result = hipGetLastError();
            if (result) {
                throw std::runtime_error(std::string("[FT][ERROR] ROCM runtime error: ") + (_hipGetErrorEnum(result))
                                         + " " + file + ":" + std::to_string(line) + " \n");
            }
            FT_LOG_DEBUG(fmtstr("run syncAndCheck at %s:%d", file, line));
        }
    }

#ifndef NDEBUG
    check_hip_error(hipDeviceSynchronize());
    hipError_t result = hipGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] ROCM runtime error: ") + (_hipGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
#endif
}

/* **************************** device tools ********************************* */
inline int getComputeCapability() {
    int device{-1};
    check_hip_error(hipGetDevice(&device));
    int major = 0;
    int minor = 0;
    check_hip_error(hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, device));
    check_hip_error(hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, device));
    return major * 10 + minor;
}

/* ***************************** common utils ******************************* */


}  // namespace rocm
}  // namespace fastertransformer

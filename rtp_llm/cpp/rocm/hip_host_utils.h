#pragma once

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/rocm/hip_capture_check.h"

#include <hip/hip_runtime.h>
#include "cuda_shims.h"

#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

namespace rtp_llm {
namespace rocm {

#define HIPBLAS_WORKSPACE_SIZE (512L * 1024L * 1024L)  // C*splitK
#define ROCM_RUNTIME_MEM_SIZE (HIPBLAS_WORKSPACE_SIZE + 512L * 1024L * 1024L)

#define ROCM_CHECK(val) rocm::check((val), __FILE__, __LINE__)
#define ROCM_CHECK_ERROR() rocm::syncAndCheckInDebug(__FILE__, __LINE__)
#define ROCM_CHECK_VALUE(val, info, ...)                                                                               \
    do {                                                                                                               \
        bool is_valid_val = (val);                                                                                     \
        if (!is_valid_val) {                                                                                           \
            rocm::throwRocmError(__FILE__, __LINE__, rtp_llm::fmtstr(info, ##__VA_ARGS__));                            \
        }                                                                                                              \
    } while (0)
#define ROCM_FAIL(info, ...) rocm::throwRocmError(__FILE__, __LINE__, rtp_llm::fmtstr(info, ##__VA_ARGS__))

void throwRocmError(const char* const file, int const line, std::string const& info = "");
template<typename T>
void check(T result, const char* const file, int const line);
void syncAndCheckInDebug(const char* const file, int const line);

int get_sm();
int getDevice();
int getDeviceCount();
int getMultiProcessorCount(int device_id = -1);
int getMaxSharedMemoryPerMultiprocessor(int device_id = -1);

}  // namespace rocm

}  // namespace rtp_llm

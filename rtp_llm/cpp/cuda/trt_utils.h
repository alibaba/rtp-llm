#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/math_utils.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

#pragma once
#define TLLM_LOG_TRACE RTP_LLM_LOG_TRACE
#define TLLM_LOG_DEBUG RTP_LLM_LOG_DEBUG
#define TLLM_LOG_INFO RTP_LLM_LOG_INFO
#define TLLM_LOG_WARNING RTP_LLM_LOG_WARNING
#define TLLM_LOG_ERROR RTP_LLM_LOG_ERROR
#define TLLM_LOG_EXCEPTION RTP_LLM_LOG_EXCEPTION

#define TLLM_CHECK_WITH_INFO RTP_LLM_CHECK_WITH_INFO
#define TLLM_CHECK RTP_LLM_CHECK
#define TLLM_CUDA_CHECK check_cuda_value
#define TLLM_THROW(...)                                                                                                \
    do {                                                                                                               \
        throw FT_EXCEPTION(__VA_ARGS__);                                                                               \
    } while (0)

namespace nvinfer1 {
enum class DataType : int32_t {
    kFLOAT = 0,
    kHALF  = 1,
    kINT8  = 2,
    kINT32 = 3,
    kBOOL  = 4,
    kUINT8 = 5,
    kFP8   = 6,
    kBF16  = 7,
    kINT64 = 8,
    kINT4  = 9,
};
}  // namespace nvinfer1

namespace tensorrt_llm::common {

using rtp_llm::ceilDiv;
using rtp_llm::get_sm;
constexpr static size_t getDTypeSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kINT64:
            return 8;
        case nvinfer1::DataType::kINT32:
            [[fallthrough]];
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kBF16:
            [[fallthrough]];
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
            [[fallthrough]];
        case nvinfer1::DataType::kUINT8:
            [[fallthrough]];
        case nvinfer1::DataType::kINT8:
            [[fallthrough]];
        case nvinfer1::DataType::kINT4:
            [[fallthrough]];
        case nvinfer1::DataType::kFP8:
            return 1;
    }
    return 0;
}

std::uintptr_t constexpr kCudaMemAlign = 128;

namespace {

inline int8_t* alignPtr(int8_t* ptr, uintptr_t to) {
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % to) {
        addr += to - addr % to;
    }
    return (int8_t*)addr;
}

inline int8_t* nextWorkspacePtrCommon(int8_t* ptr, uintptr_t previousWorkspaceSize, const uintptr_t alignment) {
    uintptr_t addr = (uintptr_t)ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*)addr, alignment);
}

inline int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize) {
    return nextWorkspacePtrCommon(ptr, previousWorkspaceSize, kCudaMemAlign);
}

inline int8_t* nextWorkspacePtr(int8_t* const   base,
                                uintptr_t&      offset,
                                const uintptr_t size,
                                const uintptr_t alignment = kCudaMemAlign) {
    uintptr_t curr_offset = offset;
    uintptr_t next_offset = curr_offset + ((size + alignment - 1) / alignment) * alignment;
    int8_t*   newptr      = size == 0 ? nullptr : base + curr_offset;
    offset                = next_offset;
    return newptr;
}

inline int8_t*
nextWorkspacePtrWithAlignment(int8_t* ptr, uintptr_t previousWorkspaceSize, const uintptr_t alignment = kCudaMemAlign) {
    return nextWorkspacePtrCommon(ptr, previousWorkspaceSize, alignment);
}

inline size_t calculateTotalWorkspaceSize(size_t* workspaces, int count, const uintptr_t alignment = kCudaMemAlign) {
    size_t total = 0;
    for (int i = 0; i < count; i++) {
        total += workspaces[i];
        if (workspaces[i] % alignment) {
            total += alignment - (workspaces[i] % alignment);
        }
    }
    return total;
}

}  // namespace

};  // namespace tensorrt_llm::common
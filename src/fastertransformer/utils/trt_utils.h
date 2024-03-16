#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/exception.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

#pragma once
#define TLLM_LOG_TRACE FT_LOG_TRACE
#define TLLM_LOG_DEBUG FT_LOG_DEBUG
#define TLLM_LOG_INFO FT_LOG_INFO
#define TLLM_LOG_WARNING FT_LOG_WARNING
#define TLLM_LOG_ERROR FT_LOG_ERROR
#define TLLM_LOG_EXCEPTION FT_LOG_EXCEPTION 


#define TLLM_CHECK_WITH_INFO FT_CHECK_WITH_INFO 
#define TLLM_CHECK FT_CHECK

#define TLLM_THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw NEW_FT_EXCEPTION(__VA_ARGS__);                                                                         \
    } while (0)

namespace nvinfer1
{
enum class DataType : int32_t
{
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
    kUINT8 = 5,
    kFP8 = 6,
    kBF16 = 7,
    kINT64 = 8,
};
}

namespace tensorrt_llm::common
{

constexpr static size_t getDTypeSize(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT32: [[fallthrough]];
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kBF16: [[fallthrough]];
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL: [[fallthrough]];
    case nvinfer1::DataType::kUINT8: [[fallthrough]];
    case nvinfer1::DataType::kINT8: [[fallthrough]];
    case nvinfer1::DataType::kFP8: return 1;
    }
    return 0;
}

std::uintptr_t constexpr kCudaMemAlign = 128;

namespace
{

int8_t* alignPtr(int8_t* ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t) ptr;
    if (addr % to)
    {
        addr += to - addr % to;
    }
    return (int8_t*) addr;
}

int8_t* nextWorkspacePtrCommon(int8_t* ptr, uintptr_t previousWorkspaceSize, const uintptr_t alignment)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*) addr, alignment);
}

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    return nextWorkspacePtrCommon(ptr, previousWorkspaceSize, kCudaMemAlign);
}

int8_t* nextWorkspacePtr(
    int8_t* const base, uintptr_t& offset, const uintptr_t size, const uintptr_t alignment = kCudaMemAlign)
{
    uintptr_t curr_offset = offset;
    uintptr_t next_offset = curr_offset + ((size + alignment - 1) / alignment) * alignment;
    int8_t* newptr = size == 0 ? nullptr : base + curr_offset;
    offset = next_offset;
    return newptr;
}

int8_t* nextWorkspacePtrWithAlignment(
    int8_t* ptr, uintptr_t previousWorkspaceSize, const uintptr_t alignment = kCudaMemAlign)
{
    return nextWorkspacePtrCommon(ptr, previousWorkspaceSize, alignment);
}

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count, const uintptr_t alignment = kCudaMemAlign)
{
    size_t total = 0;
    for (int i = 0; i < count; i++)
    {
        total += workspaces[i];
        if (workspaces[i] % alignment)
        {
            total += alignment - (workspaces[i] % alignment);
        }
    }
    return total;
}

} // namespace

}; // namespace tensorrt_llm::common
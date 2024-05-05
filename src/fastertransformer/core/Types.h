#pragma once

#include <stdexcept> 
#include <string>
#include <unordered_map>

#if GOOGLE_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace fastertransformer {

typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

enum class AllocationType {
    HOST   = 0,
    DEVICE = 1,
};

typedef enum datatype_enum {
    TYPE_INVALID,
    TYPE_BOOL,
    TYPE_UINT8,
    TYPE_UINT16,
    TYPE_UINT32,
    TYPE_UINT64,
    TYPE_INT8,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_FP16,
    TYPE_FP32,
    TYPE_FP64,
    TYPE_BYTES,
    TYPE_BF16,
    TYPE_FP8_E4M3,
    TYPE_STR,
    TYPE_VOID
} DataType;

template<DataType data_type>
struct DataTypeTraits {
};

#if GOOGLE_CUDA
template<>
struct DataTypeTraits<TYPE_FP16> {
    using type = half;
};

template<>
struct DataTypeTraits<TYPE_BF16> {
    using type = __nv_bfloat16;
};
#endif

template<>
struct DataTypeTraits<TYPE_FP32> {
    using type = float;
};

inline DataType getDataType(const std::string& type_str) {
    DataType data_type;
    if (type_str == "fp16") {
        data_type = TYPE_FP16;
    } else if (type_str == "bf16") {
        data_type = TYPE_BF16;
    } else if (type_str == "fp32") {
        data_type = TYPE_FP32;
    } else {
        throw std::runtime_error("wrong data type str " + type_str);
    }
    return data_type;
}

template<typename T>
DataType getTensorType();

size_t getTypeSize(DataType type);

bool isQuantify(DataType type);
bool isFloat(DataType type);

} // namespace fastertransformer


#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>


namespace fastertransformer {

enum AttentionMaskType {
    // ones matrix, for bert model.
    noMask,
    causalMask,
    promptMask,
};

typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

enum class AllocationType {
    HOST   = 0,
    DEVICE = 1,
};

enum DataType : std::uint8_t {
    TYPE_INVALID   =  0,
    TYPE_BOOL      =  1,
    TYPE_UINT8     =  2,
    TYPE_UINT16    =  3,
    TYPE_UINT32    =  4,
    TYPE_UINT64    =  5,
    TYPE_INT8      =  6,
    TYPE_INT16     =  7,
    TYPE_INT32     =  8,
    TYPE_INT64     =  9,
    TYPE_FP16      = 10,
    TYPE_FP32      = 11,
    TYPE_FP64      = 12,
    TYPE_BYTES     = 13,
    TYPE_BF16      = 14,
    TYPE_FP8_E4M3  = 15,
    TYPE_STR       = 16,
    TYPE_VOID      = 17,
    TYPE_QINT8     = 18,
    TYPE_INT4X2    = 19,
    TYPE_QINT4X2   = 20,
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

inline std::string getDataTypeStr(const DataType& data_type) {
    std::string type_str;
    if (data_type == TYPE_FP16) {
        type_str = "fp16";
    } else if (data_type == TYPE_BF16) {
        type_str = "bf16";
    } else if (data_type == TYPE_FP32) {
        type_str = "fp32";
    } else {
        throw std::runtime_error("wrong data type");
    }
    return type_str;
}

template<typename T>
DataType getTensorType();

size_t getTypeSize(DataType type);

size_t getTypeBits(DataType type);

} // namespace fastertransformer


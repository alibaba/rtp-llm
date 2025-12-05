#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace rtp_llm {

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
    TYPE_INVALID   = 0,
    TYPE_BOOL      = 1,
    TYPE_UINT8     = 2,
    TYPE_UINT16    = 3,
    TYPE_UINT32    = 4,
    TYPE_UINT64    = 5,
    TYPE_INT8      = 6,
    TYPE_INT16     = 7,
    TYPE_INT32     = 8,
    TYPE_INT64     = 9,
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
    TYPE_QFP8_E4M3 = 21,
};

inline DataType getDataType(const std::string& type_str) {
    DataType data_type;
    if (type_str == "fp16") {
        data_type = TYPE_FP16;
    } else if (type_str == "bf16") {
        data_type = TYPE_BF16;
    } else if (type_str == "fp32") {
        data_type = TYPE_FP32;
    } else if (type_str == "int8") {
        data_type = TYPE_INT8;
    } else if (type_str == "fp8") {
        data_type = TYPE_FP8_E4M3;
    } else {
        throw std::runtime_error("wrong data type str " + type_str);
    }
    return data_type;
}

inline std::string getDataTypeStr(const DataType& data_type) {
    switch (data_type) {
        case TYPE_FP16:
            return "fp16";
        case TYPE_BF16:
            return "bf16";
        case TYPE_FP32:
            return "fp32";
        case TYPE_INT8:
            return "int8";
        case TYPE_FP8_E4M3:
            return "fp8";
        default:
            throw std::runtime_error("Invalid DataType: " + std::to_string(static_cast<int>(data_type)));
    }
}

template<typename T>
DataType getTensorType();

size_t getTypeSize(DataType type);

size_t getTypeBits(DataType type);

}  // namespace rtp_llm

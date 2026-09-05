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

// Internal representation of the single request-visible
// `aux_info.prefill_cuda_graph_status` string.
enum class PrefillCudaGraphStatus : std::uint8_t {
    NOT_REQUESTED,
    REPLAYED,
    CAPTURE_UNAVAILABLE,
    ATTENTION_BACKEND_UNSUPPORTED,
    MIXED_PREFILL_DECODE_NOT_SUPPORTED,
    INPUT_METADATA_INVALID,
    REQUEST_COUNT_EXCEED_CAPTURE_LIMIT,
    PREFIX_CACHE_NOT_SUPPORTED,
    REQUEST_NOT_SUPPORTED,
    PD_CACHE_STORE_NOT_SUPPORTED,
    TOKEN_TYPE_INPUT_NOT_SUPPORTED,
    MULTIMODAL_INPUT_NOT_SUPPORTED,
    SCRATCH_KV_UNAVAILABLE,
    MODEL_NOT_SUPPORTED,
    MOE_CONFIG_NOT_SUPPORTED,
    INPUT_TOKENS_EXCEED_CAPTURE_LIMIT,
    GRAPH_INPUT_SHAPE_MISMATCH,
};

inline const char* prefillCudaGraphStatusString(PrefillCudaGraphStatus status) {
    switch (status) {
        case PrefillCudaGraphStatus::NOT_REQUESTED:
            return "not_requested";
        case PrefillCudaGraphStatus::REPLAYED:
            return "replayed";
        case PrefillCudaGraphStatus::CAPTURE_UNAVAILABLE:
            return "capture_unavailable";
        case PrefillCudaGraphStatus::ATTENTION_BACKEND_UNSUPPORTED:
            return "attention_backend_unsupported";
        case PrefillCudaGraphStatus::MIXED_PREFILL_DECODE_NOT_SUPPORTED:
            return "mixed_prefill_decode_not_supported";
        case PrefillCudaGraphStatus::INPUT_METADATA_INVALID:
            return "input_metadata_invalid";
        case PrefillCudaGraphStatus::REQUEST_COUNT_EXCEED_CAPTURE_LIMIT:
            return "request_count_exceed_capture_limit";
        case PrefillCudaGraphStatus::PREFIX_CACHE_NOT_SUPPORTED:
            return "prefix_cache_not_supported";
        case PrefillCudaGraphStatus::REQUEST_NOT_SUPPORTED:
            return "request_not_supported";
        case PrefillCudaGraphStatus::PD_CACHE_STORE_NOT_SUPPORTED:
            return "pd_cache_store_not_supported";
        case PrefillCudaGraphStatus::TOKEN_TYPE_INPUT_NOT_SUPPORTED:
            return "token_type_input_not_supported";
        case PrefillCudaGraphStatus::MULTIMODAL_INPUT_NOT_SUPPORTED:
            return "multimodal_input_not_supported";
        case PrefillCudaGraphStatus::SCRATCH_KV_UNAVAILABLE:
            return "scratch_kv_unavailable";
        case PrefillCudaGraphStatus::MODEL_NOT_SUPPORTED:
            return "model_not_supported";
        case PrefillCudaGraphStatus::MOE_CONFIG_NOT_SUPPORTED:
            return "moe_config_not_supported";
        case PrefillCudaGraphStatus::INPUT_TOKENS_EXCEED_CAPTURE_LIMIT:
            return "input_tokens_exceed_capture_limit";
        case PrefillCudaGraphStatus::GRAPH_INPUT_SHAPE_MISMATCH:
            return "graph_input_shape_mismatch";
    }
    return "unknown";
}

enum QScheme : size_t {
    NoQuantize = 0,
    Qint8WeightOnly,
    Qint8PerToken,
    Qint8PerTensor,
    Qfp8PerTensor,
    Qfp8PerTokenBlock,
    Qfp8PerToken
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
    TYPE_FP8_E8M0  = 22
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

#include "Types.h"

#include <string>
#include <cstdint>

#ifdef ENABLE_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace fastertransformer {

#define FT_FOREACH_TYPE(F) \
    F(DataType::TYPE_INVALID, void); \
    F(DataType::TYPE_BOOL, bool); \
    F(DataType::TYPE_UINT8, uint8_t); \
    F(DataType::TYPE_UINT16, uint16_t); \
    F(DataType::TYPE_UINT32, uint32_t); \
    F(DataType::TYPE_UINT64, uint64_t); \
    F(DataType::TYPE_INT8, int8_t); \
    F(DataType::TYPE_INT16, int16_t); \
    F(DataType::TYPE_INT32, int32_t); \
    F(DataType::TYPE_INT64, int64_t); \
    F(DataType::TYPE_FP32, float); \
    F(DataType::TYPE_FP64, double); \
    F(DataType::TYPE_BYTES, char); \
    F(DataType::TYPE_STR, std::string);

#ifdef ENABLE_CUDA
#define FT_FOREACH_DEVICE_TYPE(F) \
    F(DataType::TYPE_FP16, half); \
    F(DataType::TYPE_BF16, __nv_bfloat16);
#else
#define FT_FOREACH_DEVICE_TYPE(F)
#endif

template<typename T>
struct TypeTrait {
    static const DataType type = TYPE_INVALID;
    static const size_t size = 0;
};

#define DEFINE_TYPE(DT, T) \
    template<> \
    struct TypeTrait<T> { \
        static const DataType type = DT; \
        static const size_t size = sizeof(T); \
    }; \
    template DataType getTensorType<T>(); \
    template DataType getTensorType<const T>(); \
    template DataType getTensorType<const T *>();

template<typename T>
DataType getTensorType() {
    return TypeTrait<T>::type;
}

FT_FOREACH_TYPE(DEFINE_TYPE);
FT_FOREACH_DEVICE_TYPE(DEFINE_TYPE);
DEFINE_TYPE(DataType::TYPE_UINT64, unsigned long long int);

size_t getTypeSize(DataType type) {
#define CASE(DT, T) { \
    case DT: { \
        return TypeTrait<T>::size; \
    } \
}

    switch (type) {
        FT_FOREACH_TYPE(CASE);
        FT_FOREACH_DEVICE_TYPE(CASE);
        default:
            return 0;
    }

}


bool isFloat(DataType type) {
    return (type == DataType::TYPE_BF16) ||
           (type == DataType::TYPE_FP16) ||
           (type == DataType::TYPE_FP32) ||
           (type == DataType::TYPE_FP64);
}

bool isQuantify(DataType type) {
    return (type == DataType::TYPE_INT8);
}

} // namespace fastertransformer


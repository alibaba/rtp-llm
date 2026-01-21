#include "Types.h"
#include <string>
#include <cstdint>

#if defined(__x86_64__)
#include <immintrin.h>
#elif BUILDING_ARM_ONLY
#include "rtp_llm/cpp/devices/arm_impl/type_bf16/hie_bfloat16.hpp"
#endif

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace rtp_llm {

#define FT_FOREACH_TYPE(F)                                                                                             \
    F(DataType::TYPE_BOOL, bool);                                                                                      \
    F(DataType::TYPE_UINT8, uint8_t);                                                                                  \
    F(DataType::TYPE_UINT16, uint16_t);                                                                                \
    F(DataType::TYPE_UINT32, uint32_t);                                                                                \
    F(DataType::TYPE_UINT64, uint64_t);                                                                                \
    F(DataType::TYPE_INT8, int8_t);                                                                                    \
    F(DataType::TYPE_INT16, int16_t);                                                                                  \
    F(DataType::TYPE_INT32, int32_t);                                                                                  \
    F(DataType::TYPE_INT64, int64_t);                                                                                  \
    F(DataType::TYPE_FP32, float);                                                                                     \
    F(DataType::TYPE_FP64, double);                                                                                    \
    F(DataType::TYPE_BYTES, char);                                                                                     \
    F(DataType::TYPE_STR, std::string);

#if USING_CUDA || USING_ROCM
#define FT_FOREACH_DEVICE_TYPE(F)                                                                                      \
    F(DataType::TYPE_FP16, half);                                                                                      \
    F(DataType::TYPE_BF16, __nv_bfloat16);

#elif BUILDING_ARM_ONLY
#define FT_FOREACH_DEVICE_TYPE(F)                                                                                      \
    F(DataType::TYPE_FP16, __fp16);                                                                                    \
    F(DataType::TYPE_BF16, hie::bfloat16);
#else
struct fake_half {
    uint16_t x;
};
struct fake_bfloat16 {
    uint16_t x;
};
#define FT_FOREACH_DEVICE_TYPE(F)                                                                                      \
    F(DataType::TYPE_FP16, fake_half);                                                                                 \
    F(DataType::TYPE_BF16, fake_bfloat16);
#endif

template<typename T>
struct TypeTrait {
    static const DataType type = TYPE_INVALID;
    static const size_t   size = 0;
};

#define DECLARE_TYPE_TRAIT(DT, T)                                                                                      \
    template<>                                                                                                         \
    struct TypeTrait<T> {                                                                                              \
        static const DataType type = DT;                                                                               \
        static const size_t   size = sizeof(T);                                                                        \
    };

#define DECLARE_GET_TYPE(T)                                                                                            \
    template DataType getTensorType<T>();                                                                              \
    template DataType getTensorType<const T>();                                                                        \
    template DataType getTensorType<const T*>();

#define DEFINE_TYPE(DT, T)                                                                                             \
    DECLARE_TYPE_TRAIT(DT, T);                                                                                         \
    DECLARE_GET_TYPE(T)

template<typename T>
DataType getTensorType() {
    return TypeTrait<T>::type;
}

FT_FOREACH_TYPE(DEFINE_TYPE);
FT_FOREACH_DEVICE_TYPE(DEFINE_TYPE);

#ifdef ENABLE_FP8
DEFINE_TYPE(DataType::TYPE_FP8_E4M3, __nv_fp8_e4m3);
#endif
#if USING_ROCM
DEFINE_TYPE(DataType::TYPE_FP8_E4M3, __hip_fp8_e4m3_fnuz);
#endif

DEFINE_TYPE(DataType::TYPE_UINT64, unsigned long long int);
DECLARE_GET_TYPE(void);

size_t getTypeSize(DataType type) {
#define CASE(DT, T)                                                                                                    \
    {                                                                                                                  \
        case DT: {                                                                                                     \
            return TypeTrait<T>::size;                                                                                 \
        }                                                                                                              \
    }

    switch (type) {
        FT_FOREACH_TYPE(CASE);
        FT_FOREACH_DEVICE_TYPE(CASE);
#ifdef ENABLE_FP8
        CASE(DataType::TYPE_FP8_E4M3, __nv_fp8_e4m3);
        CASE(DataType::TYPE_QFP8_E4M3, __nv_fp8_e4m3);
#endif
#if USING_ROCM
        CASE(DataType::TYPE_FP8_E4M3, __hip_fp8_e4m3_fnuz);
        CASE(DataType::TYPE_QFP8_E4M3, __hip_fp8_e4m3_fnuz);
#endif
        CASE(DataType::TYPE_QINT8, int8_t);
        case DataType::TYPE_INT4X2:
        case DataType::TYPE_QINT4X2:
        case DataType::TYPE_FP8_E8M0:
            return 1;
        default:
            return 0;
    }
#undef CASE
}

size_t getTypeBits(DataType type) {
#define CASE(DT, T)                                                                                                    \
    {                                                                                                                  \
        case DT: {                                                                                                     \
            return TypeTrait<T>::size * 8;                                                                             \
        }                                                                                                              \
    }

    switch (type) {
        FT_FOREACH_TYPE(CASE);
        FT_FOREACH_DEVICE_TYPE(CASE);
        CASE(DataType::TYPE_QINT8, int8_t);
#ifdef ENABLE_FP8
        CASE(DataType::TYPE_FP8_E4M3, __nv_fp8_e4m3);
        CASE(DataType::TYPE_QFP8_E4M3, __nv_fp8_e4m3);
#endif
#if USING_ROCM
        CASE(DataType::TYPE_FP8_E4M3, __hip_fp8_e4m3_fnuz);
        CASE(DataType::TYPE_QFP8_E4M3, __hip_fp8_e4m3_fnuz);
#endif
        case TYPE_QINT4X2: {
            return 4;
        }
        case TYPE_INT4X2: {
            return 4;
        }
        default:
            return 0;
    }
#undef CASE
}

}  // namespace rtp_llm

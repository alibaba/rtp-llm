#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int8_t cast_to_int8(float val) {
    // https://github.com/vllm-project/vllm/blob/c5832d2ae9431a1672d547c232ec46b1a9051ff0/csrc/quantization/compressed_tensors/int8_quant_kernels.cu#L8-L25
#ifdef USING_ROCM
    static const float i8_min = static_cast<float>(std::numeric_limits<int8_t>::min());
    static const float i8_max = static_cast<float>(std::numeric_limits<int8_t>::max());
    // round
    float dst = std::nearbyint(val);
    // saturate
    dst = std::clamp(dst, i8_min, i8_max);
    return static_cast<int8_t>(dst);
#else
    union {
        int8_t  int8[2];
        int16_t int16;
    };

    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
    return int8[0];
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int32_t cast_to_int8(float2 val) {
    union {
        int8_t  int8[2];
        int32_t int32;
    };

    int8[0] = cast_to_int8(val.x);
    int8[1] = cast_to_int8(val.y);
    return int32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int32_t cast_to_int8(float4 val) {
    union {
        int8_t  int8[4];
        int32_t int32;
    };

    int8[0] = cast_to_int8(val.x);
    int8[1] = cast_to_int8(val.y);
    int8[2] = cast_to_int8(val.z);
    int8[3] = cast_to_int8(val.w);
    return int32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int64_t cast_to_int8(Float8_ val) {
    union {
        int8_t  int8[8];
        int64_t int64;
    };

    int8[0] = cast_to_int8(val.x.x);
    int8[1] = cast_to_int8(val.x.y);
    int8[2] = cast_to_int8(val.y.x);
    int8[3] = cast_to_int8(val.y.y);
    int8[4] = cast_to_int8(val.z.x);
    int8[5] = cast_to_int8(val.z.y);
    int8[6] = cast_to_int8(val.w.x);
    int8[7] = cast_to_int8(val.w.y);
    return int64;
}

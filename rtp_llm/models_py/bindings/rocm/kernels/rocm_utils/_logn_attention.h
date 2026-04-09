#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __inline__ void logn_attention(float& vec, const int seq_idx, const int logn_seq_len) {
    float logn = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        vec = vec * logn;
    }
}

__device__ __inline__ void logn_attention(float2& vec, const int seq_idx, const int logn_seq_len) {
    float logn = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        vec.x = vec.x * logn;
        vec.y = vec.y * logn;
    }
}

__device__ __inline__ void logn_attention(uint32_t& vec, const int seq_idx, const int logn_seq_len) {
    float2 result = half2_to_float2(vec);
    float  logn   = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        result.x = result.x * logn;
        result.y = result.y * logn;
    }
    vec = float2_to_half2(result);
}

__device__ __inline__ void logn_attention(float4& vec, const int seq_idx, const int logn_seq_len) {
    float logn = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        vec.x = vec.x * logn;
        vec.y = vec.y * logn;
        vec.z = vec.z * logn;
        vec.w = vec.w * logn;
    }
}

__device__ __inline__ void logn_attention(uint2& vec, const int seq_idx, const int logn_seq_len) {
    float2 result0 = half2_to_float2(vec.x);
    float2 result1 = half2_to_float2(vec.y);
    float  logn    = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        result0.x = result0.x * logn;
        result0.y = result0.y * logn;
        result1.x = result1.x * logn;
        result1.y = result1.y * logn;
    }
    vec.x = float2_to_half2(result0);
    vec.y = float2_to_half2(result1);
}

__device__ __inline__ void logn_attention(uint4& vec, const int seq_idx, const int logn_seq_len) {
    float2 result0 = half2_to_float2(vec.x);
    float2 result1 = half2_to_float2(vec.y);
    float2 result2 = half2_to_float2(vec.z);
    float2 result3 = half2_to_float2(vec.w);
    float  logn    = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        result0.x = result0.x * logn;
        result0.y = result0.y * logn;
        result1.x = result1.x * logn;
        result1.y = result1.y * logn;
        result2.x = result2.x * logn;
        result2.y = result2.y * logn;
        result3.x = result3.x * logn;
        result3.y = result3.y * logn;
    }
    vec.x = float2_to_half2(result0);
    vec.y = float2_to_half2(result1);
    vec.z = float2_to_half2(result2);
    vec.w = float2_to_half2(result3);
}

#ifdef ENABLE_BF16

__device__ __inline__ void logn_attention(__nv_bfloat162& vec, const int seq_idx, const int logn_seq_len) {
    if (seq_idx > logn_seq_len) {
        __nv_bfloat16 scalar = __nv_bfloat16((logf(seq_idx + 1) / logf(logn_seq_len)));
        vec.x                = vec.x * scalar;
        vec.y                = vec.y * scalar;
    }
}

__device__ __inline__ void logn_attention(__nv_bfloat16& vec, const int seq_idx, const int logn_seq_len) {
    if (seq_idx > logn_seq_len) {
        __nv_bfloat16 scalar = __nv_bfloat16((logf(seq_idx + 1) / logf(logn_seq_len)));
        vec                  = vec * scalar;
    }
}

__device__ __inline__ void logn_attention(bf16_8_t& vec, const int seq_idx, const int logn_seq_len) {
    if (seq_idx > logn_seq_len) {
        __nv_bfloat16  scalar  = __nv_bfloat16((logf(seq_idx + 1) / logf(logn_seq_len)));
        __nv_bfloat162 scalar2 = __nv_bfloat162(scalar, scalar);
        vec.x                  = vec.x * scalar2;
        vec.y                  = vec.y * scalar2;
        vec.z                  = vec.z * scalar2;
        vec.w                  = vec.w * scalar2;
    }
}

__device__ __inline__ void logn_attention(bf16_4_t& vec, const int seq_idx, const int logn_seq_len) {
    if (seq_idx > logn_seq_len) {
        __nv_bfloat16  scalar  = __nv_bfloat16((logf(seq_idx + 1) / logf(logn_seq_len)));
        __nv_bfloat162 scalar2 = __nv_bfloat162(scalar, scalar);
        vec.x                  = vec.x * scalar2;
        vec.y                  = vec.y * scalar2;
    }
}

#endif

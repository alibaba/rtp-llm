#pragma once

#include "decoder_masked_multihead_attention_utils.h"
#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif
#include <type_traits>

namespace rtp_llm {

template<typename scalar_t, typename vector_t>
struct vector_size {};

template<>
struct vector_size<half, uint32_t> {
    static constexpr int size = 2;
};

template<>
struct vector_size<half, uint4> {
    static constexpr int size = 8;
};

template<>
struct vector_size<float, float> {
    static constexpr int size = 1;
};

template<>
struct vector_size<float, float2> {
    static constexpr int size = 2;
};

template<>
struct vector_size<float, float4> {
    static constexpr int size = 4;
};

template<>
struct vector_size<float, Float8_> {
    static constexpr int size = 8;
};

template<>
struct vector_size<uint16_t, uint32_t> {
    static constexpr int size = 2;
};

template<>
struct vector_size<uint16_t, uint2> {
    static constexpr int size = 4;
};

template<>
struct vector_size<uint16_t, uint4> {
    static constexpr int size = 8;
};

#ifdef ENABLE_BF16
template<>
struct vector_size<__nv_bfloat16, __nv_bfloat162> {
    static constexpr int size = 2;
};

template<>
struct vector_size<__nv_bfloat16, bf16_4_t> {
    static constexpr int size = 4;
};

template<>
struct vector_size<__nv_bfloat16, bf16_8_t> {
    static constexpr int size = 8;
};
#endif

template<typename scalar_t, typename vector_t>
struct is_alignment {};

template<>
struct is_alignment<float, float> {
    static constexpr bool value = true;
};

template<>
struct is_alignment<float, float2> {
    static constexpr bool value = true;
};

template<>
struct is_alignment<float, float4> {
    static constexpr bool value = true;
};

template<>
struct is_alignment<float, Float8_> {
    static constexpr bool value = false;
};

template<>
struct is_alignment<uint16_t, uint32_t> {
    static constexpr bool value = false;
};

template<>
struct is_alignment<uint16_t, uint2> {
    static constexpr bool value = false;
};

template<>
struct is_alignment<uint16_t, uint4> {
    static constexpr bool value = false;
};

template<>
struct is_alignment<half, uint32_t> {
    static constexpr bool value = false;
};

template<>
struct is_alignment<half, uint4> {
    static constexpr bool value = false;
};

#ifdef ENABLE_BF16
template<>
struct is_alignment<__nv_bfloat16, __nv_bfloat162> {
    static constexpr bool value = true;
};

template<>
struct is_alignment<__nv_bfloat16, bf16_4_t> {
    static constexpr bool value = false;
};

template<>
struct is_alignment<__nv_bfloat16, bf16_8_t> {
    static constexpr bool value = false;
};
#endif

template<typename scalar_t, typename vector_t>
struct assign {
    static __device__ __inline__ void read(vector_t& vec, scalar_t& x) {};

    static __device__ __inline__ void read2(vector_t& vec, scalar_t& x, scalar_t& y) {
        if constexpr (is_alignment<scalar_t, vector_t>::value) {
            vec.x = x;
            vec.y = y;
        } else {
            union {
                vector_t r;
                scalar_t t[2];
            } result;
            result.t[0] = x;
            result.t[1] = y;
            vec         = result.r;
        }
    };

    static __device__ __inline__ void read4(vector_t& vec, scalar_t& x, scalar_t& y, scalar_t& z, scalar_t& w) {
        if constexpr (is_alignment<scalar_t, vector_t>::value) {
            vec.x = x;
            vec.y = y;
            vec.z = z;
            vec.w = w;
        } else {
            union {
                vector_t r;
                scalar_t t[4];
            } result;
            result.t[0] = x;
            result.t[1] = y;
            result.t[2] = z;
            result.t[3] = w;
            vec         = result.r;
        }
    };

    static __device__ __inline__ void read8(vector_t& vec,
                                            scalar_t& x,
                                            scalar_t& y,
                                            scalar_t& z,
                                            scalar_t& w,
                                            scalar_t& a,
                                            scalar_t& b,
                                            scalar_t& c,
                                            scalar_t& d) {
        if constexpr (is_alignment<scalar_t, vector_t>::value) {
            return;
        } else {
            union {
                vector_t r;
                scalar_t t[8];
            } result;
            result.t[0] = x;
            result.t[1] = y;
            result.t[2] = z;
            result.t[3] = w;
            result.t[4] = a;
            result.t[5] = b;
            result.t[6] = c;
            result.t[7] = d;
            vec         = result.r;
        }
    };

    static __device__ __inline__ void write(vector_t& vec, scalar_t& x) {};

    static __device__ __inline__ void write2(vector_t& vec, scalar_t& x, scalar_t& y) {
        if constexpr (is_alignment<scalar_t, vector_t>::value) {
            x = vec.x;
            y = vec.y;
        } else {
            union {
                vector_t r;
                scalar_t t[2];
            } result;
            result.r = vec;
            x        = result.t[0];
            y        = result.t[1];
        }
    };

    static __device__ __inline__ void write4(vector_t& vec, scalar_t& x, scalar_t& y, scalar_t& z, scalar_t& w) {
        if constexpr (is_alignment<scalar_t, vector_t>::value) {
            x = vec.x;
            y = vec.y;
            z = vec.z;
            w = vec.w;
        } else {
            union {
                vector_t r;
                scalar_t t[4];
            } result;
            result.r = vec;
            x        = result.t[0];
            y        = result.t[1];
            z        = result.t[2];
            w        = result.t[3];
        }
    };

    static __device__ __inline__ void write8(vector_t& vec,
                                             scalar_t& x,
                                             scalar_t& y,
                                             scalar_t& z,
                                             scalar_t& w,
                                             scalar_t& a,
                                             scalar_t& b,
                                             scalar_t& c,
                                             scalar_t& d) {
        if constexpr (is_alignment<scalar_t, vector_t>::value) {
            return;
        } else {
            union {
                vector_t r;
                scalar_t t[8];
            } result;
            result.r = vec;
            x        = result.t[0];
            y        = result.t[1];
            z        = result.t[2];
            w        = result.t[3];
            a        = result.t[4];
            b        = result.t[5];
            c        = result.t[6];
            d        = result.t[7];
        }
    };
};

template<typename vector_t, typename scalar_t>
__device__ __inline__ void RotaryHalfRead(vector_t& vec, scalar_t* smem, const int idx, int dim) {
    constexpr int size = vector_size<scalar_t, vector_t>::size;
    static_assert(size == 1 || size == 2 || size == 4 || size == 8, "vector size is not valid");
    if constexpr (size == 2) {
        assign<scalar_t, vector_t>::read2(vec, smem[idx], smem[idx + dim]);
    } else if constexpr (size == 4) {
        assign<scalar_t, vector_t>::read4(
            vec, smem[idx * 2], smem[idx * 2 + dim], smem[idx * 2 + 1], smem[idx * 2 + 1 + dim]);
    } else if constexpr (size == 8) {
        assign<scalar_t, vector_t>::read8(vec,
                                          smem[idx * 4],
                                          smem[idx * 4 + dim],
                                          smem[idx * 4 + 1],
                                          smem[idx * 4 + 1 + dim],
                                          smem[idx * 4 + 2],
                                          smem[idx * 4 + 2 + dim],
                                          smem[idx * 4 + 3],
                                          smem[idx * 4 + 3 + dim]);
    }
}

template<typename vector_t, typename scalar_t>
__device__ __inline__ void RotaryHalfWrite(vector_t& vec, scalar_t* smem, const int idx, int dim) {
    constexpr int size = vector_size<scalar_t, vector_t>::size;
    static_assert(size == 1 || size == 2 || size == 4 || size == 8, "vector size is not valid");
    if constexpr (size == 2) {
        assign<scalar_t, vector_t>::write2(vec, smem[idx], smem[idx + dim]);
    } else if constexpr (size == 4) {
        assign<scalar_t, vector_t>::write4(
            vec, smem[idx * 2], smem[idx * 2 + dim], smem[idx * 2 + 1], smem[idx * 2 + 1 + dim]);
    } else if constexpr (size == 8) {
        assign<scalar_t, vector_t>::write8(vec,
                                           smem[idx * 4],
                                           smem[idx * 4 + dim],
                                           smem[idx * 4 + 1],
                                           smem[idx * 4 + 1 + dim],
                                           smem[idx * 4 + 2],
                                           smem[idx * 4 + 2 + dim],
                                           smem[idx * 4 + 3],
                                           smem[idx * 4 + 3 + dim]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __inline__ float
rope_inv_freq(const int zid, const int rot_embed_dim, const float t_step, const float base) {
    return (t_step / pow(base, zid / (float)rot_embed_dim));
}

template<typename RopeInit>
__device__ __inline__ float2 rotary_embedding_coefficient(
    const int zid, const int rot_embed_dim, const float t_step, const float base, const RopeInit& rope_init) {
    float inv_freq      = rope_inv_freq(zid, rot_embed_dim, t_step, base);
    inv_freq            = rope_init(inv_freq, zid);
    float sin_cos_scale = rope_init.sin_cos_scale();
#if USING_CUDA
    float sin_i, cos_i;
#endif

#if USING_ROCM
    double sin_i, cos_i;
#endif
    sincos(inv_freq, &sin_i, &cos_i);
    return {sin_cos_scale * cos_i, sin_cos_scale * sin_i};
}

struct DefaultRope {
    __device__ __inline__ float operator()(float inv_freq, int zid) const {
        return inv_freq;
    }
    __device__ __inline__ float sin_cos_scale() const {
        return 1.0;
    }
};

struct LinearScaleRope {
    float                       scale = 1.0;
    __device__ __inline__ float operator()(float inv_freq, int zid) const {
        return inv_freq / scale;
    }

    __device__ __inline__ float sin_cos_scale() const {
        return 1.0;
    }
};

struct YarnRope {
    int   dim;
    int   base;
    int   max_pos;
    float beta_slow;
    float beta_fast;
    float scaling_factor;
    float extrapolation_factor;
    float mscale;

    static __device__ __inline__ float find_correction_dim(const int num_rotations,
                                                           const int dim,
                                                           const int base,
                                                           const int max_position_embeddings = 2048) {

        float pi = 3.141592654f;
        float t0 = dim * logf((float)max_position_embeddings / (num_rotations * 2 * pi));
        float t1 = 2 * logf((float)base);
        return (t0 / t1);
    }

    static __device__ __inline__ float2
    find_correction_range(float low_rot, float high_rot, int dim, int base, int max_position_embeddings) {
        float2 low_high;
        int    low  = floor(find_correction_dim(low_rot, dim, base, max_position_embeddings));
        int    high = ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings));
        low_high.x  = max(low, 0);
        low_high.y  = min(high, dim - 1);
        return low_high;
    }

    static __device__ __inline__ float linear_ramp_mask(float min_, float max_, int tidx) {
        if (min_ == max_) {
            max_ += 0.001;
        }
        float linear = (tidx / 2 - min_) / (max_ - min_);
        return std::min(1.f, std::max(0.f, linear));
    }

    __device__ __inline__ float operator()(float inv_freq, int zid) const {
        float2      low_high   = find_correction_range(beta_fast, beta_slow, dim, base, max_pos);
        const float inv_freq_e = inv_freq;
        const float inv_freq_i = inv_freq_e / scaling_factor;
        const float mask       = (1 - linear_ramp_mask(low_high.x, low_high.y, zid)) * extrapolation_factor;
        return inv_freq_i * (1 - mask) + inv_freq_e * mask;
    }

    __device__ __inline__ float sin_cos_scale() const {
        return mscale;
    }
};

struct Llama3Rope {
    float low_freq_factor;
    float high_freq_factor;
    float factor;
    int   old_context_len;

    __device__ __inline__ float operator()(float inv_freq, int zid) const {
        const float pi                = 3.141592654f;
        const float wavelen           = 2 * pi / inv_freq;
        const float low_freq_wavelen  = old_context_len / low_freq_factor;
        const float high_freq_wavelen = old_context_len / high_freq_factor;
        if (wavelen < high_freq_wavelen) {
            return inv_freq;
        } else if (wavelen > low_freq_wavelen) {
            return inv_freq / factor;
        } else {
            const float smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            return (1 - smooth) * inv_freq / factor + smooth * inv_freq;
        }
    }

    __device__ __inline__ float sin_cos_scale() const {
        return 1.0;
    }
};

__device__ __inline__ float2 rotary_embedding_transform(const float2 v, const float2& coef) {
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}

__device__ __inline__ float2 rotary_embedding_transform(
    const float2 v, const float2 v_permuted, const float2& coef1, const float2& coef2, const bool is_front) {
    float2 rot_v;
    rot_v.x = v.x * coef1.x + (is_front ? -v_permuted.x : v_permuted.x) * coef1.y;
    rot_v.y = v.y * coef2.x + (is_front ? -v_permuted.y : v_permuted.y) * coef2.y;
    return rot_v;
}

__device__ __inline__ Float8_ rotary_embedding_transform(const Float8_ v, const Float8_& coef) {
    Float8_ rot_v;
    rot_v.x = rotary_embedding_transform(v.x, coef.x);
    rot_v.y = rotary_embedding_transform(v.y, coef.y);
    rot_v.z = rotary_embedding_transform(v.z, coef.z);
    rot_v.w = rotary_embedding_transform(v.w, coef.w);

    return rot_v;
}

__device__ __inline__ void rotary_embedding_transform(
    Float8_& v, const Float8_& v_permuted, const Float8_& coef1, const Float8_& coef2, const bool is_front) {
    v.x = rotary_embedding_transform(v.x, v_permuted.x, coef1.x, coef1.y, is_front);
    v.y = rotary_embedding_transform(v.y, v_permuted.y, coef1.z, coef1.w, is_front);
    v.z = rotary_embedding_transform(v.z, v_permuted.z, coef2.x, coef2.y, is_front);
    v.w = rotary_embedding_transform(v.w, v_permuted.w, coef2.z, coef2.w, is_front);
}

__device__ __inline__ uint32_t rotary_embedding_transform(const uint32_t v, const float2& coef) {
    float2 fv     = half2_to_float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return float2_to_half2(rot_fv);
}

__device__ __inline__ uint4 rotary_embedding_transform(const uint4 v, const Float8_& coef) {
    Float8_ fv;
    Float8_ rot_fv;
    uint4   rot_v;
    fv.x     = half2_to_float2(v.x);
    fv.y     = half2_to_float2(v.y);
    rot_fv.x = rotary_embedding_transform(fv.x, coef.x);
    fv.z     = half2_to_float2(v.z);
    rot_fv.y = rotary_embedding_transform(fv.y, coef.y);
    rot_v.x  = float2_to_half2(rot_fv.x);
    fv.w     = half2_to_float2(v.w);
    rot_fv.z = rotary_embedding_transform(fv.z, coef.z);
    rot_v.y  = float2_to_half2(rot_fv.y);
    rot_fv.w = rotary_embedding_transform(fv.w, coef.w);
    rot_v.z  = float2_to_half2(rot_fv.z);
    rot_v.w  = float2_to_half2(rot_fv.w);

    return rot_v;
}

__device__ __inline__ void rotary_embedding_transform(
    uint4& v, const uint4& v_permuted, const Float8_& coef1, const Float8_& coef2, const bool is_front) {
    Float8_ fv;
    Float8_ fv_permuted;
    Float8_ rot_fv;
    fv.x          = half2_to_float2(v.x);
    fv_permuted.x = half2_to_float2(v_permuted.x);
    fv.y          = half2_to_float2(v.y);
    fv_permuted.y = half2_to_float2(v_permuted.y);
    rot_fv.x      = rotary_embedding_transform(fv.x, fv_permuted.x, coef1.x, coef1.y, is_front);
    fv.z          = half2_to_float2(v.z);
    fv_permuted.z = half2_to_float2(v_permuted.z);
    rot_fv.y      = rotary_embedding_transform(fv.y, fv_permuted.y, coef1.z, coef1.w, is_front);
    v.x           = float2_to_half2(rot_fv.x);
    fv.w          = half2_to_float2(v.w);
    fv_permuted.w = half2_to_float2(v_permuted.w);
    rot_fv.z      = rotary_embedding_transform(fv.z, fv_permuted.z, coef2.x, coef2.y, is_front);
    v.y           = float2_to_half2(rot_fv.y);
    rot_fv.w      = rotary_embedding_transform(fv.w, fv_permuted.w, coef2.z, coef2.w, is_front);
    v.z           = float2_to_half2(rot_fv.z);
    v.w           = float2_to_half2(rot_fv.w);
}

/**
 * Rotary position embedding
 * Reference: https://arxiv.org/abs/2309.00071
 *
 * Decoder:
 *      F(x, pos)  = dot(R(pos, ReIndex(x)), Re(x)) + dot(R'(pos, IeIndex(x)), Ie(x))
 *      R(pos, i)  = [cos(pos * Base(i)), -sin(pos * Base(i)]
 *      R'(pos, i) = [sin(pos * Base(i)), cos(pos * Base()]
 *         Base(i) = 1 / (base^(2*i / dim))
 *
 */
#ifdef ENABLE_BF16
__device__ __inline__ __nv_bfloat162 rotary_embedding_transform(const __nv_bfloat162 v, const float2& coef) {
    float2 fv     = bf1622float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return __floats2bfloat162_rn(rot_fv.x, rot_fv.y);
}

__device__ __inline__ bf16_8_t rotary_embedding_transform(const bf16_8_t v, const Float8_& coef) {
    Float8_  fv;
    Float8_  rot_fv;
    bf16_8_t rot_v;
    fv.x     = bf1622float2(v.x);
    fv.y     = bf1622float2(v.y);
    rot_fv.x = rotary_embedding_transform(fv.x, coef.x);
    fv.z     = bf1622float2(v.z);
    rot_fv.y = rotary_embedding_transform(fv.y, coef.y);
    rot_v.x  = __floats2bfloat162_rn(rot_fv.x.x, rot_fv.x.y);
    fv.w     = bf1622float2(v.w);
    rot_fv.z = rotary_embedding_transform(fv.z, coef.z);
    rot_v.y  = __floats2bfloat162_rn(rot_fv.y.x, rot_fv.y.y);
    rot_fv.w = rotary_embedding_transform(fv.w, coef.w);
    rot_v.z  = __floats2bfloat162_rn(rot_fv.z.x, rot_fv.z.y);
    rot_v.w  = __floats2bfloat162_rn(rot_fv.w.x, rot_fv.w.y);

    return rot_v;
}

__device__ __inline__ void rotary_embedding_transform(
    bf16_8_t& v, const bf16_8_t& v_permuted, const Float8_& coef1, const Float8_& coef2, const bool is_front) {
    Float8_ fv;
    Float8_ fv_permuted;
    Float8_ rot_fv;
    fv.x          = bf1622float2(v.x);
    fv_permuted.x = bf1622float2(v_permuted.x);
    fv.y          = bf1622float2(v.y);
    fv_permuted.y = bf1622float2(v_permuted.y);
    rot_fv.x      = rotary_embedding_transform(fv.x, fv_permuted.x, coef1.x, coef1.y, is_front);
    fv.z          = bf1622float2(v.z);
    fv_permuted.z = bf1622float2(v_permuted.z);
    rot_fv.y      = rotary_embedding_transform(fv.y, fv_permuted.y, coef1.z, coef1.w, is_front);
    v.x           = __floats2bfloat162_rn(rot_fv.x.x, rot_fv.x.y);
    fv.w          = bf1622float2(v.w);
    fv_permuted.w = bf1622float2(v_permuted.w);
    rot_fv.z      = rotary_embedding_transform(fv.z, fv_permuted.z, coef2.x, coef2.y, is_front);
    v.y           = __floats2bfloat162_rn(rot_fv.y.x, rot_fv.y.y);
    rot_fv.w      = rotary_embedding_transform(fv.w, fv_permuted.w, coef2.z, coef2.w, is_front);
    v.z           = __floats2bfloat162_rn(rot_fv.z.x, rot_fv.z.y);
    v.w           = __floats2bfloat162_rn(rot_fv.w.x, rot_fv.w.y);
}
#endif

template<typename RopeInit>
__device__ __inline__ void apply_rotary_embedding(float2&         q,
                                                  int             tid,
                                                  int             rot_embed_dim,
                                                  int             t_step,
                                                  float           base,
                                                  const RopeInit& rope_init,
                                                  const float2*   cos_sin_cache = nullptr) {
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    float2 coef;
    if (cos_sin_cache) {
        coef = cos_sin_cache[t_step * rot_embed_dim / 2 + tid];
    } else {
        coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step, base, rope_init);
    }
    q = rotary_embedding_transform(q, coef);
}

template<typename RopeInit>
__device__ __inline__ void apply_rotary_embedding(float4&         q,
                                                  int             tid,
                                                  int             rot_embed_dim,
                                                  int             t_step,
                                                  float           base,
                                                  const RopeInit& rope_init,
                                                  const float2*   cos_sin_cache = nullptr) {
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
    float2   coef0;
    float2   coef1;
    if (cos_sin_cache) {
        coef0 = cos_sin_cache[t_step * rot_embed_dim / 2 + 2 * tid];
        coef1 = cos_sin_cache[t_step * rot_embed_dim / 2 + 2 * tid + 1];
    } else {
        coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step, base, rope_init);
        coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step, base, rope_init);
    }
    q_.x = rotary_embedding_transform(q_.x, coef0);
    q_.y = rotary_embedding_transform(q_.y, coef1);
}

template<typename RopeInit>
__device__ __inline__ void apply_rotary_embedding(uint32_t&       q,
                                                  int             tid,
                                                  int             rot_embed_dim,
                                                  int             t_step,
                                                  float           base,
                                                  const RopeInit& rope_init,
                                                  const float2*   cos_sin_cache = nullptr) {
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    float2 coef;
    if (cos_sin_cache) {
        coef = cos_sin_cache[t_step * rot_embed_dim / 2 + tid];
    } else {
        coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step, base, rope_init);
    }
    q = rotary_embedding_transform(q, coef);
}

template<typename RopeInit>
__device__ __inline__ void apply_rotary_embedding(uint2&          q,
                                                  int             tid,
                                                  int             rot_embed_dim,
                                                  int             t_step,
                                                  float           base,
                                                  const RopeInit& rope_init,
                                                  const float2*   cos_sin_cache = nullptr) {
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    float2 coef0;
    float2 coef1;
    if (cos_sin_cache) {
        coef0 = cos_sin_cache[t_step * rot_embed_dim / 2 + 2 * tid];
        coef1 = cos_sin_cache[t_step * rot_embed_dim / 2 + 2 * tid + 1];
    } else {
        coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step, base, rope_init);
        coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step, base, rope_init);
    }
    q.x = rotary_embedding_transform(q.x, coef0);
    q.y = rotary_embedding_transform(q.y, coef1);
}

template<typename RopeInit>
__device__ __inline__ void apply_rotary_embedding(uint4&          q,
                                                  int             tid,
                                                  int             rot_embed_dim,
                                                  int             t_step,
                                                  float           base,
                                                  const RopeInit& rope_init,
                                                  const float2*   cos_sin_cache = nullptr) {
    if (8 * tid >= rot_embed_dim) {
        return;
    }

    float2 coef0;
    float2 coef1;
    float2 coef2;
    float2 coef3;
    if (cos_sin_cache) {
        coef0 = cos_sin_cache[t_step * rot_embed_dim / 2 + 4 * tid];
        coef1 = cos_sin_cache[t_step * rot_embed_dim / 2 + 4 * tid + 1];
        coef2 = cos_sin_cache[t_step * rot_embed_dim / 2 + 4 * tid + 2];
        coef3 = cos_sin_cache[t_step * rot_embed_dim / 2 + 4 * tid + 3];
    } else {
        coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step, base, rope_init);
        coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step, base, rope_init);
        coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step, base, rope_init);
        coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step, base, rope_init);
    }

    q.x = rotary_embedding_transform(q.x, coef0);
    q.y = rotary_embedding_transform(q.y, coef1);
    q.z = rotary_embedding_transform(q.z, coef2);
    q.w = rotary_embedding_transform(q.w, coef3);
}

#ifdef ENABLE_BF16

template<typename RopeInit>
__device__ __inline__ void apply_rotary_embedding(__nv_bfloat162& q,
                                                  int             tid,
                                                  int             rot_embed_dim,
                                                  int             t_step,
                                                  float           base,
                                                  const RopeInit& rope_init,
                                                  const float2*   cos_sin_cache = nullptr) {
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    float2 coef;
    if (cos_sin_cache) {
        coef = cos_sin_cache[t_step * rot_embed_dim / 2 + tid];
    } else {
        coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step, base, rope_init);
    }
    q = rotary_embedding_transform(q, coef);
}

template<typename RopeInit>
__device__ __inline__ void apply_rotary_embedding(bf16_4_t&       q,
                                                  int             tid,
                                                  int             rot_embed_dim,
                                                  int             t_step,
                                                  float           base,
                                                  const RopeInit& rope_init,
                                                  const float2*   cos_sin_cache = nullptr) {
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    float2 coef0;
    float2 coef1;
    if (cos_sin_cache) {
        coef0 = cos_sin_cache[t_step * rot_embed_dim / 2 + 2 * tid];
        coef1 = cos_sin_cache[t_step * rot_embed_dim / 2 + 2 * tid + 1];
    } else {
        coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step, base, rope_init);
        coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step, base, rope_init);
    }
    q.x = rotary_embedding_transform(q.x, coef0);
    q.y = rotary_embedding_transform(q.y, coef1);
}

template<typename RopeInit>
__device__ __inline__ void apply_rotary_embedding(bf16_8_t&       q,
                                                  int             tid,
                                                  int             rot_embed_dim,
                                                  int             t_step,
                                                  float           base,
                                                  const RopeInit& rope_init,
                                                  const float2*   cos_sin_cache = nullptr) {
    if (8 * tid >= rot_embed_dim) {
        return;
    }

    float2 coef0;
    float2 coef1;
    float2 coef2;
    float2 coef3;
    if (cos_sin_cache) {
        coef0 = cos_sin_cache[t_step * rot_embed_dim / 2 + 4 * tid];
        coef1 = cos_sin_cache[t_step * rot_embed_dim / 2 + 4 * tid + 1];
        coef2 = cos_sin_cache[t_step * rot_embed_dim / 2 + 4 * tid + 2];
        coef3 = cos_sin_cache[t_step * rot_embed_dim / 2 + 4 * tid + 3];
    } else {
        coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step, base, rope_init);
        coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step, base, rope_init);
        coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step, base, rope_init);
        coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step, base, rope_init);
    }
    q.x = rotary_embedding_transform(q.x, coef0);
    q.y = rotary_embedding_transform(q.y, coef1);
    q.z = rotary_embedding_transform(q.z, coef2);
    q.w = rotary_embedding_transform(q.w, coef3);
}

#endif  // ENABLE_BF16

template<typename RopeInit, typename scalar_t, typename vector_t>
__device__ __inline__ void normal_rope(vector_t&       x,
                                       scalar_t*       smem,
                                       const int       tidx,
                                       const int       seqidx,
                                       const int       dim,
                                       const float     base,
                                       const RopeInit& rope_init,
                                       const int       offset        = 0,
                                       const float2*   cos_sin_cache = nullptr) {
    const int  vec_size  = vector_size<scalar_t, vector_t>::size;
    const int  rope_idx  = tidx * vec_size - offset;
    const bool work      = (rope_idx >= 0 && rope_idx < dim);
    const int  rope_tidx = rope_idx / vec_size;

    if (work) {
        reinterpret_cast<vector_t*>(smem)[rope_tidx] = x;
    }

    __syncthreads();
    if (work) {
        RotaryHalfRead(x, smem, rope_tidx, dim / 2);
        apply_rotary_embedding(x, rope_tidx, dim, seqidx, base, rope_init, cos_sin_cache);
        RotaryHalfWrite(x, smem, rope_tidx, dim / 2);
    }

    __syncthreads();

    if (work) {
        x = reinterpret_cast<vector_t*>(smem)[rope_tidx];
    }
}

template<typename vector_t, typename scalar_t, typename rope_t>
__device__ __inline__ void normal_rope_with_cache(
    vector_t& x, scalar_t* smem, const int tidx, const int dim, const rope_t& coef, const bool work) {
    if (work) {
        if constexpr (std::is_same_v<rope_t, Float8_>
                      && (std::is_same_v<scalar_t, half> || std::is_same_v<scalar_t, __nv_bfloat16>)) {
            reinterpret_cast<int4*>(smem)[tidx] = *reinterpret_cast<int4*>(&x);
        } else {
            reinterpret_cast<vector_t*>(smem)[tidx] = x;
        }
    }

    __syncthreads();

    if (work) {
        RotaryHalfRead(x, smem, tidx, dim / 2);
        x = rotary_embedding_transform(x, coef);
        RotaryHalfWrite(x, smem, tidx, dim / 2);
    }

    __syncthreads();

    if (work) {
        if constexpr (std::is_same_v<rope_t, Float8_>
                      && (std::is_same_v<scalar_t, half> || std::is_same_v<scalar_t, __nv_bfloat16>)) {
            *reinterpret_cast<int4*>(&x) = reinterpret_cast<int4*>(smem)[tidx];
        } else {
            x = reinterpret_cast<vector_t*>(smem)[tidx];
        }
    }

    __syncthreads();
}

template<typename vector_t, typename scalar_t, typename rope_t>
__device__ __inline__ void normal_rope_with_cache(vector_t&       x,
                                                  const vector_t& x_permuted,
                                                  const rope_t&   coef1,
                                                  const rope_t&   coef2,
                                                  const bool      work,
                                                  const bool      is_front) {
    if (work) {
        rotary_embedding_transform(x, x_permuted, coef1, coef2, is_front);
    }
}

template<typename RopeInit, typename scalar_t, typename vector_t>
__device__ __inline__ void
glm2_rope(vector_t& x, const int tidx, const int seqidx, const int dim, const float base, const RopeInit& rope_init) {
    const int  vec_size = vector_size<scalar_t, vector_t>::size;
    const bool work     = (tidx * vec_size < dim);

    if (work) {
        apply_rotary_embedding(x, tidx, dim, seqidx, base, rope_init);
    }
}

__device__ __inline__ float
get_dynamic_ntk_base(const int dim, const float base, const int seq_len, const float scale, const int max_pos) {
    float new_base = base * 1.0f * pow(((scale * seq_len / max_pos) - (scale - 1)), (dim / (dim - 2.0f)));
    return new_base;
}

__device__ __inline__ float
get_qwen_dynamic_ntk_base(const int dim, const float base, const int seq_len, const int max_pos) {
    float context_value = logf((float)seq_len / max_pos) / logf(2.0) + 1.0;
    float ntk_scalar    = pow(2.0, ceil(context_value)) - 1;
    ntk_scalar          = max(ntk_scalar, 1.0);
    float new_base      = base * pow((float)ntk_scalar, (float)dim / (dim - 2));
    return new_base;
}

template<typename scalar_t, typename vector_t, RopeStyle ROPE_STYLE>
__device__ inline void apply_rope(RopeConfig    rope_config,
                                  vector_t&     x,
                                  scalar_t*     smem,
                                  int           tidx,
                                  int           seqidx,
                                  int           seq_len,
                                  const float2* cos_sin_cache = nullptr) {
    auto base = rope_config.base;
    auto dim  = rope_config.dim;
    switch (ROPE_STYLE) {
        case RopeStyle::No:
            break;
        case RopeStyle::Base:
            normal_rope(x, smem, tidx, seqidx, dim, base, LinearScaleRope{rope_config.scale}, 0, cos_sin_cache);
            break;
        case RopeStyle::Glm2:
            // only do rotary embedding for [..., d / 2]
            glm2_rope<LinearScaleRope, scalar_t, vector_t>(
                x, tidx, seqidx, dim / 2, base, LinearScaleRope{rope_config.scale});
            break;
        case RopeStyle::DynamicNTK:
            if (seq_len > rope_config.max_pos) {
                base = get_dynamic_ntk_base(dim, base, seq_len, rope_config.scale, rope_config.max_pos);
            }
            normal_rope(x, smem, tidx, seqidx, dim, base, DefaultRope{});
            break;
        case RopeStyle::Yarn:
            normal_rope(x,
                        smem,
                        tidx,
                        seqidx,
                        dim,
                        base,
                        YarnRope{rope_config.dim,
                                 rope_config.base,
                                 rope_config.max_pos,
                                 rope_config.factor1,
                                 rope_config.factor2,
                                 rope_config.scale,
                                 rope_config.extrapolation_factor,
                                 rope_config.mscale},
                        rope_config.offset,
                        cos_sin_cache);
            break;
        case RopeStyle::QwenDynamicNTK:
            if (seq_len > rope_config.max_pos) {
                base = get_qwen_dynamic_ntk_base(dim, base, seq_len, rope_config.max_pos);
            }
            normal_rope(x, smem, tidx, seqidx, dim, base, DefaultRope{});
            break;
        case RopeStyle::Llama3:
            normal_rope(x,
                        smem,
                        tidx,
                        seqidx,
                        dim,
                        base,
                        Llama3Rope{rope_config.factor1, rope_config.factor2, rope_config.scale, rope_config.max_pos});
            break;
        case RopeStyle::Mrope:
            normal_rope(x, smem, tidx, seqidx, dim, base, LinearScaleRope{rope_config.scale});
            break;
        default:
            break;
    }
}

template<typename vector_t, typename scalar_t, typename rope_t, RopeStyle ROPE_STYLE>
__device__ inline void
apply_rope_with_cache(vector_t& x, scalar_t* smem, const int tidx, const int dim, const rope_t& coef, const bool work) {
    switch (ROPE_STYLE) {
        case RopeStyle::Base:
        case RopeStyle::Yarn:
            normal_rope_with_cache<vector_t, scalar_t, rope_t>(x, smem, tidx, dim, coef, work);
            break;

        default:
            break;
    }
}

template<typename vector_t, typename scalar_t, typename rope_t, RopeStyle ROPE_STYLE>
__device__ inline void apply_rope_with_cache(vector_t&       x,
                                             const vector_t& x_permuted,
                                             const rope_t&   coef1,
                                             const rope_t&   coef2,
                                             const bool      work,
                                             const bool      is_front) {
    switch (ROPE_STYLE) {
        case RopeStyle::Base:
        case RopeStyle::Yarn:
            normal_rope_with_cache<vector_t, scalar_t, rope_t>(x, x_permuted, coef1, coef2, work, is_front);
            break;

        default:
            break;
    }
}

template<typename scalar_t, typename vector_t, RopeStyle ROPE_STYLE>
__device__ inline void context_rope(RopeConfig    rope_config,
                                    vector_t&     q,
                                    vector_t&     k,
                                    scalar_t*     smem,
                                    int           tidx,
                                    int           seqidx,
                                    int           position_id,
                                    int           seq_len,
                                    int           input_len,
                                    bool          PREFIX_PROMPT,
                                    int           prefix_prompt_length,
                                    int           count_length,
                                    const float2* cos_sin_cache = nullptr) {
    if (PREFIX_PROMPT && count_length) {
        input_len = input_len + prefix_prompt_length;
        seqidx    = seqidx + prefix_prompt_length;
    }
    if (position_id > 0) {
        seqidx = position_id;
    }

    apply_rope<scalar_t, vector_t, ROPE_STYLE>(rope_config, q, smem, tidx, seqidx, seq_len, cos_sin_cache);

    apply_rope<scalar_t, vector_t, ROPE_STYLE>(rope_config, k, smem, tidx, seqidx, seq_len, cos_sin_cache);
}

template<typename scalar_t, typename vector_t, RopeStyle ROPE_STYLE>
__device__ inline void attention_rope(RopeConfig rope_config,
                                      vector_t&  q,
                                      vector_t&  k,
                                      scalar_t*  smem,
                                      int        tidx,
                                      int        tlength,
                                      int        timestep,
                                      int        seq_len,
                                      int        position_id,
                                      int        input_len,
#pragma nv_diagnostic push
#pragma nv_diag_suppress 550
                                      [[maybe_unused]] int prefix_prompt_length,
#pragma nv_diagnostic pop
                                      int           count_prefix_length,
                                      bool          handle_kv,
                                      const float2* cos_sin_cache = nullptr) {
    if (count_prefix_length) {
        prefix_prompt_length = 0;
    }

    if (position_id > 0) {
        tlength = position_id;
    }

    if constexpr (ROPE_STYLE == RopeStyle::Glm2) {
        tlength = tlength - prefix_prompt_length;
    }

    apply_rope<scalar_t, vector_t, ROPE_STYLE>(rope_config, q, smem, tidx, tlength, seq_len, cos_sin_cache);

    if (handle_kv) {
        apply_rope<scalar_t, vector_t, ROPE_STYLE>(rope_config, k, smem, tidx, tlength, seq_len, cos_sin_cache);
    }
}

template<typename scalar_t, typename vector_t>
__global__ void launchApplyRopeKernel(scalar_t*  input,
                                      RopeConfig rope_config,
                                      int        head_num,
                                      int        head_size,
                                      int        seq_len,
                                      const int* padding_offset,
                                      const int* prefill_length) {
    extern __shared__ __align__(sizeof(float2)) char smem[];

    const int token_idx            = blockIdx.x;
    const int head_num_idx         = blockIdx.y;
    const int tidx                 = threadIdx.x;
    const int head_size_idx        = tidx * 2;
    const int token_padding_offset = padding_offset == nullptr ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int batch_idx            = tgt_token_idx / seq_len;
    const int token_prefill_length = prefill_length == nullptr ? 0 : prefill_length[batch_idx];
    const int seq_idx              = tgt_token_idx % seq_len + token_prefill_length;

    const bool work = (head_num_idx < head_num && head_size_idx < head_size);

    if (work) {
        vector_t  x;
        const int offset = token_idx * head_num * head_size + head_num_idx * head_size + head_size_idx;

        x = *reinterpret_cast<vector_t*>(&input[offset]);

        FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {
            apply_rope<scalar_t, vector_t, ROPE_STYLE>(
                rope_config, x, reinterpret_cast<scalar_t*>(smem), tidx, seq_idx, seq_len);
        });

        *reinterpret_cast<vector_t*>(&input[offset]) = x;
    }
}
}  // namespace rtp_llm

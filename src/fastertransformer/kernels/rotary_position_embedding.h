#include "decoder_masked_multihead_attention_utils.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"
#if USING_CUDA
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "src/fastertransformer/cuda/cuda_utils.h"
#endif

#if USING_ROCM
#include "src/fastertransformer/rocm/hip_type_utils.cuh"
#include "src/fastertransformer/rocm/hip_utils.h"
#endif
#include <type_traits>

namespace fastertransformer {

template<typename scalar_t, typename vector_t>
struct vector_size {};

template<>
struct vector_size<half, uint32_t> {
    static constexpr int size = 2;
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
    static __device__ __inline__ void read(vector_t& vec, scalar_t& x){};

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
            a        = result.t[5];
            b        = result.t[6];
            c        = result.t[7];
            d        = result.t[8];
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

inline __device__ float2 rotary_embedding_coefficient(const int   zid,
                                                      const int   rot_embed_dim,
                                                      const float t_step,
                                                      const float base           = 10000.0f,
                                                      const float scaling_factor = 1.f) {
    const float inv_freq = (t_step / pow(base, zid / (float)rot_embed_dim)) / scaling_factor;
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float linear_ramp_mask(float min_, float max_, int tidx) {
    if (min_ == max_) {
         max_ += 0.001;
    }
    float linear = (tidx / 2 - min_) / (max_ - min_);
    return std::min(1.f, std::max(0.f, linear));
}

inline __device__ float2 rotary_embedding_coefficient_yarn(const int   zid,
                                                           const int   rot_embed_dim,
                                                           const float t_step,
                                                           const float base,
                                                           const float scaling_factor,
                                                           const float mscale,
                                                           const float low,
                                                           const float high,
                                                           const float extrapolation_factor=1.f){
    const float inv_freq_e = (t_step / pow(base, zid / (float)rot_embed_dim));
    const float inv_freq_i = inv_freq_e / scaling_factor;

    const float mask     = (1 - linear_ramp_mask(low, high, zid)) * extrapolation_factor;
    const float inv_freq = inv_freq_i * (1 - mask) + inv_freq_e * mask;

    return {mscale * cos(inv_freq), mscale * sin(inv_freq)};
}

inline __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef) {
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}

inline __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef) {
    float2 fv     = half2_to_float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return float2_to_half2(rot_fv);
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
inline __device__ __nv_bfloat162 rotary_embedding_transform(const __nv_bfloat162 v, const float2 coef) {
    float2 fv     = bf1622float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return __floats2bfloat162_rn(rot_fv.x, rot_fv.y);
}
#endif

inline __device__ void apply_rotary_embedding(float& q,
                                              int    zid,
                                              int    rot_embed_dim,
                                              int    t_step,
                                              float  base                 = 10000.0f,
                                              float  scaling_factor       = 1.f,
                                              bool   yarn                 = false,
                                              float  mscale               = 1.f,
                                              float  low                  = 0.f,
                                              float  high                 = 0.f,
                                              float  extrapolation_factor = 1.f) {
    return;
}

inline __device__ void apply_rotary_embedding(float2& q,
                                              int     tid,
                                              int     rot_embed_dim,
                                              int     t_step,
                                              float   base                 = 10000.0f,
                                              float   scaling_factor       = 1.f,
                                              bool    yarn                 = false,
                                              float   mscale               = 1.f,
                                              float   low                  = 0.f,
                                              float   high                 = 0.f,
                                              float   extrapolation_factor = 1.f) {
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    float2 coef = {0.f, 0.f};
    if (yarn) {
        coef = rotary_embedding_coefficient_yarn(
            2 * tid, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);

    } else {
        coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step, base, scaling_factor);
    }
    q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(float4& q,
                                              int     tid,
                                              int     rot_embed_dim,
                                              int     t_step,
                                              float   base                 = 10000.0f,
                                              float   scaling_factor       = 1.f,
                                              bool    yarn                 = false,
                                              float   mscale               = 1.f,
                                              float   low                  = 0.f,
                                              float   high                 = 0.f,
                                              float   extrapolation_factor = 1.f) {

    if (4 * tid >= rot_embed_dim) {
        return;
    }

    Float4_& q_    = *reinterpret_cast<Float4_*>(&q);
    float2   coef0 = {0.f, 0.f};
    float2   coef1 = {0.f, 0.f};
    if (yarn) {
        coef0 = rotary_embedding_coefficient_yarn(
            4 * tid, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
        coef1 = rotary_embedding_coefficient_yarn(
            4 * tid + 2, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);

    } else {
        coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step, base, scaling_factor);
        coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step, base, scaling_factor);
    }
    q_.x = rotary_embedding_transform(q_.x, coef0);
    q_.y = rotary_embedding_transform(q_.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint32_t&   q,
                                              int         tid,
                                              int         rot_embed_dim,
                                              int         t_step,
                                              float       base                 = 10000.0f,
                                              const float scaling_factor       = 1.f,
                                              bool        yarn                 = false,
                                              float       mscale               = 1.f,
                                              float       low                  = 0.f,
                                              float       high                 = 0.f,
                                              float       extrapolation_factor = 1.f) {
    if (2 * tid >= rot_embed_dim) {
        return;
    }
        float2 coef = {0.f, 0.f};
    if (yarn) {
        coef = rotary_embedding_coefficient_yarn(
            2 * tid, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
    } else {
        coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step, base, scaling_factor);
    }
    q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(uint2&      q,
                                              int         tid,
                                              int         rot_embed_dim,
                                              int         t_step,
                                              float       base                 = 10000.0f,
                                              const float scaling_factor       = 1.f,
                                              bool        yarn                 = false,
                                              float       mscale               = 1.f,
                                              float       low                  = 0.f,
                                              float       high                 = 0.f,
                                              float       extrapolation_factor = 1.f) {
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    float2 coef0 = {0.f, 0.f};
    float2 coef1 = {0.f, 0.f};
    if (yarn) {
        coef0 = rotary_embedding_coefficient_yarn(
            4 * tid, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
        coef1 = rotary_embedding_coefficient_yarn(
            4 * tid + 2, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);

    } else {
        coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step, base, scaling_factor);
        coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step, base, scaling_factor);
    }
    q.x = rotary_embedding_transform(q.x, coef0);
    q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint4&      q,
                                              int         tid,
                                              int         rot_embed_dim,
                                              int         t_step,
                                              float       base                 = 10000.0f,
                                              const float scaling_factor       = 1.f,
                                              bool        yarn                 = false,
                                              float       mscale               = 1.f,
                                              float       low                  = 0.f,
                                              float       high                 = 0.f,
                                              float       extrapolation_factor = 1.f) {
    if (8 * tid >= rot_embed_dim) {
        return;
    }

    float2 coef0 = {0.f, 0.f};
    float2 coef1 = {0.f, 0.f};
    float2 coef2 = {0.f, 0.f};
    float2 coef3 = {0.f, 0.f};

    if (yarn) {
        coef0 = rotary_embedding_coefficient_yarn(
            8 * tid, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
        coef1 = rotary_embedding_coefficient_yarn(
            8 * tid + 2, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
        coef2 = rotary_embedding_coefficient_yarn(
            8 * tid + 4, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
        coef3 = rotary_embedding_coefficient_yarn(
            8 * tid + 6, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
    } else {
        const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step, base, scaling_factor);
        const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step, base, scaling_factor);
        const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step, base, scaling_factor);
        const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step, base, scaling_factor);
    }
    q.x = rotary_embedding_transform(q.x, coef0);
    q.y = rotary_embedding_transform(q.y, coef1);
    q.z = rotary_embedding_transform(q.z, coef2);
    q.w = rotary_embedding_transform(q.w, coef3);
}

#ifdef ENABLE_BF16
inline __device__ void apply_rotary_embedding(__nv_bfloat162& q,
                                              int             tid,
                                              int             rot_embed_dim,
                                              int             t_step,
                                              float           base                 = 10000.0f,
                                              const float     scaling_factor       = 1.f,
                                              bool            yarn                 = false,
                                              float           mscale               = 1.f,
                                              float           low                  = 0.f,
                                              float           high                 = 0.f,
                                              float           extrapolation_factor = 1.f) {
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    float2 coef = {0.f, 0.f};
    if (yarn) {
        coef = rotary_embedding_coefficient_yarn(
            2 * tid, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
    } else {
        coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step, base, scaling_factor);
    }
    q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(bf16_4_t&   q,
                                              int         tid,
                                              int         rot_embed_dim,
                                              int         t_step,
                                              float       base                 = 10000.0f,
                                              const float scaling_factor       = 1.f,
                                              bool        yarn                 = false,
                                              float       mscale               = 1.f,
                                              float       low                  = 0.f,
                                              float       high                 = 0.f,
                                              float       extrapolation_factor = 1.f) {
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    float2 coef0 = {0.f, 0.f};
    float2 coef1 = {0.f, 0.f};
    if (yarn) {
        coef0 = rotary_embedding_coefficient_yarn(
            4 * tid, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
        coef1 = rotary_embedding_coefficient_yarn(
            4 * tid + 2, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
    } else {
        coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step, base, scaling_factor);
        coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step, base, scaling_factor);
    }
    q.x = rotary_embedding_transform(q.x, coef0);
    q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(bf16_8_t&   q,
                                              int         tid,
                                              int         rot_embed_dim,
                                              int         t_step,
                                              float       base                 = 10000.0f,
                                              const float scaling_factor       = 1.f,
                                              bool        yarn                 = false,
                                              float       mscale               = 1.f,
                                              float       low                  = 0.f,
                                              float       high                 = 0.f,
                                              float       extrapolation_factor = 1.f) {
    if (8 * tid >= rot_embed_dim) {
        return;
    }
    float2 coef0 = {0.f, 0.f};
    float2 coef1 = {0.f, 0.f};
    float2 coef2 = {0.f, 0.f};
    float2 coef3 = {0.f, 0.f};

    if (yarn) {
        coef0 = rotary_embedding_coefficient_yarn(
            8 * tid, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
        coef1 = rotary_embedding_coefficient_yarn(
            8 * tid + 2, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
        coef2 = rotary_embedding_coefficient_yarn(
            8 * tid + 4, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
        coef3 = rotary_embedding_coefficient_yarn(
            8 * tid + 6, rot_embed_dim, t_step, base, scaling_factor, mscale, low, high, extrapolation_factor);
    } else {
        coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step, base, scaling_factor);
        coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step, base, scaling_factor);
        coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step, base, scaling_factor);
        coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step, base, scaling_factor);
    }

    q.x = rotary_embedding_transform(q.x, coef0);
    q.y = rotary_embedding_transform(q.y, coef1);
    q.z = rotary_embedding_transform(q.z, coef2);
    q.w = rotary_embedding_transform(q.w, coef3);
}
#endif  // ENABLE_BF16

enum class RotaryEmbeddingStyle : int8_t {
    NoRope        = 0,
    LinearScalar  = 1,
    NTKScalar     = 2,
    QWenNTKScalar = 3,
    GLM2          = 4,
    Yarn          = 5,
};

template<typename scalar_t, typename vector_t, RotaryEmbeddingStyle style>
class Rope {};

template<typename scalar_t, typename vector_t>
class Rope<scalar_t, vector_t, RotaryEmbeddingStyle::NoRope> {

public:
    static __device__ __inline__ void
    impl(vector_t& x, scalar_t* smem, const int tidx, const int seqidx, const int dim, const int base = 10000) {
        return;
    }
};

template<typename scalar_t, typename vector_t>
class Rope<scalar_t, vector_t, RotaryEmbeddingStyle::LinearScalar> {

public:
    static __device__ __inline__ void impl(vector_t&   x,
                                           scalar_t*   smem,
                                           const int   tidx,
                                           const int   seqidx,
                                           const int   dim,
                                           const float base    = 10000.f,
                                           const float scale   = 1.0,
                                           const int   seq_len = 0,
                                           const int   max_pos = 2048) {

        const int  vec_size = vector_size<scalar_t, vector_t>::size;
        const bool work     = (tidx * vec_size < dim);

        if (work) {
            reinterpret_cast<vector_t*>(smem)[tidx] = x;
        }

        __syncthreads();

        if (work) {
            RotaryHalfRead(x, smem, tidx, dim / 2);
            apply_rotary_embedding(x, tidx, dim, seqidx, base, scale);
            RotaryHalfWrite(x, smem, tidx, dim / 2);
        }

        __syncthreads();

        if (work) {
            x = reinterpret_cast<vector_t*>(smem)[tidx];
        }

        __syncthreads();
    }

};

template<typename scalar_t, typename vector_t>
class Rope<scalar_t, vector_t, RotaryEmbeddingStyle::NTKScalar> {

    static __device__ inline float get_neox_dynamic_embedding_base(
        const float base, const int seq_len, const int rot_embed_dim, const float scale, const int max_pos) {
        if (max_pos == 0 || seq_len <= max_pos)
            return base * 1.0f;
        else {
            return base * 1.0f
                   * pow(((scale * seq_len / max_pos) - (scale - 1)), (rot_embed_dim / (rot_embed_dim - 2.0f)));
        }
    }

public:
    static __device__ __inline__ void impl(vector_t&   x,
                                           scalar_t*   smem,
                                           const int   tidx,
                                           const int   seqidx,
                                           const int   dim,
                                           const float base    = 10000.f,
                                           const float scale   = 1.0,
                                           const int   seq_len = 0,
                                           const int   max_pos = 2048) {

        const int  vec_size = vector_size<scalar_t, vector_t>::size;
        const bool work     = (tidx * vec_size < dim);

        if (work) {
            reinterpret_cast<vector_t*>(smem)[tidx] = x;
        }

        __syncthreads();

        if (work) {
            float newbase = get_neox_dynamic_embedding_base(base, seq_len, dim, scale, max_pos);
            RotaryHalfRead(x, smem, tidx, dim / 2);
            apply_rotary_embedding(x, tidx, dim, seqidx, newbase);
            RotaryHalfWrite(x, smem, tidx, dim / 2);
        }

        __syncthreads();

        if (work) {
            x = reinterpret_cast<vector_t*>(smem)[tidx];
        }

        __syncthreads();
    }
};

template<typename scalar_t, typename vector_t>
class Rope<scalar_t, vector_t, RotaryEmbeddingStyle::QWenNTKScalar> {

private:
    static __device__ float
    get_qwen_dynamic_embedding_base(const int dim, const float base, const int seq_len, const int max_logn_seq_len) {
        float context_value = logf((float)seq_len / max_logn_seq_len) / logf(2.0) + 1.0;
        float ntk_scalar    = pow(2.0, ceil(context_value)) - 1;
        ntk_scalar          = max(ntk_scalar, 1.0);
        float new_base      = base * pow((float)ntk_scalar, (float)dim / (dim - 2));

        return new_base;
    }

public:
    static __device__ __inline__ void impl(vector_t&   x,
                                           scalar_t*   smem,
                                           const int   tidx,
                                           const int   seqidx,
                                           const int   dim,
                                           const float base         = 10000.f,
                                           const float scalar       = 1.0,
                                           const int   seq_len      = 0,
                                           const int   max_logn_seq = 2048) {

        const int  vec_size = vector_size<scalar_t, vector_t>::size;
        const bool work     = (tidx * vec_size < dim);

        if (work) {
            reinterpret_cast<vector_t*>(smem)[tidx] = x;
        }

        __syncthreads();

        if (work) {
            float newbase = get_qwen_dynamic_embedding_base(dim, base, seq_len, max_logn_seq);
            RotaryHalfRead(x, smem, tidx, dim / 2);
            apply_rotary_embedding(x, tidx, dim, seqidx, newbase);
            RotaryHalfWrite(x, smem, tidx, dim / 2);
        }

        __syncthreads();

        if (work) {
            x = reinterpret_cast<vector_t*>(smem)[tidx];
        }
    }
};

template<typename scalar_t, typename vector_t>
class Rope<scalar_t, vector_t, RotaryEmbeddingStyle::GLM2> {

public:
    static __device__ __inline__ void impl(vector_t&   x,
                                           const int   tidx,
                                           const int   seqidx,
                                           const int   dim,
                                           const float base           = 10000.f,
                                           const float scaling_factor = 1.f,
                                           const int   base_scale     = 1) {

        const int  vec_size = vector_size<scalar_t, vector_t>::size;
        const bool work     = (tidx * vec_size < dim);

        float new_base = base * base_scale;

        if (work) {
            apply_rotary_embedding(x, tidx, dim, seqidx, new_base, scaling_factor);
        }
    }
};

template<typename scalar_t, typename vector_t>
class Rope<scalar_t, vector_t, RotaryEmbeddingStyle::Yarn>{
private:
    static inline __device__ float find_correction_dim(const int num_rotations, const int dim, const int base=10000, 
                                                const int max_position_embeddings=2048) {
    
        float pi = 3.141592654f;
        float t0 = dim * logf((float)max_position_embeddings / (num_rotations * 2 * pi));
        float t1 = 2 * logf((float)base);
        return (t0 / t1);
    }
    static inline __device__ float2 find_correction_range(float low_rot, float high_rot, int dim, int base=10000, 
                                                   int max_position_embeddings=2048) {
        float2 low_high;
        int low = floor(find_correction_dim(low_rot, dim, base, max_position_embeddings));
        int high = ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings));
        low_high.x = max(low, 0);
        low_high.y = min(high, dim-1);
        return low_high;
    }
    static inline __device__ float get_mscale(float scale=1) {
        if (scale <= 1) {
            return 1.0;
        } else {
            return 0.1 * log(scale) + 1.0;
        }
    }

public:
    static __device__ __inline__ void impl(vector_t&   x,
                                           scalar_t*   smem,
                                           const int   tidx,
                                           const int   seqidx,
                                           const int   dim,
                                           const int   base                 = 10000,
                                           const float scale                = 1.0,
                                           const int   max_pos              = 2048,
                                           const int   org_max_pos          = 2048,
                                           float       extrapolation_factor = 1,
                                           float       beta_fast            = 32,
                                           float       beta_slow            = 1) {

        float      mscale   = get_mscale(scale);

        float2     low_high               = find_correction_range(beta_fast, beta_slow, dim, base, org_max_pos);

        const int  vec_size = vector_size<scalar_t, vector_t>::size;
        const bool work     = (tidx * vec_size < dim);

        if (work) {
            reinterpret_cast<vector_t*>(smem)[tidx] = x;
        }
        __syncthreads();

        RotaryHalfRead(x, smem, tidx, dim / 2);
        apply_rotary_embedding(x, tidx, dim, seqidx, base, scale, true, mscale, low_high.x, low_high.y, extrapolation_factor);
        RotaryHalfWrite(x, smem, tidx, dim / 2);
        __syncthreads();
        if (work) {
            x = reinterpret_cast<vector_t*>(smem)[tidx];
        }
    }
};


template<typename scalar_t, typename vector_t>
__device__ inline void context_rope(int       RopeStyle,
                                    vector_t& q,
                                    vector_t& k,
                                    scalar_t* smem,
                                    int       tidx,
                                    int       seqidx,
                                    int       position_id,
                                    int       dim,
                                    int       seq_len,
                                    float     base,
                                    float     scaling_factor,
                                    int       dynamic_embedding_max_pos,
                                    int       org_embedding_pos,
                                    int       base_scale,
                                    int       input_len,
                                    bool      PREFIX_PROMPT,
                                    int       prefix_prompt_length,
                                    int       count_length,
                                    int       logn_length = 0) {
    if (PREFIX_PROMPT && count_length) {
        input_len = input_len + prefix_prompt_length;
        seqidx    = seqidx + prefix_prompt_length;
    }
    if (position_id > 0) {
        seqidx = position_id;
    }

    switch (RopeStyle) {
        case 0:

            // NoRope
            break;

        case 1:

            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::NTKScalar>::impl(
                q, smem, tidx, seqidx, dim, base, scaling_factor, seq_len, dynamic_embedding_max_pos);
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::NTKScalar>::impl(
                k, smem, tidx, seqidx, dim, base, scaling_factor, seq_len, dynamic_embedding_max_pos);
            break;
        case 3:
            // glm2 rotary embedding
            // only do rotary embedding for [..., d / 2]

            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::GLM2>::impl(
                q, tidx, seqidx, dim / 2, base, scaling_factor, base_scale);
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::GLM2>::impl(
                k, tidx, seqidx, dim / 2, base, scaling_factor, base_scale);
            break;

        case 4:
            // qwen rorary embedding
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::QWenNTKScalar>::impl(
                q, smem, tidx, seqidx, dim, base, scaling_factor, seq_len, logn_length);
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::QWenNTKScalar>::impl(
                k, smem, tidx, seqidx, dim, base, scaling_factor, seq_len, logn_length);
            break;
        
        case 5:
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::LinearScalar>::impl(
                q, smem, tidx, seqidx, dim, base, scaling_factor, seq_len, dynamic_embedding_max_pos);
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::LinearScalar>::impl(
                k, smem, tidx, seqidx, dim, base, scaling_factor, seq_len, dynamic_embedding_max_pos);
            break;
        
        case 6:
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::Yarn>::impl(
                q, smem, tidx, seqidx, dim, base, scaling_factor, org_embedding_pos);
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::Yarn>::impl(
                k, smem, tidx, seqidx, dim, base, scaling_factor, org_embedding_pos);
            break;

        default:
            break;
    }
}

template<typename scalar_t, typename vector_t>
__device__ inline void attention_rope(int       RopeStyle,
                                      vector_t& q,
                                      vector_t& k,
                                      scalar_t* smem,
                                      int       tidx,
                                      int       tlength,
                                      int       timestep,
                                      int       dim,
                                      int       seq_len,
                                      float     base,
                                      float     scaling_factor,
                                      int       max_pos,
                                      int       org_embedding_pos,
                                      int       base_scale,
                                      int       position_id,
                                      int       input_len,
                                      int       prefix_prompt_length,
                                      int       count_prefix_length,
                                      int       logn_seq_len,
                                      bool      handle_kv) {

    if (count_prefix_length) {
        prefix_prompt_length = 0;
    }
    int gen_len = tlength - input_len;

    if (position_id > 0) {
        tlength = position_id;
    }

    switch (RopeStyle) {
        case 0:

            // NoRope
            break;

        case 1:

            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::NTKScalar>::impl(
                q, smem, tidx, tlength, dim, base, scaling_factor, seq_len, max_pos);
            if (handle_kv) {
                Rope<scalar_t, vector_t, RotaryEmbeddingStyle::NTKScalar>::impl(
                    k, smem, tidx, tlength, dim, base, scaling_factor, seq_len, max_pos);
            }
            break;

        case 3:
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::GLM2>::impl(
                q, tidx, tlength - prefix_prompt_length, dim / 2, base, scaling_factor, base_scale);
            if (handle_kv) {
                Rope<scalar_t, vector_t, RotaryEmbeddingStyle::GLM2>::impl(
                    k, tidx, tlength - prefix_prompt_length, dim / 2, base, scaling_factor, base_scale);
            }
            break;

        case 4:
            // qwen rorary embedding
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::QWenNTKScalar>::impl(
                q, smem, tidx, tlength, dim, base, scaling_factor, seq_len, logn_seq_len);
            if (handle_kv) {
                Rope<scalar_t, vector_t, RotaryEmbeddingStyle::QWenNTKScalar>::impl(
                    k, smem, tidx, tlength, dim, base, scaling_factor, seq_len, logn_seq_len);
            }
            break;
        
        case 5:
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::LinearScalar>::impl(
                q, smem, tidx, tlength, dim, base, scaling_factor, seq_len, max_pos);
            if (handle_kv) {
                Rope<scalar_t, vector_t, RotaryEmbeddingStyle::LinearScalar>::impl(
                    k, smem, tidx, tlength, dim, base, scaling_factor, seq_len, max_pos);
            }
            break;
        
        case 6:
            Rope<scalar_t, vector_t, RotaryEmbeddingStyle::Yarn>::impl(
                q, smem, tidx, tlength, dim, base, scaling_factor, org_embedding_pos);
            if (handle_kv) {
                Rope<scalar_t, vector_t, RotaryEmbeddingStyle::Yarn>::impl(
                    k, smem, tidx, tlength, dim, base, scaling_factor, org_embedding_pos);
            }
            break;

        default:
            break;
    }
}

}  // namespace fastertransformer

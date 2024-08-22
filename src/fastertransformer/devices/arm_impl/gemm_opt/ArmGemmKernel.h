#pragma once
#include "../type_bf16/hie_bfloat16.hpp"
#include "arm_common.h"
#include <arm_sve.h>
#include <iostream>
#include "src/fastertransformer/core/Buffer.h"

namespace fastertransformer {

ConstBufferPtr prepareGemmWeight(const std::string& key, ConstBufferPtr input);
BufferPtr prepareGemmOptWeight(ConstBufferPtr input, bool isTranspose = false);
BufferPtr transposeWeight(ConstBufferPtr input);

template<typename Ta, typename Tb, typename Tc, typename Tp>
class GemmPartParam {
public:
    GemmPartParam(int    M,
                  int    N,
                  int    K_pack,
                  Ta*    a_ptr,
                  Tb*    b_ptr,
                  Tc*    c_ptr,
                  float* bias_ptr,
                  Tp*    wei_scale,
                  Tp*    wei_scaleXzp,
                  int    GroupSize,
                  int    with_bias,
                  int    actType):
        M(M),
        N(N),
        K_pack(K_pack),
        a_ptr(a_ptr),
        b_ptr(b_ptr),
        c_ptr(c_ptr),
        bias_ptr(bias_ptr),
        wei_scale(wei_scale),
        wei_scaleXzp(wei_scaleXzp),
        GroupSize(GroupSize),
        with_bias(with_bias),
        actType(actType) {}

    GemmPartParam(
        int M, int N, int K_pack, Ta* a_ptr, Tb* b_ptr, Tc* c_ptr, float* bias_ptr, int with_bias, int actType):
        M(M),
        N(N),
        K_pack(K_pack),
        a_ptr(a_ptr),
        b_ptr(b_ptr),
        c_ptr(c_ptr),
        bias_ptr(bias_ptr),
        wei_scale(nullptr),
        wei_scaleXzp(nullptr),
        GroupSize(K_pack),
        with_bias(with_bias),
        actType(actType) {}

    GemmPartParam() {}
    ~GemmPartParam() {}

    int    M, N, K_pack;
    Ta*    a_ptr;
    Tb*    b_ptr;
    Tc*    c_ptr;
    float* bias_ptr;
    Tp *   wei_scale, *wei_scaleXzp;
    int    GroupSize;
    int    with_bias;
    int    actType = 0;
    int    do_act  = 1;
};

enum UnaryType : int {
    UNARYTYPE_UNDEFINED = 0,
    TANH = 1,
    GELU_ERF = 2,
    GELU_TANH = 3,
    RELU = 4,
    SILU = 5,
    UnaryType_INT_MIN_SENTINEL_DO_NOT_USE_ = std::numeric_limits<int32_t>::min(),
    UnaryType_INT_MAX_SENTINEL_DO_NOT_USE_ = std::numeric_limits<int32_t>::max()
};

class GemmKernel {
private:
    void pack_input_impl_parallel_simd(int M, int N, int K, int lda, int K_pack, float* a_fp32, hie::bfloat16* a_bf16);
    void pack_input_fp16tobf16_impl_parallel_simd(int M, int N, int K, int lda, int K_pack, float16_t* a_fp16, hie::bfloat16* a_bf16);

    void thread_block_bf16_m8(GemmPartParam<hie::bfloat16, hie::bfloat16, float, float>& p, int m, int n, int k, int k_tile);
    void
    thread_block_bf16_m8_mres(GemmPartParam<hie::bfloat16, hie::bfloat16, float, float>& p, int m, int n, int k, int k_tile);
    void
    thread_block_bf16_m8_nres(GemmPartParam<hie::bfloat16, hie::bfloat16, float, float>& p, int m, int n, int k, int k_tile);
    void
    thread_block_bf16_m8_res(GemmPartParam<hie::bfloat16, hie::bfloat16, float, float>& p, int m, int n, int k, int k_tile);

    void thread_block_bf16_m8(GemmPartParam<hie::bfloat16, hie::bfloat16, float16_t, float>& p, int m, int n, int k, int k_tile);
    void
    thread_block_bf16_m8_mres(GemmPartParam<hie::bfloat16, hie::bfloat16, float16_t, float>& p, int m, int n, int k, int k_tile);
    void
    thread_block_bf16_m8_nres(GemmPartParam<hie::bfloat16, hie::bfloat16, float16_t, float>& p, int m, int n, int k, int k_tile);
    void
    thread_block_bf16_m8_res(GemmPartParam<hie::bfloat16, hie::bfloat16, float16_t, float>& p, int m, int n, int k, int k_tile);

    void pack_input_arm(int M, int N, int K, int lda, int K_pack, float* a_fp32, hie::bfloat16* a_bf16);


    void gemm_thread_block_bf16(
        GemmPartParam<hie::bfloat16, hie::bfloat16, float, float> p, int m, int n, int m_tile, int n_tile, int k_tile);

    void gemm_thread_block_bf16(
        GemmPartParam<hie::bfloat16, hie::bfloat16, float16_t, float> p, int m, int n, int m_tile, int n_tile, int k_tile);
    
    template<typename Tc>
    void gemm_thread_block_bf16(
        GemmPartParam<hie::bfloat16, hie::bfloat16, Tc, float> p, int m, int n, int m_tile, int n_tile, int k_tile);

    template<typename Tc>
    void gemm_thread_strategy(GemmPartParam<hie::bfloat16, hie::bfloat16, Tc, float>& p);

public:
    void gemm_pack_weight_FP32toBF16_arm(int N, int K, int K_pack, const float* b_fp32, hie::bfloat16* b_bf16);

    void gemm_pack_weight_FP16toBF16_arm(int N, int K, int K_pack, const float16_t* b_fp16, hie::bfloat16* b_bf16);

    template <typename Ta, typename Tc>
    void gemm_kernel_arm(int            M,
                         int            N,
                         int            K,
                         int            k_pack,
                         int            lda,
                         Ta*            a,
                         hie::bfloat16* b_bf16,
                         Tc*            c_fp32,
                         float*         bias_fp32,
                         int            actType,
                         void*          workspace);
};

template <typename Ta, typename Tc>
void GemmKernel::gemm_kernel_arm(int            M,
                                 int            N,
                                 int            K,
                                 int            k_pack,
                                 int            lda,
                                 Ta*            a,
                                 hie::bfloat16* b_bf16,
                                 Tc*            c,
                                 float*         bias_fp32,
                                 int            actType,
                                 void*          workspace) {

    int K_pack    = k_pack;
    int with_bias = bias_fp32 == nullptr ? 0 : 1;

    hie::bfloat16* a_bf16 = reinterpret_cast<hie::bfloat16*>(workspace);
    // int            a_bf16_size = (M * K_pack + M % 2 * K_pack) * 2;  // 括号内确保对齐需要额外增加的存储空间，M是奇数的时候多加一行K_pack, * 2是因为sizeof(bf16) = 2
    // memset(a_bf16, 0, a_bf16_size);

    if constexpr (std::is_same<Ta, float>::value) {
        pack_input_arm(M, N, K, lda, K_pack, a, a_bf16);
    } else if constexpr (std::is_same<Ta, float16_t>::value) {
        pack_input_fp16tobf16_impl_parallel_simd(M, N, K, lda, K_pack, a, a_bf16);
    } else {
        std::cerr << "Unsupported data type for A" << std::endl;
        return;
    }
    
    int k_pack_compute = std::ceil(K / 8.0) * 8;
    GemmPartParam<hie::bfloat16, hie::bfloat16, Tc, float> p(
        M, N, k_pack_compute, a_bf16, b_bf16, c, bias_fp32, with_bias, actType);
    
    if constexpr (std::is_same<Tc, float>::value) {
        gemm_thread_strategy<float>(p);
    } else if constexpr (std::is_same<Tc, float16_t>::value) {
        gemm_thread_strategy<float16_t>(p);
    } else {
        std::cerr << "Unsupported data type for C" << std::endl;
        return;
    }
    return;
}

template<typename Tc>
void GemmKernel::gemm_thread_strategy(GemmPartParam<hie::bfloat16, hie::bfloat16, Tc, float>& p) {
    int m_tile = 32;
    int n_tile = 64;
    int k_tile = 2560;
    if (p.K_pack == 5120)
        k_tile = 5120;

    int m_max = (p.M + m_tile - 1) / m_tile;
    int n_max = (p.N + n_tile - 1) / n_tile;
    parallel_for(m_max, n_max, [&](int m, int n) {
        gemm_thread_block_bf16<Tc>(p, m, n, m_tile, n_tile, k_tile);
    });
    return;
}

template<typename Tc>
void GemmKernel::gemm_thread_block_bf16(
    GemmPartParam<hie::bfloat16, hie::bfloat16, Tc, float> p, int m, int n, int m_tile, int n_tile, int k_tile) {
    int nn_max     = std::min(p.N, (n + 1) * n_tile);
    int mm_max     = std::min(p.M, (m + 1) * m_tile);
    int last_block = 0;
    if (mm_max == p.M && p.M % 8 != 0)
        last_block |= 0x1;

    if (nn_max == p.N && p.N % 8 != 0)
        last_block |= 0x2;

    for (int k = 0; k < p.K_pack; k += k_tile) {
        p.do_act = 0;
        if ((k + k_tile) >= p.K_pack)
            p.do_act = 0;  // TODO: change back to 1
        for (int nn = n * n_tile; nn < nn_max; nn += 8) {
            for (int mm = m * m_tile; mm < mm_max; mm += 8) {
                if (LIKELY(last_block == 0x0)) {
                    thread_block_bf16_m8(p, mm, nn, k, k_tile);
                } else if (last_block == 0x1) {
                    thread_block_bf16_m8_mres(p, mm, nn, k, k_tile);
                } else if (last_block == 0x2) {
                    thread_block_bf16_m8_nres(p, mm, nn, k, k_tile);
                } else {
                    thread_block_bf16_m8_res(p, mm, nn, k, k_tile);
                }
            }
        }
    }
}

}  // namespace fastertransformer
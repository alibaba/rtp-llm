#pragma once
#include <rtp_kernel/glm5/deep_gemm/fp8_gemm.cuh>
#include <rtp_kernel/glm5/indexer_k_cache.cuh>
#include <cuda_bf16.h>

namespace hy4_indexer_k {
struct Args {
    const __nv_bfloat16* weight;
    const __nv_bfloat16* bias;
    const float* cos_sin;
    const int32_t* positions;
    const int64_t* slots;
    uint8_t* cache;
    __nv_bfloat16* projected;
    __nv_bfloat16* normed;
    __nv_bfloat16* key;
    int rows;
    int blocks;
};

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int mask=16;mask;mask>>=1) v+=__shfl_xor_sync(0xffffffffu,v,mask);
    return v;
}

template<class Smem>
__device__ __forceinline__ float read_bf16(const Smem& smem,uint32_t stage,int row,int col) {
    const int chunk=col/8;
    const int offset=(chunk/8)*16*128+row*128+((chunk%8)^(row%8))*16;
    const auto* ptr=reinterpret_cast<const __nv_bfloat16*>(
        reinterpret_cast<const uint8_t*>(smem[stage])+offset);
    return __bfloat162float(ptr[col%8]);
}

template<class Smem>
__device__ __forceinline__ void finish_row(const Smem& smem,uint32_t stage,const Args& a,
                                         int local_row,int row,int lane) {
    if(row>=a.rows) return;
    float v[4];
    #pragma unroll
    for(int half=0;half<2;++half) {
        #pragma unroll
        for(int j=0;j<2;++j) {
            int col=half*64+lane*2+j;
            v[half*2+j]=read_bf16(smem,stage,local_row,col);
            a.projected[row*128+col]=__float2bfloat16_rn(v[half*2+j]);
        }
    }
    // Match native 128-thread LN: 64 BF16 pairs in two live warps, then
    // blockReduceSumV2 merges warp0+warp1 (the remaining two warps are zero).
    float sums[2], squares[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
        float x=v[half*2],y=v[half*2+1];
        sums[half]=__shfl_sync(0xffffffffu,warp_sum(x+y),0);
        squares[half]=__shfl_sync(0xffffffffu,warp_sum(x*x+y*y),0);
    }
    const float mean=(sums[0]+sums[1])/128.0f;
    const float variance=(squares[0]+squares[1])/128.0f-mean*mean;
    const float inverse=rsqrtf(variance+1e-6f);
    #pragma unroll
    for(int half=0;half<2;++half) {
        #pragma unroll
        for(int j=0;j<2;++j) {
            int col=half*64+lane*2+j;
            float n=(v[half*2+j]-mean)*inverse*__bfloat162float(a.weight[col])
                +__bfloat162float(a.bias[col]);
            auto bf=__float2bfloat16_rn(n);
            a.normed[row*128+col]=bf;
            v[half*2+j]=__bfloat162float(bf);
        }
    }
    const int64_t pos=int64_t(a.positions[row])*64;
    const float cosine=a.cos_sin[pos+lane],sine=a.cos_sin[pos+32+lane];
    const float x=v[0],y=v[1];
    // Native FlashInfer rounds the sine product before cosine-product FMA.
    // The M64/M256 regression fixtures distinguish these BF16 halfway cases.
    v[0]=__bfloat162float(__float2bfloat16_rn(__fmaf_rn(x,cosine,-__fmul_rn(y,sine))));
    v[1]=__bfloat162float(__float2bfloat16_rn(__fmaf_rn(y,cosine,__fmul_rn(x,sine))));
    float maximum=0.0f;
    #pragma unroll
    for(int i=0;i<4;++i) maximum=fmaxf(maximum,fabsf(v[i]));
    #pragma unroll
    for(int mask=16;mask;mask>>=1) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,mask));
    const float scale=exp2f(ceilf(log2f(fmaxf(maximum,1e-4f)/448.0f)));
    const int64_t slot=a.slots[row];
    #pragma unroll
    for(int half=0;half<2;++half) {
        #pragma unroll
        for(int j=0;j<2;++j) {
            const int col=half*64+lane*2+j;
            const float value=v[half*2+j];
            a.key[row*128+col]=__float2bfloat16_rn(value);
            if(slot>=0 && slot<int64_t(a.blocks)*64) {
                uint16_t packed;
                const float quantized=value/scale;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(packed) : "f"(quantized),"f"(0.0f));
                a.cache[(slot/64)*64*132+(slot%64)*128+col]=uint8_t(packed);
            }
        }
    }
    if(lane==0 && slot>=0 && slot<int64_t(a.blocks)*64)
        *reinterpret_cast<float*>(a.cache+(slot/64)*64*132+64*128+(slot%64)*4)=scale;
}

struct Epilogue {
    static constexpr bool kUsesTensorMapCd=false;
    template<uint32_t BM,uint32_t BN,uint32_t SM,uint32_t SN,uint32_t SW,uint32_t Stages,
             uint32_t Threads,deep_gemm::GemmType Type,bool Accum,class CD,class Transform,class Pattern>
    CUTLASS_DEVICE static void store_swap_ab(
        const deep_gemm::utils::PatternVisitor<Pattern>& smem,uint32_t& stage,uint32_t tmem,
        uint32_t base_m,uint32_t,uint32_t,uint32_t,uint32_t warp,uint32_t lane,
        const cutlass::arch::ClusterTransactionBarrier* empty,const cute::TmaDescriptor&,const void* opaque) {
        static_assert(BM==16 && BN==128 && SM==16 && SN==128 && SW==128 && Stages==2 && Threads==256);
        rtp_kernel::glm5::detail::stage_indexer_k_bf16(smem,stage,tmem,warp,lane,empty);
        cudaGridDependencySynchronize();
        const auto& args=*static_cast<const Args*>(opaque);
        finish_row(smem,stage,args,warp,base_m+warp,lane);
        finish_row(smem,stage,args,warp+8,base_m+warp+8,lane);
        cutlass::arch::NamedBarrier::sync(256,1);
        stage=(stage+1)%Stages;
    }
};
using Config=deep_gemm::Fp8GemmConfig<cute::UMMA::Major::K,cute::UMMA::Major::K,
    32,32,128,0,128,6144,16,128,128,1,128,128,128,11,128,256,1,false,
    Epilogue,true,true,148,true,true,deep_gemm::GemmType::Normal,false,
    cutlass::float_e4m3_t,cutlass::float_e4m3_t,cutlass::bfloat16_t,
    deep_gemm::epilogue::transform::EpilogueIdentity>;
} // namespace hy4_indexer_k

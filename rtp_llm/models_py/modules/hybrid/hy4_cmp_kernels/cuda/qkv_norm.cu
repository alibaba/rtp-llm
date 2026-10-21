#include "qkv_norm.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
__device__ __forceinline__ unsigned ue8m0(float value) {
    const unsigned bits = __float_as_uint(value);
    const unsigned exponent = (bits >> 23) & 255, mantissa = bits & 0x7fffff;
    const unsigned bump = mantissa != 0 && !(exponent == 0 && mantissa <= 0x400000);
    return value <= 0.0f ? 0u : min(exponent + bump, 254u);
}

// Original Triton cached PTX uses contiguous per-thread values and a sequential
// square FMA chain, then XOR16/8/4/2/1, then the original H-selected warp count.
// Do not substitute a balanced reduction over stride-THREADS elements: those
// trees differ at FP32 rounding boundaries, even if the final FP8 stays equal.
__device__ __forceinline__ float reduce(float value, float* partial, int active_warps) {
#pragma unroll
    for (int delta=16; delta; delta>>=1)
        value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, delta));
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x/32] = value;
    __syncthreads();
    float total = partial[threadIdx.x & (active_warps-1)];
    for (int delta=active_warps/2; delta; delta>>=1)
        total = __fadd_rn(total, __shfl_xor_sync(0xffffffffu, total, delta));
    return total;
}

__device__ __forceinline__ int source_threads(int rank) {
    return rank <= 512 ? 64 : rank <= 2048 ? 128 : 256;
}

template<int THREADS, bool PDL, bool BF16, bool CACHE, int RELEASE>
__global__ __launch_bounds__(THREADS) void hy4_qkv_norm_mxfp8(
    const __nv_bfloat16* x, const __nv_bfloat16* qw, const __nv_bfloat16* kw,
    unsigned char* q8, unsigned char* scales, __nv_bfloat16* qb, __nv_bfloat16* kb,
    int qr, int kr, int64_t stride, int64_t scale_leading, float qe, float ke,
    __nv_bfloat16* cache, const int64_t* slots, const int32_t* positions, const float* cos_sin,
    int64_t block_size, int64_t page_stride, int64_t slot_stride,
    int64_t slots_stride, int64_t positions_stride, int64_t cos_stride) {
    __shared__ float qpartial[8], kpartial[8];
    const int t = threadIdx.x, row = blockIdx.x;
    const int qt = source_threads(qr), kt = source_threads(kr);
    const int qn = qr/qt, kn = kr/kt;
    // Every CTA releases launch permission before waiting for its own producer.
    // This is not publication: consumers must execute GDC before any dynamic read.
    if constexpr(PDL && RELEASE == 1) {
        if (t == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    if constexpr(PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
    if constexpr(PDL && RELEASE == 2) {
        if (t == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    int64_t cache_offset=0, slot=-1;
    if constexpr(CACHE) {
        slot=slots[int64_t(row)*slots_stride];
        if(slot>=0) cache_offset=(slot/block_size)*page_stride+(slot%block_size)*slot_stride;
    }
    float q[16], k[8];
    float qs = 0, ks = 0;
    if (t < qt) {
        for (int i=0; i<qn; ++i)
            q[i] = __bfloat162float(x[row*stride + t*qn+i]);
        // The original PTX starts x1^2 then contracts x0^2 into it.
        qs = __fmul_rn(q[1], q[1]);
        qs = __fmaf_rn(q[0], q[0], qs);
        for (int i=2; i<qn; ++i) qs = __fmaf_rn(q[i], q[i], qs);
    }
    if (t < kt) {
        for (int i=0; i<kn; ++i)
            k[i] = __bfloat162float(x[row*stride + qr+t*kn+i]);
        ks = __fmul_rn(k[1], k[1]);
        ks = __fmaf_rn(k[0], k[0], ks);
        for (int i=2; i<kn; ++i) ks = __fmaf_rn(k[i], k[i], ks);
    }
    const float qsum = reduce(qs, qpartial, qt/32), ksum = reduce(ks, kpartial, kt/32);
    float qinv, kinv;
    const float qvariance = __fadd_rn(__fdiv_rn(qsum,float(qr)), qe);
    const float kvariance = __fadd_rn(__fdiv_rn(ksum,float(kr)), ke);
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(qinv) : "f"(qvariance));
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(kinv) : "f"(kvariance));
    if (t < kt) {
        for (int i=0; i<kn; ++i) {
            const int col = t*kn+i;
            const auto rounded = __float2bfloat16_rn(
                __fmul_rn(__fmul_rn(k[i],kinv),__bfloat162float(kw[col])));
            kb[row*kr+col] = rounded;
            if constexpr(CACHE) if(slot>=0) cache[cache_offset+col]=rounded;
        }
    }
    if (t < qt) {
        float maximum = 0;
        for (int i=0; i<qn; ++i) {
            const int col = t*qn+i;
            const auto rounded = __float2bfloat16_rn(
                __fmul_rn(__fmul_rn(q[i],qinv),__bfloat162float(qw[col])));
            if constexpr(BF16) qb[row*qr+col] = rounded;
            q[i] = __bfloat162float(rounded);
            maximum = fmaxf(maximum, fabsf(q[i]));
        }
        const int group_threads = 32/qn;
        for (int delta=group_threads/2; delta; delta>>=1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta, group_threads));
        const unsigned exponent = ue8m0(__fmul_rn(maximum,1.0f/448.0f));
        const float inverse = exponent == 0 ? 0.0f : __uint_as_float((254-exponent)<<23);
        for (int i=0; i<qn; ++i) {
            const int col = t*qn+i;
            const float scaled = fminf(fmaxf(__fmul_rn(q[i],inverse), -448.0f), 448.0f);
            unsigned short pair;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(pair) : "f"(scaled), "f"(0.0f));
            q8[row*qr+col] = static_cast<unsigned char>(pair);
        }
        if (t % group_threads == 0) {
            const int group = t*qn/32;
            scales[(int64_t(group/4)*scale_leading+row)*4 + group%4] = exponent;
        }
    }
    if constexpr(CACHE) {
        if(t<32 && slot>=0) {
            const int64_t position=positions[int64_t(row)*positions_stride];
            const float cosine=cos_sin[position*cos_stride+t];
            const float sine=cos_sin[position*cos_stride+32+t];
            const float even=__bfloat162float(x[int64_t(row)*stride+qr+kr+2*t]);
            const float odd=__bfloat162float(x[int64_t(row)*stride+qr+kr+2*t+1]);
            cache[cache_offset+512+2*t]=__float2bfloat16_rn(__fmaf_rn(even,cosine,-__fmul_rn(odd,sine)));
            cache[cache_offset+512+2*t+1]=__float2bfloat16_rn(__fmaf_rn(odd,cosine,__fmul_rn(even,sine)));
        }
    }
    if constexpr(PDL && RELEASE == 0) {
        __syncthreads();
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

template<int THREADS, bool PDL, bool BF16, bool CACHE, int RELEASE>
void dispatch(const at::Tensor& x, const at::Tensor& qw, const at::Tensor& kw,
              const at::Tensor& q8, const at::Tensor& scale,
              const std::optional<at::Tensor>& qb, const at::Tensor& kb,
              int qr, int kr, float qe, float ke, std::uint64_t event, cudaStream_t stream,
                        const std::optional<at::Tensor>& cache, const std::optional<at::Tensor>& slots,
                        const std::optional<at::Tensor>& positions, const std::optional<at::Tensor>& cos_sin) {
    cudaLaunchAttribute attrs[2]{};
    cudaLaunchConfig_t config{};
    config.gridDim=dim3(x.size(0)); config.blockDim=dim3(THREADS); config.stream=stream;
    config.attrs=attrs;
    if constexpr(PDL) {
        attrs[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed=1;
        config.numAttrs=1;
        if (event) {
            attrs[1].id=cudaLaunchAttributeProgrammaticEvent;
            attrs[1].val.programmaticEvent.event=reinterpret_cast<cudaEvent_t>(event);
            attrs[1].val.programmaticEvent.flags=0;
            attrs[1].val.programmaticEvent.triggerAtBlockStart=0;
            config.numAttrs=2;
        }
    }
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, hy4_qkv_norm_mxfp8<THREADS,PDL,BF16,CACHE,RELEASE>,
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(qw.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kw.data_ptr()),
        static_cast<unsigned char*>(q8.data_ptr()), static_cast<unsigned char*>(scale.data_ptr()),
        qb ? reinterpret_cast<__nv_bfloat16*>(qb->data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(kb.data_ptr()), qr, kr, x.stride(0), scale.stride(1), qe, ke,
        cache ? reinterpret_cast<__nv_bfloat16*>(cache->data_ptr()) : nullptr,
        slots ? slots->data_ptr<int64_t>() : nullptr,
        positions ? positions->data_ptr<int32_t>() : nullptr,
        cos_sin ? cos_sin->data_ptr<float>() : nullptr,
        cache ? (cache->dim()==3 ? cache->size(1) : cache->size(0)) : int64_t(0),
        cache ? (cache->dim()==3 ? cache->stride(0) : int64_t(0)) : int64_t(0),
        cache ? cache->stride(-2) : int64_t(0),
        slots ? slots->stride(0) : int64_t(0), positions ? positions->stride(0) : int64_t(0),
        cos_sin ? cos_sin->stride(0) : int64_t(0)));
    if (!PDL && event) C10_CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), stream));
}
}

void hy4_qkv_norm_launch(const at::Tensor& x, const at::Tensor& qw, const at::Tensor& kw,
                        const at::Tensor& q8, const at::Tensor& scale,
                        const std::optional<at::Tensor>& qb, const at::Tensor& kb,
                        int qr, int kr, float qe, float ke, bool pdl, std::uint64_t event,
                        const std::optional<at::Tensor>& cache, const std::optional<at::Tensor>& slots,
                        const std::optional<at::Tensor>& positions, const std::optional<at::Tensor>& cos_sin, int release) {
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream=at::cuda::getCurrentCUDAStream();
#define DISPATCH_CACHE(T,C,E) do { \
    if (pdl && qb) dispatch<T,true,true,C,E>(x,qw,kw,q8,scale,qb,kb,qr,kr,qe,ke,event,stream,cache,slots,positions,cos_sin); \
    else if (pdl) dispatch<T,true,false,C,E>(x,qw,kw,q8,scale,qb,kb,qr,kr,qe,ke,event,stream,cache,slots,positions,cos_sin); \
    else if (qb) dispatch<T,false,true,C,0>(x,qw,kw,q8,scale,qb,kb,qr,kr,qe,ke,event,stream,cache,slots,positions,cos_sin); \
    else dispatch<T,false,false,C,0>(x,qw,kw,q8,scale,qb,kb,qr,kr,qe,ke,event,stream,cache,slots,positions,cos_sin); \
} while(false)
#define DISPATCH_RELEASE(T,E) do { if(cache) { DISPATCH_CACHE(T,true,E); } else { DISPATCH_CACHE(T,false,E); } } while(false)
#define DISPATCH(T) do { if(pdl && release == 1) { DISPATCH_RELEASE(T,1); } else if(pdl && release == 2) { DISPATCH_RELEASE(T,2); } else { DISPATCH_RELEASE(T,0); } } while(false)
    const int rank = qr > kr ? qr : kr;
    if (rank <= 512) { DISPATCH(64); }
    else if (rank <= 2048) { DISPATCH(128); }
    else { DISPATCH(256); }
#undef DISPATCH
#undef DISPATCH_CACHE
#undef DISPATCH_RELEASE
}

namespace {
template<bool PDL,bool Early>
__global__ void copy_kernel(const std::uint8_t* src,std::uint8_t* dst,std::int64_t size) {
    if constexpr(PDL && Early) if(threadIdx.x==0) cudaTriggerProgrammaticLaunchCompletion();
    if constexpr(PDL) cudaGridDependencySynchronize();
    for(std::int64_t i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        i<size;i+=static_cast<std::int64_t>(gridDim.x)*blockDim.x) dst[i]=src[i];
    __syncthreads();
    if constexpr(PDL && !Early) if(threadIdx.x==0) cudaTriggerProgrammaticLaunchCompletion();
}
}
void qkv_test_copy(const at::Tensor& src,const at::Tensor& dst,bool pdl,bool early) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.device()==dst.device() &&
      src.is_contiguous() && dst.is_contiguous() && src.sizes()==dst.sizes() && src.scalar_type()==dst.scalar_type(),
      "test copy requires identical contiguous tensors");
    const c10::cuda::CUDAGuard guard(src.device());
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t config{};config.gridDim=dim3(32);config.blockDim=dim3(256);
    config.stream=at::cuda::getCurrentCUDAStream();config.attrs=pdl ? &attr : nullptr;config.numAttrs=pdl ? 1 : 0;
    const void* kernel=pdl ? (early ? reinterpret_cast<const void*>(&copy_kernel<true,true>)
                                                  : reinterpret_cast<const void*>(&copy_kernel<true,false>))
                           : reinterpret_cast<const void*>(&copy_kernel<false,false>);
    const auto* source=static_cast<const std::uint8_t*>(src.data_ptr());
    auto* destination=static_cast<std::uint8_t*>(dst.data_ptr());
    auto size=static_cast<std::int64_t>(src.numel()*src.element_size());
    void* args[]={&source,&destination,&size};
    C10_CUDA_CHECK(cudaLaunchKernelExC(&config,kernel,args));
}

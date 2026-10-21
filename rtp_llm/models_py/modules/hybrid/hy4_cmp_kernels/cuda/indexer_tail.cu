#include "indexer_tail.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer/pos_enc.cuh>
namespace {
template<bool ROPE,bool PDL,bool EARLY>
__global__ void tail(const __nv_bfloat16* q,const int32_t* pos,const float* cs,
                     __nv_bfloat16* rounded,uint8_t* q8,float* scale,
                     const float* raw_gate,float* weighted_gate,float gate_scale) {
 const int t=threadIdx.x, row=blockIdx.x, head=blockIdx.y;
 if constexpr(PDL && EARLY) if(t==0) cudaTriggerProgrammaticLaunchCompletion();
 if constexpr(PDL) cudaGridDependencySynchronize();
 const int offset=(row*32+head)*128;
 if(t<16) {
  flashinfer::vec_t<float,8> values;
  if constexpr(ROPE) {
   if(t<8) {
    flashinfer::vec_t<float,8> cos,sin;
    // Original helper only uses indices0..3. Avoid its redundant table overread.
    #pragma unroll
    for(int i=0;i<8;++i) {cos[i]=0;sin[i]=0;}
    #pragma unroll
    for(int i=0;i<4;++i) {cos[i]=cs[int64_t(pos[row])*64+t*4+i];sin[i]=cs[int64_t(pos[row])*64+32+t*4+i];}
    values=flashinfer::vec_apply_llama_rope_cos_sin_interleave_reuse_half<8,8>(q+offset,cos,sin,64);
   } else values.cast_load(q+offset+t*8);
  } else values.cast_load(q+offset+t*8);
  float maximum=1e-10f;
  #pragma unroll
  for(int i=0;i<8;++i) {
   const auto bf=(ROPE && t<8) ? __float2bfloat16_rn(values[i]) : q[offset+t*8+i];
   rounded[offset+t*8+i]=bf;
   values[i]=__bfloat162float(bf);
   maximum=fmaxf(maximum,fabsf(values[i]));
  }
  #pragma unroll
  for(int delta=8;delta;delta>>=1) maximum=fmaxf(maximum,__shfl_xor_sync(0xffff,maximum,delta));
  // Exact selected legacy query policy; not MXFP8 and not the K-cache floor.
  float s=__fdiv_rn(maximum,448.0f);
  s=exp2f(ceilf(log2f(fmaxf(s,1e-10f))));
  // Precise exp2 of integral ceil(log2) is a normal power of two (floor1e-10).
  // Extract its exponent instead of a second log2 + negative-int truncation.
  const unsigned exponent=(__float_as_uint(s)>>23)&255u;
  if(t==0) {
   const float qs=__uint_as_float(unsigned(exponent)<<23);
   scale[row*32+head]=qs;
   if(raw_gate) {
    float scaled,value;
    // Preserve the native (acc * qscale) * constant rounding, without FTZ.
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled) : "f"(raw_gate[row*32+head]),"f"(qs));
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(scaled),"f"(gate_scale));
    weighted_gate[row*32+head]=value;
   }
  }
  #pragma unroll
  for(int i=0;i<8;++i) {
   const float v=fminf(fmaxf(__fdiv_rn(values[i],s),-448.0f),448.0f);
   unsigned short pair;
   asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(pair) : "f"(v), "f"(0.0f));
   q8[offset+t*8+i]=static_cast<uint8_t>(pair);
  }
 }
 if constexpr(PDL && !EARLY) {__syncthreads();if(t==0) cudaTriggerProgrammaticLaunchCompletion();}
}
template<bool R,bool P,bool E>
void launch(const at::Tensor& q,const at::Tensor& pos,const at::Tensor& cs,const at::Tensor& rounded,
            const at::Tensor& q8,const at::Tensor& scale,std::uint64_t event,cudaStream_t stream,
            const at::Tensor& raw_gate,const at::Tensor& weighted_gate,float gate_scale) {
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(q.size(0),32);cfg.blockDim=dim3(32);cfg.stream=stream;
 cudaLaunchAttribute attrs[2]{};cfg.attrs=attrs;
 if constexpr(P) {
  attrs[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;attrs[0].val.programmaticStreamSerializationAllowed=1;cfg.numAttrs=1;
  if(event) {attrs[1].id=cudaLaunchAttributeProgrammaticEvent;attrs[1].val.programmaticEvent.event=reinterpret_cast<cudaEvent_t>(event);
   attrs[1].val.programmaticEvent.flags=0;attrs[1].val.programmaticEvent.triggerAtBlockStart=0;cfg.numAttrs=2;}
 }
 C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,tail<R,P,E>,reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),pos.data_ptr<int32_t>(),
  cs.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(rounded.data_ptr()),static_cast<uint8_t*>(q8.data_ptr()),scale.data_ptr<float>(),
  raw_gate.defined()?raw_gate.data_ptr<float>():nullptr,weighted_gate.defined()?weighted_gate.data_ptr<float>():nullptr,gate_scale));
 if(!P && event) C10_CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),stream));
}
}
void indexer_q_launch(const at::Tensor& q,const at::Tensor& pos,const at::Tensor& cs,const at::Tensor& rounded,
 const at::Tensor& q8,const at::Tensor& scale,bool rope,bool pdl,bool early,std::uint64_t event,
 const at::Tensor& raw_gate,const at::Tensor& weighted_gate,float gate_scale) {
 const c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();
 #define RUN(R,P,E) launch<R,P,E>(q,pos,cs,rounded,q8,scale,event,stream,raw_gate,weighted_gate,gate_scale)
 #define MODE(R) do {if(!pdl) {RUN(R,false,false);} else if(early) {RUN(R,true,true);} else {RUN(R,true,false);}} while(false)
 if(rope) {MODE(true);} else {MODE(false);}
 #undef MODE
 #undef RUN
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
void indexer_test_copy(const at::Tensor& src,const at::Tensor& dst,bool pdl,bool early) {
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

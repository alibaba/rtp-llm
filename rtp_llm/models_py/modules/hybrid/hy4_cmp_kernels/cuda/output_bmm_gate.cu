#include "output_bmm_gate.cuh"
#include <hy4_output/bf16_gemm.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <array>
#include <map>
#include <memory>
#include <algorithm>

namespace hy4_output_host {
CUtensorMap make_map(void* pointer, std::uint64_t inner, std::uint64_t outer, std::uint64_t batch,
                     std::uint32_t box_inner, std::uint32_t box_outer,
                     std::uint64_t outer_stride, std::uint64_t batch_stride) {
    CUtensorMap map{};
    const cuuint64_t dimensions[]={inner,outer,batch},strides[]={outer_stride,batch_stride};
    const cuuint32_t box[]={box_inner,box_outer,1},elements[]={1,1,1};
    C10_CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,pointer,
        dimensions,strides,box,elements,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

template<int Heads,int BM,int Stages,int EW,bool PDL>
const void* address() {
    return reinterpret_cast<const void*>(&deep_gemm::rtp_bf16_gemm_impl<
        cute::UMMA::Major::K,cute::UMMA::Major::MN,0,256,512,BM,128,64,Heads,
        128,128,128,Stages,128,EW,2,true,148,128,true,true,deep_gemm::GemmType::Batched,
        false,cutlass::bfloat16_t,hy4_output::Bf16SchedulerKind::kDeepGemm,true,
        hy4_output::PdlTriggerPoint::kBeforeWait,true,100,PDL>);
}
template<int BM,int Stages,int EW,bool PDL>
const void* address_for_heads(int heads) {
    return heads==8 ? address<8,BM,Stages,EW,PDL>() : address<64,BM,Stages,EW,PDL>();
}
std::size_t raw_span_bytes(const at::Tensor& raw) {
    return ((raw.size(0)-1)*raw.stride(0)+raw.size(1)*512)*raw.element_size();
}
struct Plan {
    std::array<at::Tensor,6> tensors;
    std::array<CUtensorMap,3> maps;
    std::array<const void*,2> kernels;
    int rows,heads,bm,stages,epilogue_threads,smem,workers,device,tuning;
    std::map<std::string,int> metadata() const {
        return {{"rows",rows},{"heads",heads},{"k",512},{"n",256},{"block_m",bm},{"block_n",128},
            {"stages",stages},{"threads",128+epilogue_threads},{"cluster",2},{"workers",workers},
            {"dynamic_shared_bytes",smem},{"group_size",32},{"kernel_count",1},{"split_weight_loader",1},
            {"early_notification",1},{"dynamic_raw_gate_gdc",1},{"expanded_bf16_materialized",1},{"dynamic_raw_address",1},{"dynamic_gate_address",1},{"tuning_epilogue4",tuning==1 || tuning==2},{"tuning_id",tuning},{"epilogue_warpgroups",epilogue_threads/128}};
    }
    void run(const at::Tensor& raw,const at::Tensor& gate_input,bool pdl,std::uint64_t event) const {
        const c10::cuda::CUDAGuard guard(device);
        auto actual=tensors;
        actual[0]=raw;
        actual[2]=gate_input;
        for(int i : {0,2}) {
            const auto& input=actual[i];
            TORCH_CHECK(input.device()==tensors[i].device() && input.scalar_type()==at::kBFloat16 &&
                        input.sizes()==tensors[i].sizes() && input.strides()==tensors[i].strides(),
                        "dynamic input shape/stride/dtype/device changed at index ",i);
            const auto begin=reinterpret_cast<std::uintptr_t>(input.data_ptr());
            TORCH_CHECK(begin%16==0,"dynamic input base alignment requires16bytes");
            TORCH_CHECK(input.storage().nbytes()-input.storage_offset()*input.element_size()>=(i==0 ? raw_span_bytes(input) : input.nbytes()),
                        "dynamic input storage too small");
            for(int j=0;j<6;++j) {
                if(i==j) continue;
                const auto other=reinterpret_cast<std::uintptr_t>(actual[j].data_ptr());
                const auto bytes=j==0 ? raw_span_bytes(actual[j]) : j==5 ? actual[j].stride(1)*heads*2*sizeof(std::int32_t) : actual[j].nbytes();
                TORCH_CHECK(begin+(i==0 ? raw_span_bytes(input) : input.nbytes())<=other || other+bytes<=begin,"dynamic input/output alias unsupported");
            }
        }
        // A is returned by an allocating attention op. Capture copies this descriptor
        // by value into its kernel node; replay needs no host update or GPU staging.
        auto map_a=make_map(raw.data_ptr(),512,rows,heads,64,bm/2,raw.stride(0)*2,raw.stride(1)*2);
        int* grouped=nullptr;
        std::uint32_t m=rows,n=256,k=512;
        auto* quant=static_cast<std::uint8_t*>(tensors[4].data_ptr());
        auto* scale=tensors[5].data_ptr<std::int32_t>();
        const auto* gate=reinterpret_cast<const __nv_bfloat16*>(gate_input.data_ptr());
        auto* expanded=reinterpret_cast<__nv_bfloat16*>(tensors[3].data_ptr());
        int scale_stride=tensors[5].stride(1),output_stride=heads*256;
        void* args[]={&grouped,&m,&n,&k,&map_a,
            const_cast<CUtensorMap*>(&maps[1]),const_cast<CUtensorMap*>(&maps[2]),
            &quant,&scale,&scale_stride,&output_stride,&gate,&expanded};
        cudaLaunchAttribute attrs[3]{}; int count=0;
        attrs[count].id=cudaLaunchAttributeClusterDimension;attrs[count++].val.clusterDim={2,1,1};
        if(pdl) {
            attrs[count].id=cudaLaunchAttributeProgrammaticStreamSerialization;
            attrs[count++].val.programmaticStreamSerializationAllowed=1;
            if(event) {
                attrs[count].id=cudaLaunchAttributeProgrammaticEvent;
                attrs[count].val.programmaticEvent.event=reinterpret_cast<cudaEvent_t>(event);
                attrs[count].val.programmaticEvent.flags=0;
                attrs[count++].val.programmaticEvent.triggerAtBlockStart=0;
            }
        }
        cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(workers);cfg.blockDim=dim3(128+epilogue_threads);
        cfg.dynamicSmemBytes=smem;cfg.stream=at::cuda::getCurrentCUDAStream();cfg.attrs=attrs;cfg.numAttrs=count;
        C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg,kernels[pdl],args));
        if(!pdl && event) C10_CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),cfg.stream));
    }
};
std::shared_ptr<Plan> prepare(const at::Tensor& raw,const at::Tensor& weight,const at::Tensor& gate,
                            const at::Tensor& expanded,const at::Tensor& quant,const at::Tensor& scale,const std::string& tuning) {
    TORCH_CHECK(tuning=="baseline" || tuning=="epilogue4" || tuning=="tile64_epilogue4" || tuning=="epilogue2",
                "tuning must be baseline, epilogue4, tile64_epilogue4 or epilogue2");
    TORCH_CHECK(raw.is_cuda() && raw.dim()==3 && raw.size(0)>=1 && raw.size(0)<=256 &&
                (raw.size(1)==8 || raw.size(1)==64) && raw.size(2)==512,"raw requires CUDA BF16 [M,H,512], H8/64, 1<=M<=256");
    TORCH_CHECK(raw.stride(0)>=raw.size(1)*512 && raw.stride(0)%8==0 && raw.stride(1)==512 && raw.stride(2)==1,
                "raw requires positive row stride >= H*512 aligned8, head stride512, last stride1");
    const int m=raw.size(0),heads=raw.size(1),aligned=(m+3)/4*4;
    const c10::cuda::CUDAGuard guard(raw.device());
    cudaStreamCaptureStatus status;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(),&status));
    TORCH_CHECK(status==cudaStreamCaptureStatusNone,"prepare outside capture required");
    const auto* props=at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major==10 && props->minor==3,"SM103 required");
    TORCH_CHECK(weight.sizes()==at::IntArrayRef({heads,512,256}),"weight requires [H,512,256]");
    TORCH_CHECK(gate.sizes()==at::IntArrayRef({m,heads*256}) && expanded.sizes()==gate.sizes() && quant.sizes()==gate.sizes(),
                "gate/expanded/quant require [M,H*256]");
    TORCH_CHECK(scale.sizes()==at::IntArrayRef({m,heads*2}) && scale.stride(0)==1 && scale.stride(1)==aligned,
                "scale requires packed native [M,H*2] stride(1,align4(M))");
    const std::array<at::Tensor,6> tensors={raw,weight,gate,expanded,quant,scale};
    const at::ScalarType types[]={at::kBFloat16,at::kBFloat16,at::kBFloat16,at::kBFloat16,at::kFloat8_e4m3fn,at::kInt};
    std::array<std::size_t,6> bytes;
    for(int i=0;i<6;++i) {
        TORCH_CHECK(tensors[i].device()==raw.device() && tensors[i].scalar_type()==types[i],"dtype/device mismatch");
        TORCH_CHECK(i==0 || i==5 || tensors[i].is_contiguous(),"contiguous tensors required");
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensors[i].data_ptr())%16==0,"base alignment requires16bytes");
        bytes[i]=i==0 ? raw_span_bytes(raw) : i==5 ? aligned*heads*2*sizeof(std::int32_t) : tensors[i].nbytes();
        TORCH_CHECK(tensors[i].storage().nbytes()-tensors[i].storage_offset()*tensors[i].element_size()>=bytes[i],
                    "insufficient padded storage");
        for(int j=0;j<i;++j) {
            auto a=reinterpret_cast<std::uintptr_t>(tensors[i].data_ptr());
            auto b=reinterpret_cast<std::uintptr_t>(tensors[j].data_ptr());
            TORCH_CHECK(a+bytes[i]<=b || b+bytes[j]<=a,"input/output alias unsupported");
        }
    }
    auto plan=std::make_shared<Plan>();plan->tensors=tensors;plan->rows=m;plan->heads=heads;plan->device=raw.get_device();plan->tuning=tuning=="baseline" ? 0 : tuning=="epilogue4" ? 1 : tuning=="tile64_epilogue4" ? 2 : 3;
    if(m<=16) {plan->bm=16;plan->stages=12;plan->epilogue_threads=128;
        plan->kernels={address_for_heads<16,12,128,false>(heads),address_for_heads<16,12,128,true>(heads)};}
    else if(m<=32) {plan->bm=32;plan->stages=12;plan->epilogue_threads=256;
        plan->kernels={address_for_heads<32,12,256,false>(heads),address_for_heads<32,12,256,true>(heads)};}
    else if(m<=64) {plan->bm=64;plan->stages=10;plan->epilogue_threads=512;
        plan->kernels={address_for_heads<64,10,512,false>(heads),address_for_heads<64,10,512,true>(heads)};}
    else if(plan->tuning==2) {plan->bm=64;plan->stages=8;plan->epilogue_threads=512;
        plan->kernels={address_for_heads<64,8,512,false>(heads),address_for_heads<64,8,512,true>(heads)};}
    else if(plan->tuning==3) {plan->bm=128;plan->stages=8;plan->epilogue_threads=256;
        plan->kernels={address_for_heads<128,8,256,false>(heads),address_for_heads<128,8,256,true>(heads)};}
    else if(plan->tuning==1) {plan->bm=128;plan->stages=8;plan->epilogue_threads=512;
        plan->kernels={address_for_heads<128,8,512,false>(heads),address_for_heads<128,8,512,true>(heads)};}
    else {plan->bm=128;plan->stages=8;plan->epilogue_threads=128;
        plan->kernels={address_for_heads<128,8,128,false>(heads),address_for_heads<128,8,128,true>(heads)};}
    plan->smem=hy4_output::bf16_gemm_smem_bytes(plan->bm,128,64,plan->stages,2,2,true,std::max(2,plan->epilogue_threads/128));
    TORCH_CHECK(plan->smem<=props->sharedMemPerBlockOptin,"shared memory exceeds device limit");
    plan->workers=std::min(148,heads*2*((m+plan->bm-1)/plan->bm));
    for(auto kernel:plan->kernels)
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,plan->smem));
    plan->maps[1]=make_map(weight.data_ptr(),256,512,heads,64,64,weight.stride(1)*2,weight.stride(0)*2);
    plan->maps[2]=make_map(expanded.data_ptr(),256,m,heads,64,16,heads*256*2,256*2);
    return plan;
}
namespace {
template<bool PDL,bool EARLY>
__global__ void copy_kernel(const std::uint8_t* src,std::uint8_t* dst,std::int64_t size) {
    if constexpr(PDL && EARLY) if(threadIdx.x==0) cudaTriggerProgrammaticLaunchCompletion();
    if constexpr(PDL) cudaGridDependencySynchronize();
    for(std::int64_t i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        i<size;i+=static_cast<std::int64_t>(gridDim.x)*blockDim.x) dst[i]=src[i];
    __syncthreads();
    if constexpr(PDL && !EARLY) if(threadIdx.x==0) cudaTriggerProgrammaticLaunchCompletion();
}
}
void test_copy(const at::Tensor& src,const at::Tensor& dst,bool pdl,bool early) {
    TORCH_CHECK(src.is_cuda() && dst.device()==src.device() && src.is_contiguous() && dst.is_contiguous() &&
        src.sizes()==dst.sizes() && src.scalar_type()==dst.scalar_type(),"test copy requires identical contiguous tensors");
    const c10::cuda::CUDAGuard guard(src.device());
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t config{};config.gridDim=dim3(32);config.blockDim=dim3(256);
    config.stream=at::cuda::getCurrentCUDAStream();config.attrs=pdl ? &attr : nullptr;config.numAttrs=pdl ? 1 : 0;
    const void* kernel=pdl ? (early ? reinterpret_cast<const void*>(&copy_kernel<true,true>) : reinterpret_cast<const void*>(&copy_kernel<true,false>)) : reinterpret_cast<const void*>(&copy_kernel<false,false>);
    const auto* source=static_cast<const std::uint8_t*>(src.data_ptr());auto* destination=static_cast<std::uint8_t*>(dst.data_ptr());
    auto size=static_cast<std::int64_t>(src.nbytes());void* args[]={&source,&destination,&size};
    C10_CUDA_CHECK(cudaLaunchKernelExC(&config,kernel,args));
}
}
void bind_output_bmm_gate(pybind11::module_& m) {
    using namespace hy4_output_host;
    pybind11::class_<Plan,std::shared_ptr<Plan>>(m,"Plan",pybind11::module_local())
        .def("run",&Plan::run).def("metadata",&Plan::metadata);
    m.def("prepare",&prepare);m.def("_test_copy",&test_copy);
}

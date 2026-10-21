#include "indexer_k_fused.cuh"
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <rtp_kernel/glm5/deep_gemm/fp8_gemm_config.cuh>
#include <array>
#include <map>
#include <memory>

namespace hy4_indexer_k_host {
using Tensors=std::array<at::Tensor,13>;
CUtensorMap map(void* ptr,CUtensorMapDataType dtype,uint64_t inner,uint64_t outer,
                uint32_t box_inner,uint32_t box_outer,uint64_t pitch,CUtensorMapSwizzle swizzle) {
    CUtensorMap out{};
    const cuuint64_t sizes[]={inner,outer},strides[]={pitch};
    const cuuint32_t boxes[]={box_inner,box_outer},elements[]={1,1};
    C10_CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(&out,dtype,2,ptr,sizes,strides,boxes,elements,
        CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return out;
}
void validate(const Tensors& tensors) {
    const auto& a=tensors[0]; const auto& sa=tensors[1]; const auto& b=tensors[2]; const auto& sb=tensors[3];
    const auto& weight=tensors[4]; const auto& bias=tensors[5]; const auto& pos=tensors[6];
    const auto& cs=tensors[7]; const auto& slots=tensors[8]; const auto& cache=tensors[9];
    const at::ScalarType types[]={at::kFloat8_e4m3fn,at::kInt,at::kFloat8_e4m3fn,at::kInt,
        at::kBFloat16,at::kBFloat16,at::kInt,at::kFloat,at::kLong,at::kByte,at::kBFloat16,at::kBFloat16,at::kBFloat16};
    TORCH_CHECK(a.is_cuda() && a.dim()==2 && a.size(1)==6144 && a.size(0)>=1 && a.size(0)<=256,"requires A[M,6144],1<=M<=256");
    const int m=a.size(0),aligned=(m+3)/4*4;
    for(int i=0;i<13;++i) {
        TORCH_CHECK(tensors[i].is_cuda() && tensors[i].device()==a.device() && tensors[i].scalar_type()==types[i],"dtype/device at index ",i);
        if(i!=1 && i!=3) TORCH_CHECK(tensors[i].is_contiguous(),"contiguous tensor required at index ",i);
    }
    TORCH_CHECK(b.sizes()==at::IntArrayRef({128,6144}),"WK shape");
    TORCH_CHECK(sa.sizes()==at::IntArrayRef({m,48}) && sa.stride(0)==1 && sa.stride(1)==aligned,"activation scale layout");
    TORCH_CHECK(sb.sizes()==at::IntArrayRef({128,48}) && sb.stride(0)==1 && sb.stride(1)==128,"weight scale layout");
    TORCH_CHECK(weight.sizes()==at::IntArrayRef({128}) && bias.sizes()==at::IntArrayRef({128}),"norm weight/bias shape");
    TORCH_CHECK(pos.sizes()==at::IntArrayRef({m}) && slots.sizes()==at::IntArrayRef({m}),"position/slot shape");
    TORCH_CHECK(cs.dim()==2 && cs.size(1)==64 && cs.size(0)>0,"cos_sin shape");
    TORCH_CHECK((cache.dim()==3 && cache.size(1)==64 && cache.size(2)==132) ||
        (cache.dim()==4 && cache.size(1)==64 && cache.size(2)==1 && cache.size(3)==132),"cache page layout");
    TORCH_CHECK(cache.size(0)>0,"nonempty cache required");
    for(int i=10;i<13;++i) TORCH_CHECK(tensors[i].sizes()==at::IntArrayRef({m,128}),"diagnostic output shape");
    std::array<size_t,13> sizes;
    for(int i=0;i<13;++i) {
        sizes[i]=(i==1 || i==3) ? tensors[i].stride(1)*48*4 : tensors[i].numel()*tensors[i].element_size();
        TORCH_CHECK(tensors[i].storage().nbytes()-tensors[i].storage_offset()*tensors[i].element_size()>=sizes[i],"insufficient storage/padding");
        if(i<4) TORCH_CHECK(reinterpret_cast<uintptr_t>(tensors[i].data_ptr())%16==0,"TMA pointer alignment");
        for(int j=0;j<i;++j) {
            auto x=reinterpret_cast<uintptr_t>(tensors[i].data_ptr()),y=reinterpret_cast<uintptr_t>(tensors[j].data_ptr());
            TORCH_CHECK(x+sizes[i]<=y || y+sizes[j]<=x,"tensor storage overlaps");
        }
    }
}
struct Plan {
    Tensors t;
    std::array<CUtensorMap,5> maps;
    hy4_indexer_k::Args args;
    int device,rows,workers,smem;
    const void* kernel;
    std::map<std::string,int> metadata() const {
        return {{"rows",rows},{"block_m",16},{"block_n",128},{"stages",11},{"threads",384},
            {"grid",workers},{"dynamic_shared_bytes",smem},{"granularity_a",32},{"granularity_b",32},
            {"input_gdc",1},{"split_weight_loader",1},{"early_programmatic_trigger",1},
            {"kernel_launches",1},{"diagnostic_outputs",3},{"hadamard",0}};
    }
    void run(const Tensors& actual,bool pdl,uint64_t event) const {
        const c10::cuda::CUDAGuard guard(device);
        validate(actual);
        for(int i=0;i<13;++i) {
            TORCH_CHECK(actual[i].sizes()==t[i].sizes() && actual[i].strides()==t[i].strides() &&
                        actual[i].device()==t[i].device() && actual[i].scalar_type()==t[i].scalar_type(),"plan layout changed");
            if(i!=0 && i!=1 && i!=6 && i!=8 && i!=9)
                TORCH_CHECK(actual[i].data_ptr()==t[i].data_ptr(),"static plan pointer changed");
        }
        auto launch_maps=maps;
        launch_maps[0]=map(actual[0].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,6144,rows,128,16,6144,CU_TENSOR_MAP_SWIZZLE_128B);
        int aligned=(rows+3)/4*4;
        launch_maps[2]=map(actual[1].data_ptr(),CU_TENSOR_MAP_DATA_TYPE_INT32,aligned,48,16,1,aligned*4,CU_TENSOR_MAP_SWIZZLE_NONE);
        auto launch_args=args;
        launch_args.positions=actual[6].data_ptr<int32_t>();
        launch_args.slots=actual[8].data_ptr<int64_t>();
        launch_args.cache=actual[9].data_ptr<uint8_t>();
        int* grouped=nullptr;
        uint32_t m=rows,n=128,k=6144;
        void* argv[]={&grouped,&launch_args,&m,&n,&k,
            &launch_maps[0],&launch_maps[1],
            &launch_maps[2],&launch_maps[3],&launch_maps[4]};
        cudaLaunchAttribute attrs[2]{};
        cudaLaunchConfig_t cfg{};
        cfg.gridDim=dim3(workers);cfg.blockDim=dim3(384);cfg.dynamicSmemBytes=smem;
        cfg.stream=at::cuda::getCurrentCUDAStream();cfg.attrs=attrs;
        if(pdl) {
            attrs[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
            attrs[0].val.programmaticStreamSerializationAllowed=1;cfg.numAttrs=1;
            if(event) {
                attrs[1].id=cudaLaunchAttributeProgrammaticEvent;
                attrs[1].val.programmaticEvent.event=reinterpret_cast<cudaEvent_t>(event);
                attrs[1].val.programmaticEvent.flags=0;attrs[1].val.programmaticEvent.triggerAtBlockStart=0;
                cfg.numAttrs=2;
            }
        }
        C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg,kernel,argv));
        if(!pdl && event) C10_CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),cfg.stream));
    }
};
std::shared_ptr<Plan> prepare(const at::Tensor& a,const at::Tensor& sa,const at::Tensor& b,const at::Tensor& sb,
    const at::Tensor& weight,const at::Tensor& bias,const at::Tensor& pos,const at::Tensor& cs,
    const at::Tensor& slots,const at::Tensor& cache,const at::Tensor& projected,const at::Tensor& normed,const at::Tensor& key) {
    const Tensors tensors={a,sa,b,sb,weight,bias,pos,cs,slots,cache,projected,normed,key};
    validate(tensors);
    const int m=a.size(0),aligned=(m+3)/4*4;
    const c10::cuda::CUDAGuard guard(a.device());
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(),&capture));
    TORCH_CHECK(capture==cudaStreamCaptureStatusNone,"prepare outside capture required");
    cudaDeviceProp prop{};C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,a.get_device()));
    TORCH_CHECK(prop.major==10 && prop.minor==3 && prop.multiProcessorCount==148,"SM103/148 SMs required");
    auto p=std::make_shared<Plan>();p->t=tensors;p->device=a.get_device();p->rows=m;p->workers=(m+15)/16;
    p->smem=rtp_kernel::glm5::deep_gemm_config::shared_bytes(16,128,16,128,11,128,2,true);
    TORCH_CHECK(p->smem<=232448,"shared memory limit");
    p->kernel=reinterpret_cast<const void*>(&deep_gemm::fp8_gemm_epilogue_impl<hy4_indexer_k::Config,hy4_indexer_k::Args>);
    C10_CUDA_CHECK(cudaFuncSetAttribute(p->kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,p->smem));
    p->args={reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
        cs.data_ptr<float>(),pos.data_ptr<int32_t>(),slots.data_ptr<int64_t>(),cache.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(projected.data_ptr()),reinterpret_cast<__nv_bfloat16*>(normed.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(key.data_ptr()),m,int(cache.size(0))};
    p->maps[0]=map(a.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,6144,m,128,16,6144,CU_TENSOR_MAP_SWIZZLE_128B);
    p->maps[1]=map(b.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,6144,128,128,128,6144,CU_TENSOR_MAP_SWIZZLE_128B);
    p->maps[2]=map(sa.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_INT32,aligned,48,16,1,aligned*4,CU_TENSOR_MAP_SWIZZLE_NONE);
    p->maps[3]=map(sb.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_INT32,128,48,128,1,128*4,CU_TENSOR_MAP_SWIZZLE_NONE);
    p->maps[4]={};
    return p;
}
void bind(pybind11::module_& module) {
    pybind11::class_<Plan,std::shared_ptr<Plan>>(module,"Plan",pybind11::module_local()).def("run",&Plan::run).def("metadata",&Plan::metadata);
    module.def("prepare_plan",&prepare);
}
} // namespace hy4_indexer_k_host

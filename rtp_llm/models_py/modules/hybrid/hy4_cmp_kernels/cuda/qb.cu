#include "qb.cuh"
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <rtp_kernel/glm5/deep_gemm/fp8_gemm_config.cuh>
#include <array>
#include <map>
#include <memory>
#include <algorithm>
namespace hy4_qb_host {
std::size_t tensor_bytes(const at::Tensor& t,bool scale=false) {
    return scale ? static_cast<std::size_t>(t.stride(1))*t.size(1)*t.element_size()
                 : t.numel()*t.element_size();
}
void validate(const std::array<at::Tensor,5>& tensors,int role) {
    const auto& a=tensors[0]; const auto& sa=tensors[1];
    const auto& b=tensors[2]; const auto& sb=tensors[3]; const auto& out=tensors[4];
    TORCH_CHECK(a.is_cuda() && a.dim()==2 && b.dim()==2,"A and B must be CUDA rank2");
    TORCH_CHECK(a.size(0)>=1 && a.size(0)<=256,"requires 1<=M<=256");
    TORCH_CHECK(a.size(1)==b.size(1),"A/B K mismatch");
    TORCH_CHECK(a.size(1)==2048 && (b.size(0)==2048 || b.size(0)==16384),"QB shape unsupported");
    const int m=a.size(0), n=b.size(0), k=a.size(1);
    const at::ScalarType expected[]={at::kFloat8_e4m3fn,at::kInt,at::kFloat8_e4m3fn,at::kInt,at::kBFloat16};
    for(int i=0;i<5;++i) {
        TORCH_CHECK(tensors[i].is_cuda() && tensors[i].device()==a.device() &&
                    tensors[i].scalar_type()==expected[i],"tensor dtype/device mismatch at index ",i);
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensors[i].data_ptr())%16==0,
                    "TMA tensor base must be 16-byte aligned");
    }
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),"A/B/out must be contiguous");
    TORCH_CHECK(out.sizes()==at::IntArrayRef({m,n}),"out must have shape [M,N]");
    const int scale_rows[]={m,n};
    const at::Tensor* scales[]={&sa,&sb};
    for(int i=0;i<2;++i) {
        const auto& scale=*scales[i]; const int rows=scale_rows[i], aligned=(rows+3)/4*4;
        TORCH_CHECK(scale.sizes()==at::IntArrayRef({rows,k/128}) &&
                    scale.stride(0)==1 && scale.stride(1)==aligned,
                    "MXFP8 scale requires [rows,K/128], stride(1,round_up(rows,4))");
        const auto remaining=scale.storage().nbytes()-scale.storage_offset()*scale.element_size();
        TORCH_CHECK(remaining>=tensor_bytes(scale,true),"packed scale must own full TMA row padding");
    }
    // W/SFB are prefetched before GDC: they must not alias producer-owned A/SFA.
    // Out must not overlap any input. Pairwise disjointness makes this explicit.
    for(int i=0;i<5;++i) for(int j=i+1;j<5;++j) {
        const auto left=reinterpret_cast<std::uintptr_t>(tensors[i].data_ptr());
        const auto right=reinterpret_cast<std::uintptr_t>(tensors[j].data_ptr());
        const auto left_end=left+tensor_bytes(tensors[i],i==1 || i==3);
        const auto right_end=right+tensor_bytes(tensors[j],j==1 || j==3);
        TORCH_CHECK(left_end<=right || right_end<=left,"GEMM tensors must not overlap");
    }
}
CUtensorMap make_map(void* ptr,CUtensorMapDataType dtype,std::uint64_t inner,
                    std::uint64_t outer,std::uint32_t box_inner,std::uint32_t box_outer,
                    std::uint64_t pitch,CUtensorMapSwizzle swizzle) {
    CUtensorMap map{};
    const cuuint64_t dimensions[]={inner,outer},strides[]={pitch};
    const cuuint32_t box[]={box_inner,box_outer},elements[]={1,1};
    C10_CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(&map,dtype,2,ptr,dimensions,strides,box,elements,
        CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}


template<int Heads,int BM,int Stages>
const void* address() {
 return reinterpret_cast<const void*>(&hy4_qb::q_b_proj_impl<2048,Heads,256,192,64,576,BM,128,Stages,148,true,2>);
}
struct Plan {
 std::array<at::Tensor,8> tensors;
 std::array<CUtensorMap,5> maps;
 const void* kernel;
 int rows,heads,bm,stages,smem,workers,device;
 std::map<std::string,int> metadata() const {
  return {{"rows",rows},{"heads",heads},{"compiled_n",heads*256},{"compiled_k",2048},
   {"block_m",bm},{"block_n",128},{"stages",stages},{"cluster",2},{"threads",384},
   {"grid",workers},{"dynamic_shared_bytes",smem},{"granularity_a",32},{"granularity_b",32},
   {"input_gdc",1},{"early_programmatic_trigger",1},{"q_stride_head",256},{"transformed_stride_head",576}};
 }
 void run(const at::Tensor& positions,bool pdl,std::uint64_t event) const {
  TORCH_CHECK(positions.sizes()==tensors[5].sizes() && positions.strides()==tensors[5].strides() &&
   positions.scalar_type()==tensors[5].scalar_type() && positions.device()==tensors[5].device(),"position layout changed");
  for(int i=0;i<8;++i) if(i!=5) {
   auto x=reinterpret_cast<uintptr_t>(positions.data_ptr()),y=reinterpret_cast<uintptr_t>(tensors[i].data_ptr());
   TORCH_CHECK(x+positions.nbytes()<=y || y+tensor_bytes(tensors[i],i==1 || i==3)<=x,"positions alias input/output");
  }
  const c10::cuda::CUDAGuard guard(device);
  auto m=static_cast<std::uint32_t>(rows);
  const auto* cos=tensors[6].data_ptr<float>();
  const auto* pos=positions.data_ptr<std::int32_t>();
  auto* q=tensors[4].data_ptr(); auto* transformed=tensors[7].data_ptr();
  void* args[]={&m,const_cast<CUtensorMap*>(&maps[0]),const_cast<CUtensorMap*>(&maps[1]),
   const_cast<CUtensorMap*>(&maps[2]),const_cast<CUtensorMap*>(&maps[3]),const_cast<CUtensorMap*>(&maps[4]),
   &cos,&pos,&q,&transformed};
  cudaLaunchAttribute attrs[3]{}; int count=0;
  attrs[count].id=cudaLaunchAttributeClusterDimension; attrs[count++].val.clusterDim={2,1,1};
  if(pdl) {attrs[count].id=cudaLaunchAttributeProgrammaticStreamSerialization;
   attrs[count++].val.programmaticStreamSerializationAllowed=1;}
  if(pdl && event) {attrs[count].id=cudaLaunchAttributeProgrammaticEvent;
   attrs[count].val.programmaticEvent.event=reinterpret_cast<cudaEvent_t>(event);
   attrs[count].val.programmaticEvent.flags=0;attrs[count++].val.programmaticEvent.triggerAtBlockStart=0;}
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(workers);cfg.blockDim=dim3(384);cfg.dynamicSmemBytes=smem;
  cfg.stream=at::cuda::getCurrentCUDAStream();cfg.attrs=attrs;cfg.numAttrs=count;
  C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg,kernel,args));
  if(!pdl && event) C10_CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),cfg.stream));
 }
};
std::shared_ptr<Plan> prepare(const at::Tensor& a,const at::Tensor& sa,const at::Tensor& b,const at::Tensor& sb,
 const at::Tensor& pos,const at::Tensor& cos,const at::Tensor& q,const at::Tensor& transformed,int heads) {
 TORCH_CHECK(heads==8 || heads==64,"heads must be 8 or 64");
 TORCH_CHECK(q.sizes()==at::IntArrayRef({a.size(0),heads,256}),"Q shape mismatch");
 TORCH_CHECK(b.size(0)==heads*256,"weight/head mismatch");
 const c10::cuda::CUDAGuard guard(a.device());
 cudaStreamCaptureStatus status;
 C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(),&status));
 TORCH_CHECK(status==cudaStreamCaptureStatusNone,"prepare must run outside capture");
 validate({a,sa,b,sb,q.view({a.size(0),heads*256})},1);
 TORCH_CHECK(pos.is_cuda() && pos.device()==a.device() && pos.scalar_type()==at::kInt &&
  pos.sizes()==at::IntArrayRef({a.size(0)}) && pos.is_contiguous(),"positions require contiguous int32[M]");
 TORCH_CHECK(cos.is_cuda() && cos.device()==a.device() && cos.scalar_type()==at::kFloat &&
  cos.dim()==2 && cos.size(1)==64 && cos.size(0)>0 && cos.is_contiguous(),"cos_sin requires float32[P,64]");
 TORCH_CHECK(transformed.is_cuda() && transformed.device()==a.device() && transformed.scalar_type()==at::kBFloat16 &&
  transformed.sizes()==at::IntArrayRef({a.size(0),heads,576}) && transformed.is_contiguous(),"transformed layout mismatch");
 auto plan=std::make_shared<Plan>();
 plan->tensors={a,sa,b,sb,q,pos,cos,transformed};
 for(int i=0;i<8;++i) for(int j=i+1;j<8;++j) {
  auto x=reinterpret_cast<std::uintptr_t>(plan->tensors[i].data_ptr());
  auto y=reinterpret_cast<std::uintptr_t>(plan->tensors[j].data_ptr());
  TORCH_CHECK(x+tensor_bytes(plan->tensors[i],i==1 || i==3)<=y ||
   y+tensor_bytes(plan->tensors[j],j==1 || j==3)<=x,"QB tensors must not overlap");
 }
 cudaDeviceProp prop{};C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,a.get_device()));
 TORCH_CHECK(prop.major==10 && prop.minor==3 && prop.multiProcessorCount==148,"QB requires SM103/148 SMs");
 plan->device=a.get_device();plan->rows=a.size(0);plan->heads=heads;
 plan->bm=plan->rows<=32 ? 16 : plan->rows<=64 ? 32 : 128;
 plan->stages=plan->rows<=32 ? 12 : plan->rows<=64 ? 11 : 8;
 #define SELECT(H) (plan->bm==16 ? address<H,16,12>() : plan->bm==32 ? address<H,32,11>() : address<H,128,8>())
 plan->kernel=heads==64 ? SELECT(64) : SELECT(8);
 #undef SELECT
 plan->smem=rtp_kernel::glm5::deep_gemm_config::shared_bytes(plan->bm,128,plan->bm/2,128,plan->stages,128,2,true);
 TORCH_CHECK(plan->smem<=232448,"shared memory overflow");
 plan->workers=std::min(148,((plan->rows+plan->bm-1)/plan->bm)*(heads*2));
 C10_CUDA_CHECK(cudaFuncSetAttribute(plan->kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,plan->smem));
 int am=(plan->rows+3)/4*4,n=heads*256;
 plan->maps[0]=make_map(a.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,2048,plan->rows,128,plan->bm/2,2048,CU_TENSOR_MAP_SWIZZLE_128B);
 plan->maps[1]=make_map(b.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,2048,n,128,128,2048,CU_TENSOR_MAP_SWIZZLE_128B);
 plan->maps[2]=make_map(sa.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_INT32,am,16,plan->bm,1,am*4,CU_TENSOR_MAP_SWIZZLE_NONE);
 plan->maps[3]=make_map(sb.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_INT32,n,16,128,1,n*4,CU_TENSOR_MAP_SWIZZLE_NONE);
 plan->maps[4]=make_map(q.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,n,plan->rows,64,16,n*2,CU_TENSOR_MAP_SWIZZLE_128B);
 return plan;
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
void gemm_test_copy(const at::Tensor& src,const at::Tensor& dst,bool pdl,bool early) {
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

void bind(pybind11::module_& module) {
 pybind11::class_<Plan,std::shared_ptr<Plan>>(module,"Plan",pybind11::module_local()).def("run",&Plan::run).def("metadata",&Plan::metadata);
 module.def("prepare_plan",&prepare);
 module.def("_test_copy",&gemm_test_copy);
}
} // namespace hy4_qb_host

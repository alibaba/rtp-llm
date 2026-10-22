// Fixed V4.1 Q GEMM, non-swap 2CTA geometry; accumulator stages depend on BN.
// Uses existing cuda_extension/CUTLASS instruction wrappers, not DeepGEMM launch.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "wq_b_support.cuh"
#include "nonswap_mma.cuh"
#include "q_swap.cuh"
#ifndef NS_NUM_EPI_STAGES
#define NS_NUM_EPI_STAGES 2
#endif
namespace rtp_dsv41_qb_ns {
namespace ptx=rtp_dsv41_qb::ptx;
using Barrier=cutlass::arch::ClusterTransactionBarrier;
template<int BN> struct Config {
    static constexpr int BM=128,NE=(BN==256?1:NS_NUM_EPI_STAGES),NS=6,SA=128*128,SB=(BN/2)*128;
    static constexpr int SFA=512,SFB=((BN+127)/128)*512;
    static constexpr int STORE_N=32,CD=128*STORE_N*2,TMEM=(NE*BN+12>256?512:256);
    static constexpr int TSFA=NE*BN,TSFB=NE*BN+4;
    static_assert(TSFB+SFB/128<=TMEM);
};
template<int BN> struct Shared {
    using C=Config<BN>;
    alignas(1024) uint8_t cd[2*C::CD];
    alignas(1024) uint8_t a[C::NS*C::SA];
    alignas(1024) uint8_t b[C::NS*C::SB];
    alignas(128) uint8_t sfa[C::NS*C::SFA];
    alignas(128) uint8_t sfb[C::NS*C::SFB];
    alignas(16) Barrier full[C::NS],empty[C::NS],sf_full[C::NS],tfull[C::NE],tempty[C::NE];
    alignas(16) uint32_t tptr;
};
template<int BN,bool MFast,int Prefetch=1,bool Side=false>
__global__ __launch_bounds__(Side?512:256,1) void q_nonswap(
    const __grid_constant__ CUtensorMap da,const __grid_constant__ CUtensorMap db,
    const __grid_constant__ CUtensorMap dd,const __grid_constant__ CUtensorMap dsa,
    const __grid_constant__ CUtensorMap dsb,int m,int nt,int clusters,rtp_dsv41_qb::SideArgs side) {
    using C=Config<BN>;extern __shared__ __align__(1024) uint8_t raw[];
    auto& s=*reinterpret_cast<Shared<BN>*>(raw);
    const int warp=threadIdx.x/32,lane=threadIdx.x%32,rank=ptx::block_rank_in_cluster(),cid=blockIdx.x/2;
    const int total=((m+255)/256)*nt;
    if(side.wait_pdl)cudaGridDependencySynchronize();
    ptx::cluster_sync();
    if(warp==0){cute::prefetch_tma_descriptor(&da);cute::prefetch_tma_descriptor(&db);cute::prefetch_tma_descriptor(&dd);cute::prefetch_tma_descriptor(&dsa);cute::prefetch_tma_descriptor(&dsb);}
    if(warp==1&&ptx::elect_one_sync()) {
        #pragma unroll
        for(int i=0;i<C::NS;++i){s.full[i].init(1);s.empty[i].init(1);s.sf_full[i].init(64);}
        #pragma unroll
        for(int i=0;i<C::NE;++i){s.tfull[i].init(1);s.tempty[i].init(256);}
        cutlass::arch::fence_barrier_init();
    }else if(warp==2)ptx::tcgen05_alloc_2sm(static_cast<uint32_t>(__cvta_generic_to_shared(&s.tptr)),C::TMEM);
    ptx::cluster_sync();
    if(warp==0&&ptx::elect_one_sync()) {
        int st=0,ph=0;
        for(int task=cid;task<total;task+=clusters){
            int mi=(MFast?(m>256?(task&1)*256:0):(task/nt)*256)+rank*128;
            int ni=(MFast?(m>256?task>>1:task):task%nt)*BN;
            for(int k=0;k<10;++k){
                s.empty[st].wait(ph^1);
                cute::SM90_TMA_LOAD_2D::copy(&da,reinterpret_cast<uint64_t*>(&s.full[st]),static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),s.a+st*C::SA,k*128,mi);
                cute::SM90_TMA_LOAD_2D::copy(&db,reinterpret_cast<uint64_t*>(&s.full[st]),static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST),s.b+st*C::SB,k*128,ni+rank*(BN/2));
                cute::SM90_TMA_LOAD_2D::copy(&dsa,reinterpret_cast<uint64_t*>(&s.full[st]),static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),s.sfa+st*C::SFA,mi,k);
                cute::SM90_TMA_LOAD_2D::copy(&dsb,reinterpret_cast<uint64_t*>(&s.full[st]),static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),s.sfb+st*C::SFB,ni,k);
                s.full[st].arrive_and_expect_tx(C::SA+C::SB+C::SFA+C::SFB);
                if(++st==C::NS){st=0;ph^=1;}
            }
        }
    }else if(warp==2){
        int st=0,ph=0;
        for(int task=cid;task<total;task+=clusters)for(int k=0;k<10;++k){
            s.full[st].wait(ph);
            auto transpose=[&](uint32_t* ptr){uint32_t v[4];
                #pragma unroll
                for(int i=0;i<4;++i)v[i]=ptx::ld_shared_u32(ptr+i*32+lane);
                __syncwarp();ptx::st_shared_v4_u32(ptr+lane*4,v[0],v[1],v[2],v[3]);};
            transpose(reinterpret_cast<uint32_t*>(s.sfa+st*C::SFA));
            #pragma unroll
            for(int i=0;i<C::SFB/512;++i)transpose(reinterpret_cast<uint32_t*>(s.sfb+st*C::SFB+i*512));
            cutlass::arch::fence_view_async_shared();s.sf_full[st].arrive(0u);
            if(++st==C::NS){st=0;ph^=1;}
        }
    }else if(warp==1&&rank==0){
        using Engine=cluster_mma_fp8::ClusterMmaFP8BlockScale<128,C::NS,128,BN,false>;
        auto ds=Engine::init_desc(reinterpret_cast<__nv_fp8_e4m3*>(s.a),reinterpret_cast<__nv_fp8_e4m3*>(s.b),lane);
        uint32_t st=0,ph=0,it=0;
        for(int task=cid;task<total;task+=clusters,++it){
            int acc=it%C::NE,phase=(it/C::NE)&1;
            Engine::template run_tile<false>(ds,s.sf_full,s.empty,s.tfull[acc],s.tempty[acc],s.sfa,s.sfb,
                acc*BN,C::TSFA,C::TSFB,10,phase,st,ph);
        }
        if(it)s.tempty[(it-1)%C::NE].wait(((it-1)/C::NE)&1);
    }else if(warp==1&&rank==1){
        // Follower remains free; no MMA work is issued here.
    }else if(warp==3 || (warp>=8&&warp<=10)){
        if constexpr(Side){
            const int worker=blockIdx.x*4+(warp==3?0:warp-7);
            if(side.kv)for(int row=worker;row<side.rows;row+=gridDim.x*4)rtp_dsv41_qb::kv_row(side,row);
        }
    }else if(warp>=11){
        if constexpr(Side)rtp_dsv41_qb::index_work(side,blockIdx.x*5+warp-11,gridDim.x*5);
    }else if(warp>=8){
        }else if(warp>=4){
        int ew=warp-4;uint32_t it=0,store=0;
        for(int task=cid;task<total;task+=clusters,++it){
            int mi=(MFast?(m>256?(task&1)*256:0):(task/nt)*256)+rank*128;
            int ni=(MFast?(m>256?task>>1:task):task%nt)*BN;
            int acc=it%C::NE,phase=(it/C::NE)&1;s.tfull[acc].wait(phase);ptx::tcgen05_fence_after_sync();
            #pragma unroll
            for(int col=0;col<BN;col+=32,store^=1){
                auto* dst=s.cd+store*C::CD;
                if(ew==0)cute::tma_store_wait<1>();
                cutlass::arch::NamedBarrier::sync(128,0);
                #pragma unroll
                for(int ib=0;ib<4;ib+=Prefetch){
                    uint32_t values[Prefetch][8];
                    #pragma unroll
                    for(int j=0;j<Prefetch;++j){
                        auto* v=values[j];ptx::tmem_load_32dp32b8x(acc*BN+col+(ib+j)*8,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
                    }
                    cutlass::arch::fence_view_async_tmem_load();
                    #pragma unroll
                    for(int j=0;j<Prefetch;++j){
                    int i=ib+j;auto* v=values[j];
                    uint32_t packed[4];
                    #pragma unroll
                    for(int j=0;j<4;++j){auto bf=__floats2bfloat162_rn(__uint_as_float(v[j*2]),__uint_as_float(v[j*2+1]));packed[j]=*reinterpret_cast<uint32_t*>(&bf);}
                    int group=i+lane*4,row=group/8,c=(group%8)^(row%4);
                    ptx::st_shared_v4_u32(reinterpret_cast<uint32_t*>(dst+ew*32*64+row*128+c*16),packed[0],packed[1],packed[2],packed[3]);
                }
                }
                if(col==BN-32){ptx::tcgen05_fence_before_sync();s.tempty[acc].arrive(0u);}
                cute::tma_store_fence();cutlass::arch::NamedBarrier::sync(128,0);
                if(ew==0&&ptx::elect_one_sync()){cute::SM90_TMA_STORE_2D::copy(&dd,dst,ni+col,mi);cute::tma_store_arrive();}
                __syncwarp();
            }
        }
        if(ew==0)cute::tma_store_wait<0>();
    }
    ptx::cluster_sync();if(warp==0)ptx::tcgen05_dealloc_2sm(0,C::TMEM);
}
CUtensorMap payload(void* ptr,int rows,int cols,int boxrows,int boxcols,bool bf16,bool sw64=false){
    CUtensorMap d{};uint64_t dims[]={uint64_t(cols),uint64_t(rows)},strides[]={uint64_t(cols)*(bf16?2u:1u)};uint32_t box[]={uint32_t(boxcols),uint32_t(boxrows)},elem[]={1,1};
    auto status=cuTensorMapEncodeTiled(&d,bf16?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_UINT8,2,ptr,dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,sw64?CU_TENSOR_MAP_SWIZZLE_64B:CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"TMA payload encode failed: ",int(status));return d;
}
CUtensorMap scales(torch::Tensor sf,int rows){
    CUtensorMap d{};uint64_t dims[]={uint64_t(sf.size(0)),10},strides[]={uint64_t(sf.stride(1))*4};uint32_t box[]={uint32_t(rows),1},elem[]={1,1};
    auto status=cuTensorMapEncodeTiled(&d,CU_TENSOR_MAP_DATA_TYPE_UINT32,2,sf.data_ptr(),dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"TMA scales encode failed: ",int(status));return d;
}
template<int BN,bool MFast,int Prefetch=1,bool Side=false>void launch(torch::Tensor x,torch::Tensor xs,torch::Tensor w,torch::Tensor ws,torch::Tensor out,int clusters,rtp_dsv41_qb::SideArgs side={}){
    int m=x.size(0),n=w.size(0);using C=Config<BN>;
    auto a=payload(x.data_ptr(),m,1280,128,128,false),b=payload(w.data_ptr(),n,1280,BN/2,128,false);
    auto d=payload(out.data_ptr(),m,n,128,32,true,true),sa=scales(xs,128),sb=scales(ws,C::SFB/4);
    auto kernel=q_nonswap<BN,MFast,Prefetch,Side>;constexpr int smem=sizeof(Shared<BN>);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    cudaLaunchAttribute attr[2]{};attr[0].id=cudaLaunchAttributeClusterDimension;attr[0].val.clusterDim={2,1,1};
    attr[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;attr[1].val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(clusters*2);cfg.blockDim=dim3(Side?512:256);cfg.dynamicSmemBytes=smem;cfg.stream=at::cuda::getCurrentCUDAStream();cfg.attrs=attr;cfg.numAttrs=side.wait_pdl?2:1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,kernel,a,b,d,sa,sb,m,(n+BN-1)/BN,clusters,side));
}

} // namespace rtp_dsv41_qb_ns

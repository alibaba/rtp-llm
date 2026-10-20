// V4.1 Q up-projection, derived from cuda_extension dsv4_mega/wq_b CSA.
// Source lock and adaptation notes: README.md. No Q normalization or RoPE here.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "wq_b_support.cuh"
#include "cluster_mma_group32.cuh"
#include "side_workers.cuh"

namespace rtp_dsv41_qb {
using namespace wq_b;
using Barrier = mma_desc::Barrier;
template <int M_TPL>
struct SharedStorage {
    using D = SwapDims<M_TPL>;
    static constexpr int NS   = D::NUM_STAGES;
    static constexpr int NE   = D::NUM_EPI_STAGES;
    static constexpr int SA   = D::SMEM_A_PER_STAGE;
    static constexpr int SSFA = D::SMEM_SFA_PER_STAGE;

    alignas(1024) uint8_t smem_cd[D::SMEM_CD_TOTAL];
    alignas(1024) uint8_t smem_a[NS * SA];                    // activation e4m3
    alignas(1024) uint8_t smem_b[NS * SMEM_B_PER_STAGE];      // weight     e4m3
    alignas(128)  uint8_t smem_sfa[NS * SSFA];                // expanded K32 activation SF
    alignas(128)  uint8_t smem_sfb[NS * SMEM_SFB_PER_STAGE];  // expanded K32 weight SF

    // Barriers
    alignas(16) Barrier full_barriers[NS];            // A/B TMA done (per-CTA, init 1)
    alignas(16) Barrier empty_barriers[NS];           // stage smem reusable
    alignas(16) Barrier with_sf_full_barriers[NS];    // expanded SF ready for UTCCP
    alignas(16) Barrier tmem_full_barriers[NE];
    alignas(16) Barrier tmem_empty_barriers[NE];

    // TMEM base address
    alignas(16) uint32_t tmem_base;
};



template<int BM, bool TmaSF, bool Side=false, bool IndexTeams=false>
__global__ __launch_bounds__(IndexTeams ? 512 : 256, 1) void q_gemm_group32(
    const __grid_constant__ CUtensorMap desc_A,
    const __grid_constant__ CUtensorMap desc_B,
    const __grid_constant__ CUtensorMap desc_D,
    const __grid_constant__ CUtensorMap desc_XS,
    const __grid_constant__ CUtensorMap desc_WS,
    const uint32_t* __restrict__ x_sf, const uint32_t* __restrict__ w_sf,
    int64_t xs0, int64_t xs1, int64_t ws0, int64_t ws1,
    int problem_m, int n_tiles, int num_clusters,SideArgs side) {
    using Dims=SwapDims<BM>;
    constexpr int NS=Dims::NUM_STAGES, NE=Dims::NUM_EPI_STAGES;
    constexpr int SA=Dims::SMEM_A_PER_STAGE, SSFA=Dims::SMEM_SFA_PER_STAGE;
    constexpr int NUM_K_TILES=1280/BLOCK_K;
    static_assert(sizeof(SharedStorage<BM>) <= SMEM_CAPACITY);
    extern __shared__ __align__(1024) uint8_t storage[];
    auto& s=*reinterpret_cast<SharedStorage<BM>*>(storage);
    const int warp_id=threadIdx.x/32, lane_id=ptx::get_lane_idx();
    const int cta_rank=ptx::block_rank_in_cluster(), cluster_id=blockIdx.x/2;
    const int num_tiles_total=((problem_m+BM-1)/BM)*n_tiles;
    if(side.wait_pdl)cudaGridDependencySynchronize();
    ptx::cluster_sync();
    if (warp_id==0) {
        cute::prefetch_tma_descriptor(&desc_A);
        cute::prefetch_tma_descriptor(&desc_B);
        cute::prefetch_tma_descriptor(&desc_D);
        if constexpr(TmaSF) {
            cute::prefetch_tma_descriptor(&desc_XS);
            cute::prefetch_tma_descriptor(&desc_WS);
        }
    }
    if (warp_id==1 && ptx::elect_one_sync()) {
        for (int i=0;i<NS;++i) {
            s.full_barriers[i].init(1);
            s.empty_barriers[i].init(1);
            s.with_sf_full_barriers[i].init(64);
        }
        for (int i=0;i<NE;++i) {
            s.tmem_full_barriers[i].init(1);
            s.tmem_empty_barriers[i].init(256);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_id==2) {
        ptx::tcgen05_alloc_2sm(static_cast<uint32_t>(__cvta_generic_to_shared(&s.tmem_base)), Dims::NUM_TMEM_COLS);
    }
    ptx::cluster_sync();

    if(side.trigger_pdl && threadIdx.x==0)cudaTriggerProgrammaticLaunchCompletion();


    if (warp_id==0 && ptx::elect_one_sync()) {
        uint32_t stage=0,phase=0;
        for (int task=cluster_id;task<num_tiles_total;task+=num_clusters) {
            const int m0=(task/n_tiles)*BM, n0=(task%n_tiles)*256;
            for (int k=0;k<NUM_K_TILES;++k) {
                s.empty_barriers[stage].wait(phase^1);
                tma::copy_2d_fp8(&desc_A,&s.full_barriers[stage],
                    reinterpret_cast<__nv_fp8_e4m3*>(s.smem_a+stage*SA),k*128,m0+cta_rank*(BM/2));
                tma::copy_2d_fp8(&desc_B,&s.full_barriers[stage],
                    reinterpret_cast<__nv_fp8_e4m3*>(s.smem_b+stage*SMEM_B_PER_STAGE),k*128,n0+cta_rank*128);
                if constexpr(TmaSF) {
                    cute::SM90_TMA_LOAD_2D::copy(&desc_XS,reinterpret_cast<uint64_t*>(&s.full_barriers[stage]),
                        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                        s.smem_sfa+stage*SSFA,m0,k);
                    cute::SM90_TMA_LOAD_2D::copy(&desc_WS,reinterpret_cast<uint64_t*>(&s.full_barriers[stage]),
                        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                        s.smem_sfb+stage*SMEM_SFB_PER_STAGE,n0+cta_rank*128,k);
                }
                s.full_barriers[stage].arrive_and_expect_tx(SA+SMEM_B_PER_STAGE+(TmaSF?SSFA+SMEM_SFB_PER_STAGE:0));
                stage=(stage+1)%NS;if(stage==0)phase^=1;
            }
        }
    } else if (warp_id==2) {
        uint32_t stage=0,phase=0;
        constexpr int ATOMS=Dims::SF_BLOCK_M/128;
        for (int task=cluster_id;task<num_tiles_total;task+=num_clusters) {
            const int m0=(task/n_tiles)*BM, n0=(task%n_tiles)*256+cta_rank*128;
            for (int k=0;k<NUM_K_TILES;++k) {
                uint32_t va[ATOMS][4],vb[4];
                auto* sfa=reinterpret_cast<uint32_t*>(s.smem_sfa+stage*SSFA);
                auto* sfb=reinterpret_cast<uint32_t*>(s.smem_sfb+stage*SMEM_SFB_PER_STAGE);
                if constexpr(TmaSF) s.full_barriers[stage].wait(phase);
                #pragma unroll
                for(int atom=0;atom<ATOMS;++atom) {
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        const int row=m0+atom*128+i*32+lane_id;
                        if constexpr(TmaSF) va[atom][i]=ptx::ld_shared_u32(sfa+atom*128+i*32+lane_id);
                        else va[atom][i]=row<problem_m ? x_sf[row*xs0+k*xs1] : 0x7f7f7f7fu;
                    }
                }
                #pragma unroll
                for(int i=0;i<4;++i) {
                    if constexpr(TmaSF) vb[i]=ptx::ld_shared_u32(sfb+i*32+lane_id);
                    else vb[i]=w_sf[(n0+i*32+lane_id)*ws0+k*ws1];
                }
                // Complete every read before the in-place warp transpose overwrites SMEM.
                if constexpr(TmaSF) __syncwarp();
                else s.full_barriers[stage].wait(phase);
                #pragma unroll
                for(int atom=0;atom<ATOMS;++atom)
                    ptx::st_shared_v4_u32(sfa+atom*128+lane_id*4,va[atom][0],va[atom][1],va[atom][2],va[atom][3]);
                ptx::st_shared_v4_u32(sfb+lane_id*4,vb[0],vb[1],vb[2],vb[3]);
                cutlass::arch::fence_view_async_shared();
                s.with_sf_full_barriers[stage].arrive(0u);
                stage=(stage+1)%NS;if(stage==0)phase^=1;
            }
        }
    } else if(warp_id==1 && cta_rank==0) {
        using CM=cluster_mma_fp8::ClusterMmaFP8BlockScale<128,NS,BM,128,true>;
        auto ds=CM::init_desc(reinterpret_cast<__nv_fp8_e4m3*>(s.smem_a),reinterpret_cast<__nv_fp8_e4m3*>(s.smem_b),lane_id);
        uint32_t stage=0,phase=0,iter=0;
        for(int task=cluster_id;task<num_tiles_total;task+=num_clusters,++iter) {
            const uint32_t a=iter%NE,p=(iter/NE)&1;
            CM::template run_tile<false>(ds,s.with_sf_full_barriers,s.empty_barriers,
                s.tmem_full_barriers[a],s.tmem_empty_barriers[a],s.smem_sfa,s.smem_sfb,
                a*BM,Dims::TMEM_START_SFA,Dims::TMEM_START_SFB,NUM_K_TILES,p,stage,phase);
        }
        if(iter) s.tmem_empty_barriers[(iter-1)%NE].wait(((iter-1)/NE)&1);
    } else if(warp_id==1 && cta_rank==1) {
        // Follower CTA never issues MMA or touches its operand pipeline.

    } else if(warp_id==3) {
        if constexpr(Side) {
            if(side.kv)for(int row=blockIdx.x;row<side.rows;row+=gridDim.x)kv_row(side,row);
            if constexpr(!IndexTeams)index_work(side,blockIdx.x,gridDim.x);

        }
    } else if(warp_id>=8) {
        if constexpr(IndexTeams)index_work(side,blockIdx.x*8+warp_id-8,gridDim.x*8);
    } else if(warp_id>=4 && warp_id<8) {
        constexpr int STORE_M_T=Dims::STORE_M;
        constexpr int NUM_STORES=BM/STORE_M_T, NUM_TMEM_SUBROWS=STORE_M_T/8;
        constexpr int NUM_N_STORE_ATOMS=2, SMEM_CD_PER_STAGE_T=Dims::SMEM_CD_PER_STAGE;
        const uint32_t epi_warp_idx=warp_id-4;
        uint32_t tma_store_idx=0,iter=0;
        for(int task=cluster_id;task<num_tiles_total;task+=num_clusters,++iter) {
            const int m0=(task/n_tiles)*BM,out_n=(task%n_tiles)*256+cta_rank*128;
            const uint32_t accum_stage=iter%NE,accum_phase=(iter/NE)&1;
            s.tmem_full_barriers[accum_stage].wait(accum_phase);
            ptx::tcgen05_fence_after_sync();
            const uint32_t tmem_base=accum_stage*BM;
                        for (int st = 0; st < NUM_STORES; ++st,
                 tma_store_idx = (tma_store_idx + 1) % NUM_TMA_STORE_STAGES) {
                auto* smem_cd_ptr = reinterpret_cast<uint8_t*>(
                    s.smem_cd + tma_store_idx * SMEM_CD_PER_STAGE_T);

                if (epi_warp_idx == 0)
                    cute::tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                // Read FP32 accumulators from TMEM and round them once to BF16.
                #pragma unroll
                for (int i = 0; i < NUM_TMEM_SUBROWS; ++i) {
                    uint32_t tmem_addr = tmem_base + st * STORE_M_T + i * 8;

                    uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                    ptx::tmem_load_16dp256b1x(tmem_addr, v0, v1, v2, v3);
                    ptx::tmem_load_16dp256b1x(
                        tmem_addr | 0x00100000u, v4, v5, v6, v7);
                    cutlass::arch::fence_view_async_tmem_load();

                    const float f0 = __uint_as_float(v0);
                    const float f1 = __uint_as_float(v1);
                    const float f2 = __uint_as_float(v2);
                    const float f3 = __uint_as_float(v3);
                    const float f4 = __uint_as_float(v4);
                    const float f5 = __uint_as_float(v5);
                    const float f6 = __uint_as_float(v6);
                    const float f7 = __uint_as_float(v7);

                    const __nv_bfloat162 b0 = __floats2bfloat162_rn(f0, f1);
                    const __nv_bfloat162 b1 = __floats2bfloat162_rn(f2, f3);
                    const __nv_bfloat162 b2 = __floats2bfloat162_rn(f4, f5);
                    const __nv_bfloat162 b3 = __floats2bfloat162_rn(f6, f7);
                    const uint32_t row = lane_id & 7u;
                    const uint32_t col = (epi_warp_idx & 1u) * 4u + (lane_id >> 3);
                    uint8_t* smem_base_ptr = smem_cd_ptr
                        + (epi_warp_idx >> 1) * (STORE_M_T * SWIZZLE_CD)
                        + i * (8 * SWIZZLE_CD);
                    uint8_t* dst = smem_base_ptr
                        + row * (16 * 8) + ((col ^ row) * 16);
                    ptx::stmatrix_x4_trans(
                        dst,
                        *reinterpret_cast<const uint32_t*>(&b0),
                        *reinterpret_cast<const uint32_t*>(&b1),
                        *reinterpret_cast<const uint32_t*>(&b2),
                        *reinterpret_cast<const uint32_t*>(&b3));

                }

                if (st == NUM_STORES - 1) {
                    ptx::tcgen05_fence_before_sync();
                    s.tmem_empty_barriers[accum_stage].arrive(0u);
                }

                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                if (epi_warp_idx == 0 && ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int i = 0; i < NUM_N_STORE_ATOMS; ++i) {
                        auto* smem_ptr = reinterpret_cast<__nv_bfloat16*>(smem_cd_ptr)
                            + i * (STORE_M_T * STORE_BLOCK_N_ATOM);
                        const int n_idx = out_n + i * STORE_BLOCK_N_ATOM;
                        const int m_idx = m0 + st * STORE_M_T;
                        tma::store_2d(&desc_D, smem_ptr, n_idx, m_idx);
                    }
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }


        }
        if(epi_warp_idx==0) cute::tma_store_wait<0>();
    }
    ptx::cluster_sync();
    if(warp_id==0) ptx::tcgen05_dealloc_2sm(0,Dims::NUM_TMEM_COLS);
}

CUtensorMap tensor_map(void* ptr,int rows,int cols,int box_rows,int box_cols,bool bf16) {
    CUtensorMap desc{};
    uint64_t dims[]={static_cast<uint64_t>(cols),static_cast<uint64_t>(rows)};
    uint64_t strides[]={static_cast<uint64_t>(cols)*(bf16?2:1)};
    uint32_t box[]={static_cast<uint32_t>(box_cols),static_cast<uint32_t>(box_rows)},elem[]={1,1};
    auto status=cuTensorMapEncodeTiled(&desc,bf16?CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,ptr,dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"cuTensorMapEncodeTiled: ",static_cast<int>(status));
    return desc;
}

CUtensorMap sf_tensor_map(torch::Tensor sf,int box_rows) {
    CUtensorMap desc{};
    uint64_t dims[]={static_cast<uint64_t>(sf.size(0)),10};
    uint64_t strides[]={static_cast<uint64_t>(sf.stride(1))*4};
    uint32_t box[]={static_cast<uint32_t>(box_rows),1},elem[]={1,1};
    auto status=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_UINT32,2,sf.data_ptr(),dims,strides,box,elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"SF tensor map: ",static_cast<int>(status));
    return desc;
}

template<int BM,bool TmaSF,bool Side=false>
void launch(torch::Tensor x,torch::Tensor xs,torch::Tensor w,torch::Tensor ws,torch::Tensor out,int cluster_override,SideArgs side={}) {
    const int m=x.size(0), n=w.size(0),nt=n/256;
    auto a=tensor_map(x.data_ptr(),m,1280,BM/2,128,false);
    auto b=tensor_map(w.data_ptr(),n,1280,128,128,false);
    auto d=tensor_map(out.data_ptr(),m,n,SwapDims<BM>::STORE_M,64,true);
    int sms=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    TORCH_CHECK(cluster_override>=0&&cluster_override<=sms/2,"cluster override exceeds resident SM pairs");
    int clusters=std::min(((m+BM-1)/BM)*nt,cluster_override?cluster_override:sms/2);
    CUtensorMap sx{},sw{};
    if constexpr(TmaSF) {sx=sf_tensor_map(xs,SwapDims<BM>::SF_BLOCK_M);sw=sf_tensor_map(ws,128);}
    auto kernel=q_gemm_group32<BM,TmaSF,Side>;
    if constexpr(Side) {
        if(side.global_indices)kernel=q_gemm_group32<BM,TmaSF,Side,true>;
    }
    constexpr int smem=sizeof(SharedStorage<BM>);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id=cudaLaunchAttributeClusterDimension;attrs[0].val.clusterDim={2,1,1};
    attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim=dim3(clusters*2);cfg.blockDim=dim3(Side && side.global_indices ? 512 : 256);cfg.dynamicSmemBytes=smem;
    cfg.stream=at::cuda::getCurrentCUDAStream();cfg.attrs=attrs;cfg.numAttrs=(side.wait_pdl||side.trigger_pdl)?2:1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,kernel,a,b,d,sx,sw,
        reinterpret_cast<const uint32_t*>(xs.data_ptr()),reinterpret_cast<const uint32_t*>(ws.data_ptr()),
        xs.stride(0),xs.stride(1),ws.stride(0),ws.stride(1),m,nt,clusters,side));
}

} // namespace rtp_dsv41_qb

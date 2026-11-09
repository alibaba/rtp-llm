#pragma once
#include "mqa_logits_utils.cuh"
#include "paged_mqa_logits_scheduler.cuh"

namespace cutlass::gemm::kernel {

// ============================================================================
// FP4 Paged MQA Logits Kernel
// ============================================================================
// Key differences from FP8 PPUPagedMqaLogits:
//   - ElementQK is uint8_t (packed FP4, 2 elements per byte)
//   - kHeadDim is the *packed* dimension (original_head_dim / 2 = 64)
//   - Additional q_sf (Q scale factor, uint8 e8m0) parameter
//   - k_sf is uint8 e8m0 (not float32 per-token scale)
//   - Q/K/q_sf/k_sf/weights all use AIU load
//   - Output dtype controlled by ElementLogits (float or bfloat16)
//   - MMA atom: F32F4F4F32 (16x16x64)
// ============================================================================

template<typename ElementQK,
         typename ElementAcc,
         typename ElementLogits,
         typename ElementWeights,
         uint32_t kNextN,
         uint32_t kNumHeads,
         uint32_t kHeadDim,
         uint32_t BLOCK_KV,
         uint32_t WARP_KV,
         uint32_t kNumQStages,
         uint32_t kNumKVStages,
         uint32_t SPLIT_KV,
         bool     SPLIT_MBLOCK,
         bool     WITH_WITNESS = false>
class C4PagedScoreKernel {
public:
    static_assert(cute::is_same_v<ElementQK, uint8_t>, "FP4 paged MQA logits requires uint8_t ElementQK");
    static_assert(kHeadDim == 64, "FP4 packed head_dim must be 64 (original 128 / 2)");

    static constexpr int BLOCK_M = BLOCK_KV;
    static constexpr int BLOCK_N = kNextN * kNumHeads;
    static constexpr int BLOCK_K = 64;  // kHeadDim = 64
    static constexpr int WARP_M  = WARP_KV;
    static constexpr int WARP_N  = kNumHeads;

    static constexpr int kNumMathWarpGroups = SPLIT_KV / BLOCK_KV;

    using TileShape              = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
    static constexpr int WarpOnM = BLOCK_M / WARP_M;
    static constexpr int WarpOnN = BLOCK_N / WARP_N;

    // FP4 MMA: 16x16x64 F32F4F4F32
    using MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;
    // For FP4 (uint8, 1 byte), MMA K dimension is 32 (same as FP8 int8)
    using MmaK_type = _32;

    static constexpr int InstM    = 16;
    static constexpr int InstN    = 16;
    static constexpr int MmaIterM = WARP_M / InstM;
    static constexpr int MmaIterN = WARP_N / InstN;
    using PermutationMNK =
        Tile<Layout<Shape<Int<InstM>, Int<WarpOnM>, Int<MmaIterM>>, Stride<_1, Int<WARP_M>, Int<InstM>>>,
             Layout<Shape<Int<InstN>, Int<WarpOnN>, Int<MmaIterN>>, Stride<_1, Int<WARP_N>, Int<InstN>>>,
             MmaK_type>;
    using TiledMma = TiledMMA<MMA_Atom<MmaInst>, Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>, PermutationMNK>;

    static constexpr int  MaxThreadsPerBlock         = kNumMathWarpGroups * size(TiledMma{});
    static constexpr int  WarpOnGroup                = size(TiledMma{}) / 32;
    static constexpr int  MinBlocksPerMultiprocessor = 1;
    static constexpr bool WarpInterleaving           = (MaxThreadsPerBlock == 512);
    static_assert(!(WarpInterleaving && kNumQStages == 1), "warp-interleave does not support stage_q = 1");

    using DefaultOperandA = cutlass::gemm::config::
        DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementQK, false, Int<BLOCK_M>, Int<BLOCK_K>, false>;
    using DefaultOperandB = cutlass::gemm::config::
        DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementQK, false, Int<BLOCK_N>, Int<BLOCK_K>, true>;
    // A (K/V)
    using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
    using SmemCopyAtomA   = typename DefaultOperandA::SmemCopyAtom;
    using GmemTiledCopyA  = typename DefaultOperandA::GmemTiledCopy;
    // B (Q)
    using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
    using SmemCopyAtomB   = typename DefaultOperandB::SmemCopyAtom;
    using GmemTiledCopyB  = typename DefaultOperandB::GmemTiledCopy;

    static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{}, Shape<Int<BLOCK_M>, Int<BLOCK_K>, Int<kNumKVStages>, Int<kNumMathWarpGroups>>{}));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<BLOCK_N>, Int<BLOCK_K>, Int<kNumQStages>>{}));

    // ===========================================================================
    // FP4 Scale (SFA/SFB) gmem tiled copy — async copy, no pre-rearrangement
    // ===========================================================================
    // K_sf (SFA): per-row e8m0 scale, shape [num_kv_blocks * block_kv, kv_sf_dim]
    //   kv_sf_dim = kHeadDim / 16 = 4  (for original head_dim=128, packed=64)
    //   We load kv_sf_dim = 4 bytes per KV row as one uint32_t
    // Q_sf (SFB): per-head-group e8m0 scale, shape [batch * next_n, num_heads, q_sf_dim]
    //   q_sf_dim = kHeadDim / 16 = 4 bytes per head, loaded as one uint32_t
    // Weights: float32 per (batch * next_n, num_heads), same as FP8 paged MQA

    // SFA: k_sf — async copy of uint32_t (4 bytes per KV row)
    // Layout: BLOCK_M rows × 1 column of uint32_t
    using SFCopyAtomWidth = cute::uint32_t;

    // SFA (k_sf): one uint32_t per KV row
    using DefaultOperandSFA = cutlass::gemm::config::
        DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, uint32_t, false, _1, Int<BLOCK_M>, false, 0, false>;
    using SmemLayoutAtomSFA = typename DefaultOperandSFA::SmemLayoutAtom;
    using GmemTiledCopySFA  = typename DefaultOperandSFA::GmemTiledCopy;
    using SmemLayoutSFA     = decltype(tile_to_shape(
        SmemLayoutAtomSFA{}, make_shape(_1{}, Int<BLOCK_M>{}, Int<kNumKVStages>{}, Int<kNumMathWarpGroups>{})));

    // SFB (q_sf/weights): one uint32_t/float32 per Q head
    using DefaultOperandSFB = cutlass::gemm::config::
        DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, uint32_t, false, _1, Int<BLOCK_N>, false, 0, false>;
    using SmemLayoutAtomSFB = typename DefaultOperandSFB::SmemLayoutAtom;
    using GmemTiledCopySFB  = typename DefaultOperandSFB::GmemTiledCopy;
    using SmemLayoutSFB =
        decltype(tile_to_shape(SmemLayoutAtomSFB{}, make_shape(_1{}, Int<BLOCK_N>{}, Int<kNumQStages>{})));

    using DefaultOperandWeight = cutlass::gemm::config::
        DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementWeights, false, _1, Int<BLOCK_N>, false, 0, false>;
    using SmemLayoutAtomWeight = typename DefaultOperandWeight::SmemLayoutAtom;
    using GmemTiledCopyWeight  = typename DefaultOperandWeight::GmemTiledCopy;
    using SmemLayoutWeight =
        decltype(tile_to_shape(SmemLayoutAtomWeight{}, make_shape(_1{}, Int<BLOCK_N>{}, Int<kNumQStages>{})));

    // SFA/SFB/Weight s2r layouts
    MQA_DEFINE_FP4_S2R_LAYOUTS(WarpOnM, MmaIterM, kNumKVStages, BLOCK_M, WarpOnN, MmaIterN, kNumQStages, BLOCK_N)
    // SmemTiledCopyWeights: use conditional copy atom type (same as fp4_mqa_logits.cuh)
    using WeightCopyAtomType = cute::conditional_t<cute::is_same_v<ElementWeights, __nv_bfloat16>, uint32_t, uint64_t>;
    using SmemTiledCopyWeights =
        decltype(make_tiled_copy(Copy_Atom<UniversalCopy<WeightCopyAtomType>, ElementWeights>{},
                                 Layout<Shape<Shape<_4, Int<WarpOnN>>, _1>, Stride<Stride<_1, _4>, _1>>{},
                                 Layout<Shape<_2, _1>>{}));

    static_assert(kNumQStages <= 2 && kNumKVStages >= 3, "q_stage <= 2 and kv_stage >= 3");

    // Kernel level shared memory storage
    struct SharedStorage {
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutA>>           smem_k;     // KV packed FP4
        cute::array_aligned<ElementQK, cute::cosize_v<SmemLayoutB>>           smem_q;     // Q packed FP4
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFA>>          smem_k_sf;  // KV e8m0 scales (uint32_t)
        cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutSFB>>          smem_q_sf;  // Q e8m0 scales (uint32_t)
        cute::array_aligned<ElementWeights, cute::cosize_v<SmemLayoutWeight>> smem_weight;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    struct Arguments {
        const ElementQK*      ptr_q;
        const uint32_t*       q_sf;  // Q scale factor (packed e8m0, 4×uint8 per int32)
        const ElementQK*      ptr_k;
        const uint32_t*       k_sf;  // K scale factor (packed e8m0, uint32_t)
        const ElementWeights* weights;
        const uint32_t        batch_size;
        const uint64_t        logits_stride;
        const uint64_t        kv_cache_stride_bytes;
        const uint32_t        block_table_stride;
        const uint32_t*       context_lens;
        ElementLogits*        logits;
        const uint32_t*       block_table;
        const uint32_t*       schedule_meta;
        const uint32_t        num_pages;
        const int32_t*        row_ends;
        __nv_bfloat16*        witness;
    };

    // Kernel entry point API
    using Params = Arguments;

    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;

    GmemTiledCopySFA    gmem_tiled_copy_SFA;
    GmemTiledCopySFB    gmem_tiled_copy_SFB;
    GmemTiledCopyWeight gmem_tiled_copy_weight;
    using StrideAB  = Stride<Int<BLOCK_K>, _1>;
    using StrideSFA = Stride<Int<BLOCK_M>, _1>;
    using StrideSFB = Stride<Int<BLOCK_N>, _1>;

    CUTLASS_DEVICE void init_aiu_copy() {
        using TilerA   = typename GmemTiledCopyA::Tiler_MN;
        using TilerB   = typename GmemTiledCopyB::Tiler_MN;
        using TilerSFA = typename GmemTiledCopySFA::Tiler_MN;
        using TilerSFB = typename GmemTiledCopySFB::Tiler_MN;

        using TilerWeight = typename GmemTiledCopyWeight::Tiler_MN;

        gmem_tiled_copy_A.desc_.template init<ElementQK, false, get<0>(TilerA{}), get<1>(TilerA{})>(
            nullptr, BLOCK_M, BLOCK_K, StrideAB{});
        gmem_tiled_copy_B.desc_.template init<ElementQK, false, get<0>(TilerB{}), get<1>(TilerB{})>(
            nullptr, BLOCK_N, BLOCK_K, StrideAB{});
        gmem_tiled_copy_SFA.desc_.template init<uint32_t, false, get<0>(TilerSFA{}), get<1>(TilerSFA{})>(
            nullptr, 1, BLOCK_M, StrideSFA{});
        gmem_tiled_copy_SFB.desc_.template init<uint32_t, false, get<0>(TilerSFB{}), get<1>(TilerSFB{})>(
            nullptr, 1, BLOCK_N, StrideSFB{});
        gmem_tiled_copy_weight.desc_.template init<ElementWeights, false, get<0>(TilerWeight{}), get<1>(TilerWeight{})>(
            nullptr, 1, BLOCK_N, StrideSFB{});
    };

    CUTLASS_DEVICE auto load_init(Params const& params) {
        // load init A (K/V — packed FP4)
        Tensor mA_mk =
            make_tensor(make_gmem_ptr(params.ptr_k), Shape<Int<BLOCK_M>, Int<BLOCK_K>>{}, StrideAB{});  // (m,k)
        Tensor mA_mk_mix = make_mix_tensor_like(mA_mk);
        Tensor gA = local_tile(mA_mk_mix, TileShape{}, make_coord(0, 0, _), Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)

        // load init B (Q — packed FP4)
        Tensor mB_nk =
            make_tensor(make_gmem_ptr(params.ptr_q), Shape<Int<BLOCK_N>, Int<BLOCK_K>>{}, StrideAB{});  // (n,k)
        Tensor mB_nk_mix = make_mix_tensor_like(mB_nk);
        Tensor gB = local_tile(mB_nk_mix, TileShape{}, make_coord(0, _, 0), Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)

        // load init k_sf (SFA — e8m0, no pre-rearrangement)
        Tensor mSFA_m     = make_tensor(make_gmem_ptr(params.k_sf), Shape<_1, Int<BLOCK_M>>{}, StrideSFA{});
        Tensor mSFA_m_mix = make_mix_tensor_like(mSFA_m);
        Tensor gSFA       = local_tile(mSFA_m_mix, Shape<_1, Int<BLOCK_M>>{}, make_coord(_, 0));

        // load init q_sf (SFB — e8m0, no pre-rearrangement)
        Tensor mSFB_n     = make_tensor(make_gmem_ptr(params.q_sf), Shape<_1, Int<BLOCK_N>>{}, StrideSFB{});
        Tensor mSFB_n_mix = make_mix_tensor_like(mSFB_n);
        Tensor gSFB       = local_tile(mSFB_n_mix, Shape<_1, Int<BLOCK_N>>{}, make_coord(0, _));

        // weights (float32, same as FP8 paged MQA)
        Tensor mW_n     = make_tensor(make_gmem_ptr(params.weights), Shape<_1, Int<BLOCK_N>>{}, StrideSFB{});
        Tensor mW_n_mix = make_mix_tensor_like(mW_n);
        Tensor gW       = local_tile(mW_n_mix, Shape<_1, Int<BLOCK_N>>{}, make_coord(0, _));

        return cute::make_tuple(gA, gB, gSFA, gSFB, gW);
    }

    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {
#if __HGGC_ARCH__ == 100
        return;
#endif
        int warp_idx                                            = canonical_warp_idx_sync();
        int thread_idx                                          = int(threadIdx.x);
        auto [warp_group_idx, local_warp_idx, local_thread_idx] = [&]() {
            if constexpr (kNumMathWarpGroups == 1) {
                return cute::make_tuple(0, warp_idx, thread_idx);
            } else {
                int warp_group_idx   = warp_idx / WarpOnGroup;
                int local_warp_idx   = warp_idx % WarpOnGroup;
                int local_thread_idx = thread_idx % (32 * WarpOnGroup);
                return cute::make_tuple(warp_group_idx, local_warp_idx, local_thread_idx);
            }
        }();
        int warp_m_idx    = local_warp_idx % WarpOnM;
        int warp_n_idx    = local_warp_idx / WarpOnM;
        int lane_idx      = get_lane_idx();
        int warp_group_id = warp_idx / 8;

        // Kernel level shared memory storage
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        // init aiu copy and async copy
        init_aiu_copy();

        // init input tensors
        auto [gA, gB, gSFA, gSFB, gW] = load_init(params);  // gSFA=k_sf, gSFB=q_sf, gW=weights

        // 4D KV smem: (BLOCK_M, BLOCK_K, kNumKVStages, kNumMathWarpGroups)
        // Slice by warp_group_idx to get this warp_group's 3D region: (BLOCK_M, BLOCK_K, kNumKVStages)
        Tensor sA_full = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutA{});  // 4D
        Tensor sA      = sA_full(_, _, _, warp_group_idx);  // 3D slice for this warp_group
        Tensor sB      = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
        // Partition the copying of A and B tiles across the threads (use local_thread_idx for 128-thread group)
        auto   gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(local_thread_idx);
        auto   gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(local_thread_idx);
        Tensor tAgA            = gmem_thr_copy_A.partition_S(gA);
        Tensor tAsA            = gmem_thr_copy_A.partition_D(sA);
        Tensor tBgB            = gmem_thr_copy_B.partition_S(gB);
        Tensor tBsB            = gmem_thr_copy_B.partition_D(sB);

        // 4D SFA smem: (_1, BLOCK_M, kNumKVStages, kNumMathWarpGroups)
        // Slice by warp_group_idx to get this warp_group's 3D region
        Tensor sSFA_full = make_tensor(make_smem_ptr(shared_storage.smem_k_sf.data()), SmemLayoutSFA{});  // 4D
        Tensor sSFA      = sSFA_full(_, _, _, warp_group_idx);  // 3D slice for this warp_group
        Tensor sSFB      = make_tensor(make_smem_ptr(shared_storage.smem_q_sf.data()), SmemLayoutSFB{});
        Tensor sW        = make_tensor(make_smem_ptr(shared_storage.smem_weight.data()), SmemLayoutWeight{});

        auto gmem_thr_copy_SFA    = gmem_tiled_copy_SFA.get_thread_slice(local_thread_idx);
        auto gmem_thr_copy_SFB    = gmem_tiled_copy_SFB.get_thread_slice(local_thread_idx);
        auto gmem_thr_copy_weight = gmem_tiled_copy_weight.get_thread_slice(local_thread_idx);

        Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
        Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

        Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
        Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

        Tensor tWgW = gmem_thr_copy_weight.partition_S(gW);
        Tensor tWsW = gmem_thr_copy_weight.partition_D(sW);

        TiledMma tiled_mma;
        Tensor   accum   = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
        auto     thr_mma = tiled_mma.get_thread_slice(local_thread_idx);
        Tensor   tCrA    = thr_mma.partition_fragment_A(sA(_, _, 0));
        Tensor   tCrB    = thr_mma.partition_fragment_B(sB(_, _, 0));

        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

        auto   smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto   smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(local_warp_idx * 32);
        Tensor tCsA              = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));
        Tensor tCrA_copy_view    = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));

        auto   smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
        auto   smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(local_warp_idx * 32);
        Tensor tCsB              = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));
        Tensor tCrB_copy_view    = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));

        // SFA s2r: recast smem as uint16_t, then apply transposed layout for MMA consumption
        Tensor sSFAUint16 = recast<uint16_t>(sSFA);
        Tensor sSFATrans  = make_tensor(sSFAUint16.data(), sSFATransLayout{});

        SmemTiledCopySFA smem_tiled_copy_SFA;
        auto             smem_thr_copy_SFA = smem_tiled_copy_SFA.get_slice(warp_m_idx * 32 + lane_idx);
        Tensor           tCsSFA            = smem_thr_copy_SFA.partition_S(sSFATrans);
        Tensor           tCrSFA_stage      = smem_thr_copy_SFA.partition_D(sSFATrans(_, _, 0));
        using SFARegType                   = decltype(make_fragment_like(tCrSFA_stage));
        SFARegType tCrSFA_copy_view;

        // SFB s2r: recast smem as uint16_t, then apply transposed layout for MMA consumption
        Tensor sSFBUint16 = recast<uint16_t>(sSFB);
        Tensor sSFBTrans  = make_tensor(sSFBUint16.data(), sSFBTransLayout{});

        SmemTiledCopySFB smem_tiled_copy_SFB;
        auto             smem_thr_copy_SFB = smem_tiled_copy_SFB.get_slice(warp_n_idx * 32 + lane_idx);
        Tensor           tCsSFB            = smem_thr_copy_SFB.partition_S(sSFBTrans);
        Tensor           tCrSFB_stage      = smem_thr_copy_SFB.partition_D(sSFBTrans(_, _, 0));
        using SFBRegType                   = decltype(make_fragment_like(tCrSFB_stage));
        SFBRegType tCrSFB_copy_view;

        // FP4 scale as uint32_t for the 6-operand gemm
        Tensor tCrSFA = recast<uint32_t>(tCrSFA_copy_view);
        Tensor tCrSFB = recast<uint32_t>(tCrSFB_copy_view);

        constexpr bool enable_print = false;
        bool           thread_print = enable_print && cute::thread(0, 0);

        // Scheduler
        auto scheduler = deep_gemm::PagedMQALogitsScheduler<BLOCK_KV, kNumMathWarpGroups, kNextN>(
            params.batch_size, blockIdx.x, params.context_lens, params.schedule_meta);
        DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

        constexpr int        elem_weights = kNumHeads / 4;
        ElementLogits        weights[elem_weights];
        Tensor               sW_copy = make_tensor(sW.data(), sWCopyLayout{});
        SmemTiledCopyWeights smem_tiled_copy_weights;
        auto                 smem_thr_copy_weights = smem_tiled_copy_weights.get_slice(warp_n_idx * 4 + lane_idx % 4);
        Tensor               tCsW                  = smem_thr_copy_weights.partition_S(sW_copy);
        Tensor               tCrW_copy_view        = make_tensor_like(tCsW(_, _, _, 0));
        using WeightRegLayout = Layout<Shape<Shape<_2, _2, _2>, Int<MmaIterN>>, Stride<Stride<_1, _0, _2>, _4>>;
        Tensor tCrW           = make_tensor(static_cast<ElementLogits*>(weights), WeightRegLayout{});

        clear(accum);
        auto tKgK      = tAgA;
        auto tSFKgSFK  = tSFAgSFA;
        auto tQgQ      = tBgB;
        auto tSFQgSFQ  = tSFBgSFB;
        auto tWgW_base = tWgW;

        const auto& warp_offset = warp_m_idx * WARP_M;
        const auto& v_0_offset  = lane_idx / 4 + 0;
        const auto& v_1_offset  = lane_idx / 4 + 8;

        uint32_t q_idx = scheduler.current_q_idx;
        uint32_t kv_idx_base;
        uint32_t kv_idx_array[kNumKVStages];
        uint32_t smem_pipe_read_q = 0, smem_pipe_read_kv = 0;
        uint32_t smem_pipe_write_q = 0, smem_pipe_write_kv = 0;

        // Load Q (packed FP4) + q_sf (e8m0) + weights from gmem to smem
        auto load_q_g2s = [&](uint32_t q_idx) {
            auto q_offset   = q_idx * BLOCK_N;
            tBgB.data()     = tQgQ.data() + q_offset * kHeadDim;
            tSFBgSFB.data() = tSFQgSFQ.data() + q_offset;
            tWgW.data()     = tWgW_base.data() + q_offset;
            copy_aiu(gmem_tiled_copy_B, tBgB(_, _, _, 0), tBsB(_, _, _, smem_pipe_write_q), warp_idx);
            copy_aiu<true>(gmem_tiled_copy_SFB,
                           tSFBgSFB(_, _, _, 0),
                           tSFBsSFB(_, _, _, smem_pipe_write_q),
                           gmem_tiled_copy_weight,
                           tWgW(_, _, _, 0),
                           tWsW(_, _, _, smem_pipe_write_q),
                           warp_idx);

            if (thread_print) {
                printf("  copy_q q_idx = %d, q_offset = %d, stage = %d\n", q_idx, q_offset, smem_pipe_write_q);
            }
            smem_pipe_write_q = (smem_pipe_write_q + 1) % kNumQStages;
        };

        // Load K (packed FP4) + k_sf (e8m0) from gmem to smem
        // Each warp_group loads into its own 3D smem slice, at stage = smem_pipe_write_kv
        auto load_kv_g2s = [&](uint32_t q_idx, uint32_t kv_idx) {
            auto kv_offset           = __ldg(params.block_table + q_idx * params.block_table_stride + kv_idx);
            kv_offset                = kv_offset > 0 && kv_offset < params.num_pages ? kv_offset : 0;
            tAgA.data()              = tKgK.data() + kv_offset * params.kv_cache_stride_bytes;
            tSFAgSFA.data()          = tSFKgSFK.data() + kv_offset * params.kv_cache_stride_bytes / sizeof(uint32_t);
            constexpr bool SPLIT_AIU = (BLOCK_KV > WARP_KV);
            copy_aiu<SPLIT_AIU>(gmem_tiled_copy_A,
                                tAgA(_, _, _, 0),
                                tAsA(_, _, _, smem_pipe_write_kv),
                                gmem_tiled_copy_SFA,
                                tSFAgSFA(_, _, _, 0),
                                tSFAsSFA(_, _, _, smem_pipe_write_kv),
                                local_warp_idx);

            if (thread_print) {
                printf("  copy_k q_idx = %d, kv_idx = %d, kv_offset = %d, wg = %d, stage = %d\n",
                       q_idx,
                       kv_idx,
                       kv_offset,
                       warp_group_idx,
                       smem_pipe_write_kv);
            }
        };

        // Load Q from smem to vreg + read weights
        auto load_q_s2r = [&]() {
            copy(smem_tiled_copy_B, tCsB(_, _, _, smem_pipe_read_q), tCrB_copy_view);
            copy(smem_tiled_copy_SFB, tCsSFB(_, _, _, smem_pipe_read_q), tCrSFB_copy_view);
            copy(smem_tiled_copy_weights, tCsW(_, _, _, smem_pipe_read_q), tCrW_copy_view);
            deep_gemm::load_weights_from_copy_view<ElementWeights, ElementLogits>(
                tCrW_copy_view, weights, elem_weights);

            if (thread_print) {
                printf("    copy q to vreg, q_stage_idx = %d,\n", smem_pipe_read_q);
            }
            smem_pipe_read_q = (smem_pipe_read_q + 1) % kNumQStages;
        };

        auto load_next_qk_g2s = [&](bool load_q) {
            uint32_t q_idx, kv_idx_base, num_kv;
            if (scheduler.fetch_next_task(q_idx, kv_idx_base, num_kv)) {
                if (load_q)
                    load_q_g2s(q_idx);
                uint32_t actual_kv = kv_idx_base + warp_group_idx;
                if (actual_kv < num_kv)
                    load_kv_g2s(q_idx, actual_kv);
            }
            kv_idx_array[smem_pipe_write_kv] = kv_idx_base;
            smem_pipe_write_kv               = (smem_pipe_write_kv + 1) % kNumKVStages;
            if (thread_print)
                printf("cp_async commit, write_stage = %d\n", smem_pipe_write_kv);
            cp_async_fence();
        };

        // Load KV (A + SFA) from smem to vreg — all M blocks for one k_block (float path)
        auto load_kv_s2r = [&](int k_block) {
            copy(smem_tiled_copy_A, tCsA(_, _, k_block, smem_pipe_read_kv), tCrA_copy_view(_, _, k_block));
            copy(smem_tiled_copy_SFA, tCsSFA(_, _, k_block, smem_pipe_read_kv), tCrSFA_copy_view(_, _, k_block));
        };

        // Load KV (A + SFA) from smem to vreg — single m_block for one k_block (bf16 path)
        auto load_kv_s2r_mblock = [&](int k_block, int m_block) {
            copy(smem_tiled_copy_A, tCsA(_, m_block, k_block, smem_pipe_read_kv), tCrA_copy_view(_, m_block, k_block));
            copy(smem_tiled_copy_SFA,
                 tCsSFA(_, m_block, k_block, smem_pipe_read_kv),
                 tCrSFA_copy_view(_, m_block, k_block));
        };

        auto epilogue_mblock = [&](int q_idx, int kv_idx, int m_block) {
            static_assert(kNumMathWarpGroups == 1 && !WarpInterleaving && !SPLIT_MBLOCK);
            static_assert(kNumHeads == 64 && BLOCK_M == 64 && WARP_M == 32 && kNextN == 3);
            uint32_t physical = __ldg(params.block_table + q_idx * params.block_table_stride + kv_idx);
            bool     readable = physical > 0 && physical < params.num_pages;
            uint32_t prefix   = __ldg(params.context_lens + q_idx * kNextN + kNextN - 1);
            // This formula must pass the independent actual CUTE coordinate gate.
            // Keep direct weight addressing identical to the shared-buffer version;
            // reuse of the original tCrW weight layout is a separate experiment.
            auto product = [&](int j, int n) {
                int key   = kv_idx * BLOCK_KV + warp_m_idx * WARP_M + m_block * 16 + lane_idx / 4 + ((j / 2) & 1) * 8;
                int qhead = warp_n_idx * 64 + n * 16 + 2 * (lane_idx & 3) + (j & 1) + (j / 4) * 8;
                __nv_bfloat16 rounded_dot = __float2bfloat16_rn(readable && key < prefix ? accum(j, m_block, n) : 0.0f);
                if constexpr (WITH_WITNESS) {
                    if (key < params.logits_stride)
                        params.witness[(q_idx * BLOCK_N + qhead) * params.logits_stride + key] = rounded_dot;
                }
                float dot    = __bfloat162float(rounded_dot);
                float weight = __bfloat162float(__float2bfloat16_rn(params.weights[q_idx * BLOCK_N + qhead]));
                float relu   = dot < 0.0f ? 0.0f : dot;
                return __bfloat162float(__float2bfloat16_rn(__fmul_rn(relu, weight)));
            };
            CUTLASS_PRAGMA_UNROLL
            for (int key_part = 0; key_part < 2; ++key_part) {
                int   j  = key_part * 2;
                float ae = __fadd_rn(0.0f, product(j + 0, 0));
                float ao = __fadd_rn(0.0f, product(j + 1, 0));
                float be = __fadd_rn(0.0f, product(j + 4, 0));
                float bo = __fadd_rn(0.0f, product(j + 5, 0));
                CUTLASS_PRAGMA_UNROLL
                for (int n = 1; n < 4; ++n) {
                    ae = __fadd_rn(ae, product(j + 0, n));
                    ao = __fadd_rn(ao, product(j + 1, n));
                    be = __fadd_rn(be, product(j + 4, n));
                    bo = __fadd_rn(bo, product(j + 5, n));
                }
                // Lane0/1 pair with lane2/3 to form the original four vt0=4
                // partials. Preserve ((v0+v1)+v2)+v3, not a reassociated sum.
                float pae  = __shfl_xor_sync(0xffffffffu, ae, 2);
                float pao  = __shfl_xor_sync(0xffffffffu, ao, 2);
                float pbe  = __shfl_xor_sync(0xffffffffu, be, 2);
                float pbo  = __shfl_xor_sync(0xffffffffu, bo, 2);
                float even = __fadd_rn(__fadd_rn(__fadd_rn(ae, pae), be), pbe);
                float odd  = __fadd_rn(__fadd_rn(__fadd_rn(ao, pao), bo), pbo);
                // Only lane0 of each four-lane group stores:
                // (partial0+partial2)+(partial1+partial3).
                float even_pair = __fadd_rn(even, __shfl_xor_sync(0xffffffffu, even, 1));
                float odd_pair  = __fadd_rn(odd, __shfl_xor_sync(0xffffffffu, odd, 1));
                float score     = __fadd_rn(even_pair, odd_pair);
                if ((lane_idx & 3) == 0) {
                    int column = kv_idx * BLOCK_KV + warp_m_idx * WARP_M + m_block * 16 + lane_idx / 4 + key_part * 8;
                    int row    = q_idx * kNextN + warp_n_idx;
                    if (column >= params.row_ends[row])
                        score = -INFINITY;
                    if (column < params.logits_stride)
                        params.logits[row * params.logits_stride + column] = __float2bfloat16_rn(score);
                }
            }
        };

        // load qk
        for (int i = 0; i < kNumKVStages - 1; i++) {
            load_next_qk_g2s(i == 0);
        }
        kv_idx_base = kv_idx_array[0];
        // wait AIU Q and first K
        cp_async_wait<kNumKVStages - 2>();
        __syncthreads();

        deep_gemm::warp_interleave_start<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);

        auto handle_q_change = [&]() {
            if constexpr (kNumQStages > 1) {
                if (scheduler.exist_q_idx(q_idx + 1)) {
                    load_q_g2s(q_idx + 1);
                }
                cp_async_fence();
            }
            load_q_s2r();
            if constexpr (kNumQStages == 1) {  // not support warp interleaving
                __syncthreads();
                if (scheduler.exist_q_idx(q_idx + 1)) {
                    load_q_g2s(q_idx + 1);
                }
                cp_async_fence();
            }
        };

        // Make `first` a compile-time constant via std::integral_constant<bool, ...>
        // so the compiler can fold the `first || kv_idx_base == 0` branch and avoid
        // the runtime check on the first iteration.
        auto mainloop = [&](auto first_ic) {
            constexpr bool first = decltype(first_ic)::value;
            if (thread_print) {
                printf("q_idx = %d, kv_idx_base = %d, wg = %d, read_stage = %d\n",
                       q_idx,
                       kv_idx_base,
                       warp_group_idx,
                       smem_pipe_read_kv);
            }

            deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);

            // Handle Q change: on the first iteration always load the next Q;
            // afterwards only when kv_idx_base == 0 (first group of new q).
            if constexpr (first) {
                handle_q_change();
            } else {
                if (kv_idx_base == 0) {
                    handle_q_change();
                }
            }

            // Issue next KV load (overlap with current compute)
            load_next_qk_g2s(false);

            // s2r KV + compute GEMM from current read stage
            constexpr int K_BLOCK_MAX = size<2>(tCrA);
            constexpr int M_BLOCK     = size<1>(accum);
            constexpr int N_BLOCK     = size<2>(accum);
            uint32_t      actual_kv   = kv_idx_base + warp_group_idx;
            if constexpr (not SPLIT_MBLOCK) {
                for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                    load_kv_s2r(k_block);
                    cute::gemm(tiled_mma,
                               accum,
                               tCrA(_, _, k_block),
                               tCrSFA(_, _, k_block),
                               tCrB(_, _, k_block),
                               tCrSFB(_, _, k_block),
                               accum);
                });
                deep_gemm::warp_interleave_arrive<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);
#pragma unroll
                for (int m_block = 0; m_block < size<1>(accum); m_block++) {
                    epilogue_mblock(q_idx, actual_kv, m_block);
                }
            } else {  // Interleaved mma and epilogue
                constexpr int m_group = M_BLOCK / size<1>(tCrSFA);
                constexpr int n_group = N_BLOCK / size<1>(tCrSFB);
                for_each(make_int_sequence<M_BLOCK>{}, [&](auto m_block) {
                    if constexpr (m_block > 0)
                        deep_gemm::warp_interleave_sync<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);
                    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
                        load_kv_s2r_mblock(k_block, m_block);
                        for_each(make_int_sequence<N_BLOCK>{}, [&](auto n_block) {
                            Tensor            s = make_tensor<uint32_t>(Int<4>{});
                            MMA_Atom<MmaInst> mma_atom;
                            Tensor            d = accum(_, m_block, n_block);
                            Tensor            a = tCrA(_, m_block, k_block);
                            Tensor            b = tCrB(_, n_block, k_block);
                            Tensor            c = accum(_, m_block, n_block);
                            s[0]                = tCrSFA(_, _, k_block)[m_block / m_group];
                            s[1]                = tCrSFB(_, _, k_block)[n_block / n_group];
                            s[2]                = m_block % m_group;
                            s[3]                = n_block % n_group;
                            cute::mma_unpack(mma_atom, d, a, b, c, s);
                        });
                    });
                    deep_gemm::warp_interleave_arrive<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);
                    epilogue_mblock(q_idx, actual_kv, m_block);
                });
            }

            cp_async_wait<kNumKVStages - 2>();
            if (!WarpInterleaving) {
                __syncthreads();
            }
            // move to next kv_block
            smem_pipe_read_kv = (smem_pipe_read_kv + 1) % kNumKVStages;
            kv_idx_base       = kv_idx_array[smem_pipe_read_kv];
            if (kv_idx_base == 0)
                q_idx++;

            clear(accum);
        };

        if (!scheduler.is_last_task(q_idx, kv_idx_base)) {
            mainloop(cute::true_type{});
        }
        while (!scheduler.is_last_task(q_idx, kv_idx_base)) {
            mainloop(cute::false_type{});
        }

        deep_gemm::warp_interleave_end<WarpInterleaving>(warp_group_id, MaxThreadsPerBlock);

        cp_async_wait<0>();
        __syncthreads();
    }
};

}  // namespace cutlass::gemm::kernel

namespace deep_gemm {

template<typename ElementQK,
         typename ElementAcc,
         typename ElementLogits,
         typename ElementWeights,
         uint32_t kNextN,
         uint32_t kNumHeads,
         uint32_t kHeadDim,
         uint32_t BLOCK_KV,
         uint32_t WARP_KV,
         uint32_t kNumQStages,
         uint32_t kNumKVStages,
         uint32_t SPLIT_KV,
         bool     SPLIT_MBLOCK,
         bool     WITH_WITNESS = false>
class C4PagedScore {
public:
    static void run(const ElementQK*      ptr_q,
                    const uint32_t*       q_sf,
                    const ElementQK*      ptr_k,
                    const uint32_t*       k_sf,
                    const ElementWeights* weights,
                    const uint32_t        batch_size,
                    const uint64_t        logits_stride,
                    const uint64_t        kv_cache_stride_bytes,
                    const uint32_t        block_table_stride,
                    const uint32_t*       context_lens,
                    ElementLogits*        logits,
                    const uint32_t*       block_table,
                    const uint32_t*       schedule_meta,
                    cudaStream_t          stream,
                    int                   num_sms,
                    int                   num_blocks,
                    uint32_t              num_pages,
                    const int32_t*        row_ends,
                    __nv_bfloat16*        witness) {
        using AttnKernel = cutlass::gemm::kernel::C4PagedScoreKernel<ElementQK,
                                                                     ElementAcc,
                                                                     ElementLogits,
                                                                     ElementWeights,
                                                                     kNextN,
                                                                     kNumHeads,
                                                                     kHeadDim,
                                                                     BLOCK_KV,
                                                                     WARP_KV,
                                                                     kNumQStages,
                                                                     kNumKVStages,
                                                                     SPLIT_KV,
                                                                     SPLIT_MBLOCK,
                                                                     WITH_WITNESS>;

        static constexpr int BLOCK_M            = AttnKernel::BLOCK_M;
        static constexpr int BLOCK_N            = AttnKernel::BLOCK_N;
        static constexpr int BLOCK_K            = AttnKernel::BLOCK_K;
        static constexpr int WARP_M             = AttnKernel::WARP_M;
        static constexpr int WARP_N             = AttnKernel::WARP_N;
        static constexpr int MaxThreadsPerBlock = AttnKernel::MaxThreadsPerBlock;

        int max_blocks_per_cu = compute_occupancy_for_kernel<AttnKernel>();

        typename AttnKernel::Arguments arguments{ptr_q,
                                                 q_sf,
                                                 ptr_k,
                                                 k_sf,
                                                 weights,
                                                 batch_size,
                                                 logits_stride,
                                                 kv_cache_stride_bytes,
                                                 block_table_stride,
                                                 context_lens,
                                                 logits,
                                                 block_table,
                                                 schedule_meta,
                                                 num_pages,
                                                 row_ends,
                                                 witness};
        auto                           params = arguments;
        dim3 const                     block(MaxThreadsPerBlock, 1, 1);
        dim3 const                     grid(num_blocks, 1, 1);
        int                            smem_size_kernel = AttnKernel::SharedStorageSize;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_paged_mqa_logits_params("fp4",
                                                       batch_size,
                                                       kNextN,
                                                       kNumHeads,
                                                       kHeadDim * 2,
                                                       reinterpret_cast<int*>(const_cast<uint32_t*>(context_lens)),
                                                       stream);
            if constexpr (cute::is_same_v<ElementLogits, __nv_bfloat16>) {
                dg_prof_params.add_params("logits_dtype", std::string("bf16"));
            }
            if constexpr (cute::is_same_v<ElementWeights, __nv_bfloat16>) {
                dg_prof_params.add_params("weights_dtype", std::string("bf16"));
            }
        }

        char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, cutlass::device_kernel<AttnKernel>);

            printf("[paged_mqa_logits_fp4:]\n");
            printf("kNumHeads:%d, kHeadDim:%d(packed), kNextN:%d, BLOCK_KV:%d, SPLIT_KV:%d\n",
                   kNumHeads,
                   kHeadDim,
                   kNextN,
                   BLOCK_KV,
                   SPLIT_KV);

            printf("ThreadblockShape[%d, %d], WarpShape[%d, %d], kNumQStages:%d, kNumKVStages:%d\n",
                   BLOCK_M,
                   BLOCK_N,
                   WARP_M,
                   WARP_N,
                   kNumQStages,
                   kNumKVStages);

            printf("num_sms:%d, max_blocks_per_cu:%d, threadblock_count:%d, num_threads: %d\n",
                   num_sms,
                   max_blocks_per_cu,
                   num_blocks,
                   MaxThreadsPerBlock);
            printf("smem_size:%d, vreg:%d, stack:%d\n", smem_size_kernel, int(attr.numRegs), int(attr.localSizeBytes));
            printf("weights_bf16:%s\n", cute::is_same_v<ElementWeights, __nv_bfloat16> ? "true" : "false");
            if (num_sms * max_blocks_per_cu != num_blocks) {
                printf("Warning: num_blocks(%d) should equal to num_sms(%d) * max_blocks_per_cu(%d) = %d\n",
                       num_blocks,
                       num_sms,
                       max_blocks_per_cu,
                       num_sms * max_blocks_per_cu);
            }
        }

        ProfilingInterface::Instance().instrument(true, dg_prof_params);
        cutlass::device_kernel<AttnKernel><<<grid, block, smem_size_kernel, stream>>>(params);
        ProfilingInterface::Instance().instrument(false, dg_prof_params);
    }
};

};  // namespace deep_gemm

#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900)

#include <iostream>

#include "mha.h"
#include "mha_sm90.h"

#define XQA_SM90(func, a0, a1, b0, b1) \
    func##a0##a1##b0##b1(prop, nbKHeads, qScale, output, qkv, ropeCosSin, pool, kvCachePageList, maxSeqLen, seqLen, batchSize, kvCacheScale, semaphores, scratch, stream);

#define XQA_DISPATCH_GROUP_SIZE_SM90(ps, gs)   \
    if (page_size == ps && group_size == gs) { \
        XQA_SM90(xqa_sm90, _ps, ps, _gs, gs)   \
        return;                                \
    }

#define XQA_DISPATCH_PAGE_SIZE_SM90(ps)  \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 1)  \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 2)  \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 3)  \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 4)  \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 5)  \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 6)  \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 7)  \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 8)  \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 9)  \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 10) \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 11) \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 12) \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 13) \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 14) \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 15) \
    XQA_DISPATCH_GROUP_SIZE_SM90(ps, 16)

void run_xqa_sm90(uint32_t page_size, uint32_t group_size, cudaDeviceProp const& prop, uint32_t nbKHeads,
#if SLIDING_WINDOW
    uint32_t slidingWinSize,
#endif
    float qScale, OutputHead* output,
#if LOW_PREC_OUTPUT
    float const* rcpOutScale,
#endif
#if USE_INPUT_KV
    InputHead const* qkv,
#if ROPE_STYLE != 0
    Vec<float, validElemsPerHead> const* ropeCosSin,
#endif
#else
    InputHead const* q,
#endif
#if USE_PAGED_KV_CACHE
    GMemCacheHead* pool, // global pool of pages
    KVCachePageIndex const*
        kvCachePageList, // device pointer. shape: KVCachePageIndex[batchSize][beamWidth][2][maxNbPagesPerSeq].
#else
    GMemKVCacheHead* kvCacheData,
#endif
    uint32_t maxSeqLen, uint32_t const* seqLen,
#if BEAM_WIDTH > 1
    BeamSearchParams const& beamSearchParams,
#endif
    uint32_t batchSize,
    float const* __restrict__ kvCacheScale, // Device memory scalar. Same scale for K and V cache. Used only for
                                            // int8/fp8 KV cache.
#if SPEC_DEC
    SpecDecParams const& specDecParams,
#endif
    uint32_t* semaphores, void* scratch, cudaStream_t stream) {
    XQA_DISPATCH_PAGE_SIZE_SM90(16)
    XQA_DISPATCH_PAGE_SIZE_SM90(32)
    XQA_DISPATCH_PAGE_SIZE_SM90(64)
    XQA_DISPATCH_PAGE_SIZE_SM90(128)
}

#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900


def gen_inc():
    func_inc = f'''
#pragma once

#ifndef __CUDACC__
#include <cuda_runtime_api.h>
#endif
#include "defines.h"
#include "utils.h"
#if SPEC_DEC
#include "specDec.h"
#endif
'''
    return func_inc

def gen_one_decl(func_name: str):
    func_decl = f'''
void {func_name}(cudaDeviceProp const& prop, uint32_t nbKHeads,
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
    uint32_t* semaphores, void* scratch, cudaStream_t stream);
'''
    return func_decl

if __name__ == "__main__":
    with open('mha_sm90.h', 'w') as f:
        f.write(gen_inc())
        for page_size in [16, 32, 64, 128]:
            for group_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                func_name = 'xqa_sm90' + '_ps' + str(page_size) + '_gs' + str(group_size)
                f.write(gen_one_decl(func_name))

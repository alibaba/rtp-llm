import argparse
import os

def gen_inc():
    func_inc = f'''
#pragma once

#ifndef __CUDACC__
#include <cuda_runtime_api.h>
#endif
#include "3rdparty/xqa/defines.h"
#include "3rdparty/xqa/utils.h"
#if SPEC_DEC
#include "3rdparty/xqa/specDec.h"
#endif
'''
    return func_inc

def gen_one_decl(func_name: str, head_dim: int) -> str:
    func_decl = f'''
void {func_name}(cudaDeviceProp const& prop, uint32_t nbKHeads,
#if SLIDING_WINDOW
    uint32_t slidingWinSize,
#endif
    float qScale, Vec<__nv_bfloat16, {head_dim}>* output,
#if LOW_PREC_OUTPUT
    float const* rcpOutScale,
#endif
#if USE_INPUT_KV
    Vec<__nv_bfloat16, {head_dim}> const* qkv,
#if ROPE_STYLE != 0
    Vec<float, {head_dim}> const* ropeCosSin,
#endif
#else
    Vec<__nv_bfloat16, {head_dim}> const* q,
#endif
#if USE_PAGED_KV_CACHE
    Vec<__nv_fp8_e4m3, {head_dim}>* pool, // global pool of pages
    KVCachePageIndex const*
        kvCachePageList, // device pointer. shape: KVCachePageIndex[batchSize][beamWidth][2][maxNbPagesPerSeq].
#else
    Vec<__nv_fp8_e4m3, {head_dim}>* kvCacheData,
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
    parser = argparse.ArgumentParser("Generate xqa sm90 header")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    with open(args.output + '/mha_sm90.h', 'w') as f:
        f.write(gen_inc())
        for head_dim in [64, 128, 256]:
            for page_size in [16, 32, 64, 128]:
                for group_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                    func_name = 'xqa_sm90' + '_hd' + str(head_dim) + '_ps' + str(page_size) + '_gs' + str(group_size)
                    f.write(gen_one_decl(func_name, head_dim))

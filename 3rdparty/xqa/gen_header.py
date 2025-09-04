import argparse


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


def gen_xqa_decl(func_name: str, head_dim: int, input_type: str, kv_cache_type: str, output_type: str, is_spec: bool = False) -> str:
    func_decl = f'''
void {func_name}(cudaDeviceProp const& prop, uint32_t nbKHeads,
    float qScale, Vec<{output_type}, {head_dim}>* output,
'''

    if output_type == "__nv_fp8_e4m3":
        func_decl += f'''
    float const* rcpOutScale,
'''

    func_decl += f'''
    Vec<{input_type}, {head_dim}> const* q,
    Vec<{kv_cache_type}, {head_dim}>* pool, // global pool of pages
    KVCachePageIndex const*
        kvCachePageList, // device pointer. shape: KVCachePageIndex[batchSize][beamWidth][2][maxNbPagesPerSeq].
    uint32_t maxNbPagesPerSeq,
    uint32_t maxSeqLen, uint32_t const* seqLen,
    uint32_t batchSize,
    float const* __restrict__ kvCacheScale, // Device memory scalar. Same scale for K and V cache. Used only for
                                            // int8/fp8 KV cache.
'''

    if is_spec:
        func_decl += f'''
    void* specDecParams,
'''

    func_decl += f'''
    uint32_t* semaphores, void* scratch, cudaStream_t stream);
'''

    return func_decl


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate xqa sm90 header")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    args = parser.parse_args()

    with open(args.output, 'w') as f:
        f.write(gen_inc())
        for head_dim in [64, 128, 256]:
            for page_size in [16, 32, 64, 128]:
                for group_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                    for input_type in ["__nv_bfloat16", "half"]:
                        for kv_cache_type in [input_type, "__nv_fp8_e4m3"]:
                            for output_type in [input_type, "__nv_fp8_e4m3"]:
                                func_name = 'xqa_sm90' + '_hd' + \
                                    str(head_dim) + '_ps' + str(page_size) + '_gs' + str(group_size) + \
                                    '_input_' + input_type + '_kv_cache_' + kv_cache_type + '_output_' + output_type
                                if output_type != "__nv_fp8_e4m3":
                                    f.write(gen_xqa_decl(func_name, head_dim, input_type,
                                            kv_cache_type, output_type))

                                    if kv_cache_type == "__nv_fp8_e4m3":
                                        func_name += '_spec_dec'
                                        f.write(gen_xqa_decl(func_name, head_dim,
                                                input_type, kv_cache_type, output_type, True))
                                else:
                                    if kv_cache_type == "__nv_fp8_e4m3":
                                        f.write(gen_xqa_decl(func_name, head_dim,
                                                input_type, kv_cache_type, output_type))

                                        func_name += '_spec_dec'
                                        f.write(gen_xqa_decl(func_name, head_dim,
                                                input_type, kv_cache_type, output_type, True))

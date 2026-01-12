#include <torch/all.h>
#include <iostream>
#include <fstream>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_bf16.h>
#include "atrexPA.h"

#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

// helpers to check for hip errors
#define HIP_CHECK(ans)                                                                                                 \
    { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(hipError_t code, const char* file, int line) {
    if (code != hipSuccess) {
        const char* prefix = "Triton Error [HIP]: ";
        const char* str;
        hipDrvGetErrorString(code, &str);
        char err[1024] = {0};
        strcat(err, prefix);
        strcat(err, str);
        printf("%s\n", err);
        exit(code);
    }
}

#define CALL_PA_DECODE_DOT_KERNEL(head_sz, grp_sz, partition_sz, output_ptr)                                           \
    HIP_CHECK(_pa_decode_dot_kernel##head_sz##_##grp_sz##_##partition_sz(stream,                                       \
                                                                         exp_sums_ptr,                                 \
                                                                         max_logits_ptr,                               \
                                                                         output_ptr,                                   \
                                                                         query_ptr,                                    \
                                                                         key_cache_ptr,                                \
                                                                         value_cache_ptr,                              \
                                                                         block_tables_ptr,                             \
                                                                         context_lens_ptr,                             \
                                                                         scale,                                        \
                                                                         alibi_slopes_ptr,                             \
                                                                         exp_sums.stride(0),                           \
                                                                         exp_sums.stride(1),                           \
                                                                         exp_sums.stride(2),                           \
                                                                         tmp_out.stride(0),                            \
                                                                         tmp_out.stride(1),                            \
                                                                         tmp_out.stride(2),                            \
                                                                         tmp_out.stride(3),                            \
                                                                         query.stride(0),                              \
                                                                         query.stride(1),                              \
                                                                         key_cache.stride(0),                          \
                                                                         key_cache.stride(1),                          \
                                                                         key_cache.stride(2),                          \
                                                                         key_cache.stride(3),                          \
                                                                         value_cache.stride(0),                        \
                                                                         value_cache.stride(1),                        \
                                                                         value_cache.stride(2),                        \
                                                                         block_tables.stride(0),                       \
                                                                         grid[0],                                      \
                                                                         grid[1],                                      \
                                                                         grid[2]))

#define CALL_PA_DECODE_REDUCE_KERNEL(head_sz, grp_sz, partition_num)                                                   \
    HIP_CHECK(_pa_decode_reduce_kernel##head_sz##_##grp_sz##_##partition_num(stream,                                   \
                                                                             out_ptr,                                  \
                                                                             exp_sums_ptr,                             \
                                                                             max_logits_ptr,                           \
                                                                             tmp_out_ptr,                              \
                                                                             context_lens_ptr,                         \
                                                                             out.stride(0),                            \
                                                                             out.stride(1),                            \
                                                                             exp_sums.stride(0),                       \
                                                                             exp_sums.stride(1),                       \
                                                                             exp_sums.stride(2),                       \
                                                                             tmp_out.stride(0),                        \
                                                                             tmp_out.stride(1),                        \
                                                                             tmp_out.stride(2),                        \
                                                                             tmp_out.stride(3),                        \
                                                                             grid1[0],                                 \
                                                                             grid1[1],                                 \
                                                                             grid1[2]))

#define DISPATCH_HEAD_GRP_PARTITION(head_sz, grp_sz, partition_sz, output_ptr)                                         \
    do {                                                                                                               \
        if (head_sz == 64 && grp_sz == 2 && partition_sz == 256) {                                                     \
            CALL_PA_DECODE_DOT_KERNEL(64, 2, 256, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 2 && partition_sz == 512) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 2, 512, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 3 && partition_sz == 256) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 3, 256, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 3 && partition_sz == 512) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 3, 512, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 4 && partition_sz == 256) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 4, 256, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 4 && partition_sz == 512) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 4, 512, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 5 && partition_sz == 256) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 5, 256, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 5 && partition_sz == 512) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 5, 512, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 6 && partition_sz == 256) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 6, 256, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 6 && partition_sz == 512) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 6, 512, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 7 && partition_sz == 256) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 7, 256, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 7 && partition_sz == 512) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 7, 512, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 8 && partition_sz == 256) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 8, 256, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 8 && partition_sz == 512) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 8, 512, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 9 && partition_sz == 256) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 9, 256, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 9 && partition_sz == 512) {                                              \
            CALL_PA_DECODE_DOT_KERNEL(64, 9, 512, output_ptr);                                                         \
        } else if (head_sz == 64 && grp_sz == 10 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(64, 10, 256, output_ptr);                                                        \
        } else if (head_sz == 64 && grp_sz == 10 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(64, 10, 512, output_ptr);                                                        \
        } else if (head_sz == 64 && grp_sz == 11 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(64, 11, 256, output_ptr);                                                        \
        } else if (head_sz == 64 && grp_sz == 11 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(64, 11, 512, output_ptr);                                                        \
        } else if (head_sz == 64 && grp_sz == 12 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(64, 12, 256, output_ptr);                                                        \
        } else if (head_sz == 64 && grp_sz == 12 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(64, 12, 512, output_ptr);                                                        \
        } else if (head_sz == 64 && grp_sz == 16 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(64, 16, 256, output_ptr);                                                        \
        } else if (head_sz == 64 && grp_sz == 16 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(64, 16, 512, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 2 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 2, 256, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 2 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 2, 512, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 3 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 3, 256, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 3 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 3, 512, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 4 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 4, 256, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 4 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 4, 512, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 5 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 5, 256, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 5 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 5, 512, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 6 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 6, 256, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 6 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 6, 512, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 7 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 7, 256, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 7 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 7, 512, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 8 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 8, 256, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 8 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 8, 512, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 9 && partition_sz == 256) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 9, 256, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 9 && partition_sz == 512) {                                             \
            CALL_PA_DECODE_DOT_KERNEL(128, 9, 512, output_ptr);                                                        \
        } else if (head_sz == 128 && grp_sz == 10 && partition_sz == 256) {                                            \
            CALL_PA_DECODE_DOT_KERNEL(128, 10, 256, output_ptr);                                                       \
        } else if (head_sz == 128 && grp_sz == 10 && partition_sz == 512) {                                            \
            CALL_PA_DECODE_DOT_KERNEL(128, 10, 512, output_ptr);                                                       \
        } else if (head_sz == 128 && grp_sz == 11 && partition_sz == 256) {                                            \
            CALL_PA_DECODE_DOT_KERNEL(128, 11, 256, output_ptr);                                                       \
        } else if (head_sz == 128 && grp_sz == 11 && partition_sz == 512) {                                            \
            CALL_PA_DECODE_DOT_KERNEL(128, 11, 512, output_ptr);                                                       \
        } else if (head_sz == 128 && grp_sz == 12 && partition_sz == 256) {                                            \
            CALL_PA_DECODE_DOT_KERNEL(128, 12, 256, output_ptr);                                                       \
        } else if (head_sz == 128 && grp_sz == 12 && partition_sz == 512) {                                            \
            CALL_PA_DECODE_DOT_KERNEL(128, 12, 512, output_ptr);                                                       \
        } else if (head_sz == 128 && grp_sz == 16 && partition_sz == 256) {                                            \
            CALL_PA_DECODE_DOT_KERNEL(128, 16, 256, output_ptr);                                                       \
        } else if (head_sz == 128 && grp_sz == 16 && partition_sz == 512) {                                            \
            CALL_PA_DECODE_DOT_KERNEL(128, 16, 512, output_ptr);                                                       \
        } else {                                                                                                       \
            throw std::runtime_error("Unsupported combination: head_size=" + std::to_string(head_sz) + ", grp_size="   \
                                     + std::to_string(grp_sz) + ", partition_size=" + std::to_string(partition_sz));   \
        }                                                                                                              \
    } while (0)

#define DISPATCH_REDUCE_KERNEL(head_sz, grp_sz, partition_num)                                                         \
    do {                                                                                                               \
        if (head_sz == 64 && grp_sz == 2 && partition_num == 2) {                                                      \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 2, 2);                                                                    \
        } else if (head_sz == 64 && grp_sz == 2 && partition_num == 4) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 2, 4);                                                                    \
        } else if (head_sz == 64 && grp_sz == 2 && partition_num == 8) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 2, 8);                                                                    \
        } else if (head_sz == 64 && grp_sz == 2 && partition_num == 16) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 2, 16);                                                                   \
        } else if (head_sz == 64 && grp_sz == 2 && partition_num == 32) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 2, 32);                                                                   \
        } else if (head_sz == 64 && grp_sz == 3 && partition_num == 2) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 3, 2);                                                                    \
        } else if (head_sz == 64 && grp_sz == 3 && partition_num == 4) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 3, 4);                                                                    \
        } else if (head_sz == 64 && grp_sz == 3 && partition_num == 8) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 3, 8);                                                                    \
        } else if (head_sz == 64 && grp_sz == 3 && partition_num == 16) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 3, 16);                                                                   \
        } else if (head_sz == 64 && grp_sz == 3 && partition_num == 32) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 3, 32);                                                                   \
        } else if (head_sz == 64 && grp_sz == 4 && partition_num == 2) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 4, 2);                                                                    \
        } else if (head_sz == 64 && grp_sz == 4 && partition_num == 4) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 4, 4);                                                                    \
        } else if (head_sz == 64 && grp_sz == 4 && partition_num == 8) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 4, 8);                                                                    \
        } else if (head_sz == 64 && grp_sz == 4 && partition_num == 16) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 4, 16);                                                                   \
        } else if (head_sz == 64 && grp_sz == 4 && partition_num == 32) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 4, 32);                                                                   \
        } else if (head_sz == 64 && grp_sz == 5 && partition_num == 2) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 5, 2);                                                                    \
        } else if (head_sz == 64 && grp_sz == 5 && partition_num == 4) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 5, 4);                                                                    \
        } else if (head_sz == 64 && grp_sz == 5 && partition_num == 8) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 5, 8);                                                                    \
        } else if (head_sz == 64 && grp_sz == 5 && partition_num == 16) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 5, 16);                                                                   \
        } else if (head_sz == 64 && grp_sz == 5 && partition_num == 32) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 5, 32);                                                                   \
        } else if (head_sz == 64 && grp_sz == 6 && partition_num == 2) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 6, 2);                                                                    \
        } else if (head_sz == 64 && grp_sz == 6 && partition_num == 4) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 6, 4);                                                                    \
        } else if (head_sz == 64 && grp_sz == 6 && partition_num == 8) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 6, 8);                                                                    \
        } else if (head_sz == 64 && grp_sz == 6 && partition_num == 16) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 6, 16);                                                                   \
        } else if (head_sz == 64 && grp_sz == 6 && partition_num == 32) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 6, 32);                                                                   \
        } else if (head_sz == 64 && grp_sz == 7 && partition_num == 2) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 7, 2);                                                                    \
        } else if (head_sz == 64 && grp_sz == 7 && partition_num == 4) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 7, 4);                                                                    \
        } else if (head_sz == 64 && grp_sz == 7 && partition_num == 8) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 7, 8);                                                                    \
        } else if (head_sz == 64 && grp_sz == 7 && partition_num == 16) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 7, 16);                                                                   \
        } else if (head_sz == 64 && grp_sz == 7 && partition_num == 32) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 7, 32);                                                                   \
        } else if (head_sz == 64 && grp_sz == 8 && partition_num == 2) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 8, 2);                                                                    \
        } else if (head_sz == 64 && grp_sz == 8 && partition_num == 4) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 8, 4);                                                                    \
        } else if (head_sz == 64 && grp_sz == 8 && partition_num == 8) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 8, 8);                                                                    \
        } else if (head_sz == 64 && grp_sz == 8 && partition_num == 16) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 8, 16);                                                                   \
        } else if (head_sz == 64 && grp_sz == 8 && partition_num == 32) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 8, 32);                                                                   \
        } else if (head_sz == 64 && grp_sz == 9 && partition_num == 2) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 9, 2);                                                                    \
        } else if (head_sz == 64 && grp_sz == 9 && partition_num == 4) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 9, 4);                                                                    \
        } else if (head_sz == 64 && grp_sz == 9 && partition_num == 8) {                                               \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 9, 8);                                                                    \
        } else if (head_sz == 64 && grp_sz == 9 && partition_num == 16) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 9, 16);                                                                   \
        } else if (head_sz == 64 && grp_sz == 9 && partition_num == 32) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 9, 32);                                                                   \
        } else if (head_sz == 64 && grp_sz == 10 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 10, 2);                                                                   \
        } else if (head_sz == 64 && grp_sz == 10 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 10, 4);                                                                   \
        } else if (head_sz == 64 && grp_sz == 10 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 10, 8);                                                                   \
        } else if (head_sz == 64 && grp_sz == 10 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 10, 16);                                                                  \
        } else if (head_sz == 64 && grp_sz == 10 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 10, 32);                                                                  \
        } else if (head_sz == 64 && grp_sz == 11 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 11, 2);                                                                   \
        } else if (head_sz == 64 && grp_sz == 11 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 11, 4);                                                                   \
        } else if (head_sz == 64 && grp_sz == 11 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 11, 8);                                                                   \
        } else if (head_sz == 64 && grp_sz == 11 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 11, 16);                                                                  \
        } else if (head_sz == 64 && grp_sz == 11 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 11, 32);                                                                  \
        } else if (head_sz == 64 && grp_sz == 12 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 12, 2);                                                                   \
        } else if (head_sz == 64 && grp_sz == 12 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 12, 4);                                                                   \
        } else if (head_sz == 64 && grp_sz == 12 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 12, 8);                                                                   \
        } else if (head_sz == 64 && grp_sz == 12 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 12, 16);                                                                  \
        } else if (head_sz == 64 && grp_sz == 12 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 12, 32);                                                                  \
        } else if (head_sz == 64 && grp_sz == 16 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 16, 2);                                                                   \
        } else if (head_sz == 64 && grp_sz == 16 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 16, 4);                                                                   \
        } else if (head_sz == 64 && grp_sz == 16 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 16, 8);                                                                   \
        } else if (head_sz == 64 && grp_sz == 16 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 16, 16);                                                                  \
        } else if (head_sz == 64 && grp_sz == 16 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(64, 16, 32);                                                                  \
        } else if (head_sz == 128 && grp_sz == 2 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 2, 2);                                                                   \
        } else if (head_sz == 128 && grp_sz == 2 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 2, 4);                                                                   \
        } else if (head_sz == 128 && grp_sz == 2 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 2, 8);                                                                   \
        } else if (head_sz == 128 && grp_sz == 2 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 2, 16);                                                                  \
        } else if (head_sz == 128 && grp_sz == 2 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 2, 32);                                                                  \
        } else if (head_sz == 128 && grp_sz == 3 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 3, 2);                                                                   \
        } else if (head_sz == 128 && grp_sz == 3 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 3, 4);                                                                   \
        } else if (head_sz == 128 && grp_sz == 3 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 3, 8);                                                                   \
        } else if (head_sz == 128 && grp_sz == 3 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 3, 16);                                                                  \
        } else if (head_sz == 128 && grp_sz == 3 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 3, 32);                                                                  \
        } else if (head_sz == 128 && grp_sz == 4 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 4, 2);                                                                   \
        } else if (head_sz == 128 && grp_sz == 4 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 4, 4);                                                                   \
        } else if (head_sz == 128 && grp_sz == 4 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 4, 8);                                                                   \
        } else if (head_sz == 128 && grp_sz == 4 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 4, 16);                                                                  \
        } else if (head_sz == 128 && grp_sz == 4 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 4, 32);                                                                  \
        } else if (head_sz == 128 && grp_sz == 5 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 5, 2);                                                                   \
        } else if (head_sz == 128 && grp_sz == 5 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 5, 4);                                                                   \
        } else if (head_sz == 128 && grp_sz == 5 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 5, 8);                                                                   \
        } else if (head_sz == 128 && grp_sz == 5 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 5, 16);                                                                  \
        } else if (head_sz == 128 && grp_sz == 5 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 5, 32);                                                                  \
        } else if (head_sz == 128 && grp_sz == 6 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 6, 2);                                                                   \
        } else if (head_sz == 128 && grp_sz == 6 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 6, 4);                                                                   \
        } else if (head_sz == 128 && grp_sz == 6 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 6, 8);                                                                   \
        } else if (head_sz == 128 && grp_sz == 6 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 6, 16);                                                                  \
        } else if (head_sz == 128 && grp_sz == 6 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 6, 32);                                                                  \
        } else if (head_sz == 128 && grp_sz == 7 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 7, 2);                                                                   \
        } else if (head_sz == 128 && grp_sz == 7 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 7, 4);                                                                   \
        } else if (head_sz == 128 && grp_sz == 7 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 7, 8);                                                                   \
        } else if (head_sz == 128 && grp_sz == 7 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 7, 16);                                                                  \
        } else if (head_sz == 128 && grp_sz == 7 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 7, 32);                                                                  \
        } else if (head_sz == 128 && grp_sz == 8 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 8, 2);                                                                   \
        } else if (head_sz == 128 && grp_sz == 8 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 8, 4);                                                                   \
        } else if (head_sz == 128 && grp_sz == 8 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 8, 8);                                                                   \
        } else if (head_sz == 128 && grp_sz == 8 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 8, 16);                                                                  \
        } else if (head_sz == 128 && grp_sz == 8 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 8, 32);                                                                  \
        } else if (head_sz == 128 && grp_sz == 9 && partition_num == 2) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 9, 2);                                                                   \
        } else if (head_sz == 128 && grp_sz == 9 && partition_num == 4) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 9, 4);                                                                   \
        } else if (head_sz == 128 && grp_sz == 9 && partition_num == 8) {                                              \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 9, 8);                                                                   \
        } else if (head_sz == 128 && grp_sz == 9 && partition_num == 16) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 9, 16);                                                                  \
        } else if (head_sz == 128 && grp_sz == 9 && partition_num == 32) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 9, 32);                                                                  \
        } else if (head_sz == 128 && grp_sz == 10 && partition_num == 2) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 10, 2);                                                                  \
        } else if (head_sz == 128 && grp_sz == 10 && partition_num == 4) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 10, 4);                                                                  \
        } else if (head_sz == 128 && grp_sz == 10 && partition_num == 8) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 10, 8);                                                                  \
        } else if (head_sz == 128 && grp_sz == 10 && partition_num == 16) {                                            \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 10, 16);                                                                 \
        } else if (head_sz == 128 && grp_sz == 10 && partition_num == 32) {                                            \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 10, 32);                                                                 \
        } else if (head_sz == 128 && grp_sz == 11 && partition_num == 2) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 11, 2);                                                                  \
        } else if (head_sz == 128 && grp_sz == 11 && partition_num == 4) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 11, 4);                                                                  \
        } else if (head_sz == 128 && grp_sz == 11 && partition_num == 8) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 11, 8);                                                                  \
        } else if (head_sz == 128 && grp_sz == 11 && partition_num == 16) {                                            \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 11, 16);                                                                 \
        } else if (head_sz == 128 && grp_sz == 11 && partition_num == 32) {                                            \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 11, 32);                                                                 \
        } else if (head_sz == 128 && grp_sz == 12 && partition_num == 2) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 12, 2);                                                                  \
        } else if (head_sz == 128 && grp_sz == 12 && partition_num == 4) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 12, 4);                                                                  \
        } else if (head_sz == 128 && grp_sz == 12 && partition_num == 8) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 12, 8);                                                                  \
        } else if (head_sz == 128 && grp_sz == 12 && partition_num == 16) {                                            \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 12, 16);                                                                 \
        } else if (head_sz == 128 && grp_sz == 12 && partition_num == 32) {                                            \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 12, 32);                                                                 \
        } else if (head_sz == 128 && grp_sz == 16 && partition_num == 2) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 16, 2);                                                                  \
        } else if (head_sz == 128 && grp_sz == 16 && partition_num == 4) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 16, 4);                                                                  \
        } else if (head_sz == 128 && grp_sz == 16 && partition_num == 8) {                                             \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 16, 8);                                                                  \
        } else if (head_sz == 128 && grp_sz == 16 && partition_num == 16) {                                            \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 16, 16);                                                                 \
        } else if (head_sz == 128 && grp_sz == 16 && partition_num == 32) {                                            \
            CALL_PA_DECODE_REDUCE_KERNEL(128, 16, 32);                                                                 \
        } else {                                                                                                       \
            throw std::runtime_error("Unsupported reduce kernel combination: head_size=" + std::to_string(head_sz)     \
                                     + ", grp_size=" + std::to_string(grp_sz)                                          \
                                     + ", partition_num=" + std::to_string(partition_num));                            \
        }                                                                                                              \
    } while (0)

static inline uint64_t next_power_of_2(uint64_t n) {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

void paged_attention_atrex(torch::Tensor&                      out,
                           torch::Tensor&                      exp_sums,
                           torch::Tensor&                      max_logits,
                           torch::Tensor&                      tmp_out,
                           torch::Tensor&                      query,
                           torch::Tensor&                      key_cache,
                           torch::Tensor&                      value_cache,
                           torch::Tensor&                      context_lens,
                           torch::Tensor&                      block_tables,
                           float                               scale,
                           int64_t                             max_context_len,
                           const std::optional<torch::Tensor>& alibi_slopes) {
    int num_kv_heads = key_cache.size(1);
    int num_seqs     = query.size(0);
    int num_q_heads  = query.size(1);
    int kv_blk_sz    = value_cache.size(3);
    int head_sz      = query.size(2);
    int query_grp_sz = num_q_heads / num_kv_heads;

    // NOTE: alibi_slopes is optional.
    hipDeviceptr_t alibi_slopes_ptr =
        alibi_slopes ? reinterpret_cast<hipDeviceptr_t>(alibi_slopes.value().data_ptr()) : nullptr;
    float*          exp_sums_ptr     = reinterpret_cast<float*>(exp_sums.data_ptr());
    float*          max_logits_ptr   = reinterpret_cast<float*>(max_logits.data_ptr());
    __hip_bfloat16* tmp_out_ptr      = reinterpret_cast<__hip_bfloat16*>(tmp_out.data_ptr());
    __hip_bfloat16* query_ptr        = reinterpret_cast<__hip_bfloat16*>(query.data_ptr());
    __hip_bfloat16* key_cache_ptr    = reinterpret_cast<__hip_bfloat16*>(key_cache.data_ptr());
    __hip_bfloat16* value_cache_ptr  = reinterpret_cast<__hip_bfloat16*>(value_cache.data_ptr());
    int*            block_tables_ptr = block_tables.data_ptr<int>();
    int*            context_lens_ptr = context_lens.data_ptr<int>();
    __hip_bfloat16* out_ptr          = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

    const hipStream_t stream = at::hip::getCurrentHIPStream().stream();

    if (max_context_len <= 256) {
        std::vector<int32_t> grid = {num_seqs, num_kv_heads, 1};
        DISPATCH_HEAD_GRP_PARTITION(head_sz, query_grp_sz, 256, out_ptr);
    } else if (max_context_len <= 512) {
        std::vector<int32_t> grid = {num_seqs, num_kv_heads, 1};
        DISPATCH_HEAD_GRP_PARTITION(head_sz, query_grp_sz, 512, out_ptr);
    } else {
        constexpr int        _SEQ_PARTITION_SIZE = 512;
        int                  max_num_partitions  = (max_context_len + _SEQ_PARTITION_SIZE - 1) / _SEQ_PARTITION_SIZE;
        std::vector<int32_t> grid                = {num_seqs, num_kv_heads, max_num_partitions};

        DISPATCH_HEAD_GRP_PARTITION(head_sz, query_grp_sz, 512, tmp_out_ptr);

        std::vector<int32_t> grid1                    = {num_seqs, num_kv_heads, 1};
        const auto           max_num_partitions_pow_2 = next_power_of_2(max_num_partitions);

        DISPATCH_REDUCE_KERNEL(head_sz, query_grp_sz, max_num_partitions_pow_2);
    }
}

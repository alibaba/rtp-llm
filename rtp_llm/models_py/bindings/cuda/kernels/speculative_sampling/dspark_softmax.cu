// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// Adapted cached-cluster launch; see ../blackwell_softmax_vendor/README.md.
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/dspark_softmax_policy.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/flashinfer/sampling.cuh"

#if RTP_LLM_DSPARK_BLACKWELL_SOFTMAX
extern "C" {
#define DECLARE_CACHED(C, T) \
    __global__ void kernel_flashinfer_blackwell_softmax_cached_c##C##_t##T( \
        float*, float*, float*, int, int, int, int, float);
DECLARE_CACHED(1, 4) DECLARE_CACHED(1, 8)
DECLARE_CACHED(2, 4) DECLARE_CACHED(2, 8)
DECLARE_CACHED(4, 4) DECLARE_CACHED(4, 8)
DECLARE_CACHED(8, 4) DECLARE_CACHED(8, 8)
DECLARE_CACHED(16, 4) DECLARE_CACHED(16, 8)
#undef DECLARE_CACHED
}
#endif

namespace rtp_llm {

size_t dsparkSoftmaxWorkspaceBytes(const float* logits, const float* output, int64_t rows, int64_t vocab,
                                  int major, int minor) {
#if RTP_LLM_DSPARK_BLACKWELL_SOFTMAX
    if (dspark_softmax::useCached(rows, vocab, major, minor, logits, output)) return 0;
#endif
    // FlashInfer OnlineSoftmax's split route uses one {max,sum} per slice.
    return rows > 0 && rows <= 128 && vocab >= 24576 ? rows * ((vocab + 8191) / 8192)
               * sizeof(flashinfer::sampling::PartialSoftmaxResult) : 0;
}

cudaError_t invokeDSparkSoftmax(float* logits, float* output, int64_t rows, int64_t vocab,
                               int major, int minor, int sm_count, void* workspace, size_t workspace_bytes,
                               cudaStream_t stream) {
    if (rows == 0 || vocab == 0) return cudaSuccess;
#if RTP_LLM_DSPARK_BLACKWELL_SOFTMAX
    if (dspark_softmax::useCached(rows, vocab, major, minor, logits, output)) {
        const auto variant = dspark_softmax::selectVariant(rows, vocab, sm_count);
        const void* kernel = nullptr;
#define SELECT_CACHED(C, T) \
        if (variant.cluster_ctas == C && variant.max_tiles == T) \
            kernel = reinterpret_cast<const void*>(kernel_flashinfer_blackwell_softmax_cached_c##C##_t##T);
        SELECT_CACHED(1, 4) SELECT_CACHED(1, 8)
        SELECT_CACHED(2, 4) SELECT_CACHED(2, 8)
        SELECT_CACHED(4, 4) SELECT_CACHED(4, 8)
        SELECT_CACHED(8, 4) SELECT_CACHED(8, 8)
        SELECT_CACHED(16, 4) SELECT_CACHED(16, 8)
#undef SELECT_CACHED
        if (variant.cluster_ctas > 8) {
            auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            if (status != cudaSuccess) return status;
        }
        int rows_i = static_cast<int>(rows), vocab_i = static_cast<int>(vocab);
        int chunk = dspark_softmax::chunkElements(vocab, variant.cluster_ctas);
        int parameter_kind = 0;  // Already combined/scaled by add.rn / div.rn.
        float temperature = 1.0f;
        float* parameter = logits;  // Upstream ABI slot; kind=0 never reads it.
        void* args[] = {&logits, &parameter, &output, &rows_i, &vocab_i, &chunk, &parameter_kind, &temperature};
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(rows_i * variant.cluster_ctas);
        config.blockDim = dim3(256);
        config.dynamicSmemBytes = 128;
        config.stream = stream;
        cudaLaunchAttribute attrs[2]{};
        if (variant.cluster_ctas > 1) {
            attrs[0].id = cudaLaunchAttributeClusterDimension;
            attrs[0].val.clusterDim = {static_cast<unsigned int>(variant.cluster_ctas), 1, 1};
            attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
            attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
            config.attrs = attrs;
            config.numAttrs = 2;
        }
        // PDL stays disabled, matching the tested DSpark proposal dependency chain.
        return cudaLaunchKernelExC(&config, kernel, args);
    }
#endif
    return flashinfer::sampling::OnlineSoftmax<float>(logits, output, rows, vocab, nullptr, 1.0f,
                                                      workspace, workspace_bytes, false, stream);
}
}  // namespace rtp_llm
#endif

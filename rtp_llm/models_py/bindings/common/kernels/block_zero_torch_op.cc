#include "rtp_llm/models_py/bindings/common/kernels/block_zero_torch_op.h"

#if USING_CUDA || USING_ROCM
#include "rtp_llm/models_py/bindings/common/kernels/block_zero_kernels.h"
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include <hip/hip_runtime.h>
#endif
#endif

namespace rtp_llm {

void zero_incomplete_kv_cache_blocks(const torch::Tensor&                layer_base_addrs,
                                     const torch::Tensor&                kv_cache_block_id,
                                     const torch::Tensor&                token_counts,
                                     const std::optional<torch::Tensor>& layer_to_group,
                                     int64_t                             block_stride_bytes,
                                     int64_t                             seq_size_per_block) {
#if USING_CUDA || USING_ROCM
    const int64_t batch_size = token_counts.size(0);
    const int64_t layer_num  = layer_base_addrs.size(0);
    if (batch_size == 0 || layer_num == 0 || block_stride_bytes == 0)
        return;

    auto block_id_sizes = kv_cache_block_id.sizes();
    TORCH_CHECK(block_id_sizes.size() == 3, "kv_cache_block_id must be 3D [G,B,M]");
    const int64_t batch_dim            = block_id_sizes[1];
    const int64_t max_blocks_per_batch = block_id_sizes[2];

#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
#elif USING_ROCM
    cudaStream_t stream = at::hip::getCurrentHIPStream().stream();
#endif

    invokeZeroIncompleteKvCacheBlocks(
        reinterpret_cast<const void* const*>(layer_base_addrs.data_ptr<int64_t>()),
        kv_cache_block_id.data_ptr<int32_t>(),
        token_counts.data_ptr<int32_t>(),
        (layer_to_group.has_value() && layer_to_group->defined() && layer_to_group->numel() > 0)
            ? layer_to_group->data_ptr<int32_t>()
            : nullptr,
        static_cast<size_t>(batch_size),
        static_cast<size_t>(layer_num),
        static_cast<size_t>(batch_dim),
        static_cast<size_t>(max_blocks_per_batch),
        static_cast<size_t>(block_stride_bytes),
        static_cast<size_t>(seq_size_per_block),
        stream);
#endif
}

}  // namespace rtp_llm

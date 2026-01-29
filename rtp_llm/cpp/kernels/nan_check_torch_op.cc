#include "rtp_llm/cpp/kernels/nan_check_torch_op.h"
#include "rtp_llm/cpp/core/Types.h"

#if USING_CUDA || USING_ROCM
#include "rtp_llm/cpp/kernels/nan_check_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
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

namespace {

#if USING_CUDA || USING_ROCM
template<typename T>
void run_decode(const torch::Tensor&                layer_base_addrs,
                const torch::Tensor&                kv_cache_block_id,
                const torch::Tensor&                sequence_lengths,
                torch::Tensor                       nan_flag,
                int64_t                             batch_size,
                int64_t                             layer_num,
                int64_t                             num_groups,
                const std::optional<torch::Tensor>& layer_to_group,
                const std::optional<torch::Tensor>& group_types,
                int64_t                             batch_dim,
                int64_t                             batch_start,
                int64_t                             max_blocks_per_batch,
                int64_t                             local_head_num_kv,
                int64_t                             k_token_size,
                int64_t                             v_token_size,
                int64_t                             k_block_size_bytes,
                int64_t                             v_block_size_bytes,
                int64_t                             k_token_bytes,
                int64_t                             v_token_bytes,
                int64_t                             block_size_bytes,
                int64_t                             seq_size_per_block,
                void*                               stream) {
    cudaStream_t run_stream;
    if (stream != nullptr) {
        run_stream = reinterpret_cast<cudaStream_t>(stream);
    } else {
#if USING_CUDA
        run_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
#elif USING_ROCM
        run_stream = at::hip::getCurrentHIPStream().stream();
#endif
    }
    const void* const* layer_base_addr = reinterpret_cast<const void* const*>(layer_base_addrs.data_ptr<int64_t>());
    const int32_t*     block_id_ptr    = kv_cache_block_id.data_ptr<int32_t>();
    const int32_t*     seq_lengths_ptr = sequence_lengths.data_ptr<int32_t>();
    float*             nan_flag_ptr    = nan_flag.data_ptr<float>();
    const int32_t*     layer_to_group_ptr =
        layer_to_group.has_value() && layer_to_group->defined() && layer_to_group->numel() > 0 ?
                layer_to_group->data_ptr<int32_t>() :
                nullptr;
    const int32_t* group_types_ptr = group_types.has_value() && group_types->defined() && group_types->numel() > 0 ?
                                         group_types->data_ptr<int32_t>() :
                                         nullptr;

    invokeCheckAndResetNANKvCacheDecode<T>(layer_base_addr,
                                           block_id_ptr,
                                           seq_lengths_ptr,
                                           static_cast<size_t>(batch_size),
                                           static_cast<size_t>(layer_num),
                                           static_cast<size_t>(num_groups),
                                           layer_to_group_ptr,
                                           group_types_ptr,
                                           static_cast<size_t>(batch_dim),
                                           static_cast<size_t>(batch_start),
                                           static_cast<size_t>(max_blocks_per_batch),
                                           static_cast<size_t>(local_head_num_kv),
                                           static_cast<size_t>(k_token_size),
                                           static_cast<size_t>(v_token_size),
                                           static_cast<size_t>(k_block_size_bytes),
                                           static_cast<size_t>(v_block_size_bytes),
                                           static_cast<size_t>(k_token_bytes),
                                           static_cast<size_t>(v_token_bytes),
                                           static_cast<size_t>(block_size_bytes),
                                           static_cast<size_t>(seq_size_per_block),
                                           nan_flag_ptr,
                                           run_stream);
}

template<typename T>
void run_prefill(const torch::Tensor&                layer_base_addrs,
                 const torch::Tensor&                kv_cache_block_id,
                 const torch::Tensor&                prefix_lengths,
                 const torch::Tensor&                input_lengths,
                 torch::Tensor                       nan_flag,
                 int64_t                             batch_size,
                 int64_t                             layer_num,
                 int64_t                             num_groups,
                 const std::optional<torch::Tensor>& layer_to_group,
                 const std::optional<torch::Tensor>& group_types,
                 int64_t                             batch_dim,
                 int64_t                             batch_start,
                 int64_t                             max_blocks_per_batch,
                 int64_t                             local_head_num_kv,
                 int64_t                             k_token_size,
                 int64_t                             v_token_size,
                 int64_t                             k_block_size_bytes,
                 int64_t                             v_block_size_bytes,
                 int64_t                             k_token_bytes,
                 int64_t                             v_token_bytes,
                 int64_t                             block_size_bytes,
                 int64_t                             seq_size_per_block,
                 void*                               stream) {
    cudaStream_t run_stream;
    if (stream != nullptr) {
        run_stream = reinterpret_cast<cudaStream_t>(stream);
    } else {
#if USING_CUDA
        run_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
#elif USING_ROCM
        run_stream = at::hip::getCurrentHIPStream().stream();
#endif
    }
    const void* const* layer_base_addr    = reinterpret_cast<const void* const*>(layer_base_addrs.data_ptr<int64_t>());
    const int32_t*     block_id_ptr       = kv_cache_block_id.data_ptr<int32_t>();
    const int32_t*     prefix_lengths_ptr = prefix_lengths.data_ptr<int32_t>();
    const int32_t*     input_lengths_ptr  = input_lengths.data_ptr<int32_t>();
    float*             nan_flag_ptr       = nan_flag.data_ptr<float>();
    const int32_t*     layer_to_group_ptr =
        layer_to_group.has_value() && layer_to_group->defined() && layer_to_group->numel() > 0 ?
                layer_to_group->data_ptr<int32_t>() :
                nullptr;
    const int32_t* group_types_ptr = group_types.has_value() && group_types->defined() && group_types->numel() > 0 ?
                                         group_types->data_ptr<int32_t>() :
                                         nullptr;

    invokeCheckAndResetNANKvCachePrefill<T>(layer_base_addr,
                                            block_id_ptr,
                                            prefix_lengths_ptr,
                                            input_lengths_ptr,
                                            static_cast<size_t>(batch_size),
                                            static_cast<size_t>(layer_num),
                                            static_cast<size_t>(num_groups),
                                            layer_to_group_ptr,
                                            group_types_ptr,
                                            static_cast<size_t>(batch_dim),
                                            static_cast<size_t>(batch_start),
                                            static_cast<size_t>(max_blocks_per_batch),
                                            static_cast<size_t>(local_head_num_kv),
                                            static_cast<size_t>(k_token_size),
                                            static_cast<size_t>(v_token_size),
                                            static_cast<size_t>(k_block_size_bytes),
                                            static_cast<size_t>(v_block_size_bytes),
                                            static_cast<size_t>(k_token_bytes),
                                            static_cast<size_t>(v_token_bytes),
                                            static_cast<size_t>(block_size_bytes),
                                            static_cast<size_t>(seq_size_per_block),
                                            nan_flag_ptr,
                                            run_stream);
}
#endif  // USING_CUDA || USING_ROCM

}  // namespace

#if USING_CUDA || USING_ROCM
#if defined(ENABLE_FP8) || defined(USING_ROCM)
#define ENABLE_FP8_CASE_NUMERIC_NAN_CHECK_LOCAL(MACRO, ...) MACRO(DataType::TYPE_FP8_E4M3, __nv_fp8_e4m3, __VA_ARGS__)
#else
#define ENABLE_FP8_CASE_NUMERIC_NAN_CHECK_LOCAL(MACRO, ...)
#endif

#define DISPATCH_FOR_KV_CACHE_TYPE_NAN_CHECK_LOCAL(MACRO, ...)                                                         \
    ENABLE_FP8_CASE_NUMERIC_NAN_CHECK_LOCAL(MACRO, __VA_ARGS__)                                                        \
    DISPATCH_FOR_EACH_COMPUTE_TYPE(MACRO, __VA_ARGS__)

#define DISPATCH_CUDA_FUNCTION_KV_CACHE_TYPE_NAN_CHECK_LOCAL(data_type, function, ...)                                 \
    do {                                                                                                               \
        switch (data_type) {                                                                                           \
            DISPATCH_FOR_KV_CACHE_TYPE_NAN_CHECK_LOCAL(DP_FUNCTION_CALL_CASE_DIRECT, function, __VA_ARGS__)            \
        }                                                                                                              \
    } while (0)
#endif

void check_and_reset_nan_kv_cache_decode(const torch::Tensor&                layer_base_addrs,
                                         const torch::Tensor&                kv_cache_block_id,
                                         const torch::Tensor&                sequence_lengths,
                                         torch::Tensor                       nan_flag,
                                         int64_t                             cache_dtype,
                                         int64_t                             batch_size,
                                         int64_t                             layer_num,
                                         int64_t                             num_groups,
                                         const std::optional<torch::Tensor>& layer_to_group,
                                         const std::optional<torch::Tensor>& group_types,
                                         int64_t                             batch_dim,
                                         int64_t                             batch_start,
                                         int64_t                             max_blocks_per_batch,
                                         int64_t                             local_head_num_kv,
                                         int64_t                             k_token_size,
                                         int64_t                             v_token_size,
                                         int64_t                             k_block_size_bytes,
                                         int64_t                             v_block_size_bytes,
                                         int64_t                             k_token_bytes,
                                         int64_t                             v_token_bytes,
                                         int64_t                             block_size_bytes,
                                         int64_t                             seq_size_per_block) {
#if USING_CUDA || USING_ROCM
    DataType dtype = static_cast<DataType>(cache_dtype);
    DISPATCH_CUDA_FUNCTION_KV_CACHE_TYPE_NAN_CHECK_LOCAL(dtype,
                                                         run_decode,
                                                         layer_base_addrs,
                                                         kv_cache_block_id,
                                                         sequence_lengths,
                                                         nan_flag,
                                                         batch_size,
                                                         layer_num,
                                                         num_groups,
                                                         layer_to_group,
                                                         group_types,
                                                         batch_dim,
                                                         batch_start,
                                                         max_blocks_per_batch,
                                                         local_head_num_kv,
                                                         k_token_size,
                                                         v_token_size,
                                                         k_block_size_bytes,
                                                         v_block_size_bytes,
                                                         k_token_bytes,
                                                         v_token_bytes,
                                                         block_size_bytes,
                                                         seq_size_per_block,
                                                         nullptr);
#else
    (void)layer_base_addrs;
    (void)kv_cache_block_id;
    (void)sequence_lengths;
    (void)nan_flag;
    (void)cache_dtype;
    (void)batch_size;
    (void)layer_num;
    (void)num_groups;
    (void)layer_to_group;
    (void)group_types;
    (void)batch_dim;
    (void)batch_start;
    (void)max_blocks_per_batch;
    (void)local_head_num_kv;
    (void)k_token_size;
    (void)v_token_size;
    (void)k_block_size_bytes;
    (void)v_block_size_bytes;
    (void)k_token_bytes;
    (void)v_token_bytes;
    (void)block_size_bytes;
    (void)seq_size_per_block;
#endif
}

void check_and_reset_nan_kv_cache_prefill(const torch::Tensor&                layer_base_addrs,
                                          const torch::Tensor&                kv_cache_block_id,
                                          const torch::Tensor&                prefix_lengths,
                                          const torch::Tensor&                input_lengths,
                                          torch::Tensor                       nan_flag,
                                          int64_t                             cache_dtype,
                                          int64_t                             batch_size,
                                          int64_t                             layer_num,
                                          int64_t                             num_groups,
                                          const std::optional<torch::Tensor>& layer_to_group,
                                          const std::optional<torch::Tensor>& group_types,
                                          int64_t                             batch_dim,
                                          int64_t                             batch_start,
                                          int64_t                             max_blocks_per_batch,
                                          int64_t                             local_head_num_kv,
                                          int64_t                             k_token_size,
                                          int64_t                             v_token_size,
                                          int64_t                             k_block_size_bytes,
                                          int64_t                             v_block_size_bytes,
                                          int64_t                             k_token_bytes,
                                          int64_t                             v_token_bytes,
                                          int64_t                             block_size_bytes,
                                          int64_t                             seq_size_per_block) {
#if USING_CUDA || USING_ROCM
    DataType dtype = static_cast<DataType>(cache_dtype);
    DISPATCH_CUDA_FUNCTION_KV_CACHE_TYPE_NAN_CHECK_LOCAL(dtype,
                                                         run_prefill,
                                                         layer_base_addrs,
                                                         kv_cache_block_id,
                                                         prefix_lengths,
                                                         input_lengths,
                                                         nan_flag,
                                                         batch_size,
                                                         layer_num,
                                                         num_groups,
                                                         layer_to_group,
                                                         group_types,
                                                         batch_dim,
                                                         batch_start,
                                                         max_blocks_per_batch,
                                                         local_head_num_kv,
                                                         k_token_size,
                                                         v_token_size,
                                                         k_block_size_bytes,
                                                         v_block_size_bytes,
                                                         k_token_bytes,
                                                         v_token_bytes,
                                                         block_size_bytes,
                                                         seq_size_per_block,
                                                         nullptr);
#else
    (void)layer_base_addrs;
    (void)kv_cache_block_id;
    (void)prefix_lengths;
    (void)input_lengths;
    (void)nan_flag;
    (void)cache_dtype;
    (void)batch_size;
    (void)layer_num;
    (void)num_groups;
    (void)layer_to_group;
    (void)group_types;
    (void)batch_dim;
    (void)batch_start;
    (void)max_blocks_per_batch;
    (void)local_head_num_kv;
    (void)k_token_size;
    (void)v_token_size;
    (void)k_block_size_bytes;
    (void)v_block_size_bytes;
    (void)k_token_bytes;
    (void)v_token_bytes;
    (void)block_size_bytes;
    (void)seq_size_per_block;
#endif
}

}  // namespace rtp_llm

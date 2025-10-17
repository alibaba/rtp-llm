#include "rtp_llm/cpp/kernels/moe/ep_utils.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/moe_gemm/moe_kernels.inl"

namespace trt = tensorrt_llm::kernels;
namespace rtp_llm {

#define MOE_SWITCH(TYPE, ...)                                                                                          \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                                                                  \
    switch (_st) {                                                                                                     \
        __VA_ARGS__                                                                                                    \
        default:                                                                                                       \
            TORCH_CHECK(false, "[moe permute]data type dispatch fail!")                                                \
    }

#define MOE_DISPATCH_CASE(enum_type, ...)                                                                              \
    case enum_type: {                                                                                                  \
        using scalar_t = ScalarType2CudaType<enum_type>::type;                                                         \
        __VA_ARGS__();                                                                                                 \
        break;                                                                                                         \
    }
#define MOE_DISPATCH_FLOAT_CASE(...)                                                                                   \
    MOE_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                                                              \
    MOE_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                                                               \
    MOE_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)                                                           \
    MOE_DISPATCH_CASE(at::ScalarType::Float8_e5m2, __VA_ARGS__)                                                        \
    MOE_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__)                                                      \
    MOE_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)

#define MOE_DISPATCH(TYPE, ...) MOE_SWITCH(TYPE, MOE_DISPATCH_FLOAT_CASE(__VA_ARGS__))

template<at::ScalarType type>
struct ScalarType2CudaType;

template<>
struct ScalarType2CudaType<at::ScalarType::Float> {
    using type = float;
};
template<>
struct ScalarType2CudaType<at::ScalarType::Half> {
    using type = half;
};
template<>
struct ScalarType2CudaType<at::ScalarType::BFloat16> {
    using type = __nv_bfloat16;
};

template<>
struct ScalarType2CudaType<at::ScalarType::Byte> {
    using type = uint8_t;
};

template<>
struct ScalarType2CudaType<at::ScalarType::Float8_e5m2> {
    using type = __nv_fp8_e5m2;
};
template<>
struct ScalarType2CudaType<at::ScalarType::Float8_e4m3fn> {
    using type = __nv_fp8_e4m3;
};

// handle expert map kernel
__global__ void preprocessTopkIdKernel(int* topk_id_ptr, int size, const int* expert_map_ptr, int num_experts) {
    auto tidx   = threadIdx.x;
    auto bidx   = blockIdx.x;
    auto offset = bidx * blockDim.x;
    auto bound  = min(offset + blockDim.x, size);

    extern __shared__ int smem_expert_map[];
    for (int i = tidx; i < num_experts; i += blockDim.x) {
        smem_expert_map[i] = expert_map_ptr[i];
    }
    __syncthreads();

    // if global expert id = -1 in exert map, plus num_experts
    if (offset + tidx < bound) {
        auto topk_id          = topk_id_ptr[offset + tidx];
        auto local_expert_idx = smem_expert_map[topk_id];
        if (local_expert_idx == -1) {
            topk_id += num_experts;
        } else {
            topk_id = local_expert_idx;
        }
        __syncwarp();
        topk_id_ptr[offset + tidx] = topk_id;
    }
}
void preprocessTopkIdLauncher(
    int* topk_id_ptr, int size, const int* expert_map_ptr, int num_experts, cudaStream_t stream) {
    int block     = std::min(size, 1024);
    int grid      = (size + block - 1) / block;
    int smem_size = (num_experts) * sizeof(int);
    preprocessTopkIdKernel<<<grid, block, smem_size, stream>>>(topk_id_ptr, size, expert_map_ptr, num_experts);
}
template<typename T, bool CHECK_SKIPPED>
__global__ void expandInputRowsWithoutScaleKernel(T const*       unpermuted_input,
                                                  T*             permuted_output,
                                                  int*           sorted_experts,
                                                  int const*     expanded_dest_row_to_expanded_source_row,
                                                  int*           expanded_source_row_to_expanded_dest_row,
                                                  int*           permuted_idx,
                                                  int64_t*       expert_first_token_offset,
                                                  int64_t const  num_rows,
                                                  int64_t const* num_dest_rows,
                                                  int64_t const  cols,
                                                  int64_t        k) {
    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way
    // reduction and unpermuting. I need the reverse map for that reduction to
    // allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    int64_t       expanded_dest_row   = blockIdx.x;
    int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];

    if (threadIdx.x == 0) {
        assert(expanded_dest_row <= INT32_MAX);
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = static_cast<int>(expanded_dest_row);
        // skip non local expert token
        if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows) {
            permuted_idx[expanded_dest_row] = expanded_source_row;
        }
    }

    if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows) {
        // Load 128-bits per thread
        constexpr int64_t ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
        using DataElem                    = cutlass::Array<T, ELEM_PER_THREAD>;

        // Duplicate and permute rows
        int64_t const source_row = expanded_source_row / k;

        auto const* source_row_ptr = reinterpret_cast<DataElem const*>(unpermuted_input + source_row * cols);
        auto*       dest_row_ptr   = reinterpret_cast<DataElem*>(permuted_output + expanded_dest_row * cols);

        int64_t const start_offset     = threadIdx.x;
        int64_t const stride           = blockDim.x;
        int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;

        for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
            dest_row_ptr[elem_index] = source_row_ptr[elem_index];
        }
    }
}
template<typename T>
void expandInputRowsKernelWithoutScaleLauncher(T const*       unpermuted_input,
                                               T*             permuted_output,
                                               int*           sorted_experts,
                                               int const*     expanded_dest_row_to_expanded_source_row,
                                               int*           expanded_source_row_to_expanded_dest_row,
                                               int*           permuted_idx,
                                               int64_t*       expert_first_token_offset,
                                               int64_t const  num_rows,
                                               int64_t const* num_valid_tokens_ptr,
                                               int64_t const  cols,
                                               int const      k,
                                               cudaStream_t   stream) {
    int64_t const blocks  = num_rows * k;
    int64_t const threads = 256;
    auto          func    = (num_valid_tokens_ptr != nullptr) ? expandInputRowsWithoutScaleKernel<T, true> :
                                                                expandInputRowsWithoutScaleKernel<T, false>;
    func<<<blocks, threads, 0, stream>>>(unpermuted_input,
                                         permuted_output,
                                         sorted_experts,
                                         expanded_dest_row_to_expanded_source_row,
                                         expanded_source_row_to_expanded_dest_row,
                                         permuted_idx,
                                         expert_first_token_offset,
                                         num_rows,
                                         num_valid_tokens_ptr,
                                         cols,
                                         k);
}
void moe_pre_reorder(const torch::Tensor&                input,                 // [n_token, hidden]
                     const torch::Tensor&                topk_ids,              // [n_token, topk]
                     const torch::Tensor&                token_expert_indices,  // [n_token, topk]
                     const std::optional<torch::Tensor>& expert_map,            // [n_expert]
                     int64_t                             n_expert,
                     int64_t                             n_local_expert,
                     int64_t                             topk,
                     const std::optional<int64_t>&       align_block_size,
                     torch::Tensor&                      permuted_input,             // [permuted_size, hidden]
                     torch::Tensor&                      expert_first_token_offset,  // [n_local_expert + 1]
                     torch::Tensor&                      inv_permuted_idx,           // [n_token, topk]
                     torch::Tensor&                      permuted_idx                // [permute_size]
) {
    TORCH_CHECK(expert_first_token_offset.scalar_type() == at::ScalarType::Long,
                "expert_first_token_offset must be int64");
    TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int, "topk_ids must be int32");
    TORCH_CHECK(token_expert_indices.scalar_type() == at::ScalarType::Int, "token_expert_indices must be int32");
    TORCH_CHECK(inv_permuted_idx.scalar_type() == at::ScalarType::Int, "inv_permuted_idx must be int32");
    TORCH_CHECK(expert_first_token_offset.size(0) == n_local_expert + 1,
                "expert_first_token_offset shape != n_local_expert+1")
    TORCH_CHECK(inv_permuted_idx.sizes() == token_expert_indices.sizes(),
                "token_expert_indices shape must be same as inv_permuted_idx");
    auto n_token  = input.sizes()[0];
    auto n_hidden = input.sizes()[1];

    auto align_block_size_value = align_block_size.has_value() ? align_block_size.value() : -1;

    assert((align_block_size_value == -1) && "not implemented");

    auto       stream      = at::cuda::getCurrentCUDAStream().stream();
    const long sorter_size = trt::CubKeyValueSorter::getWorkspaceSize(n_token * topk, n_expert);
    auto       sort_workspace =
        torch::empty({sorter_size}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    auto copy_topk_ids       = topk_ids.clone();  // copy topk_ids for preprocess
    auto permuted_experts_id = torch::empty_like(topk_ids);
    auto sorted_row_idx      = torch::empty_like(inv_permuted_idx);

    trt::CubKeyValueSorter sorter{};
    int64_t*               valid_num_ptr = nullptr;
    // pre-process kernel for expert-parallelism:
    // no local expert id plus "n_expert" offset for priority to local expert
    // map local expert id [n, .., n+n_local_expert-1] to [0, n_local_expert -1]
    // For example, 4 expert with ep_size=2. ep_rank=1 owns global expert id
    // [2,3] with expert_map[-1, -1, 0, 1], preprocess_topk_id  process topk_ids
    // and map global expert id [2, 3] to local_expert id [0, 1] and map global
    // expert id [0, 1] ( not in ep rank=1)  to [4, 5] by plus n_expert. This map
    // operation is to make local expert high priority in following sort topk_ids
    // and scan local expert_first_token_offset for each ep rank for next group
    // gemm.
    if (expert_map.has_value()) {
        const int* expert_map_ptr = get_ptr<int>(expert_map.value());
        valid_num_ptr             = get_ptr<int64_t>(expert_first_token_offset) + n_local_expert;
        preprocessTopkIdLauncher(get_ptr<int>(copy_topk_ids), n_token * topk, expert_map_ptr, n_expert, stream);
    }

    // expert sort topk expert id and scan expert id get expert_first_token_offset

    sortAndScanSoftmaxOutput(get_ptr<int>(copy_topk_ids),
                             const_cast<int*>(get_ptr<int>(token_expert_indices)),
                             get_ptr<int>(permuted_experts_id),
                             get_ptr<int>(sorted_row_idx),
                             get_ptr<int64_t>(expert_first_token_offset),
                             n_token,
                             n_expert,
                             n_local_expert,
                             topk,
                             sorter,
                             get_ptr<int>(sort_workspace),
                             stream);
    // dispatch expandInputRowsKernelWithoutScaleLauncher
    MOE_DISPATCH(input.scalar_type(), [&] {
        expandInputRowsKernelWithoutScaleLauncher<scalar_t>(get_ptr<scalar_t>(input),
                                                            get_ptr<scalar_t>(permuted_input),
                                                            get_ptr<int>(permuted_experts_id),
                                                            get_ptr<int>(sorted_row_idx),
                                                            get_ptr<int>(inv_permuted_idx),
                                                            get_ptr<int>(permuted_idx),
                                                            get_ptr<int64_t>(expert_first_token_offset),
                                                            n_token,
                                                            valid_num_ptr,
                                                            n_hidden,
                                                            topk,
                                                            stream);
    });
}

template<class T, class OutputType>
void finalizeMoeRoutingKernelLauncher(T const*       expanded_permuted_rows,
                                      OutputType*    reduced_unpermuted_output,
                                      float const*   scales,
                                      int const*     expanded_source_row_to_expanded_dest_row,
                                      int64_t const  num_rows,
                                      int64_t const  cols,
                                      int64_t const  k,
                                      int64_t const* num_valid_ptr,
                                      cudaStream_t   stream) {
    int64_t const blocks         = num_rows;
    int64_t const threads        = 256;
    bool const    check_finished = num_valid_ptr != nullptr;

    // Currently we do not use scale_bias during moe unpermute.
    T const* bias_ptr = nullptr;

    using FuncPtr = decltype(&trt::finalizeMoeRoutingKernel<T /*UselessTypeDefine*/,
                                                            OutputType,
                                                            T /*GemmOutputType*/,
                                                            T /*ScaleBiasType*/,
                                                            ScaleMode::DEFAULT /*SCALE_MODE*/,
                                                            false /*CHECK_SKIPPED*/>);

    FuncPtr     func_map[2] = {&trt::finalizeMoeRoutingKernel<T, OutputType, T, T, ScaleMode::DEFAULT, false>,
                               &trt::finalizeMoeRoutingKernel<T, OutputType, T, T, ScaleMode::DEFAULT, true>};
    auto* const kernel      = func_map[check_finished];
    kernel<<<blocks, threads, 0, stream>>>(
        expanded_permuted_rows,
        reduced_unpermuted_output,
        bias_ptr /*a nullptr for unused param 'bias_ptr'*/,
        scales,
        expanded_source_row_to_expanded_dest_row,
        expanded_source_row_to_expanded_dest_row /*dummy ptr for unused param 'expert_for_source_row'*/,
        cols,
        k,
        num_valid_ptr);
}
void moe_post_reorder(const torch::Tensor&                permuted_hidden_states,     // [n_token * topk, hidden]
                      const torch::Tensor&                topk_weights,               // [n_token, topk]
                      const torch::Tensor&                inv_permuted_idx,           // [topk, n_token]
                      const std::optional<torch::Tensor>& expert_first_token_offset,  // [n_local_expert+1]
                      int64_t                             topk,
                      torch::Tensor&                      hidden_states  // [n_token, hidden]
) {
    TORCH_CHECK(permuted_hidden_states.scalar_type() == hidden_states.scalar_type(),
                "permuted_hidden_states dtype must be same as hidden_states");
    auto n_token  = hidden_states.size(0);
    auto n_hidden = hidden_states.size(1);
    auto stream   = at::cuda::getCurrentCUDAStream().stream();

    int64_t const* valid_ptr = nullptr;
    if (expert_first_token_offset.has_value()) {
        int n_local_expert = expert_first_token_offset.value().size(0) - 1;
        valid_ptr          = get_ptr<int64_t>(expert_first_token_offset.value()) + n_local_expert;
    }

    MOE_DISPATCH(hidden_states.scalar_type(), [&] {
        finalizeMoeRoutingKernelLauncher<scalar_t, scalar_t>(get_ptr<scalar_t>(permuted_hidden_states),
                                                             get_ptr<scalar_t>(hidden_states),
                                                             get_ptr<float>(topk_weights),
                                                             get_ptr<int>(inv_permuted_idx),
                                                             n_token,
                                                             n_hidden,
                                                             topk,
                                                             valid_ptr,
                                                             stream);
    });
}

}  // namespace rtp_llm
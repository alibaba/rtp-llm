/*
 * Routing kernels extracted from cutlass MOE code.
 * Non-GEMM CUDA kernels used by SelectTopkOp and ep_utils.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstddef>

// Cutlass types needed for finalizeMoeRoutingKernel template
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#pragma GCC diagnostic pop

namespace tensorrt_llm::kernels {

// ========================== Utility ==========================

static inline size_t pad_to_multiple_of_16(size_t const& input) {
    static constexpr int ALIGNMENT = 16;
    return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

// Device binary search function
template<class T>
__device__ inline int64_t
findTotalEltsLessThanTarget(T const* sorted_indices, int64_t const arr_length, T const target) {
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high) {
        int64_t mid = (low + high) / 2;
        if (sorted_indices[mid] >= target) {
            high = mid - 1;
        } else {
            low             = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

template<class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
    using Type = typename U::Element;
    static_assert(T::kElements == U::kElements);
    U u;
#pragma unroll
    for (int i = 0; i < U::kElements; i++) {
        u[i] = static_cast<Type>(input[i]);
    }
    return u;
}

// ========================== Enums / Structs ==========================

enum class MOEExpertScaleNormalizationMode : int {
    NONE = 0,
    RENORMALIZE,
    SPARSE_MIXER,
};

struct MOEParallelismConfig {
    int tp_size = 1;
    int tp_rank = 0;
    int ep_size = 1;
    int ep_rank = 0;

    MOEParallelismConfig() = default;
    MOEParallelismConfig(int tp_size, int tp_rank, int ep_size, int ep_rank):
        tp_size(tp_size), tp_rank(tp_rank), ep_size(ep_size), ep_rank(ep_rank) {}
};

enum class ScaleMode : int {
    NO_SCALE     = 0,
    DEFAULT      = 1,
    RENORM_SCALE = 2,
};

// ========================== CubKeyValueSorter ==========================

class CubKeyValueSorter {
public:
    CubKeyValueSorter();
    CubKeyValueSorter(int const num_experts);
    void          updateNumExperts(int const num_experts);
    static size_t getWorkspaceSize(size_t const num_key_value_pairs, int const num_experts);
    void          run(void*        workspace,
                      size_t const workspace_size,
                      int const*   keys_in,
                      int*         keys_out,
                      int const*   values_in,
                      int*         values_out,
                      size_t const num_key_value_pairs,
                      cudaStream_t stream);

private:
    static int expertsToBits(int experts);
    int        num_experts_;
    int        num_bits_;
};

// ========================== Function declarations ==========================

template<typename TOPK_T>
void invokeSelectExpertsForTokens(float const*                    input,
                                  float const*                    input_with_bias,
                                  float*                          output,
                                  float*                          mixer_temp_output,
                                  float*                          softmax_temp_output,
                                  TOPK_T*                         indices,
                                  int*                            source_row,
                                  int64_t const                   num_rows,
                                  int const                       num_experts,
                                  int const                       k,
                                  int const                       start_expert,
                                  int const                       end_expert,
                                  float                           mixer_epsilon,
                                  MOEExpertScaleNormalizationMode norm_mode,
                                  cudaStream_t                    stream);

void sortAndScanSoftmaxOutput(int*               expert_for_source_row,
                              int*               source_rows,
                              int*               permuted_experts,
                              int*               permuted_rows,
                              int64_t*           expert_first_token_offset,
                              int64_t            num_rows,
                              int64_t            num_experts,
                              int64_t            num_experts_per_node,
                              int64_t            k,
                              CubKeyValueSorter& sorter,
                              void*              sorter_ws,
                              cudaStream_t       stream);

void genSourceRow(int*         expert_rows,
                  int*         source_rows,
                  size_t       token_num,
                  size_t       top_k,
                  size_t       num_experts,
                  int          start_expert,
                  int          end_expert,
                  cudaStream_t stream);

void computeExpertFirstTokenOffset(int const*   sorted_indices,
                                   int const    total_indices,
                                   int const    num_experts,
                                   int64_t*     expert_first_token_offset,
                                   cudaStream_t stream);

// ========================== finalizeMoeRoutingKernel ==========================
// Template kernel used by ep_utils.cu

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

template<typename T,
         typename OutputType,
         class GemmOutputType,
         class ScaleBiasType,
         ScaleMode SCALE_MODE,
         bool      CHECK_SKIPPED>
__global__ void finalizeMoeRoutingKernel(GemmOutputType const* expanded_permuted_rows,
                                         OutputType*           reduced_unpermuted_output,
                                         ScaleBiasType const*  bias,
                                         float const*          scales,
                                         int const*            expanded_source_row_to_expanded_dest_row,
                                         int const*            expert_for_source_row,
                                         int64_t const         orig_cols,
                                         int64_t const         k,
                                         int64_t const*        num_valid_ptr) {
    assert(orig_cols % 4 == 0);
    int64_t const original_row    = blockIdx.x;
    int64_t const num_rows        = gridDim.x;
    auto const    offset          = original_row * orig_cols;
    OutputType*   reduced_row_ptr = reduced_unpermuted_output + offset;
    int64_t const num_valid       = num_valid_ptr ? *num_valid_ptr : 0;

    constexpr int64_t FINALIZE_ELEM_PER_THREAD =
        128 / std::min(cutlass::sizeof_bits<OutputType>::value, cutlass::sizeof_bits<GemmOutputType>::value);

    int64_t const start_offset     = threadIdx.x;
    int64_t const stride           = FINALIZE_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

    using BiasElem                       = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
    using InputElem                      = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
    using OutputElem                     = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
    using ComputeElem                    = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
    auto const* bias_v                   = reinterpret_cast<BiasElem const*>(bias);
    auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
    auto*       reduced_row_ptr_v        = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#pragma unroll
    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        bool        has_valid = false;
        ComputeElem thread_output;
        thread_output.fill(0);
        float row_rescale{0.f};
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            int64_t const expanded_original_row = original_row + k_idx * num_rows;
            int64_t const expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

            int64_t const k_offset  = original_row * k + k_idx;
            float const   row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
            if constexpr (SCALE_MODE == ScaleMode::RENORM_SCALE) {
                row_rescale = row_rescale + row_scale;
            }
            if (expanded_permuted_row < 0) {
                continue;
            }
            if (CHECK_SKIPPED && expanded_permuted_row >= num_valid) {
                continue;
            }

            auto const* expanded_permuted_rows_row_ptr =
                expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

            int64_t const expert_idx = expert_for_source_row[k_offset];

            auto const* bias_ptr = bias_v + expert_idx * num_elems_in_col;
            ComputeElem bias_value;
            if (bias) {
                bias_value = arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
            } else {
                bias_value.fill(0);
            }

            ComputeElem expert_result =
                arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
            thread_output = thread_output + row_scale * (expert_result + bias_value);
            has_valid     = true;
        }

        if (SCALE_MODE == ScaleMode::RENORM_SCALE && (!CHECK_SKIPPED || has_valid)) {
            assert(row_rescale != 0.f);
            for (auto& elem : thread_output) {
                elem /= row_rescale;
            }
        }

        OutputElem output_elem        = arrayConvert<ComputeElem, OutputElem>(thread_output);
        reduced_row_ptr_v[elem_index] = output_elem;
    }
}

template<class T, class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(GemmOutputType const*           expanded_permuted_rows,
                                      OutputType*                     reduced_unpermuted_output,
                                      ScaleBiasType const*            bias,
                                      float const*                    scales,
                                      int const*                      expanded_source_row_to_expanded_dest_row,
                                      int const*                      expert_for_source_row,
                                      int64_t const                   num_rows,
                                      int64_t const                   cols,
                                      int64_t const                   k,
                                      int64_t const*                  num_valid_ptr,
                                      MOEParallelismConfig            parallelism_config,
                                      MOEExpertScaleNormalizationMode normalization_mode,
                                      cudaStream_t                    stream) {
    int64_t const blocks  = num_rows;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;

    bool const           is_rank_0 = parallelism_config.tp_rank == 0;
    ScaleBiasType const* bias_ptr  = is_rank_0 ? bias : nullptr;

    bool const check_finished = num_valid_ptr != nullptr;

    ScaleMode renorm_scales = ScaleMode::DEFAULT;
    if (normalization_mode == MOEExpertScaleNormalizationMode::RENORMALIZE) {
        renorm_scales = k == 1 ? ScaleMode::NO_SCALE : ScaleMode::RENORM_SCALE;
    }

    using FuncPtr =
        decltype(&finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, false>);
    FuncPtr func_map[2][3] = {
        {
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, false>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, false>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::RENORM_SCALE, false>,
        },
        {
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, true>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, true>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::RENORM_SCALE, true>,
        },
    };
    auto* const func = func_map[check_finished][int(renorm_scales)];
    func<<<blocks, threads, 0, stream>>>(expanded_permuted_rows,
                                         reduced_unpermuted_output,
                                         bias_ptr,
                                         scales,
                                         expanded_source_row_to_expanded_dest_row,
                                         expert_for_source_row,
                                         cols,
                                         k,
                                         num_valid_ptr);
}

}  // namespace tensorrt_llm::kernels

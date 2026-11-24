#include "rtp_llm/cpp/models/SamplingRows.h"
#if USING_CUDA || USING_ROCM
#include "rtp_llm/models_py/bindings/common/kernels/dspark_packed_sampling.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#endif

namespace rtp_llm {
GreedyParams
gatherSamplingRows(const GreedyParams& dense_params, const torch::Tensor& row_map, const torch::Tensor& history) {
    const int64_t packed_rows   = row_map.numel();
    const auto    device        = dense_params.logits.options();
    auto          packed_params = dense_params;
    auto          floats        = torch::empty({6, packed_rows}, device.dtype(torch::kFloat32));
    auto          ints          = torch::empty({5, packed_rows}, device.dtype(torch::kInt32));
    packed_params.top_p         = floats.select(0, 0);
    packed_params.temperature   = floats.select(0, 1);
    if (dense_params.repetition_penalty)
        packed_params.repetition_penalty = floats.select(0, 2);
    if (dense_params.presence_penalty)
        packed_params.presence_penalty = floats.select(0, 3);
    if (dense_params.frequency_penalty)
        packed_params.frequency_penalty = floats.select(0, 4);
    packed_params.inverse_temperature = floats.select(0, 5);
    packed_params.top_k               = ints.select(0, 0);
    if (dense_params.no_repeat_ngram_size)
        packed_params.no_repeat_ngram_size = ints.select(0, 1);
    packed_params.input_lengths         = ints.select(0, 2);
    packed_params.sequence_lengths      = ints.select(0, 3);
    packed_params.sequence_last_indexes = ints.select(0, 4);
    packed_params.decoder_batch_size    = packed_rows;
    packed_params.seed =
        dense_params.seed.defined() ? torch::empty({packed_rows}, device.dtype(torch::kInt64)) : torch::Tensor();
    packed_params.offset =
        dense_params.offset.defined() ? torch::empty({packed_rows}, device.dtype(torch::kInt64)) : torch::Tensor();
    packed_params.do_sample = torch::empty({packed_rows}, device.dtype(torch::kBool));
    if (dense_params.cum_log_probs)
        packed_params.cum_log_probs = torch::empty({packed_rows}, device.dtype(torch::kFloat32));
    packed_params.token_ids =
        torch::empty({packed_rows, static_cast<int64_t>(dense_params.step + 1)}, device.dtype(torch::kInt32));
    if (packed_params.mode.ngram_active) {
        packed_params.output_ids_ptrs = torch::empty({packed_rows}, device.dtype(torch::kInt64));
    }
    auto history_gpu = history.defined() ? history.to(device.device(), /*non_blocking=*/true) : torch::Tensor();
#if USING_CUDA || USING_ROCM
    invokeSamplingGather(dense_params,
                         packed_params,
                         row_map,
                         history_gpu,
                         packed_params.token_ids,
                         cuda_graph::graphGetCurrentStream().stream());
#else
    TORCH_CHECK(false, "sampling row gather is unavailable on this backend");
#endif
    return packed_params;
}

SamplerOutput scatterSamplingRows(const SamplerOutput& packed, const torch::Tensor& dense_to_packed) {
    const int64_t dense_rows = dense_to_packed.numel();
    auto          result     = packed;
    result.token_ids         = torch::empty({dense_rows, 1}, packed.token_ids.options());
    const auto allocate      = [dense_rows](const torch::Tensor& t) {
        if (!t.defined())
            return torch::Tensor();
        auto shape = t.sizes().vec();
        shape[0]   = dense_rows;
        return torch::empty(shape, t.options());
    };
    result.all_probs         = allocate(packed.all_probs);
    result.all_probs_indices = allocate(packed.all_probs_indices);
    result.cum_log_probs     = allocate(packed.cum_log_probs);
    result.success           = allocate(packed.success);
    const bool shared        = packed.sampling_probs.defined() && packed.all_probs.defined()
                        && packed.sampling_probs.data_ptr() == packed.all_probs.data_ptr();
    result.sampling_probs = shared ? result.all_probs : allocate(packed.sampling_probs);
#if USING_CUDA || USING_ROCM
    SamplingResult src{packed.token_ids,
                       packed.token_ids_are_point_mass ? torch::Tensor() : packed.all_probs,
                       packed.all_probs_indices,
                       shared ? torch::Tensor() : packed.sampling_probs,
                       packed.cum_log_probs,
                       packed.success};
    SamplingResult dst{result.token_ids,
                       packed.token_ids_are_point_mass ? torch::Tensor() : result.all_probs,
                       result.all_probs_indices,
                       shared ? torch::Tensor() : result.sampling_probs,
                       result.cum_log_probs,
                       result.success};
    invokeSamplingScatter(src, dst, dense_to_packed, cuda_graph::graphGetCurrentStream().stream());
#else
    TORCH_CHECK(false, "sampling row scatter is unavailable on this backend");
#endif
    return result;
}
}  // namespace rtp_llm

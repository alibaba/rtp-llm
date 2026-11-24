#include "rtp_llm/models_py/bindings/core/SamplingParams.h"
#include "rtp_llm/models_py/bindings/core/SampleTemperature.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <algorithm>
#include <cmath>
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#elif USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/kernels/sampling/sampling.h"
#endif

namespace rtp_llm {
namespace {
SamplingMode summarizeSamplingParams(const GreedyParams& p, int64_t rows) {
    SamplingMode m;
    TORCH_CHECK(p.top_k.device().is_cpu() && p.top_p.device().is_cpu() && p.temperature.device().is_cpu(),
                "sampling dispatch summary requires CPU parameters");
    for (int64_t i = 0; i < rows; ++i) {
        const int   k     = p.top_k.data_ptr<int32_t>()[i];
        const float raw_p = p.top_p.data_ptr<float>()[i];
        const float prob  = std::abs(raw_p) < 1e-7f ? 1.0f : raw_p;
        m.all_top_k_one &= k == 1;
        m.all_top_k_unlimited &= k <= 0;
        m.min_top_k = std::min(m.min_top_k, k);
        m.max_top_k = std::max(m.max_top_k, k);
        m.all_top_p_one &= std::abs(prob - 1.0f) < 1e-7f;
        m.all_top_p_exact_one &= prob == 1.0f;
        m.any_top_p_one |= std::abs(prob - 1.0f) < 1e-7f;
        m.temperature_active |= p.temperature.data_ptr<float>()[i] != 1.0f;
        if (p.repetition_penalty) {
            m.repetition_active |= !p.presence_penalty || !p.frequency_penalty
                                   || p.repetition_penalty->data_ptr<float>()[i] != 1.0f
                                   || p.presence_penalty->data_ptr<float>()[i] != 0.0f
                                   || p.frequency_penalty->data_ptr<float>()[i] != 0.0f;
        }
        if (p.no_repeat_ngram_size && i < p.sequence_lengths.numel()) {
            m.ngram_active |= p.no_repeat_ngram_size->data_ptr<int32_t>()[i] != 0;
        }
        const bool sample = !p.do_sample || p.do_sample->data_ptr<bool>()[i];
        m.need_do_sample |= sample;
        m.has_not_do_sample |= !sample;
    }
    return m;
}

std::pair<torch::Tensor, torch::Tensor>
makeSamplingSeedOffsetTensors(const std::vector<at::Generator>& generators, int64_t batch_size, int increment) {
    RTP_LLM_CHECK_WITH_INFO(generators.size() >= static_cast<size_t>(batch_size),
                            "sampling generators size [%lu] is smaller than batch size [%ld]",
                            generators.size(),
                            batch_size);
    const auto options = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    auto       seed    = torch::empty({batch_size}, options);
    auto       offset  = torch::empty({batch_size}, options);
#if USING_CUDA
    get_sampling_seed_offsets(generators, batch_size, increment, seed.data_ptr<int64_t>(), offset.data_ptr<int64_t>());
#elif USING_ROCM
    for (int64_t row = 0; row < batch_size; ++row) {
        auto generator                = generators[row].defined() ? std::make_optional(generators[row]) : std::nullopt;
        auto [s, o]                   = get_seed_and_offset(increment, generator);
        seed.data_ptr<int64_t>()[row] = s;
        offset.data_ptr<int64_t>()[row] = o;
    }
#else
    TORCH_CHECK(false, "device sampling RNG is unavailable on this backend");
#endif
    return {seed.to(torch::kCUDA, true), offset.to(torch::kCUDA, true)};
}

}  // namespace

void prepareSamplingTokenIds(GreedyParams& params, const torch::Tensor& token_ids) {
    params.token_ids = token_ids;
    if (!params.mode.ngram_active) {
        return;
    }
    const auto rows = params.decoder_batch_size;
    auto       pointers =
        torch::empty({rows}, torch::TensorOptions().dtype(torch::kInt64).pinned_memory(token_ids.is_cuda()));
    for (int64_t row = 0; row < rows; ++row) {
        pointers.data_ptr<int64_t>()[row] =
            reinterpret_cast<int64_t>(token_ids.data_ptr<int32_t>() + row * token_ids.stride(0));
    }
    params.output_ids_ptrs = pointers.to(token_ids.device(), true);
}

GreedyParams prepareSamplingParams(const GreedyParams& params, int64_t rows) {
    auto result               = params;
    result.mode               = summarizeSamplingParams(params, rows);
    const int  force_topk     = enableCudaSampling() ? samplingCandidateWidth(params, result.mode) : 0;
    const auto host           = torch::TensorOptions().device(torch::kCPU).pinned_memory(!params.logits.is_cpu());
    auto       floats         = torch::empty({6, rows}, host.dtype(torch::kFloat32));
    auto       ints           = torch::empty({5, rows}, host.dtype(torch::kInt32));
    auto       do_sample      = torch::empty({rows}, host.dtype(torch::kBool));
    const auto optional_float = [](const std::optional<torch::Tensor>& value, int64_t row, float fallback) {
        return value ? value->data_ptr<float>()[row] : fallback;
    };
    for (int64_t row = 0; row < rows; ++row) {
        auto*       f            = floats.data_ptr<float>();
        auto*       i            = ints.data_ptr<int32_t>();
        const int   k            = params.top_k.data_ptr<int32_t>()[row];
        const float p            = params.top_p.data_ptr<float>()[row];
        const float temperature  = params.temperature.data_ptr<float>()[row];
        const int   input_length = params.input_lengths.is_cpu() ? params.input_lengths.data_ptr<int32_t>()[row] : 0;
        const int   length =
            row < params.sequence_lengths.numel() ?
                  (params.sequence_lengths.is_cpu() ? params.sequence_lengths.data_ptr<int32_t>()[row] : 0) :
                  input_length;
        f[row]            = std::abs(p) < 1e-7f ? 1.0f : p;
        f[rows + row]     = temperature;
        f[2 * rows + row] = optional_float(params.repetition_penalty, row, 1.0f);
        f[3 * rows + row] = optional_float(params.presence_penalty, row, 0.0f);
        f[4 * rows + row] = optional_float(params.frequency_penalty, row, 0.0f);
        f[5 * rows + row] = sampleTemperatureInvScale(temperature);
        i[row]            = result.mode.all_top_k_one || result.mode.all_top_k_unlimited ? k :
                            force_topk > 0 ? (k <= 0 ? force_topk : std::min(k, force_topk)) :
                                             (k <= 0 ? 1 << 30 : k);
        i[rows + row]     = params.no_repeat_ngram_size ? params.no_repeat_ngram_size->data_ptr<int32_t>()[row] : 0;
        i[2 * rows + row] = input_length;
        i[3 * rows + row] = length;
        i[4 * rows + row] = length - 1;
        do_sample.data_ptr<bool>()[row] = !params.do_sample || params.do_sample->data_ptr<bool>()[row];
    }
    // Reserve all source rows before any packing, including aliased/default
    // generators. All-greedy groups consume no entropy.
    if (!result.mode.all_top_k_one && params.logits.is_cuda()) {
        auto rng      = makeSamplingSeedOffsetTensors(params.generator, rows, 32);
        result.seed   = std::move(rng.first);
        result.offset = std::move(rng.second);
    }
    floats             = floats.to(params.logits.device(), true);
    ints               = ints.to(params.logits.device(), true);
    result.top_p       = floats.select(0, 0);
    result.temperature = floats.select(0, 1);
    if (params.repetition_penalty)
        result.repetition_penalty = floats.select(0, 2);
    if (params.presence_penalty)
        result.presence_penalty = floats.select(0, 3);
    if (params.frequency_penalty)
        result.frequency_penalty = floats.select(0, 4);
    result.inverse_temperature = floats.select(0, 5);
    result.top_k               = ints.select(0, 0);
    if (params.no_repeat_ngram_size)
        result.no_repeat_ngram_size = ints.select(0, 1);
    result.input_lengths         = ints.select(0, 2);
    result.sequence_lengths      = ints.select(0, 3);
    result.sequence_last_indexes = ints.select(0, 4);
    result.decoder_batch_size    = params.sequence_lengths.numel();
    if (params.cum_log_probs)
        result.cum_log_probs = params.cum_log_probs->to(params.logits.device(), true);
    result.do_sample = do_sample.to(params.logits.device(), true);
    // Low-level callers may already own device lengths. Preserve that contract
    // without a device-to-host read; engine gatherers supply CPU lengths above.
    if (params.input_lengths.is_cuda()) {
        result.input_lengths.copy_(params.input_lengths, true);
        result.sequence_lengths.copy_(params.input_lengths, true);
    }
    const auto decoder_rows = params.sequence_lengths.numel();
    if (decoder_rows > 0 && (params.sequence_lengths.is_cuda() || params.input_lengths.is_cuda())) {
        result.sequence_lengths.narrow(0, 0, decoder_rows).copy_(params.sequence_lengths, true);
    }
    if (result.mode.ngram_active && params.sequence_lengths.is_cuda()) {
        auto last_indexes = result.sequence_last_indexes.narrow(0, 0, decoder_rows);
        torch::sub_out(last_indexes, params.sequence_lengths, 1);
    }
    if (params.token_ids.defined()) {
        prepareSamplingTokenIds(result, params.token_ids.to(params.logits.device(), true));
    }
    if (params.logits.is_cuda())
        result.generator.clear();
    return result;
}

}  // namespace rtp_llm

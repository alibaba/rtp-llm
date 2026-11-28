#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/SamplingRows.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/SamplingParams.h"
#include <algorithm>
#include <variant>
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace std;

namespace rtp_llm {

namespace {
struct SamplingGroup {
    int64_t                                      input_offset  = 0;
    int64_t                                      output_offset = 0;
    int64_t                                      input_rows    = 0;
    int64_t                                      output_rows   = 0;
    std::variant<GreedyParams, BeamSearchParams> params;
};

std::vector<SamplingGroup> prepareSamplingGroups(const SamplerInputs& inputs) {
    std::vector<SamplingGroup> groups;
    const auto*                beams_in      = inputs.num_beams_in.data_ptr<int64_t>();
    const auto*                beams_out     = inputs.num_beams_out.data_ptr<int64_t>();
    int64_t                    output_offset = 0;
    for (int64_t begin = 0; begin < inputs.batch_size;) {
        int64_t end = begin + 1;
        while (end < inputs.batch_size && beams_in[end] == beams_in[begin] && beams_out[end] == beams_out[begin]) {
            ++end;
        }
        const auto    rows           = end - begin;
        const auto    execution_rows = inputs.row_mapping ? inputs.logits.size(0) : rows;
        SamplingGroup group{
            begin, output_offset, execution_rows, execution_rows / beams_in[begin] * beams_out[begin], {}};
        auto logits = inputs.row_mapping ? inputs.logits : inputs.logits.narrow(0, begin, rows);
        if (beams_in[begin] == 1 && beams_out[begin] == 1) {
            const auto optional_slice = [begin, rows](const torch::Tensor& value) -> std::optional<torch::Tensor> {
                return value.defined() ? std::optional<torch::Tensor>(value.narrow(0, begin, rows)) : std::nullopt;
            };
            GreedyParams params;
            params.logits        = logits;
            params.step          = inputs.step;
            params.input_lengths = inputs.input_lengths.narrow(0, begin, rows);
            const auto decoder_rows =
                std::max<int64_t>(0, std::min<int64_t>(rows, inputs.sequence_lengths.numel() - begin));
            params.sequence_lengths =
                inputs.sequence_lengths.narrow(0, std::min(begin, inputs.sequence_lengths.numel()), decoder_rows);
            params.top_k                     = inputs.top_k.narrow(0, begin, rows);
            params.top_p                     = inputs.top_p.narrow(0, begin, rows);
            params.temperature               = inputs.temperature.narrow(0, begin, rows);
            params.repetition_penalty        = optional_slice(inputs.repetition_penalty);
            params.presence_penalty          = optional_slice(inputs.presence_penalty);
            params.frequency_penalty         = optional_slice(inputs.frequency_penalty);
            params.no_repeat_ngram_size      = optional_slice(inputs.no_repeat_ngram_size);
            params.do_sample                 = optional_slice(inputs.do_sample);
            params.cum_log_probs             = optional_slice(inputs.cum_log_probs);
            params.force_topk                = inputs.force_topk;
            params.skip_logits_probs_restore = inputs.skip_logits_probs_restore;
            params.return_original_all_probs = inputs.return_original_all_probs;
            params.return_sampling_probs     = inputs.return_logprobs;
            if (inputs.generator.empty()) {
                params.generator.resize(rows);
            } else {
                params.generator = {inputs.generator.begin() + begin, inputs.generator.begin() + end};
            }
            params       = prepareSamplingParams(params, rows);
            group.params = std::move(params);
        } else {
            const auto beams = beams_in[begin];
            RTP_LLM_CHECK_WITH_INFO(
                rows % beams == 0, "sampling group rows [%ld] must be divisible by input beams [%ld]", rows, beams);
            const auto       batch  = rows / beams;
            const auto       device = logits.device();
            BeamSearchParams params;
            params.logits           = logits.reshape({batch, beams, logits.size(1)});
            params.input_lengths    = inputs.input_lengths.narrow(0, begin, rows).reshape({batch, beams}).to(device);
            params.sequence_lengths = inputs.sequence_lengths.narrow(0, begin, rows).reshape({batch, beams}).to(device);
            params.cum_log_probs =
                (inputs.cum_log_probs.defined() ? inputs.cum_log_probs.narrow(0, begin, rows).reshape({batch, beams}) :
                                                  torch::zeros({batch, beams}))
                    .to(device);
            params.num_beams_out = beams_out[begin];
            group.params         = std::move(params);
        }
        output_offset += group.output_rows;
        groups.push_back(std::move(group));
        begin = end;
    }
    return groups;
}

void samplePreparedGroups(const SamplerInputs& inputs, std::vector<SamplingGroup>& groups, SamplerOutput& output) {
    const auto mayNarrow = [](const torch::Tensor& t, int64_t offset, int64_t size) {
        return t.defined() ? t.narrow(0, offset, size) : torch::Tensor();
    };
    const auto mayOptNarrow =
        [&mayNarrow](const torch::Tensor& t, int64_t offset, int64_t size) -> std::optional<torch::Tensor> {
        return t.defined() ? std::make_optional(mayNarrow(t, offset, size)) : std::nullopt;
    };
    const bool    has_beams      = std::any_of(groups.begin(), groups.end(), [](const auto& group) {
        return std::holds_alternative<BeamSearchParams>(group.params);
    });
    const int64_t batch_size     = inputs.logits.size(0);
    const int64_t batch_size_out = output.token_ids.size(0);
    // Keep success on device; callers that need host access explicitly copy it.
    output.success       = torch::empty({batch_size}, inputs.logits.options().dtype(torch::kBool));
    output.beam_index    = has_beams ? torch::empty({batch_size_out}, torch::kInt32) : torch::Tensor();
    output.cum_log_probs = inputs.cum_log_probs.defined() ?
                               torch::empty({batch_size_out}, inputs.logits.options().dtype(torch::kFloat32)) :
                               torch::Tensor();

    // Only whole-batch single-beam outputs can use a compact distribution.
    const bool distribution = inputs.probability_output == SamplerProbabilityOutput::DISTRIBUTION
                              && enableCudaSampling() && !has_beams && inputs.skip_logits_probs_restore
                              && !inputs.return_original_all_probs && !inputs.cum_log_probs.defined();
    output.token_ids_are_point_mass =
        distribution && !inputs.return_logprobs && std::all_of(groups.begin(), groups.end(), [](const auto& group) {
            return std::get<GreedyParams>(group.params).mode.all_top_k_one;
        });
    const bool compact = distribution && inputs.force_topk > 0 && inputs.force_topk <= 4096
                         && inputs.force_topk < inputs.logits.size(1)
                         && std::all_of(groups.begin(), groups.end(), [](const auto& group) {
                                return std::get<GreedyParams>(group.params).mode.all_top_k_unlimited;
                            });
    if (inputs.probability_output != SamplerProbabilityOutput::NONE) {
        const auto width   = compact ? inputs.force_topk : static_cast<int64_t>(inputs.vocab_size);
        const auto options = inputs.logits.options().dtype(torch::kFloat32);
        output.all_probs =
            distribution ? torch::empty({batch_size, width}, options) : torch::zeros({batch_size, width}, options);
        if (compact) {
            output.all_probs_indices = torch::empty(output.all_probs.sizes(), options.dtype(torch::kInt32));
        }
    }
    std::vector<torch::Tensor> sampling_probs;
    for (auto& group : groups) {
        auto success           = output.success.narrow(0, group.input_offset, group.input_rows);
        auto token_ids_out     = output.token_ids.narrow(0, group.output_offset, group.output_rows);
        auto cum_log_probs_out = mayNarrow(output.cum_log_probs, group.output_offset, group.output_rows);
        if (auto* params = std::get_if<GreedyParams>(&group.params)) {
            params->all_probs_point_mass = output.token_ids_are_point_mass;
            params->output_all_probs     = mayOptNarrow(output.all_probs, group.input_offset, group.input_rows);
            params->output_all_probs_indices =
                mayOptNarrow(output.all_probs_indices, group.input_offset, group.input_rows);
            RTP_LLM_PROFILE_SCOPE("sampler.forward.execSampleGreedy");
            auto greedy_output = execSampleGreedy(*params);
            if (cum_log_probs_out.defined() && params->cum_log_probs) {
                cum_log_probs_out.copy_(*params->cum_log_probs);
            }
            if (inputs.return_logprobs) {
                sampling_probs.push_back(greedy_output.sampling_probs);
            }
            output.token_ids_are_point_mass = output.token_ids_are_point_mass && greedy_output.all_probs_point_mass;
            if (greedy_output.success.defined()) {
                success.copy_(greedy_output.success);
            } else {
                success.fill_(true);
            }
            // Greedy writes in place. Copy history only when beam transitions
            // require separate output storage for the batch.
            if (token_ids_out.data_ptr() != params->token_ids.data_ptr()) {
                token_ids_out.copy_(params->token_ids);
            }
        } else {
            const auto& beam_params = std::get<BeamSearchParams>(group.params);
            if (inputs.return_logprobs) {
                sampling_probs.push_back(torch::softmax(beam_params.logits.flatten(0, 1), -1));
            }
            auto beam_output = execSampleBeamSearch(beam_params);
            token_ids_out.copy_(beam_output.token_ids.flatten(0, 1));
            if (cum_log_probs_out.defined()) {
                cum_log_probs_out.copy_(beam_output.cum_log_probs.flatten());
            }
            output.beam_index.narrow(0, group.output_offset, group.output_rows)
                .copy_(beam_output.beam_indices.flatten());
            success.fill_(true);
        }
    }
    if (inputs.return_logprobs && !output.token_ids_are_point_mass) {
        output.sampling_probs = sampling_probs.size() == 1 ? sampling_probs.front() : torch::cat(sampling_probs, 0);
    }
}

}  // namespace

Sampler::Sampler(const SamplerInitParams& params):
    fixed_max_batch_size_(params.max_batch_size > 0 && params.fixed_max_batch_size),
    max_batch_size_(params.max_batch_size) {}

SamplerOutput Sampler::forward(const SamplerInputs& requested_inputs) {
    auto inputs = requested_inputs;
    if (inputs.probability_output == SamplerProbabilityOutput::NONE && inputs.all_probs.defined()) {
        inputs.probability_output = inputs.all_probs_point_mass ? SamplerProbabilityOutput::DISTRIBUTION : SamplerProbabilityOutput::DENSE;
    }
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_PROFILE_SCOPE("sampler.forward");
    RTP_LLM_CHECK_WITH_INFO(!fixed_max_batch_size_ || inputs.batch_size <= max_batch_size_,
                            "sampler batch size [%lu] exceeds initialized max batch size [%lu]",
                            inputs.batch_size,
                            max_batch_size_);
    const bool mapped = inputs.row_mapping.has_value();
    if (mapped) {
        const auto single_beam = [](const torch::Tensor& beams) {
            const auto* values = beams.data_ptr<int64_t>();
            return std::all_of(values, values + beams.numel(), [](auto n) { return n == 1; });
        };
        TORCH_CHECK(single_beam(inputs.num_beams_in) && single_beam(inputs.num_beams_out)
                        && inputs.sequence_lengths.numel() == inputs.batch_size,
                    "row selection requires single-beam decoder rows");
    }
    TORCH_CHECK(!mapped || !inputs.logits_processor_states_ptr,
                "mapped sampling requires logits processors to run before Sampler::forward");
    auto processor_errors = preprocessLogits(inputs);

    // Normalize parameters and reserve RNG in logical row order before selection.
    auto groups    = prepareSamplingGroups(inputs);
    auto token_ids = inputs.token_ids;
    if (mapped) {
        auto& params = std::get<GreedyParams>(groups.front().params);
        params       = gatherSamplingRows(params, inputs.row_mapping->packed_to_dense, inputs.token_ids);
        token_ids    = params.token_ids;
    }
#if USING_ROCM || (defined(USE_PPU) && USE_PPU)
    constexpr bool async_token_copy = false;
#else
    constexpr bool async_token_copy = true;
#endif
    token_ids = token_ids.defined() ?
                    token_ids.to(inputs.logits.device(), async_token_copy) :
                    torch::empty({static_cast<int64_t>(inputs.batch_size), static_cast<int64_t>(inputs.step + 1)},
                                 inputs.logits.options().dtype(torch::kInt32));
    for (auto& group : groups) {
        auto group_tokens = token_ids.narrow(0, group.input_offset, group.input_rows);
        if (auto* params = std::get_if<GreedyParams>(&group.params)) {
            // Gather already bound packed history and its ngram pointers.
            if (!params->token_ids.defined()) {
                prepareSamplingTokenIds(*params, group_tokens);
            }
        } else {
            auto& beam_params = std::get<BeamSearchParams>(group.params);
            beam_params.token_ids =
                group_tokens.reshape({beam_params.logits.size(0), beam_params.logits.size(1), token_ids.size(1)});
        }
    }
    // Reuse the history matrix unless a beam group changes its row count.
    // Mixed 1->2 and 2->1 groups still need separate storage when totals match.
    const bool independent_output = std::any_of(
        groups.begin(), groups.end(), [](const auto& group) { return group.input_rows != group.output_rows; });
    SamplerOutput output;
    output.token_ids =
        independent_output ?
            torch::empty({static_cast<int64_t>(inputs.batch_size_out), static_cast<int64_t>(inputs.step + 1)},
                         token_ids.options()) :
            token_ids;
    samplePreparedGroups(inputs, groups, output);
    output.processor_errors = std::move(processor_errors);
    if (mapped) {
        output = scatterSamplingRows(output, inputs.row_mapping->dense_to_packed);
    }
    output.raw_logprobs_logits = inputs.raw_logprobs_logits;
    output.raw_logprobs_row_indices = inputs.raw_logprobs_row_indices;
    return output;
}

SamplingLogprobs computeSamplingLogprobs(const torch::Tensor& probs,
                                         const torch::Tensor& candidate_ids,
                                         const torch::Tensor& point_mass_token_ids,
                                         const torch::Tensor& emitted_token_ids,
                                         int64_t              max_top_logprobs) {
    SamplingLogprobs result;
    auto             tokens = emitted_token_ids.reshape({-1}).to(torch::kLong);
    if (point_mass_token_ids.defined()) {
        auto ids                     = point_mass_token_ids.reshape({-1});
        result.token_logprobs        = tokens.eq(ids).to(torch::kFloat32).log();
        const int64_t k              = max_top_logprobs > 0 ? 1 : 0;
        result.top_logprob_token_ids = ids.to(torch::kInt32).unsqueeze(1).narrow(1, 0, k);
        result.top_logprobs          = torch::zeros({tokens.numel(), k}, result.token_logprobs.options());
        return result;
    }
    RTP_LLM_CHECK(probs.defined() && probs.dim() == 2 && probs.size(0) == tokens.numel());
    torch::Tensor token_probs;
    if (candidate_ids.defined()) {
        // A replacement token is looked up in target p, never in (p - q)+.
        token_probs = probs.masked_fill(candidate_ids.ne(tokens.unsqueeze(1)), 0).sum(1);
    } else {
        auto valid  = tokens.ge(0).logical_and(tokens.lt(probs.size(1)));
        token_probs = probs.gather(1, tokens.clamp(0, probs.size(1) - 1).unsqueeze(1)).squeeze(1);
        token_probs = token_probs.masked_fill(~valid, 0);
    }
    result.token_logprobs = token_probs.to(torch::kFloat32).log();
    const int64_t k       = std::min<int64_t>(std::max<int64_t>(max_top_logprobs, 0), probs.size(1));
    auto          top     = probs.topk(k, -1, true, true);
    result.top_logprobs   = std::get<0>(top).to(torch::kFloat32).log();
    result.top_logprob_token_ids =
        (candidate_ids.defined() ? candidate_ids.gather(1, std::get<1>(top)) : std::get<1>(top)).to(torch::kInt32);
    return result;
}

std::vector<std::optional<ErrorInfo>> Sampler::preprocessLogits(const SamplerInputs& inputs) {
    if (inputs.logits_processor_states_ptr != nullptr) {
        return inputs.logits_processor_states_ptr->batchProcess(inputs);
    }
    return {};
}

}  // namespace rtp_llm

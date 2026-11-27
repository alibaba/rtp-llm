#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeKernels.cuh"
#endif

namespace rtp_llm {
namespace speculative {

FastTopKSamplerOutput FastTopKSampler::forward(const torch::Tensor& logits, int top_k) {
    FastTopKSamplerOutput output;

    if (top_k == 1) {
        output.token_ids = torch::argmax(logits, -1, true);
        output.all_probs = torch::zeros_like(logits).scatter_(-1, output.token_ids, 1.0);
    } else {
        auto draft_probs = torch::softmax(logits, -1);
        output.token_ids = std::get<1>(torch::topk(draft_probs, top_k, -1));
        output.all_probs = std::move(draft_probs);
    }

    int batch_size = output.token_ids.size(0);
    execMappingDraft2Target({output.token_ids, d2t_map_, batch_size, 0, 1});
    output.token_ids = output.token_ids.to(torch::kInt32);

    return output;
}

SamplerOutput SpeculativeSampler::sampleDSparkDraft(const torch::Tensor& base_logits,
                                                    const torch::Tensor& anchors,
                                                    const torch::Tensor& temperature,
                                                    const torch::Tensor& markov_w1,
                                                    const torch::Tensor& markov_w2,
                                                    size_t               draft_vocab_size) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.sample_dspark_draft");
    RTP_LLM_CHECK_WITH_INFO(temperature.defined() && temperature.is_cuda() && temperature.is_contiguous()
                                && temperature.scalar_type() == torch::kFloat32 && temperature.dim() == 1,
                            "DSpARK draft temperatures must be contiguous CUDA FP32 [B]");
    const auto batch_size = temperature.numel();
    RTP_LLM_CHECK_WITH_INFO(base_logits.defined() && base_logits.is_cuda() && base_logits.stride(1) == 1
                                && (base_logits.scalar_type() == torch::kFloat32
                                    || base_logits.scalar_type() == torch::kBFloat16)
                                && base_logits.dim() == 2
                                && base_logits.size(0) == batch_size * static_cast<int64_t>(propose_step_)
                                && base_logits.size(1) >= static_cast<int64_t>(draft_vocab_size),
                            "DSpARK C++ lm_head must emit row-contiguous CUDA FP32/BF16 [B*gamma,vocab_padded] logits "
                            "with vocab_padded >= draft vocab size");
    RTP_LLM_CHECK_WITH_INFO(anchors.defined() && anchors.is_cuda() && anchors.numel() == batch_size,
                            "DSpARK anchors must be a CUDA tensor with one token per request");

    const auto gamma      = static_cast<int64_t>(propose_step_);
    const auto draft_v    = static_cast<int64_t>(draft_vocab_size);
    const auto cuda_f32   = torch::TensorOptions().dtype(torch::kFloat32).device(base_logits.device());
    const auto cuda_i32   = torch::TensorOptions().dtype(torch::kInt32).device(base_logits.device());
    auto all_probabilities = torch::empty({batch_size, gamma, draft_v}, cuda_f32);
    if (batch_size == 0 || gamma == 0 || draft_v == 0) {
        SamplerOutput output;
        output.token_ids                = torch::empty({batch_size, gamma}, cuda_i32);
        output.all_probs                = std::move(all_probabilities);
        output.token_ids_are_point_mass = false;
        return output;
    }

#if USING_CUDA
    auto token_ids       = torch::empty({batch_size, gamma}, cuda_i32);
    auto anchors_i32     = anchors.scalar_type() == torch::kInt32 ? anchors.reshape({batch_size}) :
                                                                   anchors.reshape({batch_size}).to(torch::kInt32);
    auto markov_w2_t     = markov_w2.transpose(0, 1);
    auto markov_emb_buf  = torch::empty({batch_size, markov_w1.size(1)}, markov_w1.options());
    auto markov_bias_buf = torch::empty({batch_size, draft_v}, markov_w1.options());
    auto step_logits_buf = batch_size == 1 ? torch::Tensor() : torch::empty({batch_size, draft_v}, cuda_f32);

    for (int64_t step = 0; step < gamma; ++step) {
        auto previous_tokens = step == 0 ? anchors_i32 : token_ids.select(1, step - 1);
        launchGatherMarkovEmbedding(markov_w1, previous_tokens, markov_emb_buf);
        torch::mm_out(markov_bias_buf, markov_emb_buf, markov_w2_t);

        torch::Tensor step_target = batch_size == 1 ? all_probabilities.select(1, step) : step_logits_buf;
        launchDSparkAddMarkovDivTemp(base_logits, markov_bias_buf, temperature, step_target, step, gamma, draft_v);

        // Draft q applies request temperature only. Materialize that exact
        // dense distribution once, sample from it with FlashInfer, and pass
        // the same q to rejection sampling. Request top-k/top-p stay target-side.
        auto sampling_probabilities = execSamplerSoftmaxInplace(step_target);
        auto sampled_draft_tokens   = execSampleFromProbs(sampling_probabilities);
        if (sampled_draft_tokens.scalar_type() != torch::kInt32) {
            sampled_draft_tokens = sampled_draft_tokens.to(torch::kInt32);
        }
        auto step_token_out = token_ids.select(1, step);
        if (d2t_map_.defined()) {
            if (d2t_map_.is_cuda()) {
                launchMapDraftToTargetInt32(sampled_draft_tokens, d2t_map_, step_token_out);
            } else {
                step_token_out.copy_(
                    d2t_map_.index_select(0, sampled_draft_tokens.cpu().to(torch::kLong))
                        .to(torch::kInt32)
                        .to(base_logits.device(), /*non_blocking=*/true));
            }
        } else {
            step_token_out.copy_(sampled_draft_tokens, /*non_blocking=*/true);
        }
        if (batch_size != 1 || sampling_probabilities.data_ptr() != step_target.data_ptr()) {
            all_probabilities.select(1, step).copy_(sampling_probabilities, /*non_blocking=*/true);
        }
    }

    SamplerOutput output;
    output.token_ids                = std::move(token_ids);
    output.all_probs                = std::move(all_probabilities);
    output.token_ids_are_point_mass = false;
    return output;
#else
    auto previous_tokens = anchors.reshape({batch_size}).to(torch::kLong);
    std::vector<torch::Tensor> token_columns;
    token_columns.reserve(propose_step_);
    auto proposal_logits = base_logits.narrow(1, 0, draft_vocab_size).view({batch_size, gamma, draft_v});
    if (proposal_logits.scalar_type() != torch::kFloat32) {
        proposal_logits = proposal_logits.to(torch::kFloat32);
    }
    auto temperature_column = temperature.unsqueeze(1);

    for (int64_t step = 0; step < gamma; ++step) {
        auto markov_embedding = markov_w1.index_select(0, previous_tokens);
        auto markov_bias      = torch::mm(markov_embedding, markov_w2.transpose(0, 1)).to(torch::kFloat32);
        auto logits           = proposal_logits.select(1, step) + markov_bias;

        logits.div_(temperature_column);
        auto sampling_probabilities = execSamplerSoftmaxInplace(logits);
        auto sampled_draft_tokens   = execSampleFromProbs(sampling_probabilities).to(torch::kInt32);
        auto sampled_target_tokens  = sampled_draft_tokens;
        if (d2t_map_.defined()) {
            sampled_target_tokens = d2t_map_.index_select(0, sampled_draft_tokens.to(torch::kLong)).to(torch::kInt32);
        }
        all_probabilities.select(1, step).copy_(sampling_probabilities);
        token_columns.push_back(sampled_target_tokens);
        previous_tokens = sampled_target_tokens.to(torch::kLong);
    }

    SamplerOutput output;
    output.token_ids                = torch::stack(token_columns, 1).contiguous();
    output.all_probs                = std::move(all_probabilities);
    output.token_ids_are_point_mass = false;
    return output;
#endif
}

torch::Tensor dflashDraftProbabilities(const torch::Tensor& logits, const GenerateConfig& config) {
    TORCH_CHECK(logits.dim() == 2 && logits.size(1) > 0, "DFlash logits must be [tokens,vocab]");
    auto scores = logits.to(torch::kFloat32);
    // Greedy requests never read the sampling parameters, so settle them first:
    // the repo's own smoke load sends {temperature: 0, top_p: 0} and admission
    // accepts it (see case_runner.py).
    if (!config.stochastic() || config.temperature == 0.0f) {
        return torch::zeros_like(scores).scatter_(-1, scores.argmax(-1, true), 1.0);
    }
    // Admission lets unusable sampling parameters through: the OpenAI endpoint
    // passes top_p = 0 / > 1 through unchanged and is_positive_number accepts
    // inf temperatures. This function runs inside the engine loop thread, so
    // instead of failing the whole engine, normalize admission-tolerant values
    // the same way MtpExecutor::decodeStep sanitizes DSpARK temperatures: an
    // invalid temperature or top_p falls back to 1.0 (no temperature skew, no
    // nucleus cut) and a non-positive top_k means "no top-k filtering".
    constexpr float kMinDraftTemperature = 1.0e-6f;
    auto temperature =
        std::isfinite(config.temperature) && config.temperature > 0.0f ? config.temperature : 1.0f;
    // Tiny positive temperatures (e.g. denormals) survive the normalization
    // above but overflow scores/temperature to inf, which turns softmax into
    // NaN; multinomial on NaN then fails in the engine loop thread or samples
    // a garbage token. Apply the same floor as the DSpARK sanitize.
    temperature = std::max(temperature, kMinDraftTemperature);
    const auto top_k      = config.top_k > 0 ? config.top_k : 0;
    const auto top_p      = std::isfinite(config.top_p) && config.top_p > 0.0f && config.top_p <= 1.0f ? config.top_p : 1.0f;
    scores                = scores / temperature;
    torch::Tensor indices;
    if (top_k > 0 && top_k < scores.size(1)) {
        auto top = scores.topk(top_k, -1);
        scores   = std::get<0>(top);
        indices  = std::get<1>(top);
    }
    auto probabilities = torch::softmax(scores, -1);
    if (top_p < 1.0f) {
        auto sorted               = probabilities.sort(-1, /*descending=*/true);
        auto sorted_probabilities = std::get<0>(sorted);
        // Keep the token crossing the nucleus threshold, as in the reference.
        auto keep            = sorted_probabilities.cumsum(-1) - sorted_probabilities < config.top_p;
        sorted_probabilities = sorted_probabilities * keep;
        probabilities        = torch::zeros_like(probabilities).scatter_(-1, std::get<1>(sorted), sorted_probabilities);
        probabilities        = probabilities / probabilities.sum(-1, true);
    }
    if (indices.defined()) {
        probabilities = torch::zeros_like(logits, probabilities.options()).scatter_(-1, indices, probabilities);
    }
    return probabilities;
}

SamplerOutput SpeculativeSampler::sampleDFlashDraft(const torch::Tensor&                base_logits,
                                                    const std::list<GenerateStreamPtr>& streams,
                                                    size_t                              draft_vocab_size) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.sample_dflash_draft");
    const auto batch_size = static_cast<int64_t>(streams.size());
    TORCH_CHECK(!d2t_map_.defined(), "DFlash samples in the full target vocabulary");
    TORCH_CHECK(batch_size > 0 && propose_step_ > 0 && base_logits.dim() == 2
                    && base_logits.size(0) == batch_size * static_cast<int64_t>(propose_step_)
                    && base_logits.size(1) >= static_cast<int64_t>(draft_vocab_size),
                "DFlash logits must contain only mask rows: [B*gamma,vocab_padded]");
    auto logits =
        base_logits.narrow(1, 0, draft_vocab_size)
            .reshape({batch_size, static_cast<int64_t>(propose_step_), static_cast<int64_t>(draft_vocab_size)});
    std::vector<torch::Tensor> probability_rows;
    std::vector<torch::Tensor> token_rows;
    probability_rows.reserve(batch_size);
    token_rows.reserve(batch_size);
    bool    all_greedy = true;
    int64_t row        = 0;
    for (const auto& stream : streams) {
        TORCH_CHECK(stream->maxBatchSize() == 1 && !stream->hasNumBeams(),
                    "DFlash does not support tiled or beam sampling");
        const auto config = stream->generateConfig();
        // Greedy requests short-circuit inside dflashDraftProbabilities and
        // never read the sampling parameters, so only stochastic requests need
        // the admission-tolerant sanitize; mirror MtpExecutor::decodeStep's
        // DSpARK handling and log the substitution so a misconfigured client
        // stays visible to operators instead of failing silently.
        if (config->stochastic() && config->temperature != 0.0f) {
            if (!std::isfinite(config->temperature) || config->temperature < 0.0f) {
                RTP_LLM_LOG_WARNING("DFlash received invalid sampling temperature=%g for stream=%ld; using 1.0",
                                    config->temperature,
                                    stream->streamId());
                config->temperature = 1.0f;
            }
            if (config->top_k < 0) {
                RTP_LLM_LOG_WARNING("DFlash received invalid top_k=%d for stream=%ld; using 0",
                                    config->top_k,
                                    stream->streamId());
                config->top_k = 0;
            }
            if (!std::isfinite(config->top_p) || config->top_p <= 0.0f || config->top_p > 1.0f) {
                RTP_LLM_LOG_WARNING("DFlash received invalid top_p=%g for stream=%ld; using 1.0",
                                    config->top_p,
                                    stream->streamId());
                config->top_p = 1.0f;
            }
        }
        auto       probabilities = dflashDraftProbabilities(logits[row++], *config);
        const bool greedy       = !config->stochastic() || config->temperature == 0.0f;
        all_greedy                = all_greedy && greedy;
        torch::Tensor tokens;
        if (greedy) {
            tokens = probabilities.argmax(-1);
        } else {
            auto generator = stream->getGenerator();
            tokens         = torch::multinomial(probabilities,
                                        1,
                                        false,
                                        generator.defined() ? std::optional<at::Generator>(generator) : std::nullopt)
                         .squeeze(-1);
        }
        token_rows.push_back(tokens.to(torch::kInt32));
        probability_rows.push_back(std::move(probabilities));
    }
    SamplerOutput output;
    output.token_ids                = torch::stack(token_rows).contiguous();
    output.token_ids_are_point_mass = all_greedy;
    if (!all_greedy) {
        // Mixed batches retain explicit one-hot q for their greedy rows.
        output.all_probs = torch::stack(probability_rows).contiguous();
    }
    return output;
}

SpeculativeSamplerOutput SpeculativeSampler::forward(const SpeculativeSamplingParams& params,
                                                     SamplerOutput&                   draft_sampler_output,
                                                     SamplerOutput&                   target_sampler_output) {
    // TensorHolder release point (SpeculativeSampler): advances host tensors
    // staged for rejection sampling H2D in the previous forward.
    buffer_holder_.release();
    SpeculativeSamplerOutput sample_output;
    batchSample(sample_output, params, draft_sampler_output, target_sampler_output);

    return sample_output;
}

SpeculativeSamplerOutput SpeculativeSampler::forwardWithSpecCap(const SpeculativeSamplingParams& params,
                                                                SamplerOutput&                   draft_sampler_output,
                                                                SamplerOutput&                   target_sampler_output,
                                                                const PrepareSpecCapFn&          prepare_spec_cap,
                                                                bool&                            fused_cap_applied) {
    buffer_holder_.release();
    SpeculativeSamplerOutput sample_output;
    fused_cap_applied = false;
    batchSample(
        sample_output, params, draft_sampler_output, target_sampler_output, &prepare_spec_cap, &fused_cap_applied);
    return sample_output;
}

void SpeculativeSampler::batchSample(SpeculativeSamplerOutput&        sample_output,
                                     const SpeculativeSamplingParams& params,
                                     SamplerOutput&                   draft_sampler_output,
                                     SamplerOutput&                   target_sampler_output,
                                     const PrepareSpecCapFn*          prepare_spec_cap,
                                     bool*                            fused_cap_applied) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample");
    torch::Device target_device = getTorchCudaDevice();

    const int batch_size = params.do_sample.size(0);

    auto draft_token_ids  = draft_sampler_output.token_ids;
    auto target_token_ids = target_sampler_output.token_ids;

    auto       draft_token_probs      = draft_sampler_output.all_probs;
    auto       target_token_probs     = target_sampler_output.all_probs;
    const bool draft_probs_point_mass = draft_sampler_output.token_ids_are_point_mass;

    buffer_holder_.hold_host(draft_token_ids);
    auto draft_token_ids_d_t = draft_token_ids.to(target_device, true);

    auto target_token_ids_d_t = target_sampler_output.token_ids;
    if (!target_token_ids_d_t.is_cuda()) {
        buffer_holder_.hold_host(target_token_ids_d_t);
        target_token_ids_d_t = target_token_ids_d_t.to(target_device, true);
    }

    const auto& do_sample = params.do_sample;
    buffer_holder_.hold_host(do_sample);
    auto do_sample_d = do_sample.to(target_device, true);

    auto          rand_options      = torch::TensorOptions().device(target_device).dtype(torch::kFloat);
    torch::Tensor uniform_samples_d = torch::rand({(long)batch_size, (long)propose_step_ + 1}, rand_options);

    // Override per-stream uniform samples with seeded generator when random_seed is set,
    // ensuring deterministic acceptance for reproducible iter_count.
    {
        int idx = 0;
        for (const auto& gen : params.generators) {
            if (gen.defined()) {
                uniform_samples_d[idx] = torch::rand({(long)propose_step_ + 1}, gen, std::nullopt, rand_options);
            }
            idx++;
        }
    }

    auto          draft_token_probs_d_t  = draft_token_probs;
    auto          target_token_probs_d_t = target_token_probs;
    torch::Tensor output_token_ids_d =
        torch::zeros({(long)batch_size, (long)propose_step_ + 1},
                     torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));
    torch::Tensor output_accepted_token_num_d = torch::zeros(
        {(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));

    RTP_LLM_CHECK_WITH_INFO(draft_probs_point_mass
                                || (draft_token_probs_d_t.defined() && draft_token_probs_d_t.dim() == 3),
                            "draft probabilities must be [B, steps, vocab] unless token ids define a point mass");
    if (!draft_probs_point_mass && draft_token_probs_d_t.size(2) != target_token_probs_d_t.size(2)) {
        const int64_t target_vocab_size = target_token_probs_d_t.size(2);
        const int64_t num_spec          = draft_token_probs_d_t.size(1);

        // Reuse pre-allocated padding buffer to avoid per-forward GPU allocation.
        // Grow-only along batch / num_spec dims; vocab dim must match exactly.
        const bool need_realloc = !draft_probs_padding_buffer_.defined()
                                  || draft_probs_padding_buffer_.size(0) < (int64_t)batch_size
                                  || draft_probs_padding_buffer_.size(1) < num_spec
                                  || draft_probs_padding_buffer_.size(2) != target_vocab_size
                                  || draft_probs_padding_buffer_.dtype() != draft_token_probs_d_t.dtype()
                                  || draft_probs_padding_buffer_.device() != draft_token_probs_d_t.device();
        if (need_realloc) {
            const int64_t cap_b =
                std::max((int64_t)batch_size,
                         draft_probs_padding_buffer_.defined() ? draft_probs_padding_buffer_.size(0) : (int64_t)0);
            const int64_t cap_s = std::max(
                num_spec, draft_probs_padding_buffer_.defined() ? draft_probs_padding_buffer_.size(1) : (int64_t)0);
            draft_probs_padding_buffer_ =
                torch::zeros({cap_b, cap_s, target_vocab_size}, draft_token_probs_d_t.options());
        }

        auto draft_probs_padding = draft_probs_padding_buffer_.narrow(0, 0, (int64_t)batch_size).narrow(1, 0, num_spec);
        draft_probs_padding.zero_();
        if (d2t_map_.defined()) {
            draft_probs_padding.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), d2t_map_},
                                           draft_token_probs_d_t);
        } else {
            // Full-vocabulary DFlash omits synthetic lm-head padding from q.
            // The target sampler retains padded columns, each with zero mass.
            RTP_LLM_CHECK_WITH_INFO(draft_token_probs_d_t.size(2) <= target_vocab_size,
                                    "draft vocabulary exceeds target vocabulary without a d2t map");
            draft_probs_padding.narrow(2, 0, draft_token_probs_d_t.size(2)).copy_(draft_token_probs_d_t);
        }
        draft_token_probs_d_t = draft_probs_padding;
    }

    {
        RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.execRejectionSampling");
        execRejectionSampling({
            draft_token_probs_d_t,
            draft_token_ids_d_t,
            uniform_samples_d,
            target_token_probs_d_t,
            target_token_ids_d_t,
            output_token_ids_d,
            output_accepted_token_num_d,
            do_sample_d,
            draft_probs_point_mass,
        });
    }

    RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling");

    bool          has_force    = false;
    torch::Tensor force_mask_d;
    if (params.force_accept.defined() && batch_size > 0) {
        auto        force_src = params.force_accept.contiguous();
        const auto* force_ptr = force_src.data_ptr<bool>();
        for (int idx = 0; idx < batch_size; ++idx) {
            if (force_ptr[idx]) {
                has_force = true;
                break;
            }
        }
        if (has_force) {
            auto force_mask_h = torch::empty(
                {(long)batch_size}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
            std::memcpy(force_mask_h.data_ptr<bool>(), force_ptr, static_cast<size_t>(batch_size) * sizeof(bool));
            buffer_holder_.hold_host(force_mask_h);
            force_mask_d = force_mask_h.to(target_device, /*non_blocking=*/true);
        }
    }

    torch::Tensor cap_gpu;
    if (prepare_spec_cap != nullptr && *prepare_spec_cap) {
        cap_gpu = (*prepare_spec_cap)(target_device);
    }

#if USING_CUDA
    if (output_token_ids_d.is_cuda() && output_accepted_token_num_d.is_cuda() && output_token_ids_d.is_contiguous()
        && output_accepted_token_num_d.is_contiguous()
        && (!cap_gpu.defined()
            || (cap_gpu.is_cuda() && cap_gpu.is_contiguous()
                && (cap_gpu.scalar_type() == torch::kInt32 || cap_gpu.scalar_type() == torch::kInt64)))) {
        auto draft_for_force = has_force ? draft_token_ids_d_t.contiguous() : torch::Tensor();
        auto target_for_post = (has_force || cap_gpu.defined()) ?
                                   (target_token_ids_d_t.scalar_type() == torch::kInt32 ?
                                        target_token_ids_d_t :
                                        target_token_ids_d_t.to(torch::kInt32)) :
                                   torch::Tensor();
        launchPostRejectionAndCap(output_token_ids_d,
                                  output_accepted_token_num_d,
                                  output_token_ids_d,
                                  output_accepted_token_num_d,
                                  force_mask_d,
                                  draft_for_force,
                                  target_for_post,
                                  cap_gpu,
                                  /*clamp_minus_one=*/true);
        if (fused_cap_applied != nullptr) {
            *fused_cap_applied = cap_gpu.defined();
        }
        sample_output.accept_tokens = output_token_ids_d;
        sample_output.accept_len    = output_accepted_token_num_d;
        return;
    }
#endif

    // forceSpAccept: override rejection sampling results for streams that requested
    // forced acceptance — accept all draft tokens plus the target bonus token.
    if (has_force) {
        RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling.forceSpAccept");
        // target_token_ids_d_t layout: [batch_size * (propose_step+1), token_stride]
        // Extract the bonus token at position propose_step for each batch item.
        int64_t token_stride = target_token_ids_d_t.size(1);
        auto    target_bonus_t =
            target_token_ids_d_t.reshape({(long)batch_size, (long)(propose_step_ + 1), token_stride});
        auto target_bonus = target_bonus_t.select(1, propose_step_).select(1, token_stride - 1).unsqueeze(1);
        // forced_tokens: draft tokens [0..propose_step-1] + target bonus
        auto forced_tokens          = torch::cat({draft_token_ids_d_t, target_bonus}, 1);
        auto force_mask_2d          = force_mask_d.unsqueeze(1).expand_as(output_token_ids_d);
        output_token_ids_d          = torch::where(force_mask_2d, forced_tokens, output_token_ids_d);
        output_accepted_token_num_d = torch::where(
            force_mask_d,
            torch::full_like(output_accepted_token_num_d, (int32_t)(propose_step_ + 1)),
            output_accepted_token_num_d);
    }

    // use async sample here, we assume accept all tokens
    // so we need to reset -1 to 0 in output_token_ids_d
    output_token_ids_d.index_put_({output_token_ids_d == -1}, 0);
    sample_output.accept_tokens = output_token_ids_d;
    sample_output.accept_len    = output_accepted_token_num_d;
}

}  // namespace speculative
}  // namespace rtp_llm

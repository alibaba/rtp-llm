#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <algorithm>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <vector>
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#endif

namespace rtp_llm {
namespace speculative {

FastTopKSamplerOutput FastTopKSampler::forward(const torch::Tensor& logits, int top_k) {
    FastTopKSamplerOutput output;

    if (proposal_mode_ == DraftProposalMode::DETERMINISTIC) {
        RTP_LLM_CHECK_WITH_INFO(top_k == 1, "deterministic draft requires top_k=1, got %d", top_k);
        output.token_ids = std::get<1>(torch::max(logits, -1, true));
        // The draft token is selected deterministically. Rejection sampling must
        // therefore see the actual point-mass proposal distribution. Keep this
        // tensor draft-vocabulary aligned; batchSample owns d2t probability
        // remapping when draft and target vocabularies differ.
        output.all_probs = torch::zeros_like(logits).scatter_(-1, output.token_ids, 1.0);
    } else {
        output.all_probs = torch::softmax(logits, -1);
        output.token_ids = top_k == 1 ? std::get<1>(torch::max(output.all_probs, -1, true)) :
                                        std::get<1>(torch::topk(output.all_probs, top_k, -1));
    }

    int batch_size = output.token_ids.size(0);
    execMappingDraft2Target({output.token_ids, d2t_map_, batch_size, 0, 1});

    return output;
}

SamplerOutput SpeculativeSampler::sampleDSparkDraft(const torch::Tensor& base_logits,
                                                    const torch::Tensor& anchors,
                                                    const torch::Tensor& temperature,
                                                    const torch::Tensor& markov_w1,
                                                    const torch::Tensor& markov_w2,
                                                    size_t               draft_vocab_size,
                                                    size_t               sampled_prefix_steps) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.sample_dspark_draft");
    const size_t prefix_steps = sampled_prefix_steps == 0 ? propose_step_ : sampled_prefix_steps;
    RTP_LLM_CHECK_WITH_INFO(prefix_steps > 0 && prefix_steps <= propose_step_,
                            "DSpARK sampled prefix must be in [1,gamma]");
    RTP_LLM_CHECK_WITH_INFO(temperature.defined() && temperature.is_cuda() && temperature.is_contiguous()
                                && temperature.scalar_type() == torch::kFloat32 && temperature.dim() == 1,
                            "DSpARK draft temperatures must be contiguous CUDA FP32 [B]");
    const auto batch_size = temperature.numel();
    RTP_LLM_CHECK_WITH_INFO(base_logits.defined() && base_logits.is_cuda() && base_logits.is_contiguous()
                                && base_logits.scalar_type() == torch::kFloat32 && base_logits.dim() == 2
                                && base_logits.size(0) == batch_size * static_cast<int64_t>(propose_step_)
                                && base_logits.size(1) >= static_cast<int64_t>(draft_vocab_size),
                            "DSpARK C++ lm_head must emit contiguous CUDA FP32 [B*gamma,vocab_padded] logits with "
                            "vocab_padded >= draft vocab size");
    RTP_LLM_CHECK_WITH_INFO(anchors.defined() && anchors.is_cuda() && anchors.numel() == batch_size,
                            "DSpARK anchors must be a CUDA tensor with one token per request");
    RTP_LLM_CHECK_WITH_INFO(markov_w1.defined() && markov_w1.is_cuda() && markov_w1.dim() == 2 && markov_w1.size(0) > 0,
                            "DSpARK markov_w1 must be CUDA [target_vocab, rank]");
    RTP_LLM_CHECK_WITH_INFO(markov_w2.defined() && markov_w2.is_cuda() && markov_w2.dim() == 2
                                && markov_w2.size(0) == static_cast<int64_t>(draft_vocab_size)
                                && markov_w2.size(1) == markov_w1.size(1),
                            "DSpARK markov_w2 must be CUDA [draft_vocab, rank]");

    // The backbone and LM head still produce all gamma rows. Only CUDA's
    // FlashInfer path has a known per-call RNG reservation contract. Keep the
    // full legacy loop for HIP and empty batches, then select the prefix.
    size_t sampling_steps = propose_step_;
#if USING_CUDA
    if (batch_size > 0) {
        sampling_steps = prefix_steps;
    }
#endif
    auto previous_tokens = anchors.reshape({batch_size}).to(torch::kLong);
    auto all_probabilities =
        torch::empty({batch_size, static_cast<int64_t>(sampling_steps), static_cast<int64_t>(draft_vocab_size)},
                     torch::TensorOptions().dtype(torch::kFloat32).device(base_logits.device()));
    std::vector<torch::Tensor> token_columns;
    token_columns.reserve(sampling_steps);
    auto proposal_logits =
        base_logits.narrow(1, 0, draft_vocab_size)
            .view({batch_size, static_cast<int64_t>(propose_step_), static_cast<int64_t>(draft_vocab_size)});

    for (int64_t step = 0; step < static_cast<int64_t>(sampling_steps); ++step) {
        auto markov_embedding = markov_w1.index_select(0, previous_tokens);
        auto markov_bias      = torch::mm(markov_embedding, markov_w2.transpose(0, 1));
        auto logits           = execDSparkCombineLogits(proposal_logits.select(1, step), markov_bias, temperature);
        auto sampling_probabilities = torch::softmax(logits, -1);
        auto sampled_draft_tokens   = execSampleFromProbs(sampling_probabilities).to(torch::kInt32);
        auto sampled_target_tokens  = sampled_draft_tokens;
        if (d2t_map_.defined()) {
            sampled_target_tokens = d2t_map_.index_select(0, sampled_draft_tokens.to(torch::kLong)).to(torch::kInt32);
        }
        all_probabilities.select(1, step).copy_(sampling_probabilities);
        token_columns.push_back(sampled_target_tokens);
        previous_tokens = sampled_target_tokens.to(torch::kLong);
    }

    if (sampling_steps < propose_step_) {
        // Preserve the RNG position seen by target sampling and later rounds.
        // This must run after the retained prefix, on the sampling rank only.
        execReserveSampleFromProbsRng(base_logits, static_cast<int64_t>(propose_step_ - sampling_steps));
    }

    SamplerOutput output;
    output.token_ids = torch::stack(token_columns, 1).contiguous();
    output.all_probs = std::move(all_probabilities);
    if (prefix_steps < sampling_steps) {
        output.token_ids = output.token_ids.narrow(1, 0, prefix_steps).contiguous();
        output.all_probs = output.all_probs.narrow(1, 0, prefix_steps).contiguous();
    }
    return output;
}

torch::Tensor SpeculativeSampler::computeDSparkConfidence(const torch::Tensor& draft_hidden,
                                                          const torch::Tensor& anchors,
                                                          const torch::Tensor& sampled_tokens,
                                                          const torch::Tensor& markov_w1,
                                                          const torch::Tensor& confidence_w,
                                                          const torch::Tensor& confidence_b) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.compute_dspark_confidence");
    RTP_LLM_CHECK_WITH_INFO(sampled_tokens.defined() && sampled_tokens.dim() == 2
                                && sampled_tokens.size(1) == static_cast<int64_t>(propose_step_),
                            "DSpARK confidence requires all gamma sampled tokens");
    RTP_LLM_CHECK_WITH_INFO(draft_hidden.defined() && draft_hidden.dim() == 2
                                && draft_hidden.size(0) == sampled_tokens.numel(),
                            "DSpARK confidence hidden rows must match B*gamma");
    return execDSparkConfidence(draft_hidden, anchors, sampled_tokens, markov_w1, confidence_w, confidence_b);
}

SpeculativeSamplerOutput SpeculativeSampler::forward(const std::list<GenerateStreamPtr>& streams,
                                                     SamplerOutput&                      draft_sampler_output,
                                                     SamplerOutput&                      target_sampler_output,
                                                     const torch::Tensor&                active_verify_lengths) {
    // TensorHolder release point (SpeculativeSampler): advances host tensors
    // staged for rejection sampling H2D in the previous forward.
    buffer_holder_.release();
    SpeculativeSamplerOutput sample_output;
    batchSample(sample_output, streams, draft_sampler_output, target_sampler_output, active_verify_lengths);

    return sample_output;
}

void SpeculativeSampler::batchSample(SpeculativeSamplerOutput&           sample_output,
                                     const std::list<GenerateStreamPtr>& streams,
                                     SamplerOutput&                      draft_sampler_output,
                                     SamplerOutput&                      target_sampler_output,
                                     const torch::Tensor&                active_verify_lengths) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample");
    torch::Device target_device = getTorchCudaDevice();

    int batch_size = streams.size();

    auto draft_token_ids  = draft_sampler_output.token_ids;
    auto target_token_ids = target_sampler_output.token_ids;

    auto draft_token_probs  = draft_sampler_output.all_probs;
    auto target_token_probs = target_sampler_output.all_probs;

    buffer_holder_.hold_host(draft_token_ids);
    auto draft_token_ids_d_t = draft_token_ids.to(target_device, true);

    auto target_token_ids_d_t = target_sampler_output.token_ids;
    if (!target_token_ids_d_t.is_cuda()) {
        buffer_holder_.hold_host(target_token_ids_d_t);
        target_token_ids_d_t = target_token_ids_d_t.to(target_device, true);
    }

    torch::Tensor do_sample =
        torch::zeros({(long)batch_size}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
    int stream_idx = 0;
    for (const GenerateStreamPtr& stream : streams) {
        do_sample[stream_idx++] = !stream->generateConfig()->top1();
    }
    buffer_holder_.hold_host(do_sample);
    auto do_sample_d = do_sample.to(target_device, true);

    auto          rand_options = torch::TensorOptions().device(target_device).dtype(torch::kFloat);
    torch::Tensor uniform_samples_d;
    if (proposal_mode_ == DraftProposalMode::DETERMINISTIC) {
        // Exact-match reuses target_sampler_output and must not advance the
        // per-request generator a second time.
        uniform_samples_d = torch::zeros({(long)batch_size, (long)propose_step_ + 1}, rand_options);
    } else {
        uniform_samples_d = torch::rand({(long)batch_size, (long)propose_step_ + 1}, rand_options);
        int idx           = 0;
        for (const auto& stream : streams) {
            auto gen = stream->getGenerator();
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

    if (draft_token_probs_d_t.size(2) != target_token_probs_d_t.size(2)) {
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
        draft_probs_padding.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), d2t_map_},
                                       draft_token_probs_d_t);
        draft_token_probs_d_t = draft_probs_padding;
    }

    {
        RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.execRejectionSampling");
        torch::Tensor validation_workspace;
        if (proposal_mode_ == DraftProposalMode::SAMPLED) {
            sample_output.success = torch::ones({batch_size}, do_sample_d.options());
#if USING_CUDA
            const auto elements =
                rejectionValidationWorkspaceElements(batch_size, propose_step_, target_token_probs_d_t.size(2));
            auto&                   reserved = validation_workspaces_[cuda_graph::graphGetCurrentStream().id()];
            cudaStreamCaptureStatus capture_status;
            RTP_LLM_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(), &capture_status)
                          == cudaSuccess);
            if (!reserved.tensor.defined() || reserved.tensor.numel() < elements
                || reserved.tensor.device() != target_token_probs_d_t.device()) {
                RTP_LLM_CHECK_WITH_INFO(!reserved.captured && capture_status == cudaStreamCaptureStatusNone,
                                        "DSpARK rejection workspace must be warmed to maximum batch before capture");
                reserved.tensor = torch::empty({elements}, target_token_probs_d_t.options());
            }
            reserved.captured    = reserved.captured || capture_status != cudaStreamCaptureStatusNone;
            validation_workspace = reserved.tensor;
            validation_workspace.record_stream(cuda_graph::graphGetCurrentStream());
#endif
        }
        execRejectionSampling({
            draft_token_probs_d_t,
            draft_token_ids_d_t,
            uniform_samples_d,
            target_token_probs_d_t,
            target_token_ids_d_t,
            output_token_ids_d,
            output_accepted_token_num_d,
            do_sample_d,
            proposal_mode_ == DraftProposalMode::DETERMINISTIC,
            proposal_mode_ == DraftProposalMode::SAMPLED,
            sample_output.success,
            target_sampler_output.success,
            active_verify_lengths,
            validation_workspace,
        });
    }

    RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling");

    // forceSpAccept: override rejection sampling results for streams that requested
    // forced acceptance — accept all draft tokens plus the target bonus token.
    {
        const bool has_force =
            std::any_of(streams.begin(), streams.end(), [](const auto& stream) { return stream->forceSpAccept(); });
        if (has_force) {
            RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling.forceSpAccept");
            auto force_mask =
                torch::zeros({(long)batch_size}, torch::TensorOptions().dtype(torch::kBool).device(target_device));
            int idx = 0;
            for (const auto& stream : streams) {
                if (stream->forceSpAccept()) {
                    force_mask[idx] = true;
                }
                idx++;
            }
            // target_token_ids_d_t layout: [batch_size * (propose_step+1), token_stride]
            // Extract the bonus token at position propose_step for each batch item.
            int64_t token_stride = target_token_ids_d_t.size(1);
            auto    target_bonus_t =
                target_token_ids_d_t.reshape({(long)batch_size, (long)(propose_step_ + 1), token_stride});
            auto target_bonus = target_bonus_t.select(1, propose_step_).select(1, token_stride - 1).unsqueeze(1);
            // forced_tokens: draft tokens [0..propose_step-1] + target bonus
            auto forced_tokens = torch::cat({draft_token_ids_d_t, target_bonus}, 1);
            auto force_mask_2d = force_mask.unsqueeze(1).expand_as(output_token_ids_d);
            output_token_ids_d = torch::where(force_mask_2d, forced_tokens, output_token_ids_d);
            output_accepted_token_num_d =
                torch::where(force_mask,
                             torch::full_like(output_accepted_token_num_d, (int32_t)(propose_step_ + 1)),
                             output_accepted_token_num_d);
            // The perf-only override must not erase a failure detected by
            // normal rejection sampling.
            if (sample_output.success.defined()) {
                output_accepted_token_num_d = torch::where(
                    sample_output.success, output_accepted_token_num_d, torch::ones_like(output_accepted_token_num_d));
                output_token_ids_d = torch::where(
                    sample_output.success.unsqueeze(1), output_token_ids_d, torch::zeros_like(output_token_ids_d));
            }
        }
    }

    // use async sample here, we assume accept all tokens
    // so we need to reset -1 to 0 in output_token_ids_d
    output_token_ids_d.index_put_({output_token_ids_d == -1}, 0);
    sample_output.accept_tokens = output_token_ids_d;
    sample_output.accept_len    = output_accepted_token_num_d;

    sample_output.accept_tokens_cpu = sample_output.accept_tokens.to(torch::kCPU, true);
    sample_output.accept_len_cpu    = sample_output.accept_len.to(torch::kCPU, true);
    if (sample_output.success.defined()) {
        sample_output.success_cpu = sample_output.success.to(torch::kCPU, true);
    }
    sample_output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}

}  // namespace speculative
}  // namespace rtp_llm

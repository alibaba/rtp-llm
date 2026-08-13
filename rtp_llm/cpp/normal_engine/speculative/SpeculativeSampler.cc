#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <algorithm>
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {
namespace speculative {

FastTopKSampler::FastTopKSampler(torch::Tensor d2t_map, size_t target_vocab_size):
    d2t_map_(std::move(d2t_map)), target_vocab_size_(target_vocab_size) {
    if (!d2t_map_.defined() || d2t_map_.numel() == 0) {
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(d2t_map_.dim() == 1, "d2t_map must be 1-D");
    RTP_LLM_CHECK_WITH_INFO(d2t_map_.dtype() == torch::kInt64, "d2t_map must be int64");
    RTP_LLM_CHECK_WITH_INFO(target_vocab_size_ > 0, "target vocab size must be positive when d2t_map is provided");
    RTP_LLM_CHECK_WITH_INFO(d2t_map_.min().item<int64_t>() >= 0, "d2t_map contains a negative target token id");
    RTP_LLM_CHECK_WITH_INFO(d2t_map_.max().item<int64_t>() < static_cast<int64_t>(target_vocab_size_),
                            "d2t_map target token id exceeds target vocab size");
}

FastTopKSamplerOutput FastTopKSampler::forward(const torch::Tensor& logits, int top_k) {
    RTP_LLM_CHECK_WITH_INFO(top_k == 1, "FastTopKSampler only supports top_k=1 proposals");
    RTP_LLM_CHECK_WITH_INFO(logits.dim() == 2, "FastTopKSampler logits must be 2-D");

    FastTopKSamplerOutput output;
    auto                  draft_token_ids = torch::argmax(logits, -1, true);

    const int     batch_size        = draft_token_ids.size(0);
    const int64_t draft_vocab_size  = logits.size(1);
    const int64_t target_vocab_size = target_vocab_size_ == 0 ? draft_vocab_size : target_vocab_size_;

    if (d2t_map_.defined() && d2t_map_.numel() > 0) {
        RTP_LLM_CHECK_WITH_INFO(d2t_map_.numel() == draft_vocab_size,
                                "d2t_map size mismatch: %ld != %ld",
                                d2t_map_.numel(),
                                draft_vocab_size);
    } else {
        RTP_LLM_CHECK_WITH_INFO(draft_vocab_size == target_vocab_size,
                                "draft/target vocab mismatch requires d2t_map: %ld != %ld",
                                draft_vocab_size,
                                target_vocab_size);
    }

    // The probability tensor always describes the distribution that produced
    // token_ids. Top-1 is a point mass; a future top-p sampler can return its
    // filtered and normalized distribution through the same contract.
    output.all_probs = torch::zeros({batch_size, draft_vocab_size}, logits.options().dtype(torch::kFloat32));
    output.all_probs.scatter_(1, draft_token_ids.to(torch::kLong), 1.0f);

    output.token_ids = draft_token_ids;
    execMappingDraft2Target({output.token_ids, d2t_map_, batch_size, 0, 1});

    return output;
}

torch::Tensor SpeculativeSampler::mapDraftProbsToTarget(const torch::Tensor& draft_probs,
                                                        const torch::Tensor& d2t_map,
                                                        int64_t              target_vocab_size,
                                                        torch::Tensor*       target_probs_buffer) {
    RTP_LLM_CHECK_WITH_INFO(draft_probs.defined(), "draft proposal probabilities must be defined");
    RTP_LLM_CHECK_WITH_INFO(draft_probs.dim() == 3, "draft proposal probabilities must be 3-D");
    RTP_LLM_CHECK_WITH_INFO(draft_probs.scalar_type() == torch::kFloat32,
                            "draft proposal probabilities must be float32");
    RTP_LLM_CHECK_WITH_INFO(target_vocab_size > 0, "target vocabulary size must be positive");

    const int64_t draft_vocab_size = draft_probs.size(2);
    if (!d2t_map.defined() || d2t_map.numel() == 0) {
        RTP_LLM_CHECK_WITH_INFO(draft_vocab_size == target_vocab_size,
                                "draft/target vocab mismatch requires d2t_map: %ld != %ld",
                                draft_vocab_size,
                                target_vocab_size);
        return draft_probs;
    }

    RTP_LLM_CHECK_WITH_INFO(d2t_map.dim() == 1, "d2t_map must be 1-D");
    RTP_LLM_CHECK_WITH_INFO(d2t_map.scalar_type() == torch::kInt64, "d2t_map must be int64");
    RTP_LLM_CHECK_WITH_INFO(d2t_map.device() == draft_probs.device(),
                            "d2t_map and draft proposal probabilities must be on the same device");
    RTP_LLM_CHECK_WITH_INFO(
        d2t_map.numel() == draft_vocab_size, "d2t_map size mismatch: %ld != %ld", d2t_map.numel(), draft_vocab_size);

    const int64_t batch_size = draft_probs.size(0);
    const int64_t num_steps  = draft_probs.size(1);
    torch::Tensor local_buffer;
    auto&         buffer       = target_probs_buffer == nullptr ? local_buffer : *target_probs_buffer;
    const bool    reuse_buffer = buffer.defined() && buffer.size(0) >= batch_size && buffer.size(1) == num_steps
                              && buffer.size(2) == target_vocab_size
                              && buffer.scalar_type() == draft_probs.scalar_type()
                              && buffer.device() == draft_probs.device();
    if (!reuse_buffer) {
        const int64_t buffer_batch_size = std::max(batch_size, buffer.defined() ? buffer.size(0) : int64_t{0});
        buffer = torch::empty({buffer_batch_size, num_steps, target_vocab_size}, draft_probs.options());
    }
    auto target_probs = buffer.narrow(0, 0, batch_size).narrow(1, 0, num_steps);
    target_probs.zero_();
    auto target_indices = d2t_map.view({1, 1, draft_vocab_size}).expand_as(draft_probs);
    target_probs.scatter_add_(-1, target_indices, draft_probs);
    return target_probs;
}

SpeculativeSamplerOutput SpeculativeSampler::forward(const std::list<GenerateStreamPtr>& streams,
                                                     SamplerOutput&                      draft_sampler_output,
                                                     SamplerOutput&                      target_sampler_output) {
    // TensorHolder release point (SpeculativeSampler): advances host tensors
    // staged for rejection sampling H2D in the previous forward.
    buffer_holder_.release();
    SpeculativeSamplerOutput sample_output;
    batchSample(sample_output, streams, draft_sampler_output, target_sampler_output);

    return sample_output;
}

void SpeculativeSampler::batchSample(SpeculativeSamplerOutput&           sample_output,
                                     const std::list<GenerateStreamPtr>& streams,
                                     SamplerOutput&                      draft_sampler_output,
                                     SamplerOutput&                      target_sampler_output) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample");
    torch::Device target_device = getTorchCudaDevice();

    int batch_size = streams.size();

    auto draft_token_ids   = draft_sampler_output.token_ids;
    auto draft_token_probs = draft_sampler_output.all_probs;

    auto target_token_probs = target_sampler_output.all_probs;

    if (!draft_token_ids.is_cuda()) {
        buffer_holder_.hold_host(draft_token_ids);
    }
    auto draft_token_ids_d_t = draft_token_ids.to(target_device, torch::kInt32, true);

    auto target_token_ids_d_t = target_sampler_output.token_ids;
    if (!target_token_ids_d_t.is_cuda()) {
        buffer_holder_.hold_host(target_token_ids_d_t);
        target_token_ids_d_t = target_token_ids_d_t.to(target_device, true);
    }

    torch::Tensor do_sample =
        torch::zeros({(long)batch_size}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
    int stream_idx = 0;
    for (const GenerateStreamPtr& stream : streams) {
        do_sample[stream_idx] = stream->generateConfig()->do_sample;
        stream_idx++;
    }
    buffer_holder_.hold_host(do_sample);
    auto do_sample_d = do_sample.to(target_device, true);

    auto          rand_options      = torch::TensorOptions().device(target_device).dtype(torch::kFloat);
    torch::Tensor uniform_samples_d = torch::rand({(long)batch_size, (long)propose_step_ + 1}, rand_options);

    // Override per-stream uniform samples with seeded generator when random_seed is set,
    // ensuring deterministic acceptance for reproducible iter_count.
    {
        int idx = 0;
        for (const auto& stream : streams) {
            auto gen = stream->getGenerator();
            if (gen.defined()) {
                uniform_samples_d[idx] = torch::rand({(long)propose_step_ + 1}, gen, std::nullopt, rand_options);
            }
            idx++;
        }
    }

    auto target_token_probs_d_t = target_token_probs;
    auto draft_token_probs_d_t =
        mapDraftProbsToTarget(draft_token_probs, d2t_map_, target_token_probs_d_t.size(2), &draft_probs_target_buffer_);
    torch::Tensor output_token_ids_d =
        torch::zeros({(long)batch_size, (long)propose_step_ + 1},
                     torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));
    torch::Tensor output_accepted_token_num_d = torch::zeros(
        {(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));

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
        });
    }

    RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling");

    // forceSpAccept: override rejection sampling results for streams that requested
    // forced acceptance — accept all draft tokens plus the target bonus token.
    {
        bool has_force = false;
        auto force_mask =
            torch::zeros({(long)batch_size}, torch::TensorOptions().dtype(torch::kBool).device(target_device));
        int idx = 0;
        for (const auto& stream : streams) {
            if (stream->forceSpAccept()) {
                force_mask[idx] = true;
                has_force       = true;
            }
            idx++;
        }
        if (has_force) {
            RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling.forceSpAccept");
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
        }
    }

    // use async sample here, we assume accept all tokens
    // so we need to reset -1 to 0 in output_token_ids_d
    output_token_ids_d.index_put_({output_token_ids_d == -1}, 0);
    sample_output.accept_tokens = output_token_ids_d;
    sample_output.accept_len    = output_accepted_token_num_d;

    sample_output.accept_tokens_cpu = sample_output.accept_tokens.to(torch::kCPU, true);
    sample_output.accept_len_cpu    = sample_output.accept_len.to(torch::kCPU, true);
    sample_output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}

}  // namespace speculative
}  // namespace rtp_llm

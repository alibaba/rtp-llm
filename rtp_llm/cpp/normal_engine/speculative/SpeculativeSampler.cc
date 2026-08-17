#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <algorithm>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {
namespace speculative {

bool draftProbsNeedTargetVocabRemap(const torch::Tensor& d2t_map) {
    return d2t_map.defined() && d2t_map.numel() > 0;
}

torch::Tensor remapDraftProbsToTargetVocab(const torch::Tensor& draft_probs,
                                           const torch::Tensor& d2t_map,
                                           int64_t              batch_size,
                                           int64_t              target_vocab_size,
                                           torch::Tensor&       padding_buffer) {
    const int64_t num_spec = draft_probs.size(1);

    // Reuse pre-allocated padding buffer to avoid per-forward GPU allocation.
    // Grow-only along batch / num_spec dims; vocab dim must match exactly.
    const bool need_realloc = !padding_buffer.defined() || padding_buffer.size(0) < batch_size
                              || padding_buffer.size(1) < num_spec || padding_buffer.size(2) != target_vocab_size
                              || padding_buffer.dtype() != draft_probs.dtype()
                              || padding_buffer.device() != draft_probs.device();
    if (need_realloc) {
        const int64_t cap_b = std::max(batch_size, padding_buffer.defined() ? padding_buffer.size(0) : (int64_t)0);
        const int64_t cap_s = std::max(num_spec, padding_buffer.defined() ? padding_buffer.size(1) : (int64_t)0);
        padding_buffer      = torch::zeros({cap_b, cap_s, target_vocab_size}, draft_probs.options());
    }

    auto draft_probs_padding = padding_buffer.narrow(0, 0, batch_size).narrow(1, 0, num_spec);
    draft_probs_padding.zero_();
    draft_probs_padding.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), d2t_map}, draft_probs);
    return draft_probs_padding;
}

FastTopKSamplerOutput FastTopKSampler::forward(const torch::Tensor& logits, int top_k) {
    RTP_LLM_CHECK_WITH_INFO(top_k == 1, "greedy speculative proposal sampling requires top_k == 1, got %d", top_k);

    FastTopKSamplerOutput output;
    // softmax is monotone, so the argmax over the logits is the argmax over the probabilities.
    // Taking it on the logits avoids materialising a full-vocabulary softmax whose every value the
    // one-hot below would overwrite.
    output.token_ids = std::get<1>(torch::max(logits, -1, true));

    // Rejection sampling must receive the distribution that actually selected the proposal token.
    // The scatter happens while token_ids are still draft-vocabulary indices, because all_probs is
    // indexed in draft-vocabulary space and batchSample remaps it with the same d2t map later.
    output.all_probs = torch::zeros_like(logits).scatter_(-1, output.token_ids, 1.0);

    int batch_size = output.token_ids.size(0);
    execMappingDraft2Target({output.token_ids, d2t_map_, batch_size, 0, 1});

    return output;
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
        do_sample[stream_idx] = !stream->generateConfig()->top1();
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

    auto          draft_token_probs_d_t  = draft_token_probs;
    auto          target_token_probs_d_t = target_token_probs;
    torch::Tensor output_token_ids_d =
        torch::zeros({(long)batch_size, (long)propose_step_ + 1},
                     torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));
    torch::Tensor output_accepted_token_num_d = torch::zeros(
        {(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));

    // Whenever a draft->target vocab map is defined, all_probs left FastTopKSampler::forward in
    // draft-vocabulary space while token_ids were rewritten into target-vocabulary ids. Rejection
    // sampling indexes the proposal probabilities with the sampled target id, so they must be
    // scattered into target-vocabulary space via the same d2t_map_ regardless of whether the two
    // vocab widths happen to match. Keying this on the vocab-width inequality alone left the
    // equal-width d2t case unremapped, so q at the sampled id was a spurious 0.0 and u * p < q
    // accepted unconditionally (a rejection could never fire). The emptiness check mirrors
    // mappingDraft2Target, which treats a defined but empty map as "no mapping".
    if (draftProbsNeedTargetVocabRemap(d2t_map_)) {
        draft_token_probs_d_t = remapDraftProbsToTargetVocab(draft_token_probs_d_t,
                                                             d2t_map_,
                                                             (int64_t)batch_size,
                                                             target_token_probs_d_t.size(2),
                                                             draft_probs_padding_buffer_);
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

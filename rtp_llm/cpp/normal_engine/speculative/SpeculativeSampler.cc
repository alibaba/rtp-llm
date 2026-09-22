#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <algorithm>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <atomic>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace rtp_llm {
namespace speculative {

namespace {

const bool kDebugMtpAcceptEnabled = []() {
    const char* env = std::getenv("RTP_LLM_DEBUG_MTP_ACCEPT");
    return env != nullptr && std::string(env) != "0";
}();

bool debugMtpAcceptEnabled() {
    static const bool logged = []() {
        if (kDebugMtpAcceptEnabled) {
            RTP_LLM_LOG_WARNING("[debug-mtp-accept] enabled; this copies small sampler tensors to host");
        }
        return true;
    }();
    (void)logged;
    return kDebugMtpAcceptEnabled;
}

std::string debugTensorSummary(const torch::Tensor& tensor, int64_t limit = 24) {
    if (!tensor.defined()) {
        return "None";
    }
    std::ostringstream oss;
    oss << "shape=[";
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << tensor.size(i);
    }
    oss << "] device=" << tensor.device() << " dtype=" << tensor.dtype();
    if (tensor.numel() == 0) {
        return oss.str();
    }
    try {
        auto flat  = tensor.reshape({-1});
        auto count = std::min<int64_t>(limit, flat.numel());
        auto head  = flat.slice(0, 0, count);
        if (head.device().is_cuda()) {
            head = head.cpu();
        }
        oss << " head=" << head;
    } catch (const std::exception& e) {
        oss << " summary_error=" << e.what();
    }
    return oss.str();
}

}  // namespace

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
                                                    size_t               draft_vocab_size) const {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.sample_dspark_draft");
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

    auto previous_tokens = anchors.reshape({batch_size}).to(torch::kLong);
    auto all_probabilities =
        torch::empty({batch_size, static_cast<int64_t>(propose_step_), static_cast<int64_t>(draft_vocab_size)},
                     torch::TensorOptions().dtype(torch::kFloat32).device(base_logits.device()));
    std::vector<torch::Tensor> token_columns;
    token_columns.reserve(propose_step_);
    auto proposal_logits =
        base_logits.narrow(1, 0, draft_vocab_size)
            .view({batch_size, static_cast<int64_t>(propose_step_), static_cast<int64_t>(draft_vocab_size)});
    auto temperature_column = temperature.unsqueeze(1);

    for (int64_t step = 0; step < static_cast<int64_t>(propose_step_); ++step) {
        auto markov_embedding = markov_w1.index_select(0, previous_tokens);
        auto markov_bias      = torch::mm(markov_embedding, markov_w2.transpose(0, 1)).to(torch::kFloat32);
        auto logits           = proposal_logits.select(1, step) + markov_bias;
        logits.div_(temperature_column);
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

    SamplerOutput output;
    output.token_ids = torch::stack(token_columns, 1).contiguous();
    output.all_probs = std::move(all_probabilities);
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

    if (debugMtpAcceptEnabled()) {
        static std::atomic<int> log_budget{32};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            torch::Tensor target_sampled_ids_debug;
            torch::Tensor draft_target_match_debug;
            try {
                const int64_t target_token_stride = target_token_ids_d_t.size(1);
                auto          target_token_ids_3d = target_token_ids_d_t.reshape(
                    {(int64_t)batch_size, (int64_t)(propose_step_ + 1), target_token_stride});
                target_sampled_ids_debug = target_token_ids_3d.select(2, target_token_stride - 1).to(torch::kInt32);
                draft_target_match_debug = draft_token_ids_d_t.to(torch::kInt32)
                                               .eq(target_sampled_ids_debug.narrow(1, 0, (int64_t)propose_step_));
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("[debug-mtp-accept] failed to summarize target sampled ids: %s", e.what());
            }
            RTP_LLM_LOG_INFO("[debug-mtp-accept] batch=%d propose_step=%zu draft_token_ids=%s "
                             "target_sampled_ids=%s draft_target_match=%s target_token_ids=%s accept_len=%s "
                             "accept_tokens=%s draft_probs=%s target_probs=%s",
                             batch_size,
                             propose_step_,
                             debugTensorSummary(draft_token_ids, 32).c_str(),
                             debugTensorSummary(target_sampled_ids_debug, 32).c_str(),
                             debugTensorSummary(draft_target_match_debug, 32).c_str(),
                             debugTensorSummary(target_token_ids, 32).c_str(),
                             debugTensorSummary(sample_output.accept_len, 32).c_str(),
                             debugTensorSummary(sample_output.accept_tokens, 32).c_str(),
                             debugTensorSummary(draft_token_probs, 0).c_str(),
                             debugTensorSummary(target_token_probs, 0).c_str());
        }
    }

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

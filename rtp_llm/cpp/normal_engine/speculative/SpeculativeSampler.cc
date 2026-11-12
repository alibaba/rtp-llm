#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <algorithm>
#include "rtp_llm/cpp/utils/MtpFusion.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#endif
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <atomic>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>

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

bool FastTopKSampler::compactPointMassEnabled() const {
#if USING_CUDA
    return allow_compact_point_mass_ && mtpFusionEnabled() && (!d2t_map_.defined() || d2t_map_.numel() == 0);
#else
    return false;
#endif
}

FastTopKSamplerOutput
FastTopKSampler::forward(const torch::Tensor& logits, int top_k, const MtpDraftSampleContext* context) {
    FastTopKSamplerOutput output;
    const auto scores = use_raw_logits_ && top_k == 1 ? logits : torch::softmax(logits, -1);
    const bool            compact = top_k == 1 && compactPointMassEnabled();
#if USING_CUDA
    const torch::Tensor empty;
    if (mtpFusionEnabled() && top_k == 1
        && tryMtpGreedyPack(scores,
                            d2t_map_,
                            output.all_probs,
                            output.token_ids,
                            output.token_ids_int32,
                            context ? context->sequence_lengths : empty,
                            context ? context->hidden_to_zero : empty,
                            context && context->zero_tokens,
                            output.next_sequence_lengths,
                            cuda_graph::graphGetCurrentStream().stream(),
                            compact)) {
        output.fake_inputs_prepared = context && context->zero_tokens;
        return output;
    }
#endif

    if (top_k == 1) {
        output.token_ids = use_raw_logits_ ? torch::argmax(scores, -1, true) : std::get<1>(torch::max(scores, -1, true));
        // A deterministic top-1 proposal must be represented by its point-mass
        // distribution when rejection sampling computes its acceptance ratio.
        if (!compact) {
            output.all_probs = torch::zeros_like(scores).scatter_(-1, output.token_ids, 1.0);
        }
    } else {
        // Preserve the existing multi-candidate behavior. This path does not
        // describe the deterministic top-1 proposal fixed above.
        output.all_probs = scores;
        output.token_ids = std::get<1>(torch::topk(output.all_probs, top_k, -1));
    }

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
    auto target_token_probs = target_sampler_output.deferred_token_sampling ? target_sampler_output.verify_logits :
                                                                              target_sampler_output.all_probs;
    const auto& target_candidate_ids = target_sampler_output.all_probs_indices;
    if (target_candidate_ids.defined()) {
        RTP_LLM_CHECK(draft_sampler_output.draft_point_mass);
        RTP_LLM_CHECK(target_candidate_ids.is_cuda() && target_candidate_ids.is_contiguous()
                      && target_candidate_ids.scalar_type() == torch::kInt32
                      && target_candidate_ids.sizes() == target_token_probs.sizes());
    }

    buffer_holder_.hold_host(draft_token_ids);
    auto draft_token_ids_d_t = draft_token_ids.to(target_device, true);

    auto target_token_ids_d_t = target_sampler_output.token_ids;
    if (!target_sampler_output.deferred_token_sampling && !target_token_ids_d_t.is_cuda()) {
        buffer_holder_.hold_host(target_token_ids_d_t);
        target_token_ids_d_t = target_token_ids_d_t.to(target_device, true);
    }

    auto do_sample_d =
        target_sampler_output.deferred_token_sampling ? target_sampler_output.verify_do_sample : torch::Tensor();
    [[maybe_unused]] bool all_greedy = true;
    [[maybe_unused]] bool has_forced_accept = false;
    if (!do_sample_d.defined()) {
        auto do_sample =
            torch::empty({(long)batch_size}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
        int stream_idx = 0;
        for (const GenerateStreamPtr& stream : streams) {
            const auto& config                       = *stream->generateConfig();
            const bool  sample                       = config.do_sample && config.top_k != 1;
            do_sample.data_ptr<bool>()[stream_idx++] = sample;
            all_greedy                               = all_greedy && !sample;
            has_forced_accept                        = has_forced_accept || stream->forceSpAccept();
        }
        buffer_holder_.hold_host(do_sample);
        do_sample_d = do_sample.to(target_device, true);
    } else {
        RTP_LLM_CHECK(do_sample_d.device() == target_device && do_sample_d.scalar_type() == torch::kBool
                      && do_sample_d.dim() == 1 && do_sample_d.size(0) == batch_size && do_sample_d.is_contiguous());
    }

    auto          rand_options      = torch::TensorOptions().device(target_device).dtype(torch::kFloat);
    torch::Tensor uniform_samples_d;
    if (!target_sampler_output.deferred_token_sampling) {
        uniform_samples_d = torch::rand({(long)batch_size, (long)propose_step_ + 1}, rand_options);
    }

    // Override per-stream uniform samples with seeded generator when random_seed is set,
    // ensuring deterministic acceptance for reproducible iter_count.
    if (!target_sampler_output.deferred_token_sampling) {
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
    const auto output_options = torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false);
    // The CUDA rejection kernel writes every token slot and acceptance length,
    // including unused -1 tails, on both greedy and stochastic branches.
    torch::Tensor packed_output;
    torch::Tensor output_token_ids_d;
    torch::Tensor output_accepted_token_num_d;
    if (target_sampler_output.deferred_token_sampling) {
        const int64_t token_count = static_cast<int64_t>(batch_size) * (propose_step_ + 1);
        packed_output             = torch::empty({token_count + batch_size}, output_options);
        output_token_ids_d =
            packed_output.as_strided({batch_size, (long)propose_step_ + 1}, {(long)propose_step_ + 1, 1}, 0);
        output_accepted_token_num_d = packed_output.as_strided({batch_size}, {1}, token_count);
    } else {
        output_token_ids_d          = mtpFusionEnabled() ?
                                          torch::empty({(long)batch_size, (long)propose_step_ + 1}, output_options) :
                                          torch::zeros({(long)batch_size, (long)propose_step_ + 1}, output_options);
        output_accepted_token_num_d = mtpFusionEnabled() ? torch::empty({(long)batch_size}, output_options) :
                                                           torch::zeros({(long)batch_size}, output_options);
    }

    const int64_t target_vocab_size = target_sampler_output.verify_dense_fallback.defined() ?
                                          target_sampler_output.verify_dense_fallback.size(-1) :
                                          target_token_probs_d_t.size(2);
    if (!draft_sampler_output.draft_point_mass && draft_token_probs_d_t.size(2) != target_vocab_size) {
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

    bool greedy_finalized = false;
    {
        RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.execRejectionSampling");
#if USING_CUDA
        if (target_sampler_output.deferred_token_sampling) {
            std::vector<at::Generator> generators;
            std::vector<bool>          force_accept;
            generators.reserve(batch_size);
            force_accept.reserve(batch_size);
            for (const auto& stream : streams) {
                if (!target_sampler_output.prepared_rejection_rng.defined()) {
                    generators.push_back(stream->getGenerator());
                }
                force_accept.push_back(stream->forceSpAccept());
            }
            execRejectionSampling({draft_token_probs_d_t,
                                   draft_token_ids_d_t,
                                   uniform_samples_d,
                                   torch::Tensor(),
                                   target_token_ids_d_t,
                                   output_token_ids_d,
                                   output_accepted_token_num_d,
                                   do_sample_d,
                                   true,
                                   draft_sampler_output.draft_point_mass,
                                   generators,
                                   &buffer_holder_,
                                   target_token_probs_d_t,
                                   target_sampler_output.acceptance_cap,
                                   force_accept,
                                   target_sampler_output.cum_log_probs,
                                   target_sampler_output.verify_candidate_ids,
                                   target_sampler_output.verify_dense_fallback,
                                   target_sampler_output.verify_fallback_rows,
                                   target_sampler_output.prepared_rejection_rng,
                                   target_sampler_output.verify_log_normalizer});
        } else if (mtpFusionEnabled() && draft_sampler_output.draft_point_mass && all_greedy && !has_forced_accept) {
            greedy_finalized = tryMtpGreedyFinalize(draft_token_ids_d_t, target_token_ids_d_t,
                                                   output_token_ids_d, output_accepted_token_num_d,
                                                   sample_output.accept_metric_rows,
                                                   sample_output.accept_metric_total,
                                                   cuda_graph::graphGetCurrentStream().stream());
        }
        if (!target_sampler_output.deferred_token_sampling && !greedy_finalized
            && draft_sampler_output.draft_point_mass) {
            RTP_LLM_CHECK(!draft_token_probs_d_t.defined());
            RTP_LLM_CHECK(target_token_probs_d_t.scalar_type() == torch::kFloat32
                          && target_token_probs_d_t.is_contiguous());
            RTP_LLM_CHECK(draft_token_ids_d_t.scalar_type() == torch::kInt32 && draft_token_ids_d_t.is_contiguous());
            RTP_LLM_CHECK(target_token_ids_d_t.scalar_type() == torch::kInt32 && target_token_ids_d_t.is_contiguous());
            RTP_LLM_CHECK(invokePointMassRejectionSampling(
                              draft_token_ids_d_t.data_ptr<int32_t>(),
                              uniform_samples_d.data_ptr<float>(),
                              target_token_probs_d_t.data_ptr<float>(),
                              target_token_ids_d_t.data_ptr<int32_t>(),
                              target_token_ids_d_t.size(1),
                              output_token_ids_d.data_ptr<int32_t>(),
                              output_accepted_token_num_d.data_ptr<int32_t>(),
                              do_sample_d.data_ptr<bool>(),
                              batch_size,
                              propose_step_,
                              target_token_probs_d_t.size(2),
                              cuda_graph::graphGetCurrentStream().stream(),
                              target_candidate_ids.defined() ? target_candidate_ids.data_ptr<int32_t>() : nullptr)
                          == cudaSuccess);
        } else if (!target_sampler_output.deferred_token_sampling && !greedy_finalized)
#endif
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
    if (!target_sampler_output.deferred_token_sampling) {
        bool has_force = false;
        if (mtpFusionEnabled()) {
            has_force =
                std::any_of(streams.begin(), streams.end(), [](const auto& stream) { return stream->forceSpAccept(); });
        }
        auto force_mask =
            (!mtpFusionEnabled() || has_force) ?
                torch::zeros({(long)batch_size}, torch::TensorOptions().dtype(torch::kBool).device(target_device)) :
                torch::Tensor();
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
    bool cleaned = greedy_finalized || target_sampler_output.deferred_token_sampling;
#if USING_CUDA
    if (!cleaned) {
        cleaned = mtpFusionEnabled()
                  && tryMtpCleanRejectedTokens(output_token_ids_d, cuda_graph::graphGetCurrentStream().stream());
    }
#endif
    if (!cleaned) {
        output_token_ids_d.index_put_({output_token_ids_d == -1}, 0);
    }
    sample_output.accept_tokens = output_token_ids_d;
    sample_output.accept_len    = output_accepted_token_num_d;

    if (debugMtpAcceptEnabled() && !target_sampler_output.deferred_token_sampling) {
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

    if (packed_output.defined()) {
        auto host_output =
            torch::empty(packed_output.sizes(), torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
        host_output.copy_(packed_output, true);
        const int64_t token_count = static_cast<int64_t>(batch_size) * (propose_step_ + 1);
        sample_output.accept_tokens_cpu =
            host_output.as_strided({batch_size, (long)propose_step_ + 1}, {(long)propose_step_ + 1, 1}, 0);
        sample_output.accept_len_cpu = host_output.as_strided({batch_size}, {1}, token_count);
    } else {
        sample_output.accept_tokens_cpu = sample_output.accept_tokens.to(torch::kCPU, true);
        sample_output.accept_len_cpu    = sample_output.accept_len.to(torch::kCPU, true);
    }
    sample_output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());

}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}

}  // namespace speculative
}  // namespace rtp_llm

#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <algorithm>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {
namespace speculative {

FastTopKSamplerOutput FastTopKSampler::forward(const torch::Tensor& logits, int top_k) {
    FastTopKSamplerOutput output;
    output.all_probs = torch::softmax(logits, -1);

    std::tuple<torch::Tensor, torch::Tensor> sample_res;
    if (top_k == 1) {
        sample_res = torch::max(output.all_probs, -1, true);
    } else {
        sample_res = torch::topk(output.all_probs, top_k, -1);
    }

    output.token_ids = std::get<1>(sample_res);

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

    publishAcceptToHost(sample_output);
}

void SpeculativeSampler::publishAcceptToHost(SpeculativeSamplerOutput& sample_output) const {
    // Pinned ping-pong destinations: a pageable dst turns this cudaMemcpyAsync
    // into a synchronous staged copy on top of the mandatory wait for the
    // producing kernels.
    accept_cpu_slot_        = (accept_cpu_slot_ + 1) % accept_tokens_cpu_slots_.size();
    auto& tokens_slot       = accept_tokens_cpu_slots_[accept_cpu_slot_];
    auto& len_slot          = accept_len_cpu_slots_[accept_cpu_slot_];
    const auto& tokens_src  = sample_output.accept_tokens;
    const auto& len_src     = sample_output.accept_len;
    if (!tokens_slot.defined() || tokens_slot.numel() < tokens_src.numel() || tokens_slot.dtype() != tokens_src.dtype()) {
        tokens_slot = torch::empty(tokens_src.sizes(), tokens_src.options().device(torch::kCPU).pinned_memory(true));
    }
    if (!len_slot.defined() || len_slot.numel() < len_src.numel() || len_slot.dtype() != len_src.dtype()) {
        len_slot = torch::empty(len_src.sizes(), len_src.options().device(torch::kCPU).pinned_memory(true));
    }
    auto tokens_dst = tokens_slot.numel() == tokens_src.numel() ?
                          tokens_slot.view(tokens_src.sizes()) :
                          tokens_slot.flatten().narrow(0, 0, tokens_src.numel()).view(tokens_src.sizes());
    auto len_dst    = len_slot.numel() == len_src.numel() ?
                          len_slot.view(len_src.sizes()) :
                          len_slot.flatten().narrow(0, 0, len_src.numel()).view(len_src.sizes());
    tokens_dst.copy_(tokens_src, /*non_blocking=*/true);
    len_dst.copy_(len_src, /*non_blocking=*/true);
    sample_output.accept_tokens_cpu = tokens_dst;
    sample_output.accept_len_cpu    = len_dst;
    sample_output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
}

SpeculativeSamplerOutput SpeculativeSampler::forwardCoupled(const std::list<GenerateStreamPtr>& streams,
                                                            SamplerOutput& draft_sampler_output,
                                                            SamplerOutput& target_sampler_output,
                                                            const torch::Tensor& target_prefix_lengths) {
    RTP_LLM_PROFILE_SCOPE("speculative_sampler.forwardCoupled");
    buffer_holder_.release();
    const int64_t batch_size = static_cast<int64_t>(streams.size());
    const int64_t width      = static_cast<int64_t>(propose_step_) + 1;
    RTP_LLM_CHECK_WITH_INFO(batch_size > 0, "coupled verifier requires a non-empty batch");

    auto draft_tokens = draft_sampler_output.token_ids;
    if (!draft_tokens.is_cuda()) {
        buffer_holder_.hold_host(draft_tokens);
        draft_tokens = draft_tokens.to(getTorchCudaDevice(), /*non_blocking=*/true);
    }
    draft_tokens = draft_tokens.reshape({batch_size, width - 1}).contiguous();

    bool has_coupled_logits = false;
    for (const auto& stream : streams) {
        const auto& config = stream->generateConfig();
        has_coupled_logits |= config->do_sample && !config->top1();
        RTP_LLM_CHECK_WITH_INFO(config->random_seed.has_value(),
                                "DSpark coupled sampling requires an issued request seed");
    }

    torch::Tensor seeds_host;
    torch::Tensor temperatures_host;
    if (has_coupled_logits) {
        auto metadata_options = torch::TensorOptions().pinned_memory(true);
        seeds_host = torch::empty({batch_size * width}, metadata_options.dtype(torch::kInt64));
        temperatures_host = torch::empty({batch_size * width}, metadata_options.dtype(torch::kFloat32));
        auto* seeds_ptr = seeds_host.data_ptr<int64_t>();
        auto* temperatures_ptr = temperatures_host.data_ptr<float>();
        int64_t row = 0;
        for (const auto& stream : streams) {
            const auto& config = stream->generateConfig();
            const bool stochastic =
                config->do_sample && !config->top1() && config->temperature != 0.0F;
            for (int64_t col = 0; col < width; ++col) {
                seeds_ptr[row * width + col] = static_cast<int64_t>(config->random_seed.value());
                // Target logits already contain temperature/top-k/top-p and
                // grammar transforms. One enables noise; zero is argmax.
                temperatures_ptr[row * width + col] = stochastic ? 1.0F : 0.0F;
            }
            ++row;
        }
    }

    torch::Tensor target_tokens;
    if (has_coupled_logits) {
        RTP_LLM_CHECK_WITH_INFO(target_sampler_output.all_probs.defined(),
                                "DSpark coupled sampling requires target processed logits");
        buffer_holder_.hold_host(seeds_host);
        buffer_holder_.hold_host(temperatures_host);
        auto seeds = seeds_host.to(getTorchCudaDevice(), /*non_blocking=*/true);
        auto temperatures = temperatures_host.to(getTorchCudaDevice(), /*non_blocking=*/true);
        auto prefix = target_prefix_lengths.to(
            torch::TensorOptions().device(getTorchCudaDevice()).dtype(torch::kInt64), /*non_blocking=*/true);
        auto positions = (prefix.reshape({batch_size, 1})
                          + torch::arange(width,
                                          torch::TensorOptions()
                                              .device(getTorchCudaDevice())
                                              .dtype(torch::kInt64)))
                             .reshape({batch_size * width})
                             .contiguous();
        // The field is the existing SamplerOutput carrier, but for coupled
        // DSpark it holds processed target logits (not probabilities).
        auto processed_logits = target_sampler_output.all_probs.reshape({batch_size * width, -1}).contiguous();
        target_tokens = execGumbelSample(
                            {processed_logits, seeds, positions, temperatures, /*input_is_probs=*/false,
                             /*apply_temperature=*/false, use_fp64_gumbel_})
                            .reshape({batch_size, width});
        target_sampler_output.token_ids = target_tokens.reshape({batch_size * width, 1}).to(torch::kInt32);
    } else {
        auto target_ids = target_sampler_output.token_ids;
        RTP_LLM_CHECK_WITH_INFO(target_ids.defined(),
                                "DSpark greedy coupling requires target argmax token_ids");
        if (!target_ids.is_cuda()) {
            buffer_holder_.hold_host(target_ids);
            target_ids = target_ids.to(getTorchCudaDevice(), /*non_blocking=*/true);
        }
        const int64_t target_stride = target_ids.size(1);
        target_tokens = target_ids.reshape({batch_size, width, target_stride})
                            .select(2, target_stride - 1)
                            .contiguous();
    }

    auto verified = execCoupledTokenVerify({draft_tokens, target_tokens});
    SpeculativeSamplerOutput output;
    output.accept_tokens = std::move(verified.accept_tokens);
    output.accept_len    = std::move(verified.accept_len);

    bool has_force = false;
    auto force_mask = torch::zeros({batch_size},
                                   torch::TensorOptions().dtype(torch::kBool).device(getTorchCudaDevice()));
    int64_t index = 0;
    for (const auto& stream : streams) {
        if (stream->forceSpAccept()) {
            force_mask[index] = true;
            has_force         = true;
        }
        ++index;
    }
    if (has_force) {
        auto forced = torch::cat({draft_tokens.to(torch::kInt32), target_tokens.select(1, width - 1).unsqueeze(1).to(torch::kInt32)}, 1);
        output.accept_tokens = torch::where(force_mask.unsqueeze(1), forced, output.accept_tokens);
        output.accept_len = torch::where(force_mask,
                                         torch::full_like(output.accept_len, static_cast<int32_t>(width)),
                                         output.accept_len);
    }
    publishAcceptToHost(output);
    return output;
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}

}  // namespace speculative
}  // namespace rtp_llm

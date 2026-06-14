#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
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

    return output;
}

SpeculativeSamplerOutput SpeculativeSampler::forward(const std::list<GenerateStreamPtr>& streams,
                                                     SamplerOutput&                      draft_sampler_output,
                                                     SamplerOutput&                      target_sampler_output) {
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

    const int batch_size = static_cast<int>(streams.size());

    auto draft_token_ids  = draft_sampler_output.token_ids;
    auto target_token_ids = target_sampler_output.token_ids;

    auto draft_token_probs  = draft_sampler_output.all_probs;
    auto target_token_probs = target_sampler_output.all_probs;

    auto draft_token_ids_d_t = draft_token_ids.to(target_device).clone();

    auto target_token_ids_d_t = target_sampler_output.token_ids;
    if (!target_token_ids_d_t.is_cuda()) {
        target_token_ids_d_t = target_token_ids_d_t.to(target_device, /*non_blocking=*/true);
    }

    auto          rand_options      = torch::TensorOptions().device(target_device).dtype(torch::kFloat);
    torch::Tensor uniform_samples_d = torch::rand({(long)batch_size, (long)propose_step_ + 1}, rand_options);

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
    torch::Tensor output_emitted_token_num_d = torch::zeros(
        {(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));

    {
        RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.execChainSpeculativeSampling");
        execChainSpeculativeSampling({draft_token_probs_d_t,
                                      draft_token_ids_d_t,
                                      uniform_samples_d,
                                      target_token_probs_d_t,
                                      output_token_ids_d,
                                      output_accepted_token_num_d,
                                      output_emitted_token_num_d});
    }

    auto accept_len_d    = output_emitted_token_num_d;
    auto accept_tokens_d = output_token_ids_d;

    const int64_t token_stride = target_token_ids_d_t.size(1);
    auto          target_3d = target_token_ids_d_t.reshape({(long)batch_size, (long)propose_step_ + 1, token_stride});
    auto          target_last_col = target_3d.select(2, token_stride - 1);

    auto row_idx     = torch::arange(batch_size, target_token_ids_d_t.options().dtype(torch::kLong));
    auto col_idx     = (accept_len_d - 1).clamp(0, static_cast<int64_t>(propose_step_)).to(torch::kLong);
    auto last_tokens = target_last_col.gather(1, col_idx.unsqueeze(1)).squeeze(1);
    accept_tokens_d.index_put_({row_idx, col_idx}, last_tokens);

    bool has_force = false;
    for (const auto& stream : streams) {
        if (stream->forceSpAccept()) {
            has_force = true;
            break;
        }
    }
    if (has_force) {
        RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.forceSpAccept");
        // Build the mask on CPU and H2D once; per-element CUDA writes would lower to many tiny memcpys.
        std::vector<uint8_t> force_mask_cpu(batch_size, 0);
        {
            int idx = 0;
            for (const auto& stream : streams) {
                if (stream->forceSpAccept()) {
                    force_mask_cpu[idx] = 1;
                }
                idx++;
            }
        }
        // Pin so .to(non_blocking) is a real async DMA; pageable source strips non_blocking.
        auto force_mask_host =
            torch::from_blob(force_mask_cpu.data(), {batch_size}, torch::TensorOptions().dtype(torch::kBool))
                .clone()
                .pin_memory();
        auto force_mask    = force_mask_host.to(target_device, /*non_blocking=*/true);
        auto target_bonus  = target_last_col.select(1, static_cast<int64_t>(propose_step_));
        auto forced_tokens = torch::cat({draft_token_ids_d_t, target_bonus.unsqueeze(1)}, 1);
        auto force_mask_2d = force_mask.unsqueeze(1).expand_as(accept_tokens_d);
        accept_tokens_d    = torch::where(force_mask_2d, forced_tokens, accept_tokens_d);
        accept_len_d       = torch::where(
            force_mask, torch::full_like(accept_len_d, static_cast<int32_t>(propose_step_ + 1)), accept_len_d);
    }

    sample_output.accept_tokens = accept_tokens_d;
    sample_output.accept_len    = accept_len_d;

    sample_output.accept_tokens_cpu = sample_output.accept_tokens.to(torch::kCPU, /*non_blocking=*/true);
    sample_output.accept_len_cpu    = sample_output.accept_len.to(torch::kCPU, /*non_blocking=*/true);
    sample_output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {
    (void)sample_output;
    (void)streams;
    (void)draft_sampler_output;
    (void)target_sampler_output;
}

}  // namespace speculative
}  // namespace rtp_llm

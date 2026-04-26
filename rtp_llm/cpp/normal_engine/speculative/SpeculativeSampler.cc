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
    torch::Device host_device   = torch::Device(torch::kCPU);

    int batch_size = streams.size();

    // target_sampler_output.token_ids may be a CUDA tensor (Sampler keeps it on GPU to avoid
    // D2H sync during sampling). Move to CPU once here for data_ptr access.
    const torch::Tensor target_token_ids_cpu = target_sampler_output.token_ids.is_cuda() ?
                                                   target_sampler_output.token_ids.to(host_device, true) :
                                                   target_sampler_output.token_ids;
    const int*          new_all_token_ids    = target_token_ids_cpu.data_ptr<int32_t>();
    const size_t        token_stride         = target_token_ids_cpu.size(1);

    auto draft_token_ids  = draft_sampler_output.token_ids;
    auto target_token_ids = target_sampler_output.token_ids;

    auto draft_token_probs  = draft_sampler_output.all_probs;
    auto target_token_probs = target_sampler_output.all_probs;

    auto draft_token_ids_d_t = draft_token_ids.to(target_device, true);

    auto target_token_ids_d_t = target_sampler_output.token_ids;
    if (!target_token_ids_d_t.is_cuda()) {
        target_token_ids_d_t = target_token_ids_d_t.to(target_device, true);
    }

    torch::Tensor do_sample  = torch::zeros({(long)batch_size}, torch::TensorOptions().dtype(torch::kBool));
    int           stream_idx = 0;
    for (const GenerateStreamPtr& stream : streams) {
        do_sample[stream_idx] = !stream->generateConfig()->top1();
        stream_idx++;
    }
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

    {
        RTP_LLM_PROFILE_SCOPE("speculative_sampler.batchSample.post_rejection_sampling");
        // back to host
        torch::Tensor output_token_ids_h = output_token_ids_d.to(host_device, true);

        // sync here
        torch::Tensor output_accepted_token_num_h = output_accepted_token_num_d.to(host_device, false);

        torch::Tensor draft_token_ids_h;
        for (const GenerateStreamPtr& stream : streams) {
            if (stream->forceSpAccept()) {
                draft_token_ids_h = draft_token_ids.cpu();
                break;
            }
        }

        stream_idx = 0;
        for (const GenerateStreamPtr& stream : streams) {
            torch::Tensor accept_tokens;
            size_t        accept_len = 0;

            if (stream->forceSpAccept()) {
                accept_len    = propose_step_ + 1;
                accept_tokens = torch::empty({1, (int64_t)accept_len}, torch::TensorOptions().dtype(torch::kInt32));
                memcpy(accept_tokens.data_ptr<int>(),
                       draft_token_ids_h.data_ptr<int32_t>() + stream_idx * propose_step_,
                       sizeof(int32_t) * propose_step_);
                accept_tokens.data_ptr<int32_t>()[accept_len - 1] =
                    new_all_token_ids[(stream_idx * (propose_step_ + 1) + accept_len - 1) * token_stride + token_stride
                                      - 1];
            } else {
                accept_len    = output_accepted_token_num_h[stream_idx].item<int32_t>();
                accept_tokens = torch::empty({1, (int64_t)accept_len}, torch::TensorOptions().dtype(torch::kInt32));
                memcpy(accept_tokens.data_ptr<int>(),
                       output_token_ids_h[stream_idx].data_ptr<int32_t>(),
                       sizeof(int32_t) * accept_len);
            }

            sample_output.accept_tokens.push_back(accept_tokens);
            sample_output.accept_len.push_back(accept_len);
            stream_idx++;
        }
    }
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}

}  // namespace speculative
}  // namespace rtp_llm

#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"

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
    torch::Device target_device = getTorchCudaDevice();
    torch::Device host_device   = torch::Device(torch::kCPU);

    int batch_size = streams.size();

    auto draft_token_ids  = draft_sampler_output.token_ids;
    auto target_token_ids = target_sampler_output.token_ids;

    auto draft_token_probs  = draft_sampler_output.all_probs;
    auto target_token_probs = target_sampler_output.all_probs;

    // Rejection sampling consumes target tokens directly. Keeping the
    // batch/score-row mapping inside the kernel avoids rebuilding the bonus
    // token with host-side offsets when multiple MTP streams share a batch.
    auto          draft_token_ids_d_t    = draft_token_ids.to(target_device).clone();
    auto          target_token_ids_d_t   = target_token_ids.to(target_device).clone();
    auto          draft_token_probs_d_t  = draft_token_probs;
    auto          target_token_probs_d_t = target_token_probs;
    auto          do_sample_h            = torch::empty(
        {(long)batch_size}, torch::TensorOptions().dtype(torch::kBool).pinned_memory(true));
    int stream_idx = 0;
    for (const auto& stream : streams) {
        do_sample_h[stream_idx++] =
            stream->generateConfig()->do_sample && !stream->generateConfig()->top1();
    }
    auto          do_sample_d            = do_sample_h.to(target_device).clone();
    auto          rand_options           = torch::TensorOptions().device(target_device).dtype(torch::kFloat);
    torch::Tensor uniform_samples_d      = torch::rand({(long)batch_size, (long)propose_step_ + 1}, rand_options);

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
    torch::Tensor output_token_ids_d     = torch::zeros({(long)batch_size, (long)propose_step_ + 1},
                                                    torch::TensorOptions().device(target_device).dtype(torch::kInt32));
    torch::Tensor output_accepted_token_num_d =
        torch::zeros({(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32));
    execRejectionSampling({draft_token_probs_d_t,
                           draft_token_ids_d_t,
                           uniform_samples_d,
                           target_token_probs_d_t,
                           target_token_ids_d_t,
                           output_token_ids_d,
                           output_accepted_token_num_d,
                           do_sample_d});

    // back to host
    torch::Tensor output_token_ids_h         = output_token_ids_d.to(host_device, true);
    torch::Tensor output_accepted_token_num_h = output_accepted_token_num_d.to(host_device);  // implicit sync here

    torch::Tensor draft_token_ids_h;
    torch::Tensor target_token_ids_h;
    for (const GenerateStreamPtr& stream : streams) {
        if (stream->forceSpAccept()) {
            draft_token_ids_h = draft_token_ids.cpu().clone();
            target_token_ids_h = target_token_ids_d_t.to(host_device);
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
            const size_t token_stride = target_token_ids_h.size(1);
            accept_tokens.data_ptr<int>()[accept_len - 1] =
                target_token_ids_h.data_ptr<int32_t>()[
                    (stream_idx * (propose_step_ + 1) + propose_step_) * token_stride + token_stride - 1];
        } else {
            accept_len    = output_accepted_token_num_h[stream_idx].item<int32_t>();
            accept_tokens = torch::empty({1, (int64_t)accept_len}, torch::TensorOptions().dtype(torch::kInt32));
            memcpy(accept_tokens.data_ptr<int>(),
                   output_token_ids_h[stream_idx].data_ptr<int32_t>(),
                   sizeof(int32_t) * accept_len);
        }

        sample_output.accept_tokens.push_back(std::move(accept_tokens));
        sample_output.accept_len.push_back(accept_len);
        stream_idx++;
    }
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}
}  // namespace speculative
}  // namespace rtp_llm
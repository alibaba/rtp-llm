#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {
namespace speculative {
SpeculativeSamplerOutput SpeculativeSampler::forward(const std::list<GenerateStreamPtr>& streams,
                                                     SamplerOutput&                      draft_sampler_output,
                                                     SamplerOutput&                      target_sampler_output) const {
    // bool contain_topp        = false;
    // bool force_stream_sample = device_->initParams().sp_config.force_stream_sample;
    // torch::Device            target_device       = device_->getTorchDevice();
    SpeculativeSamplerOutput sample_output;

    // for (const GenerateStreamPtr& stream : streams) {
    //     contain_topp |= !stream->generateConfig()->top1();
    // }
    // if (!contain_topp || force_stream_sample || target_device != torch::Device(torch::kCUDA)) {
    //     streamSample(sample_output, streams, draft_sampler_output, target_sampler_output);
    // } else {
    batchSample(sample_output, streams, draft_sampler_output, target_sampler_output);
    // }
    return sample_output;
}

void SpeculativeSampler::batchSample(SpeculativeSamplerOutput&           sample_output,
                                     const std::list<GenerateStreamPtr>& streams,
                                     SamplerOutput&                      draft_sampler_output,
                                     SamplerOutput&                      target_sampler_output) const {
    // TODO(yinzhi): stream from prefill part should not use batch sample
    torch::Device target_device = device_->getTorchDevice();
    torch::Device host_device   = torch::Device(torch::kCPU);

    int batch_size = streams.size();

    const int*   new_all_token_ids = target_sampler_output.token_ids->data<int32_t>();
    const size_t token_stride      = target_sampler_output.token_ids->shape()[1];

    auto draft_token_ids  = draft_sampler_output.token_ids;
    auto target_token_ids = target_sampler_output.token_ids;

    auto draft_token_probs  = draft_sampler_output.all_probs;
    auto target_token_probs = target_sampler_output.all_probs;

    // TODO(yinzhi): should change to device buffer to avoid copy data
    auto draft_token_ids_d  = device_->clone({*draft_token_ids, AllocationType::DEVICE});
    auto target_token_ids_d = device_->clone({*target_token_ids, AllocationType::DEVICE});

    // note target token probs is already on device
    auto target_token_probs_d = target_token_probs;
    auto draft_token_probs_d  = draft_token_probs;

    // prepare data for chain speculative sampling
    auto          draft_token_ids_d_t    = Buffer2torchTensor(draft_token_ids_d, false);
    auto          draft_token_probs_d_t  = Buffer2torchTensor(draft_token_probs_d, false);
    auto          target_token_probs_d_t = Buffer2torchTensor(target_token_probs_d, false);
    torch::Tensor uniform_samples_d      = torch::rand({(long)batch_size, (long)propose_step_ + 1},
                                                  torch::TensorOptions().device(target_device).dtype(torch::kFloat));
    torch::Tensor output_token_ids_d     = torch::zeros({(long)batch_size, (long)propose_step_ + 1},
                                                    torch::TensorOptions().device(target_device).dtype(torch::kInt32));
    torch::Tensor output_accepted_token_num_d =
        torch::zeros({(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32));
    torch::Tensor output_emitted_token_num_d =
        torch::zeros({(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32));

    device_->chainSpeculativeSampling({draft_token_probs_d_t,
                                       draft_token_ids_d_t,
                                       uniform_samples_d,
                                       target_token_probs_d_t,
                                       output_token_ids_d,
                                       output_accepted_token_num_d,
                                       output_emitted_token_num_d});

    // back to host
    torch::Tensor output_token_ids_h         = output_token_ids_d.to(host_device).contiguous();
    torch::Tensor output_emitted_token_num_h = output_emitted_token_num_d.to(host_device).contiguous();

    for (int stream_idx = 0; stream_idx < batch_size; stream_idx++) {
        size_t accept_len = output_emitted_token_num_h[stream_idx].item<int32_t>();

        BufferPtr accept_tokens = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {1, accept_len}, rtp_llm::AllocationType::HOST}, {"accept_tokens"});

        memcpy(accept_tokens->data(), output_token_ids_h[stream_idx].data_ptr<int32_t>(), sizeof(int32_t) * accept_len);

        if (accept_len == propose_step_ + 1) {
            *accept_tokens->dataWithOffset<int32_t>(accept_len - 1) =
                new_all_token_ids[(stream_idx * (propose_step_ + 1) + accept_len - 1) * token_stride + token_stride
                                  - 1];
        }

        sample_output.accept_tokens.push_back(accept_tokens);
        sample_output.accept_len.push_back(accept_len);
    }
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}
}  // namespace speculative
}  // namespace rtp_llm
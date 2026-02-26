#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/cpp/kernels/vocab_prune/mapping.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

#if USING_CUDA
#include "rtp_llm/cpp/kernels/speculative_sampling/sampling.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/speculative_sampling/sampling.cuh"
#endif

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
    mappingDraft2Target({torchTensor2Buffer(output.token_ids), d2t_map_, batch_size, 0, 1});

    return output;
}

SpeculativeSamplerOutput SpeculativeSampler::forward(const std::list<GenerateStreamPtr>& streams,
                                                     SamplerOutput&                      draft_sampler_output,
                                                     SamplerOutput&                      target_sampler_output) {
    SpeculativeSamplerOutput sample_output;
    batchSample(sample_output, streams, draft_sampler_output, target_sampler_output);

    return sample_output;
}

void FastTopKSampler::mappingDraft2Target(const MappingDraft2TargetParams& params) const {
    if (!params.d2t_map || !params.d2t_map->size()) {
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(params.tokens->size() == params.batch_size * params.token_stride,
                            "tokens size mismatch(expect: %d, actual: %d)",
                            params.batch_size * params.token_stride,
                            params.tokens->size());

    if (params.tokens->where() != MemoryType::MEMORY_GPU && params.d2t_map->where() != MemoryType::MEMORY_GPU) {
        RTP_LLM_CHECK_WITH_INFO(params.tokens->size() == params.batch_size * params.token_stride,
                                "tokens size mismatch(expect: %d, actual: %d)",
                                params.batch_size * params.token_stride,
                                params.tokens->size());
        int*     tokens  = params.tokens->data<int32_t>();
        int64_t* d2t_map = params.d2t_map->data<int64_t>();

        for (int i = 0; i < params.batch_size; i++) {
            for (int j = params.token_offset; j < params.token_stride; j++) {
                int idx     = i * params.token_stride + j;
                tokens[idx] = d2t_map[tokens[idx]];
            }
        }
    } else if (params.tokens->where() == MemoryType::MEMORY_GPU && params.d2t_map->where() == MemoryType::MEMORY_GPU) {
        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        if (params.tokens->type() == DataType::TYPE_INT32) {
            invokeMappingDraft2Target(params.tokens->data<int32_t>(),
                                      params.batch_size,
                                      params.token_offset,
                                      params.token_stride,
                                      params.d2t_map->data<int64_t>(),
                                      stream);
        } else if (params.tokens->type() == DataType::TYPE_INT64) {
            invokeMappingDraft2Target(params.tokens->data<int64_t>(),
                                      params.batch_size,
                                      params.token_offset,
                                      params.token_stride,
                                      params.d2t_map->data<int64_t>(),
                                      stream);
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

void SpeculativeSampler::batchSample(SpeculativeSamplerOutput&           sample_output,
                                     const std::list<GenerateStreamPtr>& streams,
                                     SamplerOutput&                      draft_sampler_output,
                                     SamplerOutput&                      target_sampler_output) const {
    torch::Device target_device = device_->getTorchDevice();
    torch::Device host_device   = torch::Device(torch::kCPU);

    int batch_size = streams.size();

    const int*   new_all_token_ids = target_sampler_output.token_ids->data<int32_t>();
    const size_t token_stride      = target_sampler_output.token_ids->shape()[1];

    auto draft_token_ids  = draft_sampler_output.token_ids;
    auto target_token_ids = target_sampler_output.token_ids;

    auto draft_token_probs  = draft_sampler_output.all_probs;
    auto target_token_probs = target_sampler_output.all_probs;

    auto      draft_token_ids_d = device_->clone({*draft_token_ids, AllocationType::DEVICE});
    BufferPtr target_token_ids_d;

    if (target_token_ids->where() == MemoryType::MEMORY_CPU) {
        target_token_ids_d = device_->clone({*target_token_ids, AllocationType::DEVICE});
    } else {
        target_token_ids_d = target_token_ids;
    }

    torch::Tensor do_sample  = torch::zeros({(long)batch_size}, torch::TensorOptions().dtype(torch::kBool));
    int           stream_idx = 0;
    for (const GenerateStreamPtr& stream : streams) {
        do_sample[stream_idx] = !stream->generateConfig()->top1();
        stream_idx++;
    }
    auto do_sample_d = do_sample.to(target_device);

    // note target token probs is already on device
    auto target_token_probs_d = target_token_probs;
    auto draft_token_probs_d  = draft_token_probs;

    // prepare data for chain speculative sampling
    auto          draft_token_ids_d_t    = Buffer2torchTensor(draft_token_ids_d, false);
    auto          draft_token_probs_d_t  = Buffer2torchTensor(draft_token_probs_d, false);
    auto          target_token_ids_d_t   = Buffer2torchTensor(target_token_ids_d, false);
    auto          target_token_probs_d_t = Buffer2torchTensor(target_token_probs_d, false);
    torch::Tensor uniform_samples_d =
        torch::rand({(long)batch_size, (long)propose_step_ + 1},
                    torch::TensorOptions().device(target_device).dtype(torch::kFloat).requires_grad(false));
    torch::Tensor output_token_ids_d =
        torch::zeros({(long)batch_size, (long)propose_step_ + 1},
                     torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));
    torch::Tensor output_accepted_token_num_d = torch::zeros(
        {(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32).requires_grad(false));

    if (draft_token_probs_d_t.size(2) != target_token_probs_d_t.size(2)) {
        auto draft_probs_padding =
            torch::zeros({(long)batch_size, draft_token_probs_d_t.size(1), target_token_probs_d_t.size(2)},
                         draft_token_probs_d_t.options());
        torch::Tensor d2t_map_d_t = Buffer2torchTensor(d2t_map_, false);
        // draft_probs_padding[:, :, d2t_map_d_t] = draft_probs_d_t
        draft_probs_padding.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), d2t_map_d_t},
                                       draft_token_probs_d_t);
        draft_token_probs_d_t = draft_probs_padding;
    }

    rejectionSampling({
        draft_token_probs_d_t,
        draft_token_ids_d_t,
        uniform_samples_d,
        target_token_probs_d_t,
        target_token_ids_d_t,
        output_token_ids_d,
        output_accepted_token_num_d,
        do_sample_d,
    });

    // back to host
    torch::Tensor output_token_ids_h          = output_token_ids_d.to(host_device).contiguous();
    torch::Tensor output_accepted_token_num_h = output_accepted_token_num_d.to(host_device).contiguous();

    BufferPtr draft_token_ids_h;
    for (const GenerateStreamPtr& stream : streams) {
        if (stream->forceSpAccept()) {
            draft_token_ids_h = device_->clone({*draft_token_ids, AllocationType::HOST});
            break;
        }
    }

    stream_idx = 0;
    for (const GenerateStreamPtr& stream : streams) {
        BufferPtr accept_tokens;
        size_t    accept_len = 0;

        if (stream->forceSpAccept()) {
            accept_len    = propose_step_ + 1;
            accept_tokens = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, accept_len}, rtp_llm::AllocationType::HOST}, {"accept_tokens"});
            memcpy(accept_tokens->data(),
                   draft_token_ids_h->dataWithOffset<int32_t>(stream_idx * propose_step_),
                   sizeof(int32_t) * propose_step_);
            *accept_tokens->dataWithOffset<int32_t>(accept_len - 1) =
                new_all_token_ids[(stream_idx * (propose_step_ + 1) + accept_len - 1) * token_stride + token_stride
                                  - 1];
        } else {
            accept_len    = output_accepted_token_num_h[stream_idx].item<int32_t>();
            accept_tokens = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, accept_len}, rtp_llm::AllocationType::HOST}, {"accept_tokens"});
            memcpy(accept_tokens->data(),
                   output_token_ids_h[stream_idx].data_ptr<int32_t>(),
                   sizeof(int32_t) * accept_len);
        }

        sample_output.accept_tokens.push_back(accept_tokens);
        sample_output.accept_len.push_back(accept_len);
        stream_idx++;
    }
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams,
                                      SamplerOutput&                      draft_sampler_output,
                                      SamplerOutput&                      target_sampler_output) const {}

void SpeculativeSampler::rejectionSampling(const RejectionSamplingParams& params) const {
    RTP_LLM_CHECK(params.draft_probs_d.is_cuda());
    RTP_LLM_CHECK(params.draft_token_ids_d.is_cuda());
    RTP_LLM_CHECK(params.target_probs_d.is_cuda());

    RTP_LLM_CHECK(params.draft_probs_d.dtype() == torch::kFloat32);
    RTP_LLM_CHECK(params.draft_token_ids_d.dtype() == torch::kInt32);
    RTP_LLM_CHECK(params.target_probs_d.dtype() == torch::kFloat32);

    RTP_LLM_CHECK(params.draft_probs_d.dim() == 3);
    RTP_LLM_CHECK(params.draft_token_ids_d.dim() == 2);
    RTP_LLM_CHECK(params.target_probs_d.dim() == 3);

    int  batch_size             = params.draft_probs_d.size(0);
    int  num_speculative_tokens = params.draft_probs_d.size(1);
    int  target_vocab_size      = params.target_probs_d.size(2);
    int  target_token_stride    = params.target_token_ids_d.size(1);
    auto stream                 = c10::cuda::getCurrentCUDAStream().stream();

    RTP_LLM_CHECK(params.draft_token_ids_d.size(0) == batch_size);
    RTP_LLM_CHECK(params.draft_token_ids_d.size(1) == num_speculative_tokens);
    RTP_LLM_CHECK(params.target_probs_d.size(0) == batch_size);
    RTP_LLM_CHECK(params.target_probs_d.size(1) == num_speculative_tokens + 1);
    RTP_LLM_CHECK(params.draft_probs_d.size(2) == target_vocab_size);

    check_cuda_value(invokeRejectionSampling(params.draft_probs_d.data_ptr<float>(),
                                             params.draft_token_ids_d.data_ptr<int32_t>(),
                                             params.uniform_samples_d.data_ptr<float>(),
                                             params.target_probs_d.data_ptr<float>(),
                                             params.target_token_ids_d.data_ptr<int32_t>(),
                                             target_token_stride,
                                             params.output_token_ids_d.data_ptr<int32_t>(),
                                             params.output_accepted_token_num_d.data_ptr<int32_t>(),
                                             params.do_sample_d.data_ptr<bool>(),
                                             batch_size,
                                             num_speculative_tokens,
                                             target_vocab_size,
                                             stream));
}
}  // namespace speculative
}  // namespace rtp_llm
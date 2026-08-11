#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <algorithm>
#include <exception>
#include <limits>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"

namespace rtp_llm {
namespace speculative {

int validateSpeculativeEmittedTokenCount(int emitted_token_count, size_t propose_step) {
    RTP_LLM_CHECK_WITH_INFO(propose_step < static_cast<size_t>(std::numeric_limits<int>::max()),
                            "speculative sampling proposal step does not fit emitted token count: %zu",
                            propose_step);
    const int max_emitted_token_count = static_cast<int>(propose_step) + 1;
    RTP_LLM_CHECK_WITH_INFO(emitted_token_count >= 1 && emitted_token_count <= max_emitted_token_count,
                            "speculative sampling emitted token count must be in [1, %d], got %d",
                            max_emitted_token_count,
                            emitted_token_count);
    return emitted_token_count;
}

ValidatedSpeculativeSamplerInputs validateSpeculativeSamplerInputs(size_t               batch_size,
                                                                   size_t               propose_step,
                                                                   const SamplerOutput& draft_sampler_output,
                                                                   const SamplerOutput& target_sampler_output,
                                                                   bool copy_draft_token_ids_to_cpu) {
    RTP_LLM_CHECK_WITH_INFO(batch_size > 0, "speculative sampling requires a non-empty batch");
    RTP_LLM_CHECK_WITH_INFO(propose_step > 0, "speculative sampling requires propose_step > 0");
    RTP_LLM_CHECK_WITH_INFO(batch_size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                            "speculative sampling batch size is too large: %zu",
                            batch_size);
    RTP_LLM_CHECK_WITH_INFO(propose_step < static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                            "speculative sampling proposal step is too large: %zu",
                            propose_step);
    RTP_LLM_CHECK_WITH_INFO(propose_step < static_cast<size_t>(std::numeric_limits<int>::max()),
                            "speculative sampling proposal step does not fit accept_len: %zu",
                            propose_step);

    const auto& draft_token_ids    = draft_sampler_output.token_ids;
    const auto& target_token_ids   = target_sampler_output.token_ids;
    const auto& draft_token_probs  = draft_sampler_output.all_probs;
    const auto& target_token_probs = target_sampler_output.all_probs;

    RTP_LLM_CHECK_WITH_INFO(draft_token_ids.defined(), "draft token IDs are undefined");
    RTP_LLM_CHECK_WITH_INFO(target_token_ids.defined(), "target token IDs are undefined");
    RTP_LLM_CHECK_WITH_INFO(draft_token_probs.defined(), "draft probability tensor is undefined");
    RTP_LLM_CHECK_WITH_INFO(target_token_probs.defined(), "target probability tensor is undefined");

    RTP_LLM_CHECK_WITH_INFO(draft_token_probs.dim() == 3,
                            "draft probabilities must have shape [batch, step, vocab], got rank %d",
                            draft_token_probs.dim());
    RTP_LLM_CHECK_WITH_INFO(target_token_probs.dim() == 3,
                            "target probabilities must have shape [batch, step + 1, vocab], got rank %d",
                            target_token_probs.dim());
    RTP_LLM_CHECK_WITH_INFO(draft_token_probs.size(0) == static_cast<int64_t>(batch_size)
                                && draft_token_probs.size(1) == static_cast<int64_t>(propose_step),
                            "draft probability shape mismatch: expected [%zu, %zu, vocab], got [%ld, %ld, %ld]",
                            batch_size,
                            propose_step,
                            draft_token_probs.size(0),
                            draft_token_probs.size(1),
                            draft_token_probs.size(2));
    RTP_LLM_CHECK_WITH_INFO(target_token_probs.size(0) == static_cast<int64_t>(batch_size)
                                && target_token_probs.size(1) == static_cast<int64_t>(propose_step + 1),
                            "target probability shape mismatch: expected [%zu, %zu, vocab], got [%ld, %ld, %ld]",
                            batch_size,
                            propose_step + 1,
                            target_token_probs.size(0),
                            target_token_probs.size(1),
                            target_token_probs.size(2));

    const int64_t draft_vocab_size  = draft_token_probs.size(2);
    const int64_t target_vocab_size = target_token_probs.size(2);
    RTP_LLM_CHECK_WITH_INFO(draft_vocab_size > 0, "draft probability vocab dimension must be positive");
    RTP_LLM_CHECK_WITH_INFO(target_vocab_size > 0, "target probability vocab dimension must be positive");
    RTP_LLM_CHECK_WITH_INFO(draft_vocab_size == target_vocab_size,
                            "target/proposal probability vocab mismatch before speculative sampling: target=%ld, "
                            "proposal=%ld",
                            target_vocab_size,
                            draft_vocab_size);
    RTP_LLM_CHECK_WITH_INFO(draft_token_probs.scalar_type() == torch::kFloat32
                                && target_token_probs.scalar_type() == torch::kFloat32,
                            "speculative sampling probabilities must be float32: draft=%s, target=%s",
                            c10::toString(draft_token_probs.scalar_type()),
                            c10::toString(target_token_probs.scalar_type()));
    RTP_LLM_CHECK_WITH_INFO(draft_token_probs.is_cuda() && target_token_probs.is_cuda(),
                            "speculative sampling probabilities must be accelerator tensors: draft=%s, target=%s",
                            draft_token_probs.device().str().c_str(),
                            target_token_probs.device().str().c_str());
    RTP_LLM_CHECK_WITH_INFO(draft_token_probs.device() == target_token_probs.device(),
                            "draft and target probabilities must be on the same device: draft=%s, target=%s",
                            draft_token_probs.device().str().c_str(),
                            target_token_probs.device().str().c_str());
    RTP_LLM_CHECK_WITH_INFO(draft_token_probs.is_contiguous() && target_token_probs.is_contiguous(),
                            "speculative sampling probability tensors must be contiguous");

    RTP_LLM_CHECK_WITH_INFO(draft_token_ids.dim() == 2,
                            "draft token IDs must have shape [batch, step], got rank %d",
                            draft_token_ids.dim());
    RTP_LLM_CHECK_WITH_INFO(draft_token_ids.size(0) == static_cast<int64_t>(batch_size)
                                && draft_token_ids.size(1) == static_cast<int64_t>(propose_step),
                            "draft token ID shape mismatch: expected [%zu, %zu], got [%ld, %ld]",
                            batch_size,
                            propose_step,
                            draft_token_ids.size(0),
                            draft_token_ids.size(1));
    RTP_LLM_CHECK_WITH_INFO(draft_token_ids.scalar_type() == torch::kInt32,
                            "draft token IDs must be int32, got %s",
                            c10::toString(draft_token_ids.scalar_type()));
    RTP_LLM_CHECK_WITH_INFO(draft_token_ids.is_contiguous(),
                            "draft token IDs must be contiguous for speculative sampling");
    RTP_LLM_CHECK_WITH_INFO(draft_token_ids.is_cpu() || draft_token_ids.device() == draft_token_probs.device(),
                            "draft token IDs must be on CPU or the probability tensor device: ids=%s, probs=%s",
                            draft_token_ids.device().str().c_str(),
                            draft_token_probs.device().str().c_str());

    const uint64_t target_row_count = static_cast<uint64_t>(batch_size) * (static_cast<uint64_t>(propose_step) + 1);
    RTP_LLM_CHECK_WITH_INFO(target_row_count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                            "target token row count is too large: %zu * (%zu + 1)",
                            batch_size,
                            propose_step);
    RTP_LLM_CHECK_WITH_INFO(target_token_ids.dim() == 2,
                            "target token IDs must have shape [batch * (step + 1), token_stride], got rank %d",
                            target_token_ids.dim());
    RTP_LLM_CHECK_WITH_INFO(target_token_ids.size(0) == static_cast<int64_t>(target_row_count)
                                && target_token_ids.size(1) > 0,
                            "target token ID shape mismatch: expected [%lu, token_stride>0], got [%ld, %ld]",
                            target_row_count,
                            target_token_ids.size(0),
                            target_token_ids.size(1));
    RTP_LLM_CHECK_WITH_INFO(target_token_ids.scalar_type() == torch::kInt32,
                            "target token IDs must be int32, got %s",
                            c10::toString(target_token_ids.scalar_type()));
    RTP_LLM_CHECK_WITH_INFO(target_token_ids.is_contiguous(),
                            "target token IDs must be contiguous for speculative sampling");
    RTP_LLM_CHECK_WITH_INFO(target_token_ids.is_cpu() || target_token_ids.device() == target_token_probs.device(),
                            "target token IDs must be on CPU or the probability tensor device: ids=%s, probs=%s",
                            target_token_ids.device().str().c_str(),
                            target_token_probs.device().str().c_str());

    torch::Tensor draft_token_ids_cpu;
    if (copy_draft_token_ids_to_cpu || draft_token_ids.is_cpu()) {
        draft_token_ids_cpu          = draft_token_ids.to(torch::kCPU);
        const int64_t draft_min_token = draft_token_ids_cpu.min().item<int64_t>();
        const int64_t draft_max_token = draft_token_ids_cpu.max().item<int64_t>();
        RTP_LLM_CHECK_WITH_INFO(draft_min_token >= 0 && draft_max_token < target_vocab_size,
                                "draft token IDs are outside [0, vocab_size): min=%ld, max=%ld, vocab_size=%ld",
                                draft_min_token,
                                draft_max_token,
                                target_vocab_size);
    }

    if (target_token_ids.is_cpu()) {
        const torch::Tensor target_candidates = target_token_ids.select(1, target_token_ids.size(1) - 1);
        const int64_t       target_min_token   = target_candidates.min().item<int64_t>();
        const int64_t       target_max_token   = target_candidates.max().item<int64_t>();
        RTP_LLM_CHECK_WITH_INFO(target_min_token >= 0 && target_max_token < target_vocab_size,
                                "target token IDs are outside [0, vocab_size): min=%ld, max=%ld, vocab_size=%ld",
                                target_min_token,
                                target_max_token,
                                target_vocab_size);
    }

    return {draft_token_ids_cpu,
            static_cast<size_t>(target_token_ids.size(1)),
            static_cast<size_t>(target_vocab_size)};
}

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

    const bool force_sp_accept = std::any_of(streams.begin(), streams.end(), [](const GenerateStreamPtr& stream) {
        return stream->forceSpAccept();
    });
    const size_t batch_size_value = streams.size();
    const auto   validated_inputs = validateSpeculativeSamplerInputs(
        batch_size_value, propose_step_, draft_sampler_output, target_sampler_output, force_sp_accept);
    RTP_LLM_CHECK_WITH_INFO(batch_size_value <= static_cast<size_t>(std::numeric_limits<int>::max()),
                            "speculative sampling batch size does not fit int: %zu",
                            batch_size_value);
    const int    batch_size   = static_cast<int>(batch_size_value);
    const size_t token_stride = validated_inputs.token_stride;

    auto draft_token_ids = draft_sampler_output.token_ids;

    auto draft_token_probs  = draft_sampler_output.all_probs;
    auto target_token_probs = target_sampler_output.all_probs;

    // prepare data for chain speculative sampling
    auto          draft_token_ids_d_t    = draft_token_ids.to(target_device).clone();
    auto          draft_token_probs_d_t  = draft_token_probs;
    auto          target_token_probs_d_t = target_token_probs;
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
    torch::Tensor output_emitted_token_num_d =
        torch::zeros({(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32));

    execChainSpeculativeSampling({draft_token_probs_d_t,
                                  draft_token_ids_d_t,
                                  uniform_samples_d,
                                  target_token_probs_d_t,
                                  output_token_ids_d,
                                  output_accepted_token_num_d,
                                  output_emitted_token_num_d});

    // back to host
    torch::Tensor output_token_ids_h         = output_token_ids_d.to(host_device, true);
    torch::Tensor target_token_ids_h         = target_sampler_output.token_ids.to(host_device, true);
    torch::Tensor output_emitted_token_num_h = output_emitted_token_num_d.to(host_device);  // implicit sync here

    // Any row's final column can become the fallback target token. This scan runs
    // after the existing emitted-count synchronization, so it adds no D2H sync.
    const torch::Tensor target_candidates = target_token_ids_h.select(1, target_token_ids_h.size(1) - 1);
    const int64_t       target_min_token   = target_candidates.min().item<int64_t>();
    const int64_t       target_max_token   = target_candidates.max().item<int64_t>();
    RTP_LLM_CHECK_WITH_INFO(target_min_token >= 0
                                && static_cast<uint64_t>(target_max_token) < validated_inputs.vocab_size,
                            "target token IDs are outside [0, vocab_size): min=%ld, max=%ld, vocab_size=%zu",
                            target_min_token,
                            target_max_token,
                            validated_inputs.vocab_size);
    const int* new_all_token_ids = target_token_ids_h.data_ptr<int32_t>();

    int stream_idx = 0;
    for (const GenerateStreamPtr& stream : streams) {
        torch::Tensor accept_tokens;
        int           accept_len = 0;

        if (stream->forceSpAccept()) {
            accept_len    = static_cast<int>(propose_step_) + 1;
            accept_tokens = torch::empty({1, (int64_t)accept_len}, torch::TensorOptions().dtype(torch::kInt32));
            memcpy(accept_tokens.data_ptr<int>(),
                   validated_inputs.draft_token_ids_cpu.data_ptr<int32_t>() + stream_idx * propose_step_,
                   sizeof(int32_t) * propose_step_);
        } else {
            try {
                accept_len = validateSpeculativeEmittedTokenCount(
                    output_emitted_token_num_h[stream_idx].item<int32_t>(), propose_step_);
            } catch (const std::exception& e) {
                // The remainder of this executor batch cannot be assembled safely. Surface an
                // error to every affected request before propagating the invariant violation.
                for (const auto& affected_stream : streams) {
                    affected_stream->reportError(ErrorCode::EXECUTION_EXCEPTION, e.what());
                }
                throw;
            }
            accept_tokens = torch::empty({1, (int64_t)accept_len}, torch::TensorOptions().dtype(torch::kInt32));
            memcpy(accept_tokens.data_ptr<int>(),
                   output_token_ids_h[stream_idx].data_ptr<int32_t>(),
                   sizeof(int32_t) * accept_len);
        }

        // The device kernel already writes the q-p correction token when a draft is rejected.
        // It leaves only the bonus slot empty when every draft token is accepted.
        if (accept_len == static_cast<int>(propose_step_) + 1) {
            accept_tokens.data_ptr<int>()[accept_len - 1] =
                new_all_token_ids[(stream_idx * (propose_step_ + 1) + accept_len - 1) * token_stride + token_stride - 1];
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

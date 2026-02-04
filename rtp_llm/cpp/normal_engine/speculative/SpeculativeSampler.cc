#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

#include <cstdlib>

namespace rtp_llm {

namespace {

// 方案1：松弛投机采样 - 通过缩小 draft_probs 来提高接受率
// 接受条件: u * p < q，缩小 p 使条件更容易满足
// SPECULATIVE_RELAXATION_FACTOR < 1 时放松接受条件（如 0.1 表示接受率提高约10倍）
// SPECULATIVE_RELAXATION_FACTOR > 1 时收紧接受条件
float getSpeculativeRelaxationFactor() {
    static float value = []() {
        const char* env = std::getenv("SPECULATIVE_RELAXATION_FACTOR");
        if (env) {
            RTP_LLM_LOG_INFO("SPECULATIVE_RELAXATION_FACTOR = %f (方案1: 直接缩放已启用)", std::atof(env));
        }
        return env ? std::atof(env) : 1.0f;
    }();
    return value;
}

// 方案3：松弛投机采样 - 通过 Temperature 调整 draft_probs 分布形状
// SPECULATIVE_DRAFT_TEMPERATURE > 1 时分布变平坦，高概率 token 的 p 降低，接受率提高
// SPECULATIVE_DRAFT_TEMPERATURE < 1 时分布变尖锐，高概率 token 的 p 升高，接受率降低
// 公式: new_probs = softmax(log(probs) / temperature)
float getSpeculativeDraftTemperature() {
    static float value = []() {
        const char* env = std::getenv("SPECULATIVE_DRAFT_TEMPERATURE");
        if (env) {
            RTP_LLM_LOG_INFO("SPECULATIVE_DRAFT_TEMPERATURE = %f (方案3: Temperature调整已启用)", std::atof(env));
        }
        return env ? std::atof(env) : 1.0f;
    }();
    return value;
}

}  // namespace
namespace speculative {
SpeculativeSamplerOutput SpeculativeSampler::forward(const std::list<GenerateStreamPtr>& streams,
                                                     SamplerOutput&                      draft_sampler_output,
                                                     SamplerOutput&                      target_sampler_output) const {
    SpeculativeSamplerOutput sample_output;
    batchSample(sample_output, streams, draft_sampler_output, target_sampler_output);

    return sample_output;
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

    // note target token probs is already on device
    auto target_token_probs_d = target_token_probs;
    auto draft_token_probs_d  = draft_token_probs;

    // prepare data for chain speculative sampling
    auto          draft_token_ids_d_t    = Buffer2torchTensor(draft_token_ids_d, false);
    auto          draft_token_probs_d_t  = Buffer2torchTensor(draft_token_probs_d, false);
    auto          target_token_probs_d_t = Buffer2torchTensor(target_token_probs_d, false);
    torch::Tensor uniform_samples_d      = torch::rand({(long)batch_size, (long)propose_step_ + 1},
                                                  torch::TensorOptions().device(target_device).dtype(torch::kFloat));

    // 方案3：Temperature 调整 draft_probs 分布形状
    // T > 1 分布变平坦，高概率 token 被压低，接受率提高
    // T < 1 分布变尖锐，高概率 token 被放大，接受率降低
    float temperature = getSpeculativeDraftTemperature();
    if (temperature != 1.0f) {
        // new_probs = softmax(log(probs + eps) / temperature, dim=-1)
        constexpr float eps   = 1e-10f;
        draft_token_probs_d_t = torch::softmax(torch::log(draft_token_probs_d_t + eps) / temperature, -1);
    }

    // 方案1：直接缩放 draft_probs
    // factor < 1 放松接受条件，factor > 1 收紧接受条件
    float relaxation_factor = getSpeculativeRelaxationFactor();
    if (relaxation_factor != 1.0f) {
        draft_token_probs_d_t = draft_token_probs_d_t * relaxation_factor;
    }

    torch::Tensor output_token_ids_d = torch::zeros({(long)batch_size, (long)propose_step_ + 1},
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

    BufferPtr draft_token_ids_h;
    for (const GenerateStreamPtr& stream : streams) {
        if (stream->forceSpAccept()) {
            draft_token_ids_h = device_->clone({*draft_token_ids, AllocationType::HOST});
            break;
        }
    }

    int stream_idx = 0;
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
        } else {
            accept_len    = output_emitted_token_num_h[stream_idx].item<int32_t>();
            accept_tokens = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, accept_len}, rtp_llm::AllocationType::HOST}, {"accept_tokens"});
            memcpy(accept_tokens->data(),
                   output_token_ids_h[stream_idx].data_ptr<int32_t>(),
                   sizeof(int32_t) * accept_len);
        }

        // always use target token as the last token
        *accept_tokens->dataWithOffset<int32_t>(accept_len - 1) =
            new_all_token_ids[(stream_idx * (propose_step_ + 1) + accept_len - 1) * token_stride + token_stride - 1];

        sample_output.accept_tokens.push_back(accept_tokens);
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
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "autil/StringUtil.h"

using namespace std;

namespace rtp_llm {

const float BaseLogitsProcessor::neg_inf = -std::numeric_limits<float>::max();

BaseLogitsProcessor::BaseLogitsProcessor(rtp_llm::DeviceBase* device): device_(device) {};

void BaseLogitsProcessor::memFill(const rtp_llm::BufferPtr& new_tokens_logits, size_t vocab_size, size_t index) {
    auto shapes = new_tokens_logits->shape();
    RTP_LLM_CHECK(shapes.size() == 1);
    auto tensor = Buffer2torchTensor(*new_tokens_logits, false);
    tensor.fill_(neg_inf);
    tensor[index] = 1;
}

rtp_llm::BufferPtr BaseLogitsProcessor::generateVocabMask(
    size_t batch_size, size_t vocab_size, const std::vector<std::vector<size_t>>& batch_candidate_token_ids) {
    RTP_LLM_CHECK(batch_candidate_token_ids.size() == batch_size);
    std::vector<uint8_t> vocab_mask_cpu(batch_size * vocab_size, 1);

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const auto& candidate_token_ids = batch_candidate_token_ids[batch_idx];
        for (const auto& token_id : candidate_token_ids) {
            if (token_id < vocab_size) {
                vocab_mask_cpu[batch_idx * vocab_size + token_id] = 0;
            }
        }
    }

    BufferPtr vocab_mask_buffer_cpu = vector2Buffer(vocab_mask_cpu);
    auto      buffer_reshape        = vocab_mask_buffer_cpu->reshape({batch_size, vocab_size});
    return device_->clone({buffer_reshape, rtp_llm::AllocationType::DEVICE});
}

void BaseLogitsProcessor::maskLogits(const rtp_llm::BufferPtr& new_tokens_logits,
                                     const rtp_llm::BufferPtr& vocab_mask) {
    RTP_LLM_CHECK(new_tokens_logits->shape().size() == 2);
    RTP_LLM_CHECK(vocab_mask->shape().size() == 2);
    RTP_LLM_CHECK(new_tokens_logits->shape()[0] == vocab_mask->shape()[0]);
    RTP_LLM_CHECK(new_tokens_logits->shape()[1] == vocab_mask->shape()[1]);
    device_->maskLogits(*new_tokens_logits, *vocab_mask);
}

SparseMaskLogitsParams
BaseLogitsProcessor::generateSparseVocabMask(size_t                                  batch_size,
                                             size_t                                  vocab_size,
                                             const std::vector<std::vector<size_t>>& batch_candidate_token_ids,
                                             const rtp_llm::BufferPtr&               batch_logits) {
    RTP_LLM_CHECK(batch_candidate_token_ids.size() == batch_size);

    int total_num = 0;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        total_num += batch_candidate_token_ids[batch_idx].size();
    }

    std::vector<int> h_batch_indices;  // batch id
    std::vector<int> h_vocab_mask;     // vocab mask
    h_batch_indices.reserve(batch_size * 2);
    h_vocab_mask.reserve(total_num);

    size_t cur_total_num = 0;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const std::vector<size_t>& candidate_token_ids = batch_candidate_token_ids[batch_idx];
        if (candidate_token_ids.empty()) {
            continue;
        }
        for (const auto& token_id : candidate_token_ids) {
            h_vocab_mask.push_back(token_id);
        }
        cur_total_num += candidate_token_ids.size();
        h_batch_indices.push_back(cur_total_num);
        h_batch_indices.push_back(batch_idx);
    }

    BufferPtr d_batch_indices = vector2Buffer(h_batch_indices);
    BufferPtr d_vocab_mask    = vector2Buffer(h_vocab_mask);

    SparseMaskLogitsParams params;
    params.batch_indices = device_->clone({*d_batch_indices, rtp_llm::AllocationType::DEVICE});
    params.mask_indices  = device_->clone({*d_vocab_mask, rtp_llm::AllocationType::DEVICE});
    params.logits        = batch_logits;
    params.valid_scores =
        device_->allocateBuffer({batch_logits->type(), {params.mask_indices->size()}, AllocationType::DEVICE});
    return params;
}

void BaseLogitsProcessor::sparseMaskLogits(SparseMaskLogitsParams& params) {
    RTP_LLM_CHECK(params.logits->shape().size() == 2);
    RTP_LLM_CHECK(params.batch_indices->shape().size() == 1);
    RTP_LLM_CHECK(params.mask_indices->shape().size() == 1);
    device_->sparseMaskLogits(params);
}

WeightMaskLogitsParams
BaseLogitsProcessor::generateVocabWeight(size_t                                  batch_size,
                                         size_t                                  vocab_size,
                                         const std::vector<const TokenWeights*>& batch_candidate_token_weights,
                                         const rtp_llm::BufferPtr&               batch_logits) {
    RTP_LLM_CHECK(batch_candidate_token_weights.size() == batch_size);
    WeightMaskLogitsParams params;

    int total_num = 0;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (batch_candidate_token_weights[batch_idx] != nullptr) {
            total_num += batch_candidate_token_weights[batch_idx]->token_ids.size();
        }
    }

    // std::vector<int>   h_batch_indices;  // batch id
    // std::vector<int>   h_vocab_indices;  // vocab index
    // std::vector<float> h_vocab_weight;   // weight value
    // h_batch_indices.reserve(batch_size * 2);
    // h_vocab_indices.reserve(total_num);
    // h_vocab_weight.reserve(total_num);

    params.h_batch_indices.reserve(batch_size * 2);
    params.h_vocab_indices.reserve(total_num);
    params.h_vocab_weight.reserve(total_num);

    size_t cur_total_num = 0;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const TokenWeights* tw = batch_candidate_token_weights[batch_idx];
        if (tw == nullptr) {
            continue;
        }
        const size_t count = tw->token_ids.size();
        for (size_t i = 0; i < count; ++i) {
            const int32_t token_id = tw->token_ids[i];
            const float   weight   = tw->weights[i];
            params.h_vocab_indices.push_back(token_id);
            params.h_vocab_weight.push_back(weight);
        }
        cur_total_num += count;
        params.h_batch_indices.push_back(cur_total_num);
        params.h_batch_indices.push_back(batch_idx);
    }
    // infer_info_.push_back("generateVocabWeight: " + to_string(infer_info_.size()));
    // std::string h_b = "batch_indices: " + autil::StringUtil::toString(params.h_batch_indices, ",");
    // std::string h_v = "vocab_indices: " + autil::StringUtil::toString(params.h_vocab_indices, ",");
    // std::string h_w = "vocab_weight: " + autil::StringUtil::toString(params.h_vocab_weight, ",");
    // infer_info_.push_back(h_b);
    // infer_info_.push_back(h_v);
    // infer_info_.push_back(h_w);

    // std::vector<int> b = {8,0,35,1,39,2,87,3,103,4,105,5,106,6,109,7,110,8,115,9};
    // std::vector<int> v =
    // {151645,220,46622,26439,369,3691,90780,502,220,8968,26989,46622,11482,1910,39752,22380,978,33689,502,24232,5750,28202,25211,3647,3745,2797,26439,2745,6753,4158,151645,6644,90780,7845,1356,16,18,20,17,220,26747,90780,595,296,1932,1051,6753,67910,39752,11968,1173,274,7168,3647,22734,12414,17545,42238,4017,16138,39674,902,22380,53514,5297,11045,1356,348,11369,3015,43848,5763,728,95932,705,151645,2615,1936,41588,11482,4531,38644,35082,5479,12188,29845,29663,90780,369,1550,8823,5519,39944,151645,220,7987,5517,6753,502,26439,2745,894,1910,151645,151645,70,18,16,20,70,16,17,18,20,15};
    // std::vector<float> w =
    // {3.4641e-05,1.98464e-05,2.59808e-05,2.59808e-05,1.78618e-05,2.59808e-05,2.59808e-05,2.16506e-05,2.02524e-05,2.59808e-05,2.59808e-05,2.59808e-05,2.09289e-05,2.16506e-05,2.46817e-05,2.59808e-05,1.94856e-05,2.16506e-05,2.16506e-05,2.45374e-05,2.59808e-05,2.16506e-05,2.16506e-05,2.59808e-05,2.59808e-05,2.59808e-05,2.59808e-05,2.16506e-05,2.16506e-05,2.59808e-05,3.4641e-05,2.59808e-05,2.59808e-05,2.16506e-05,2.16506e-05,2.02112e-05,2.25162e-05,2.16506e-05,2.59808e-05,2.18858e-05,1.94856e-05,2.4357e-05,2.06066e-05,2.03336e-05,2.59808e-05,2.16506e-05,2.16506e-05,2.45963e-05,2.37556e-05,2.59808e-05,2.16506e-05,2.02975e-05,2.59808e-05,2.48982e-05,2.30038e-05,2.16506e-05,2.16506e-05,2.59808e-05,2.25327e-05,2.59808e-05,2.35789e-05,1.88541e-05,2.4357e-05,2.46577e-05,2.54996e-05,2.16506e-05,1.94856e-05,2.16506e-05,2.396e-05,2.31662e-05,1.94856e-05,1.8403e-05,2.59808e-05,2.16506e-05,2.16506e-05,3.4641e-05,2.16506e-05,2.03877e-05,2.16506e-05,2.16506e-05,2.59808e-05,2.59808e-05,2.59808e-05,2.59808e-05,2.59808e-05,2.16506e-05,2.59808e-05,2.59808e-05,1.78618e-05,2.16506e-05,2.59808e-05,2.38157e-05,2.59808e-05,3.4641e-05,2.02185e-05,2.59808e-05,2.59808e-05,2.16506e-05,2.16506e-05,2.59808e-05,2.16506e-05,1.94856e-05,2.16506e-05,0,0,2.41765e-05,2.10011e-05,2.16506e-05,2.16506e-05,2.59808e-05,2.02383e-05,2.09289e-05,2.28534e-05,2.21919e-05,1.8403e-05};
    // BufferPtr d_batch_indices = vector2Buffer(b);
    // BufferPtr d_vocab_indices = vector2Buffer(v);
    // BufferPtr d_vocab_weight  = vector2Buffer(w);

    BufferPtr d_batch_indices = vector2Buffer(params.h_batch_indices);
    BufferPtr d_vocab_indices = vector2Buffer(params.h_vocab_indices);
    BufferPtr d_vocab_weight  = vector2Buffer(params.h_vocab_weight);

    params.batch_indices = device_->clone({*d_batch_indices, rtp_llm::AllocationType::DEVICE});
    params.vocab_indices = device_->clone({*d_vocab_indices, rtp_llm::AllocationType::DEVICE});
    params.vocab_weights = device_->clone({*d_vocab_weight, rtp_llm::AllocationType::DEVICE});
    params.logits        = batch_logits;
    params.valid_scores =
        device_->allocateBuffer({batch_logits->type(), {params.vocab_indices->size()}, AllocationType::DEVICE});
    return params;
}

void BaseLogitsProcessor::weightLogits(WeightMaskLogitsParams& params) {
    RTP_LLM_CHECK(params.logits->shape().size() == 2);
    RTP_LLM_CHECK(params.batch_indices->shape().size() == 1);
    RTP_LLM_CHECK(params.vocab_indices->shape().size() == params.batch_indices->shape().size());
    RTP_LLM_CHECK(params.vocab_weights->shape().size() == params.batch_indices->shape().size());
    device_->weightLogits(params);
}

void BaseLogitsProcessor::finishedMaskLogits(const FinishedMaskParams& params) {
    RTP_LLM_CHECK(params.logits->shape().size() == 2);
    RTP_LLM_CHECK(params.finished_mask->shape().size() == 1);
    RTP_LLM_CHECK(params.logits->shape()[0] == params.finished_mask->shape()[0]);
    device_->finishedMaskLogits(params);
}

}  // namespace rtp_llm

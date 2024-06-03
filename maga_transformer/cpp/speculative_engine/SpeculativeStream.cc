#include "maga_transformer/cpp/speculative_engine/SpeculativeStream.h"
#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include <memory>
#include <optional>

using namespace std;

namespace rtp_llm {

SpeculativeStream::SpeculativeStream(const GenerateStreamPtr& stream, uint gen_num_per_circle, size_t vocab_size):
    GenerateStream(stream->generateInput(), stream->resourceContext(), stream->maxSeqLen()), gen_num_per_circle_(gen_num_per_circle) {
    special_tokens_ = stream->specialTokens();
    target_stream_     = stream;
    target_index_prob_ = device_->allocateBuffer(
        {ft::DataType::TYPE_FP32, {(size_t)stream->tileNum(), gen_num_per_circle, vocab_size}, ft::AllocationType::HOST});
    draft_index_prob_ = device_->allocateBuffer(
        {ft::DataType::TYPE_FP32, {(size_t)stream->tileNum(), gen_num_per_circle, vocab_size}, ft::AllocationType::HOST});
}

int SpeculativeStream::tryReleaseKVBlock(int nums) {
    nums = GenerateStream::tryReleaseKVBlock(nums);
    return target_stream_->tryReleaseKVBlock(nums);
}

bool SpeculativeStream::initKVBlock() {
    return GenerateStream::initKVBlock() && target_stream_->initKVBlock();
}

bool SpeculativeStream::incrKVBlock() {
    return GenerateStream::incrKVBlock() && target_stream_->incrKVBlock();
}

void SpeculativeStream::updateDraftToken() {
    auto new_tokens = device_->allocateBuffer(
        {complete_token_ids_->type(), {(size_t)batchSize(), (size_t)gen_num_per_circle_}, ft::AllocationType::HOST});
    for (auto i = 0; i < batchSize(); ++i) {
        size_t src_offset = seqLength() - gen_num_per_circle_;
        device_->copy({(*new_tokens)[i], (*complete_token_ids_)[i], 0, src_offset, gen_num_per_circle_});
    }
    target_stream_->update(new_tokens, gen_num_per_circle_, ft::Buffer::emptyBuffer(), ft::Buffer::emptyBuffer(), ft::Buffer::emptyBuffer(), true);
    target_stream_->incrKVBlock();
}

void SpeculativeStream::releaseResource() {
    GenerateStream::releaseResource();
    target_stream_->releaseResource();
}

void SpeculativeStream::updateTargetProb(const ft::Buffer& prob) {
    device_->copy({*target_index_prob_, prob});
}

void SpeculativeStream::updateDraftProb(const ft::Buffer& prob, uint index) {
    for (auto i = 0; i < batchSize(); ++i) {
        // auto draft_index_prob_slice_ = draft_index_prob_->view(i, 1, true);
        device_->copy({(*draft_index_prob_)[i][index], prob});
    }
}

}  // namespace rtp_llm

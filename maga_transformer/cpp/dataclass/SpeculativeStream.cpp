#include "maga_transformer/cpp/dataclass/SpeculativeStream.h"
#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include <memory>

namespace rtp_llm {
SpeculativeStream::SpeculativeStream(const std::shared_ptr<GenerateInput>& query, int max_seq_len):
    GenerateStream(query, max_seq_len)
{
    draft_stream_ = std::make_shared<GenerateStream>(query, max_seq_len);
}

int SpeculativeStream::tryReleaseKVBlock(int nums) {
    nums = GenerateStream::tryReleaseKVBlock(nums);
    return draft_stream_->tryReleaseKVBlock(nums);
}

bool SpeculativeStream::initKVBlock() {
    bool init = GenerateStream::initKVBlock();
    if (!init) {
        return init;
    }
    return draft_stream_->initKVBlock();
}

bool SpeculativeStream::incrKVBlock() {
    bool init = GenerateStream::incrKVBlock();
    if (!init) {
        return init;
    }
    return draft_stream_->incrKVBlock();
}

void SpeculativeStream::releaseResource() {
    GenerateStream::releaseResource();
    draft_stream_->releaseResource();
}

}

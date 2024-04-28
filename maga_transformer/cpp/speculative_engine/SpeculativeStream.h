#pragma once

#include "maga_transformer/cpp/dataclass/GenerateStream.h"

namespace rtp_llm {
class SpeculativeStream: public GenerateStream {
public:
    explicit SpeculativeStream(const GenerateStreamPtr& stream, uint gen_num, size_t vocab_size);
    ~SpeculativeStream() {}

public:
    int tryReleaseKVBlock(int nums) override;

    bool initKVBlock() override;

    bool incrKVBlock() override;

    void releaseResource() override;

    void updateDraftToken();

    void updateTargetProb(const ft::Buffer& prob);

    void updateDraftProb(const ft::Buffer& prob, uint index);

    GenerateStreamPtr targetStream() const {
        return target_stream_;
    }

private:
    GenerateStreamPtr target_stream_;
    ft::BufferPtr     target_index_prob_;
    ft::BufferPtr     draft_index_prob_;
    uint              gen_num_per_circle_;
};

}  // namespace rtp_llm

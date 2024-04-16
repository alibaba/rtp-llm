#pragma once

#include "maga_transformer/cpp/dataclass/GenerateStream.h"

namespace rtp_llm {
class SpeculativeStream: public GenerateStream {
public:
    explicit SpeculativeStream(const std::shared_ptr<GenerateInput>& query, int max_seq_len = 2048);
    ~SpeculativeStream() {}

public:
    int tryReleaseKVBlock(int nums);

    bool initKVBlock();

    bool incrKVBlock();

    void releaseResource();

    void targetSync(){};

    void draftSync(){};

    GenerateStreamPtr draftStream() const {
        return draft_stream_;
    }

private:
    GenerateStreamPtr draft_stream_;
};

}  // namespace rtp_llm

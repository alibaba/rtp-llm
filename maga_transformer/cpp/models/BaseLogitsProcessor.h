#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "maga_transformer/cpp/models/SampleInfos.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class BaseLogitsProcessor {
public:
    BaseLogitsProcessor(ft::DeviceBase* device);
    virtual ~BaseLogitsProcessor() {}

public:
    virtual void process(const SamplerInputs& inputs) = 0;
    virtual void updateStatus(const SamplerInputs& inputs) = 0;
    void memFill(ft::BufferPtr new_tokens_logits, size_t vocab_size, size_t index);

protected:
    ft::DeviceBase* device_;
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

} // namespace rtp_llm
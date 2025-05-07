#pragma once

#include "maga_transformer/cpp/core/Buffer.h"
#include "maga_transformer/cpp/devices/DeviceBase.h"
#include "maga_transformer/cpp/models/SampleInfos.h"



namespace rtp_llm {

class BaseLogitsProcessor {
public:
    BaseLogitsProcessor(rtp_llm::DeviceBase* device);
    virtual ~BaseLogitsProcessor() {}
    static const float neg_inf;

public:
    virtual void process(const SamplerInputs& inputs) = 0;
    virtual void updateStatus(const SamplerInputs& inputs) = 0;
    void memFill(rtp_llm::BufferPtr new_tokens_logits, size_t vocab_size, size_t index);

protected:
    rtp_llm::DeviceBase* device_;
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

} // namespace rtp_llm
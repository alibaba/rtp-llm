#pragma once

#include "absl/status/status.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class EngineBase {
public:
    EngineBase(const EngineInitParams& params);
    virtual ~EngineBase() {}

    static void initDevices(const EngineInitParams& params);
    ft::DeviceBase* getDevice() {
        return device_;
    }
    virtual std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) = 0;
    virtual absl::Status stop() = 0;

    virtual KVCacheInfo getKVCacheInfo() const {
        return {0, 0};
    }
protected:
    ft::DeviceBase* device_;
};

}  // namespace rtp_llm

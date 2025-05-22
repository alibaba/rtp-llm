#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include <gmock/gmock.h>

namespace rtp_llm {

class MockMemoryUtil: public MemoryUtil {
public:
    MockMemoryUtil(std::shared_ptr<MemoryUtil> impl): impl_(std::move(impl)) {}

public:
    MOCK_METHOD(bool, regUserMr, (void*, uint64_t, bool, uint64_t), (override));
    MOCK_METHOD(bool, isMemoryMr, (void*, uint64_t, bool, bool), (override));
    MOCK_METHOD(bool, findMemoryMr, (void*, void*, uint64_t, bool, bool), (override));

    bool isRdmaMode() {
        return impl_->isRdmaMode();
    }
    bool deregUserMr(void* buf, bool gpu) {
        return impl_->deregUserMr(buf, gpu);
    }

private:
    std::shared_ptr<MemoryUtil> impl_;
};

}  // namespace rtp_llm

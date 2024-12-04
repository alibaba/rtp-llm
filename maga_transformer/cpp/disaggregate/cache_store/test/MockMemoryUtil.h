#pragma once

#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"
#include <gmock/gmock.h>

namespace rtp_llm {

class MockMemoryUtil: public MemoryUtil {
public:
    MockMemoryUtil(std::unique_ptr<MemoryUtil> impl): impl_(std::move(impl)) {}

public:
    bool regUserMr(void* buf, uint64_t size, bool gpu) {
        return impl_->regUserMr(buf, size, gpu);
    }
    bool deregUserMr(void* buf, bool gpu) {
        return impl_->deregUserMr(buf, gpu);
    }
    bool isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted) {
        return impl_->isMemoryMr(ptr, size, gpu, adopted);
    }
    MOCK_METHOD(bool, findMemoryMr, (void*, void*, uint64_t, bool, bool), (override));
    // bool findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted) {
    //     return impl_->findMemoryMr(mem_info, buf, size, gpu, adopted);
    // }
    bool isRdmaMode() {
        return impl_->isRdmaMode();
    }

private:
    std::unique_ptr<MemoryUtil> impl_;
};

}  // namespace rtp_llm
#pragma once

#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"
#include <gmock/gmock.h>

namespace rtp_llm {

class MockMemoryUtil: public MemoryUtil {
public:
    MockMemoryUtil(std::unique_ptr<MemoryUtil> impl): MemoryUtil(std::move(impl)) {}

public:
    MOCK_METHOD(bool, gpuEventBarrier, (void*), (override));
    MOCK_METHOD(bool, findMemoryMr, (void*, void*, uint64_t, bool, bool), (override));
};

}  // namespace rtp_llm
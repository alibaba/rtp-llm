#pragma once

#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"

namespace rtp_llm {

class MockAsyncContext: public AsyncContext {
public:
    MockAsyncContext()           = default;
    ~MockAsyncContext() override = default;

public:
    MOCK_METHOD(bool, done, (), (const, override));
    MOCK_METHOD(bool, success, (), (const, override));
};

}  // namespace rtp_llm

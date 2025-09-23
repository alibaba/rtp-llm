#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"

namespace http_server {

class MockHttpResponseWriter: public HttpResponseWriter {
public:
    MockHttpResponseWriter(): HttpResponseWriter(nullptr) {}
    ~MockHttpResponseWriter() override = default;

public:
    MOCK_METHOD0(WriteDone, bool());
    MOCK_METHOD0(isConnected, bool());
    MOCK_METHOD1(Write, bool(const std::string&));
};

}  // namespace http_server

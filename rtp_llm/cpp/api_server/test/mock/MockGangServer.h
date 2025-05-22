#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/GangServer.h"

namespace rtp_llm {

class MockGangServer: public GangServer {
public:
    MockGangServer(): GangServer(py::none()) {}
    ~MockGangServer() override = default;

public:
    MOCK_METHOD3(requestWorkers, void(const std::map<std::string, std::string>&, const std::string& uri, bool is_wait));
};

}  // namespace rtp_llm

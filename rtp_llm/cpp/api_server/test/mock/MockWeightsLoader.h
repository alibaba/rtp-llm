#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/WeightsLoader.h"

namespace rtp_llm {

class MockWeightsLoader: public WeightsLoader {
public:
    MockWeightsLoader(): WeightsLoader(py::none()) {}
    ~MockWeightsLoader() override = default;
};

}  // namespace rtp_llm

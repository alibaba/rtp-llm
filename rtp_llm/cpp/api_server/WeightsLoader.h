#pragma once

#include "rtp_llm/cpp/pybind/PyUtils.h"

namespace rtp_llm {

class WeightsLoader {
public:
    WeightsLoader(py::object model_weights_loader);
    virtual ~WeightsLoader() = default;

private:
    py::object model_weights_loader_;
};

}  // namespace rtp_llm

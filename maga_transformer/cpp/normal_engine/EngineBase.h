#pragma once

#include "absl/status/status.h"
#include "torch/all.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

namespace rtp_llm {

class EngineBase {
public:
    virtual ~EngineBase() {}
    virtual absl::Status enqueue(std::shared_ptr<GenerateStream>& stream) = 0;
};

}  // namespace rtp_llm

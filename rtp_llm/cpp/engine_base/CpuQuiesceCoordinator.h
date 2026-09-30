#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include "absl/status/statusor.h"

namespace rtp_llm {

// Backend-only, on-demand coordination. The model loop never enters this group.
// Its lifetime is independent of model/NCCL resource suspend and restore.
class CpuQuiesceCoordinator {
public:
    explicit CpuQuiesceCoordinator(c10::intrusive_ptr<c10d::Backend> group);

    absl::StatusOr<uint64_t>
    targetRound(const std::string& token, uint64_t frozen_round, std::chrono::steady_clock::time_point deadline);

private:
    c10::intrusive_ptr<c10d::Backend> group_;
    std::timed_mutex                  mutex_;
    std::string                       token_;
    uint64_t                          frozen_round_ = 0;
    std::optional<uint64_t>           target_;
    std::string                       transport_failure_;
};

}  // namespace rtp_llm

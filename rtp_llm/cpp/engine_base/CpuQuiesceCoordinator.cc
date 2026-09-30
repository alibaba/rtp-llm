#include "rtp_llm/cpp/engine_base/CpuQuiesceCoordinator.h"

#include <stdexcept>
#include <ATen/Functions.h>

namespace rtp_llm {

CpuQuiesceCoordinator::CpuQuiesceCoordinator(c10::intrusive_ptr<c10d::Backend> group): group_(std::move(group)) {
    if (!group_ || group_->getBackendName() != "gloo" || group_->getSize() < 1) {
        throw std::invalid_argument("execution quiesce requires a dedicated CPU/Gloo group");
    }
}

absl::StatusOr<uint64_t> CpuQuiesceCoordinator::targetRound(const std::string&                    token,
                                                            uint64_t                              frozen_round,
                                                            std::chrono::steady_clock::time_point deadline) {
    // Compare the complete operation identity, not a collision-prone hash. MAX
    // over (byte, -byte) supplies both extrema in the SAME collective as round.
    constexpr size_t kMaxTokenBytes = 256;
    if (token.empty() || token.size() > kMaxTokenBytes || frozen_round > INT64_MAX) {
        return absl::InvalidArgumentError("invalid execution quiesce token or frozen round");
    }
    std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
    if (!lock.try_lock_until(deadline)) {
        return absl::DeadlineExceededError("CPU quiesce coordinator is busy");
    }
    if (!transport_failure_.empty()) {
        return absl::FailedPreconditionError("CPU quiesce group failed; restart required: " + transport_failure_);
    }
    if (token == token_ && target_) {
        if (frozen_round != frozen_round_) {
            return absl::FailedPreconditionError("frozen round changed within a quiesce operation");
        }
        return *target_;
    }
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
        return absl::DeadlineExceededError("CPU quiesce deadline expired before reduction");
    }
    auto  values = at::zeros({1 + 2 * (kMaxTokenBytes + 1)}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    auto* data   = values.data_ptr<int64_t>();
    data[0]      = static_cast<int64_t>(frozen_round);
    for (size_t i = 0; i <= kMaxTokenBytes; ++i) {
        const int64_t value =
            i == 0 ? token.size() : (i <= token.size() ? static_cast<unsigned char>(token[i - 1]) : 0);
        data[1 + 2 * i] = value;
        data[2 + 2 * i] = -value;
    }
    try {
        std::vector<at::Tensor> tensors{values};
        c10d::AllreduceOptions  options;
        options.reduceOp = c10d::ReduceOp::MAX;
        options.timeout  = remaining;
        auto work        = group_->allreduce(tensors, options);
        if (!work->wait(remaining)) {
            throw std::runtime_error("CPU all-reduce did not complete before its deadline");
        }
    } catch (const std::exception& error) {
        // Gloo work can poison its context on timeout/disconnect. Never reuse
        // that group or report a safe stop. GPU resources remain backed.
        transport_failure_ = error.what();
        return absl::UnavailableError("CPU quiesce all-reduce failed: " + transport_failure_);
    }
    for (size_t i = 0; i <= kMaxTokenBytes; ++i) {
        const int64_t value =
            i == 0 ? token.size() : (i <= token.size() ? static_cast<unsigned char>(token[i - 1]) : 0);
        if (data[1 + 2 * i] != value || data[2 + 2 * i] != -value) {
            return absl::FailedPreconditionError("CPU quiesce ranks disagree on lifecycle operation token");
        }
    }
    if (data[0] < static_cast<int64_t>(frozen_round)) {
        return absl::InternalError("CPU quiesce target is behind the frozen round");
    }
    token_        = token;
    frozen_round_ = frozen_round;
    target_       = static_cast<uint64_t>(data[0]);
    return *target_;
}

}  // namespace rtp_llm

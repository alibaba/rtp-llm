#pragma once

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace rtp_llm {

struct DeferredSamplingRngSlot {
    torch::Tensor host;
    torch::Tensor device;
    at::cuda::CUDAEvent ready;
    int device_index;
    std::atomic<bool> released{true};
    bool disabled = false;

    explicit DeferredSamplingRngSlot(int index): device_index(index) {}
};

// A single-consumption host packet can cross target/rejection threads. Reuse
// requires both release of every lease owner and completion of the actual consumer.
class DeferredSamplingRngLease {
public:
    DeferredSamplingRngLease(std::shared_ptr<DeferredSamplingRngSlot> slot, int64_t prefix, int64_t batch):
        slot_(std::move(slot)), batch_(batch), state_(slot_->host.narrow(0, prefix, 2 * batch).view({2, batch})) {}

    ~DeferredSamplingRngLease() {
        finishNoThrow();
        slot_->released.store(true, std::memory_order_release);
    }

    const torch::Tensor& hostState() const { return state_; }

    void begin(const at::cuda::CUDAStream& stream) {
        TORCH_CHECK(stream.device_index() == slot_->device_index, "deferred RNG lease device mismatch");
        cudaStreamCaptureStatus capture;
        C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture));
        TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "deferred RNG lease cannot be consumed during capture");
        int expected = 0;
        TORCH_CHECK(consumption_.compare_exchange_strong(expected, 1), "deferred RNG lease already consumed");
        stream_ = stream;
    }

    torch::Tensor fallbackDeviceState() {
        TORCH_CHECK(consumption_.load() == 1 && stream_.has_value(), "deferred RNG lease has no active consumer");
        c10::cuda::CUDAStreamGuard guard(*stream_);
        const int64_t count = 2 * batch_;
        if (!slot_->device.defined() || slot_->device.numel() < count) {
            slot_->device = torch::empty({count}, torch::TensorOptions().dtype(torch::kInt64)
                                                    .device(torch::Device(torch::kCUDA, slot_->device_index)));
        }
        c10::cuda::CUDACachingAllocator::recordStream(slot_->device.storage().data_ptr(), *stream_);
        auto output = slot_->device.narrow(0, 0, count).view({2, batch_});
        output.copy_(state_, /*non_blocking=*/true);
        return output;
    }

    void disableReuse() noexcept { slot_->disabled = true; }

    void finish() {
        int expected = 1;
        if (!consumption_.compare_exchange_strong(expected, 2)) return;
        try {
            slot_->ready.record(*stream_);
        } catch (...) {
            slot_->disabled = true;
            throw;
        }
    }

    void finishNoThrow() noexcept {
        try {
            finish();
        } catch (...) {
            // Failure disables this slot; cleanup must not replace the original exception.
        }
    }

private:
    std::shared_ptr<DeferredSamplingRngSlot> slot_;
    int64_t batch_;
    torch::Tensor state_;
    std::atomic<int> consumption_{0};
    std::optional<at::cuda::CUDAStream> stream_;
};

class DeferredSamplingRngPool {
public:
    std::shared_ptr<DeferredSamplingRngLease> acquire(int64_t batch, int64_t prefix, int device) {
        TORCH_CHECK(batch >= 0 && prefix >= 0 && batch <= (std::numeric_limits<int64_t>::max() - prefix) / 2,
                    "invalid deferred RNG packet extent");
        cudaStreamCaptureStatus capture;
        C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(device).stream(), &capture));
        if (capture != cudaStreamCaptureStatusNone) return {};
        std::lock_guard<std::mutex> lock(mutex_);
        std::shared_ptr<DeferredSamplingRngSlot> slot;
        for (const auto& candidate : slots_) {
            if (candidate->device_index == device && candidate->released.load(std::memory_order_acquire)
                && !candidate->disabled && candidate->ready.query()) {
                slot = candidate;
                break;
            }
        }
        if (!slot) {
            slot = std::make_shared<DeferredSamplingRngSlot>(device);
            // Busy slots never force a serving-thread wait or unbounded cache growth.
            if (slots_.size() < 3) slots_.push_back(slot);
        }
        const int64_t count = prefix + 2 * batch;
        if (!slot->host.defined() || slot->host.numel() < count) {
            slot->host = torch::empty({count}, torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        }
        auto lease = std::make_shared<DeferredSamplingRngLease>(slot, prefix, batch);
        slot->released.store(false, std::memory_order_release);
        return lease;
    }

private:
    std::mutex mutex_;
    std::vector<std::shared_ptr<DeferredSamplingRngSlot>> slots_;
};

}  // namespace rtp_llm
#endif

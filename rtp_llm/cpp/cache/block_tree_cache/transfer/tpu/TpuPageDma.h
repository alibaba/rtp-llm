#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/core/TensorBody.h>

namespace rtp_llm {

enum class TpuCopyDirection { DeviceToHost, HostToDevice };

// All offsets are bytes within the registered base storage or host region.
struct TpuCopySpan {
    std::size_t  buffer_index;
    std::int64_t device_offset;
    std::int64_t host_offset;
    std::int64_t bytes;
};

struct TpuDmaRegion;

// No DMA was enqueued. A previously registered carrier could not be
// reacquired after the model ran, so the old contents remain unpublished.
class TpuDmaAdmissionError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The WorkLoop is the sole submit/poll owner. The caller protects the selected
// page IDs from reuse and orders device computation against the transfer.
class TpuDmaBatch final {
public:
    ~TpuDmaBatch();
    TpuDmaBatch(const TpuDmaBatch&) = delete;
    TpuDmaBatch& operator=(const TpuDmaBatch&) = delete;

    void submit();
    // Pending, empty string on success, or a failure after all DMA has stopped.
    std::optional<std::string> poll();

private:
    friend class TpuPageDma;
    struct State;
    explicit TpuDmaBatch(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
};

class TpuPageDma final {
public:
    // The host tensor must own (or retain the owner of) a page-aligned,
    // page-sized multiple of contiguous CPU byte storage, exclusively granted
    // to this instance for mlock/DmaMap until close().
    TpuPageDma(const std::vector<at::Tensor>& device_buffers, at::Tensor host_region, std::int64_t num_blocks);
    TpuPageDma(const TpuPageDma&) = delete;
    TpuPageDma& operator=(const TpuPageDma&) = delete;
    std::shared_ptr<TpuDmaBatch> prepare(TpuCopyDirection direction, std::vector<TpuCopySpan> spans) const;
    std::vector<std::size_t> bufferSizes() const;
    // Call after stopping the WorkLoop and releasing its batches. A failed
    // unmap leaves the registration intact so close() can be retried.
    void close();

private:
    mutable std::mutex            mutex_;
    std::shared_ptr<TpuDmaRegion> region_;
    bool                         closing_ = false;
};

}  // namespace rtp_llm

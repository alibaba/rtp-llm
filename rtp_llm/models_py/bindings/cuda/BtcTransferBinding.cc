#include "rtp_llm/models_py/bindings/cuda/DeviceHostCopySubmit.h"

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include <unistd.h>

namespace rtp_llm {

void writeBtcTransferTrace(const char* format, ...) noexcept {
    // Keep each record within the POSIX minimum PIPE_BUF. Workers share
    // stderr, so stdio fragments can interleave at arbitrary byte boundaries.
    char line[512];
    va_list args;
    va_start(args, format);
    const int length = std::vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(line))
        return;
    while (::write(STDERR_FILENO, line, static_cast<size_t>(length)) < 0 && errno == EINTR) {}
}

}  // namespace rtp_llm

namespace {

thread_local char last_error[512];

int fail(const char* phase, cudaError_t error = cudaSuccess) noexcept {
    if (error == cudaSuccess) {
        std::snprintf(last_error, sizeof(last_error), "%s", phase);
    } else {
        std::snprintf(last_error, sizeof(last_error), "%s: %s", phase, cudaGetErrorString(error));
    }
    return -1;
}

[[noreturn]] void abortUnconfirmed(const char* phase, cudaError_t error) noexcept {
    std::fprintf(stderr, "BTC transfer CUDA completion unconfirmed at %s: %s\n", phase, cudaGetErrorString(error));
    std::fflush(stderr);
    std::abort();
}

class DeviceScope {
public:
    explicit DeviceScope(int device) noexcept {
        error_ = cudaGetDevice(&previous_);
        if (error_ == cudaSuccess && previous_ != device) {
            error_   = cudaSetDevice(device);
            changed_ = error_ == cudaSuccess;
        }
    }

    ~DeviceScope() {
        if (changed_) {
            const auto status = cudaSetDevice(previous_);
            if (status != cudaSuccess) {
                abortUnconfirmed("restore CUDA device", status);
            }
        }
    }

    cudaError_t error() const {
        return error_;
    }

private:
    int         previous_{-1};
    cudaError_t error_{};
    bool        changed_{false};
};

void drain(cudaStream_t stream) noexcept {
    const auto status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        abortUnconfirmed("stream drain", status);
    }
}

struct DrainOnFailure {
    cudaStream_t stream;
    bool         armed{false};

    ~DrainOnFailure() {
        if (armed) {
            drain(stream);
        }
    }
};

bool traceEnabled() noexcept {
    const auto* value = std::getenv("RTP_BTC_TRANSFER_TRACE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

struct Transfer {
    int          device;
    int          rank;
    int          direction;
    uint64_t     job;
    cudaStream_t stream;
    size_t       count;
    size_t       bytes{0};
    cudaEvent_t  event{nullptr};
    bool         batch{false};
    bool         trace{false};
    int          terminal{0};
    cudaError_t  error{cudaSuccess};
    std::vector<void*>       destinations;
    std::vector<const void*> sources;
    std::vector<size_t>      lengths;

    ~Transfer() {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
        }
    }

    void log(const char* stage, const char* result = "") const noexcept {
        if (trace) {
            rtp_llm::writeBtcTransferTrace(
                         "BTC_NATIVE_TRANSFER stage=%s job=%llu rank=%d direction=%s bytes=%zu spans=%zu backend=%s%s\n",
                         stage,
                         static_cast<unsigned long long>(job),
                         rank,
                         direction == 0 ? "H2D" : "D2H",
                         bytes,
                         count,
                         batch ? "cuda_batch" : "cuda_single",
                         result);
        }
    }

    // Ticket operations are serialized by the foreign worker. On a terminal
    // result no CUDA access to the borrowed buffers can still be in flight.
    int complete(bool wait) noexcept {
        if (terminal != 0) {
            return terminal > 0 ? terminal : fail("BTC transfer event failed", error);
        }
        DeviceScope scope(device);
        if (scope.error() != cudaSuccess) {
            abortUnconfirmed("select CUDA device for completion", scope.error());
        }
        const auto status = wait ? cudaEventSynchronize(event) : cudaEventQuery(event);
        if (!wait && status == cudaErrorNotReady) {
            return 0;
        }
        if (status != cudaSuccess) {
            drain(stream);
            error    = status;
            terminal = -1;
            log("complete", " success=0");
            return fail("BTC transfer event failed", status);
        }
        terminal = 1;
        log("complete", " success=1");
        return terminal;
    }
};

}  // namespace

extern "C" {

const char* btc_transfer_last_error() noexcept {
    return last_error;
}

int btc_transfer_submit(int              device_index,
                        int              rank,
                        int              direction,
                        uint64_t         job,
                        uintptr_t        stream_handle,
                        const uintptr_t* sources,
                        const uintptr_t* destinations,
                        const size_t*    lengths,
                        size_t           count,
                        void**           ticket) noexcept {
    if (ticket == nullptr) {
        return fail("BTC transfer requires a ticket output");
    }
    *ticket = nullptr;
    if (device_index < 0 || rank < 0 || (direction != 0 && direction != 1) || !stream_handle
        || (count && (!sources || !destinations || !lengths))) {
        return fail("invalid BTC transfer descriptor arrays, device, direction, or stream");
    }

    try {
        DeviceScope scope(device_index);
        if (scope.error() != cudaSuccess) {
            return fail("select CUDA device for BTC transfer", scope.error());
        }
        auto transfer = std::make_unique<Transfer>();
        transfer->device    = device_index;
        transfer->rank      = rank;
        transfer->direction = direction;
        transfer->job       = job;
        transfer->stream    = reinterpret_cast<cudaStream_t>(stream_handle);
        transfer->count     = count;
        transfer->trace     = traceEnabled();
        transfer->batch     = rtp_llm::cudaDeviceHostBatchSupport().available;

        transfer->destinations.reserve(count);
        transfer->sources.reserve(count);
        transfer->lengths.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto length = lengths[i];
            if (!sources[i] || !destinations[i] || !length
                || length > std::numeric_limits<uintptr_t>::max() - sources[i]
                || length > std::numeric_limits<uintptr_t>::max() - destinations[i]
                || length > std::numeric_limits<size_t>::max() - transfer->bytes) {
                return fail("invalid BTC transfer address range or size");
            }
            transfer->bytes += length;
            transfer->sources.push_back(reinterpret_cast<const void*>(sources[i]));
            transfer->destinations.push_back(reinterpret_cast<void*>(destinations[i]));
            transfer->lengths.push_back(length);
        }

        auto status = cudaEventCreateWithFlags(&transfer->event, cudaEventDisableTiming);
        if (status != cudaSuccess) {
            return fail("create BTC transfer event", status);
        }
        DrainOnFailure rollback{transfer->stream};
        rollback.armed = true;
        if (count && transfer->batch) {
            status = rtp_llm::submitCudaDeviceHostBatch(transfer->destinations.data(),
                                                       transfer->sources.data(),
                                                       transfer->lengths.data(),
                                                       count,
                                                       transfer->stream);
        } else {
            for (size_t i = 0; i < count && status == cudaSuccess; ++i) {
                status = rtp_llm::submitCudaDeviceHostTile(
                    transfer->destinations[i], transfer->sources[i], transfer->lengths[i], transfer->stream);
            }
        }
        if (status != cudaSuccess) {
            return fail("submit BTC device-host copy", status);
        }
        status = cudaEventRecord(transfer->event, transfer->stream);
        if (status != cudaSuccess) {
            return fail("record BTC transfer event", status);
        }
        transfer->log("submit");
        const int result = transfer->batch ? 0 : 1;
        rollback.armed   = false;
        *ticket          = transfer.release();
        return result;
    } catch (const std::exception& error) {
        return fail(error.what());
    } catch (...) {
        return fail("unknown BTC transfer submission error");
    }
}

int btc_transfer_query(void* ticket) noexcept {
    return ticket == nullptr ? fail("missing BTC transfer ticket") : static_cast<Transfer*>(ticket)->complete(false);
}

int btc_transfer_wait(void* ticket) noexcept {
    return ticket == nullptr ? fail("missing BTC transfer ticket") : static_cast<Transfer*>(ticket)->complete(true);
}

void btc_transfer_destroy(void* ticket) noexcept {
    if (ticket == nullptr) {
        return;
    }
    auto*       handle = static_cast<Transfer*>(ticket);
    DeviceScope scope(handle->device);
    if (scope.error() != cudaSuccess) {
        abortUnconfirmed("select CUDA device for ticket destruction", scope.error());
    }
    std::unique_ptr<Transfer> transfer(handle);
    (void)transfer->complete(true);
}

}  // extern "C"

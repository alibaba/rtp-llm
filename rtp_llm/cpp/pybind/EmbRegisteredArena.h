#pragma once

#include <ATen/core/Tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace rtp_llm {
namespace {

// Owns numeric extents only. The Python transport keeps the MR and stream alive
// until all active tickets have drained, including tickets awaiting a view.
class EmbRegisteredArena {
public:
    enum class Result { Acquired, LockTimeout, DeadlineWithLock, ExtentUnavailable, Closed, Quarantined };
    using Acquisition = std::tuple<Result, uint64_t, int64_t, int64_t, int64_t>;

    EmbRegisteredArena(int64_t base, int64_t bytes, int64_t block, int64_t writeEnd):
        base_(base), bytes_(bytes), block_(block), writeEnd_(writeEnd), freeBytes_(bytes) {
        if (base < 0 || bytes <= 0 || block <= 0 || bytes % block || base % 64
            || bytes > std::numeric_limits<int64_t>::max() - base
            || writeEnd < base || writeEnd > base + bytes || (writeEnd - base) % block) {
            throw std::invalid_argument("invalid registered EMB arena geometry");
        }
        const auto blocks = static_cast<size_t>(bytes / block);
        free_.reserve(blocks / 2 + 1);
        free_.push_back({base, bytes});
        tickets_.resize(blocks);
    }

    Acquisition acquire(int64_t bytes, bool forWrite, int64_t timeoutNs, bool sample) {
        return acquireExtent(bytes, forWrite, timeoutNs, sample, false);
    }

    Acquisition tryAcquireForNativeWrite(int64_t bytes, bool sample) {
        return acquireExtent(bytes, true, 0, sample, true);
    }

    std::pair<int64_t, int64_t> nativeWriteInUse() const {
        std::lock_guard<std::timed_mutex> lock(mutex_);
        return {nativeWriteActiveBytes_, nativeWriteActiveExtents_};
    }

private:
    Acquisition acquireExtent(int64_t bytes, bool forWrite, int64_t timeoutNs, bool sample, bool nativeWrite) {
        if (bytes <= 0 || bytes > bytes_ || timeoutNs < 0 || timeoutNs > 600'000'000'000LL) {
            throw std::invalid_argument("invalid registered EMB arena acquisition");
        }
        const auto capacity = ((bytes - 1) / block_ + 1) * block_;
        const auto deadline = Clock::now() + std::chrono::nanoseconds(timeoutNs);
        std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
        if (!(timeoutNs ? lock.try_lock_until(deadline) : lock.try_lock())) {
            return {Result::LockTimeout, 0, 0, 0, 0};
        }
        auto heldSince = sample ? Clock::now() : Clock::time_point{};
        int64_t heldNs = 0;
        const auto finish = [&](Result result, uint64_t ticket = 0, int64_t offset = 0, int64_t count = 0) {
            return Acquisition{result, ticket, offset, count, heldNs + (sample ? elapsed(heldSince) : 0)};
        };
        while (accepting_ && !deviceFailed_ && quarantinedBytes_ < bytes_) {
            if (timeoutNs && Clock::now() >= deadline) {
                return finish(Result::DeadlineWithLock);
            }
            for (auto it = free_.begin(); it != free_.end(); ++it) {
                const int64_t offset = it->offset;
                if (capacity > it->bytes || (forWrite && offset + capacity > writeEnd_)) {
                    continue;
                }
                const auto ticket = ++lastTicket_;
                if (ticket == 0) {
                    throw std::overflow_error("registered EMB arena ticket counter wrapped");
                }
                tickets_[index(offset)] = {ticket, capacity};
                ++active_;
                if (nativeWrite) {
                    nativeWriteActiveBytes_ += capacity;
                    ++nativeWriteActiveExtents_;
                }
                freeBytes_ -= capacity;
                if (capacity == it->bytes) {
                    free_.erase(it);
                } else {
                    it->offset += capacity;
                    it->bytes -= capacity;
                }
                return finish(Result::Acquired, ticket, offset, capacity);
            }
            if (!timeoutNs) {
                return finish(Result::ExtentUnavailable);
            }
            if (sample) {
                heldNs += elapsed(heldSince);
            }
            const auto status = changed_.wait_until(lock, deadline);
            if (sample) {
                heldSince = Clock::now();
            }
            if (status == std::cv_status::timeout) {
                return finish(Result::ExtentUnavailable);
            }
        }
        return finish(accepting_ ? Result::Quarantined : Result::Closed);
    }

public:
    int64_t release(uint64_t ticket, int64_t offset, bool reusable, bool sample) {
        return releaseExtent(ticket, offset, reusable, sample, false);
    }

    int64_t releaseWrittenExtent(uint64_t ticket, int64_t offset, bool reusable, bool sample) {
        return releaseExtent(ticket, offset, reusable, sample, true);
    }

    std::pair<int64_t, int64_t> nativeWriteQuarantine() const {
        std::lock_guard<std::timed_mutex> lock(mutex_);
        return {nativeWriteQuarantinedBytes_, nativeWriteQuarantinedExtents_};
    }

private:
    int64_t releaseExtent(uint64_t ticket, int64_t offset, bool reusable, bool sample, bool nativeWrite) {
        int64_t heldNs = 0;
        {
            std::lock_guard<std::timed_mutex> lock(mutex_);
            const auto heldSince = sample ? Clock::now() : Clock::time_point{};
            auto& entry = tickets_.at(index(offset));
            if (ticket == 0 || entry.id != ticket) {
                throw std::invalid_argument("stale registered EMB arena ticket");
            }
            if (reusable) {
                returnExtent(offset, entry.bytes);
                freeBytes_ += entry.bytes;
            } else {
                quarantinedBytes_ += entry.bytes;
                if (nativeWrite) {
                    nativeWriteQuarantinedBytes_ += entry.bytes;
                    ++nativeWriteQuarantinedExtents_;
                }
            }
            if (nativeWrite) {
                nativeWriteActiveBytes_ -= entry.bytes;
                --nativeWriteActiveExtents_;
            }
            entry = {};
            --active_;
            heldNs = sample ? elapsed(heldSince) : 0;
        }
        changed_.notify_all();
        return heldNs;
    }

public:
    std::pair<int64_t, int64_t> freeSpace() const {
        std::lock_guard<std::timed_mutex> lock(mutex_);
        int64_t largest = 0;
        for (const auto& extent : free_) {
            largest = std::max(largest, extent.bytes);
        }
        return {freeBytes_, largest};
    }

    void stop() {
        {
            std::lock_guard<std::timed_mutex> lock(mutex_);
            accepting_ = false;
        }
        changed_.notify_all();
    }

    void failDevice() {
        {
            std::lock_guard<std::timed_mutex> lock(mutex_);
            deviceFailed_ = true;
        }
        changed_.notify_all();
    }

    void waitDrained() {
        std::unique_lock<std::timed_mutex> lock(mutex_);
        changed_.wait(lock, [this] { return active_ == 0; });
    }

private:
    using Clock = std::chrono::steady_clock;
    struct Extent { int64_t offset; int64_t bytes; };
    struct Ticket { uint64_t id = 0; int64_t bytes = 0; };

    static int64_t elapsed(Clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    }

    size_t index(int64_t offset) const {
        if (offset < base_ || offset >= base_ + bytes_ || (offset - base_) % block_) {
            throw std::invalid_argument("invalid registered EMB arena extent offset");
        }
        return static_cast<size_t>((offset - base_) / block_);
    }

    void returnExtent(int64_t offset, int64_t bytes) {
        auto it = std::lower_bound(free_.begin(), free_.end(), offset, [](const Extent& extent, int64_t key) {
            return extent.offset < key;
        });
        if (it != free_.begin() && (it - 1)->offset + (it - 1)->bytes == offset) {
            --it;
            it->bytes += bytes;
        } else {
            it = free_.insert(it, {offset, bytes});
        }
        const auto next = it + 1;
        if (next != free_.end() && it->offset + it->bytes == next->offset) {
            it->bytes += next->bytes;
            free_.erase(next);
        }
    }

    const int64_t base_;
    const int64_t bytes_;
    const int64_t block_;
    const int64_t writeEnd_;
    int64_t freeBytes_;
    int64_t quarantinedBytes_ = 0;
    int64_t nativeWriteQuarantinedBytes_ = 0;
    int64_t nativeWriteQuarantinedExtents_ = 0;
    int64_t nativeWriteActiveBytes_ = 0;
    int64_t nativeWriteActiveExtents_ = 0;
    uint64_t lastTicket_ = 0;
    size_t active_ = 0;
    bool accepting_ = true;
    bool deviceFailed_ = false;
    std::vector<Extent> free_;
    std::vector<Ticket> tickets_;
    mutable std::timed_mutex mutex_;
    std::condition_variable_any changed_;
};

}  // namespace

inline void registerEmbRegisteredArena(pybind11::module& module) {
    namespace py = pybind11;
    using Arena = EmbRegisteredArena;
    using Result = Arena::Result;
    using ReleaseGil = py::call_guard<py::gil_scoped_release>;
    py::enum_<Result>(module, "EmbArenaAcquireResult")
        .value("acquired", Result::Acquired)
        .value("lock_timeout", Result::LockTimeout)
        .value("deadline_with_lock", Result::DeadlineWithLock)
        .value("extent_unavailable", Result::ExtentUnavailable)
        .value("closed", Result::Closed)
        .value("quarantined", Result::Quarantined);
    py::class_<Arena>(module, "EmbRegisteredArena")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>())
        .def("acquire", [](Arena& arena, const at::Tensor& root, int64_t bytes, bool write, int64_t timeoutNs, bool sample) {
            using namespace std::chrono;
            if (timeoutNs < 0 || timeoutNs > 600'000'000'000LL) {
                throw std::invalid_argument("invalid registered EMB arena timeout");
            }
            const auto started = steady_clock::now();
            Arena::Acquisition acquired{};
            at::Tensor view;
            const auto materialize = [&] {
                if (std::get<1>(acquired) != 0) {
                    view = root.narrow(0, std::get<2>(acquired), std::get<3>(acquired));
                }
            };
            try {
                acquired = arena.acquire(bytes, write, 0, sample);
                const auto firstResult = std::get<0>(acquired);
                const auto usedNs = duration_cast<nanoseconds>(steady_clock::now() - started).count();
                if (usedNs < timeoutNs
                    && (firstResult == Result::LockTimeout || firstResult == Result::ExtentUnavailable)) {
                    const auto fastHold = std::get<4>(acquired);
                    py::gil_scoped_release release;
                    acquired = arena.acquire(bytes, write, timeoutNs - usedNs, sample);
                    std::get<4>(acquired) += fastHold;
                    materialize();
                } else {
                    materialize();
                }
                const auto [result, ticket, offset, capacity, heldNs] = acquired;
                return py::make_tuple(result, ticket, offset, capacity, heldNs,
                                      view.defined() ? py::cast(view) : py::none());
            } catch (...) {
                if (const auto ticket = std::get<1>(acquired); ticket != 0) {
                    py::gil_scoped_release release;
                    arena.release(ticket, std::get<2>(acquired), true, false);
                }
                throw;
            }
        })
        .def("release", &Arena::release)
        .def("native_write_quarantine", &Arena::nativeWriteQuarantine, ReleaseGil())
        .def("native_write_in_use", &Arena::nativeWriteInUse, ReleaseGil())
        .def("free_space", &Arena::freeSpace, ReleaseGil())
        .def("stop", &Arena::stop, ReleaseGil())
        .def("fail_device", &Arena::failDevice, ReleaseGil())
        .def("wait_drained", &Arena::waitDrained, ReleaseGil());
}

}  // namespace rtp_llm

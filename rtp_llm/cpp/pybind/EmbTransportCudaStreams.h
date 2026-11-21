#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/ScopeExit.h>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace rtp_llm {

class EmbTransportCudaStreams {
public:
    struct Lease {
        cudaStream_t         stream;
        at::cuda::CUDAEvent* headerFence;
    };

    static EmbTransportCudaStreams& instance() {
        // The PyTorch allocators can retain stream identifiers after a cache
        // transport closes. Reuse the handles, but keep them alive until CUDA
        // tears down the process and its allocator state.
        static auto* streams = new EmbTransportCudaStreams;
        return *streams;
    }

    Lease acquire(c10::DeviceIndex device) {
        bool createAllocation;
        {
            std::lock_guard lock(mutex_);
            for (auto& [stream, entry] : transport_) {
                if (entry.device == device && !entry.inUse) {
                    entry.inUse = true;
                    return {stream, &entry.headerFence};
                }
            }
            createAllocation = allocation_.find(device) == allocation_.end();
        }
        const c10::cuda::CUDAGuard guard(device);
        cudaStream_t               allocation = nullptr;
        if (createAllocation) {
            C10_CUDA_CHECK(cudaStreamCreateWithFlags(&allocation, cudaStreamNonBlocking));
        }
        auto         releaseAllocation = c10::make_scope_exit([allocation] {
            if (allocation != nullptr) {
                cudaStreamDestroy(allocation);
            }
        });
        cudaStream_t stream;
        C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto releaseStream = c10::make_scope_exit([stream] { cudaStreamDestroy(stream); });
        {
            std::lock_guard lock(mutex_);
            if (allocation != nullptr && allocation_.emplace(device, allocation).second) {
                releaseAllocation.release();
            }
            auto entry = transport_.try_emplace(stream, device);
            releaseStream.release();
            initialized_.store(true, std::memory_order_release);
            return {stream, &entry.first->second.headerFence};
        }
    }

    void release(cudaStream_t stream) {
        std::lock_guard lock(mutex_);
        const auto      entry = transport_.find(stream);
        if (entry != transport_.end()) {
            entry->second.inUse = false;
        }
    }

    std::optional<c10::cuda::CUDAStream> allocationFor(const c10::cuda::CUDAStream& stream) {
        if (!initialized_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        std::lock_guard lock(mutex_);
        if (transport_.find(stream.stream()) == transport_.end()) {
            return std::nullopt;
        }
        const auto device = stream.device_index();
        return c10::cuda::getStreamFromExternal(allocation_.at(device), device);
    }

private:
    struct Entry {
        explicit Entry(c10::DeviceIndex index): device(index) {}

        c10::DeviceIndex    device;
        bool                inUse = true;
        at::cuda::CUDAEvent headerFence{cudaEventBlockingSync | cudaEventDisableTiming};
    };
    std::atomic<bool>                                  initialized_{false};
    std::mutex                                         mutex_;
    std::unordered_map<cudaStream_t, Entry>            transport_;
    std::unordered_map<c10::DeviceIndex, cudaStream_t> allocation_;
};

class EmbTransportCudaStream {
public:
    explicit EmbTransportCudaStream(c10::DeviceIndex device):
        lease_(EmbTransportCudaStreams::instance().acquire(device)) {}

    ~EmbTransportCudaStream() {
        if (reusable_) {
            EmbTransportCudaStreams::instance().release(lease_.stream);
        }
    }

    // An uncertain CUDA failure can leave work pending on this stream.
    void quarantine() noexcept { reusable_ = false; }

    EmbTransportCudaStream(const EmbTransportCudaStream&)            = delete;
    EmbTransportCudaStream& operator=(const EmbTransportCudaStream&) = delete;

    std::uintptr_t address() const {
        return reinterpret_cast<std::uintptr_t>(lease_.stream);
    }

    // The node is process-owned. Only the current stream lease may record
    // this fence, and a quarantined lease never makes it available again.
    at::cuda::CUDAEvent* headerFence() const { return lease_.headerFence; }

private:
    EmbTransportCudaStreams::Lease lease_;
    bool reusable_ = true;
};

}  // namespace rtp_llm

#pragma once

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "kvcm_client/transfer_client.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkTransferObserver.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::kvcm {

// One complete GPU IOV per SSD URI, with ordered D2D gather/scatter into a
// separately registered arena. No CPU payload allocation or fallback exists.
class PackedSsdTransferClient final: public kv_cache_manager::TransferClient, public SdkTransferObservable {
    using Code    = kv_cache_manager::ClientErrorCode;
    using Buffers = kv_cache_manager::BlockBuffers;
    using Uris    = kv_cache_manager::UriStrVec;
    using Trace   = std::shared_ptr<kv_cache_manager::TransferTraceInfo>;

    struct InnerIo {
        Uris uris;
        Buffers buffers;
        Trace trace;
        size_t payload_bytes{0};
    };

public:
    struct GpuPackingContext {
        explicit GpuPackingContext(std::shared_ptr<void> storage): memory(std::move(storage)) {
            free_ranges.reserve(1026);
            free_ranges.emplace_back(0, 64ULL * 1024 * 1024);
        }
        std::shared_ptr<void> memory;
        std::mutex            mutex;
        // GPU ranges, not whole tags, remain quarantined after uncertain DMA.
        // Fixed metadata capacity prevents allocation in the lease destructor.
        std::vector<std::pair<size_t, size_t>> free_ranges;
        size_t                                 active_reservations{0};
        // Accessed only while ClientFactory's device registry mutex is held.
        bool                                         registered{false};
        std::vector<kv_cache_manager::GpuMemorySpan> registered_gpu_spans;

        bool coversRegisteredGpuSpan(const kv_cache_manager::RegistSpan& span, int device) const noexcept {
            if (!registered || !span.base || span.size == 0)
                return false;
            const auto address = reinterpret_cast<uintptr_t>(span.base);
            for (const auto& gpu : registered_gpu_spans) {
                if (gpu.device_id != device || !gpu.base)
                    continue;
                const auto base = reinterpret_cast<uintptr_t>(gpu.base);
                // Subtraction avoids wrapping either end address.
                if (address >= base && address - base <= gpu.size && span.size <= gpu.size - (address - base))
                    return true;
            }
            return false;
        }
    };
    // Copy must finish before returning: external RDMA cannot rely on CUDA
    // stream ordering to observe a gather or protect an in-flight scatter.
    using Copy =
        std::function<bool(void*, kv_cache_manager::MemoryType, const void*, kv_cache_manager::MemoryType, size_t)>;
    using ReadThreadInit                            = std::function<bool()>;
    static constexpr size_t max_operation_bytes     = 64ULL * 1024 * 1024;
    static constexpr size_t max_process_bytes       = 512ULL * 1024 * 1024;
    static constexpr size_t max_read_window_bytes   = 256ULL * 1024;
    static constexpr size_t max_read_window_objects = 4;
    static constexpr size_t max_write_window_bytes  = 2ULL * 1024 * 1024;
    static constexpr size_t max_write_window_objects = 64;

    PackedSsdTransferClient(std::unique_ptr<kv_cache_manager::TransferClient> inner,
                            Copy                                              copy,
                            std::shared_ptr<GpuPackingContext>                gpu_context,
                            ReadThreadInit                                    read_thread_init = {}):
        inner_(std::move(inner)),
        copy_(std::move(copy)),
        gpu_context_(std::move(gpu_context)),
        staging_type_(kv_cache_manager::MemoryType::GPU),
        read_thread_init_(std::move(read_thread_init)) {
        if (!inner_ || !copy_ || (!gpu_context_ || !gpu_context_->memory))
            throw std::invalid_argument("missing SSD delegate/copy operation");
    }
    ~PackedSsdTransferClient() override {
        // The SDK delegate drains before staging memory (including failures)
        // is released. ClientWrapper already serializes shutdown against I/O.
        inner_.reset();
        quarantine_.reset();
    }

    void setSdkTransferObserver(SdkTransferObserver observer) override {
        std::lock_guard<std::mutex> lock(mutex_);
        transfer_observer_ = std::move(observer);
    }

    Code LoadKvCaches(const Uris& uris, const Buffers& destination, Trace trace = nullptr) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (trace && trace->block_ids.size() != uris.size())
            return kv_cache_manager::ER_INVALID_PARAMS;
        auto operation = makeOperation(uris, destination, false, read_thread_init_ ? max_read_window_objects : 1);
        if (!operation)
            return kv_cache_manager::ER_INVALID_LOCAL_BUFFERS;
        for (size_t block = 0; block < uris.size(); ++block)
            operation->calls[block]->trace = blockTrace(trace, block, uris.size());
        auto scatter = [&](size_t block) {
            size_t offset = 0;
            for (const auto& iov : destination[block].iovs) {
                if (!iov.ignore
                    && !copy_(iov.base,
                              iov.type,
                              static_cast<unsigned char*>(operation->bytes[block].get()) + offset,
                              staging_type_,
                              iov.size)) {
                    return false;
                }
                offset += iov.size;
            }
            return true;
        };
        // Bound simultaneous object leases by count and payload. Every object
        // in a window has a disjoint registered GPU span. Join all SDK calls
        // before scattering or reusing any window memory, including failures.
        for (const auto& range : operation->read_ranges) {
            try {
                auto result = kv_cache_manager::ER_OK;
                if (read_thread_init_ && range.second - range.first > 1) {
                    std::vector<std::future<Code>> reads;
                    reads.reserve(range.second - range.first);
                    for (size_t block = range.first; block < range.second; ++block) {
                        reads.emplace_back(std::async(std::launch::async,
                                                      [this,
                                                       call = operation->calls[block]]() {
                                                          if (!read_thread_init_())
                                                              return kv_cache_manager::ER_CUDAMEMCPY_ERROR;
                                                          return loadOne(*call);
                                                      }));
                    }
                    std::exception_ptr failure;
                    for (auto& read : reads) {
                        try {
                            const auto code = read.get();
                            // Any unproven peer dominates a buffer-safe failure.
                            // Join every peer before deciding whether the arena
                            // can be released or the caller may shut the SDK down.
                            if (code != kv_cache_manager::ER_OK
                                && (result == kv_cache_manager::ER_OK || !sdkCodeProvesBufferSafety(code)))
                                result = code;
                        } catch (...) {
                            if (!failure)
                                failure = std::current_exception();
                        }
                    }
                    if (failure)
                        std::rethrow_exception(failure);
                } else {
                    const auto block = range.first;
                    result = loadOne(*operation->calls[block]);
                }
                if (result != kv_cache_manager::ER_OK) {
                    if (!sdkCodeProvesBufferSafety(result))
                        quarantine(operation);
                    return result;
                }
                for (size_t block = range.first; block < range.second; ++block) {
                    if (!scatter(block)) {
                        quarantine(operation);
                        return kv_cache_manager::ER_CUDAMEMCPY_ERROR;
                    }
                }
            } catch (...) {
                quarantine(operation);
                throw;
            }
        }
        return kv_cache_manager::ER_OK;
    }

    std::pair<Code, Uris> SaveKvCaches(const Uris& uris, const Buffers& source, Trace trace = nullptr) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (trace && trace->block_ids.size() != uris.size())
            return {kv_cache_manager::ER_INVALID_PARAMS, {}};
        auto operation = makeOperation(uris, source, true);
        if (!operation)
            return {kv_cache_manager::ER_INVALID_LOCAL_BUFFERS, {}};
        // Prepare trace slices before the first gather or SDK submission.
        for (size_t window = 0; window < operation->write_ranges.size(); ++window) {
            if (trace) {
                const auto& range = operation->write_ranges[window];
                auto slice = std::make_shared<kv_cache_manager::TransferTraceInfo>(*trace);
                slice->block_ids = {trace->block_ids.begin() + range.first, trace->block_ids.begin() + range.second};
                operation->write_calls[window]->trace = std::move(slice);
            }
        }
        auto gather = [&](size_t block) {
            size_t offset = 0;
            for (const auto& iov : source[block].iovs) {
                if (!copy_(static_cast<unsigned char*>(operation->bytes[block].get()) + offset,
                           staging_type_,
                           iov.base,
                           iov.type,
                           iov.size)) {
                    return false;
                }
                offset += iov.size;
            }
            return true;
        };
        auto actual = uris;
        for (size_t window = 0; window < operation->write_ranges.size(); ++window) {
            const auto& range = operation->write_ranges[window];
            // Each URI has a disjoint GPU subspan. Complete every gather in the
            // window before submitting its single multi-URI SDK operation.
            try {
                for (size_t block = range.first; block < range.second; ++block) {
                    if (!gather(block)) {
                        quarantine(operation);
                        return {kv_cache_manager::ER_CUDAMEMCPY_ERROR, {}};
                    }
                }
                auto result = saveBatch(*operation->write_calls[window]);
                if (result.first != kv_cache_manager::ER_OK) {
                    if (!sdkCodeProvesBufferSafety(result.first))
                        quarantine(operation);
                    return {result.first, {}};
                }
                if (!result.second.empty()) {
                    if (result.second.size() != range.second - range.first) {
                        quarantine(operation);
                        return {kv_cache_manager::ER_INVALID_PARAMS, {}};
                    }
                    for (size_t i = 0; i < result.second.size(); ++i)
                        actual[range.first + i] = std::move(result.second[i]);
                }
            } catch (...) {
                quarantine(operation);
                throw;
            }
            // Only successful completion permits the next window to reuse
            // these addresses. Unknown completion retains the entire lease.
        }
        return {kv_cache_manager::ER_OK, std::move(actual)};
    }

protected:
    Code Init(const std::string&, const kv_cache_manager::InitParams&) override {
        return kv_cache_manager::ER_SERVICE_UNSUPPORTED;
    }

private:
    void observe(bool is_read, size_t bytes, bool unknown) const noexcept {
        if (transfer_observer_) {
            try {
                transfer_observer_(is_read, bytes, unknown);
            } catch (...) {
                // Telemetry must not rewrite an SDK result or create another
                // completion-unknown event for a successful SDK operation.
                try {
                    RTP_LLM_LOG_ERROR("failed to report inner SSD SDK payload");
                } catch (...) {}
            }
        }
    }
    Code loadOne(const InnerIo& call) {
        Code code;
        try {
            code = inner_->LoadKvCaches(call.uris, call.buffers, call.trace);
        } catch (...) {
            observe(true, 0, true);
            throw;
        }
        observe(true, code == kv_cache_manager::ER_OK ? call.buffers.front().iovs.front().size : 0,
                code != kv_cache_manager::ER_OK);
        return code;
    }
    std::pair<Code, Uris> saveBatch(const InnerIo& call) {
        std::pair<Code, Uris> result;
        try {
            result = inner_->SaveKvCaches(call.uris, call.buffers, call.trace);
        } catch (...) {
            observe(false, 0, true);
            throw;
        }
        observe(false, result.first == kv_cache_manager::ER_OK ? call.payload_bytes : 0,
                result.first != kv_cache_manager::ER_OK);
        return result;
    }
    static Trace blockTrace(const Trace& trace, size_t block, size_t count) {
        if (!trace)
            return {};
        auto result = std::make_shared<kv_cache_manager::TransferTraceInfo>(*trace);
        if (trace->block_ids.size() == count)
            result->block_ids = {trace->block_ids[block]};
        else
            return {};
        return result;
    }
    struct GpuRangeLease {
        std::shared_ptr<GpuPackingContext> context;
        size_t                             begin{0}, bytes{0};
        ~GpuRangeLease() {
            if (!context)
                return;
            std::lock_guard<std::mutex> guard(context->mutex);
            auto&                       ranges = context->free_ranges;
            ranges.emplace_back(begin, bytes);  // capacity reserved before any I/O
            std::sort(ranges.begin(), ranges.end());
            size_t kept = 0;
            for (size_t index = 0; index < ranges.size(); ++index) {
                const auto range = ranges[index];
                if (kept && ranges[kept - 1].first + ranges[kept - 1].second == range.first)
                    ranges[kept - 1].second += range.second;
                else
                    ranges[kept++] = range;
            }
            ranges.resize(kept);
            --context->active_reservations;
        }
    };
    struct Operation {
        size_t                                 charged = 0;
        std::shared_ptr<GpuRangeLease>         gpu_lease;
        std::vector<std::shared_ptr<void>>     bytes;
        Buffers                                buffers;
        std::vector<std::shared_ptr<InnerIo>> calls;
        std::vector<std::pair<size_t, size_t>> read_ranges;
        std::vector<std::pair<size_t, size_t>> write_ranges;
        std::vector<std::shared_ptr<InnerIo>> write_calls;
        ~Operation() {
            process_bytes_.fetch_sub(charged);
        }
    };
    void quarantine(const std::shared_ptr<Operation>& operation) {
        quarantine_ = operation;
        // Only this client's unresolved GPU range stays reserved. Independent
        // tags can continue using other ranges of the registered arena.
    }
    std::shared_ptr<Operation>
    makeOperation(const Uris& uris, const Buffers& original, bool write, size_t read_window_objects = 1) {
        // Any uncertain completion fails this client closed until shutdown;
        // failed buffers cannot be overwritten by a later operation.
        if (quarantine_ || uris.empty() || uris.size() != original.size())
            return {};
        size_t              total = 0;
        std::vector<size_t> sizes;
        for (const auto& block : original) {
            size_t size = 0;
            for (const auto& iov : block.iovs) {
                if (!iov.size || (write && iov.ignore) || (!iov.ignore && !iov.base)
                    || iov.type != kv_cache_manager::MemoryType::GPU || iov.size > max_operation_bytes - size)
                    return {};
                size += iov.size;
            }
            if (!size || size > max_operation_bytes - total)
                return {};
            total += size;
            sizes.push_back(size);
        }
        auto                operation = std::make_shared<Operation>();
        std::vector<size_t> offsets(sizes.size(), 0);
        size_t              charge = 0;
        const size_t window_objects = write ? max_write_window_objects : read_window_objects;
        const size_t window_bytes = write ? max_write_window_bytes : max_read_window_bytes;
        auto& ranges = write ? operation->write_ranges : operation->read_ranges;
        for (size_t begin = 0; begin < sizes.size();) {
            size_t end = begin, bytes = 0;
            while (end < sizes.size() && end - begin < window_objects) {
                // A single large state/SWA object is allowed alone, bounded
                // by max_operation_bytes, and cannot bring a sibling with it.
                if (end > begin && sizes[end] > window_bytes - std::min(bytes, window_bytes))
                    break;
                offsets[end] = bytes;
                bytes += sizes[end++];
            }
            ranges.emplace_back(begin, end);
            charge = std::max(charge, bytes);
            begin  = end;
        }
        // Charge the largest simultaneously live GPU window, not the whole
        // logical request. A large object still reserves its own window.
        size_t previous = process_bytes_.load();
        do {
            if (previous > max_process_bytes - charge)
                return {};
        } while (!process_bytes_.compare_exchange_weak(previous, previous + charge));
        operation->charged = charge;
        operation->bytes.resize(sizes.size());
        operation->buffers.resize(sizes.size());
        size_t offset = 0;
        {
            auto lease = std::make_shared<GpuRangeLease>();
            {
                std::lock_guard<std::mutex> guard(gpu_context_->mutex);
                if (gpu_context_->active_reservations >= 1024)
                    return {};
                auto& free = gpu_context_->free_ranges;
                auto  it =
                    std::find_if(free.begin(), free.end(), [charge](const auto& r) { return r.second >= charge; });
                if (it == free.end())
                    return {};
                offset = it->first;
                it->first += charge;
                it->second -= charge;
                if (it->second == 0)
                    free.erase(it);
                ++gpu_context_->active_reservations;
                lease->context = gpu_context_;
                lease->begin   = offset;
                lease->bytes   = charge;
            }
            operation->gpu_lease = std::move(lease);
        }
        for (size_t i = 0; i < sizes.size(); ++i) {
            operation->bytes[i] = std::shared_ptr<void>(
                gpu_context_->memory, static_cast<unsigned char*>(gpu_context_->memory.get()) + offset + offsets[i]);
            operation->buffers[i].iovs.push_back({staging_type_, operation->bytes[i].get(), sizes[i], false});
            if (!write) {
                auto call = std::make_shared<InnerIo>();
                call->uris.push_back(uris[i]);
                call->buffers.push_back(operation->buffers[i]);
                operation->calls.push_back(std::move(call));
            }
        }
        if (write) {
            operation->write_calls.reserve(operation->write_ranges.size());
            for (const auto& range : operation->write_ranges) {
                auto call = std::make_shared<InnerIo>();
                call->uris.assign(uris.begin() + range.first, uris.begin() + range.second);
                call->buffers.assign(operation->buffers.begin() + range.first,
                                     operation->buffers.begin() + range.second);
                for (size_t i = range.first; i < range.second; ++i)
                    call->payload_bytes += sizes[i];
                operation->write_calls.push_back(std::move(call));
            }
        }
        return operation;
    }
    inline static std::atomic<size_t>                 process_bytes_{0};
    std::unique_ptr<kv_cache_manager::TransferClient> inner_;
    Copy                                              copy_;
    std::shared_ptr<GpuPackingContext>                gpu_context_;
    kv_cache_manager::MemoryType                      staging_type_;
    ReadThreadInit                                    read_thread_init_;
    std::mutex                                        mutex_;
    SdkTransferObserver                               transfer_observer_;
    std::shared_ptr<Operation>                        quarantine_;
};
}  // namespace rtp_llm::kvcm

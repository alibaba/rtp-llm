#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include <grpcpp/grpcpp.h>
#include <grpcpp/impl/codegen/sync_stream.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

// Per-request in-memory response buffer owned by a single Prefill DP worker.
// The engine-side detached worker appends GenerateOutputsPB via a writer; the
// Frontend drains via FetchResponse on the same DP worker. NOT shared across
// DP ranks — each DP worker keeps its own registry.
struct ResponseBufferEntry {
    std::deque<GenerateOutputsPB> queue;
    std::atomic<bool>             done{false};
    std::atomic<bool>             cancelled{false};
    std::optional<grpc::Status>   error_status;
    std::mutex                    mu;
    std::condition_variable       cv;
    int64_t                       last_activity_us{0};
};

// Registry is keyed by request_id. Entries live until a fetch drains to done
// / error / cancel and then the caller erases, or until TTL GC sweeps idle
// entries left behind by a crashed Frontend.
class ResponseBufferRegistry {
public:
    ResponseBufferRegistry() = default;

    // Returns the existing entry if request_id is already registered
    // (duplicate DpInternalEnqueue → caller must treat as ALREADY_EXISTS),
    // otherwise creates and returns a fresh entry.
    std::shared_ptr<ResponseBufferEntry> create(int64_t request_id);

    std::shared_ptr<ResponseBufferEntry> get(int64_t request_id);

    void erase(int64_t request_id);

    // Remove entries older than ttl whose queue is empty AND done/cancelled.
    // Called by a background sweep thread. Returns count swept.
    size_t gc(std::chrono::microseconds ttl);

    size_t size() const;

private:
    mutable std::mutex                                                mu_;
    std::unordered_map<int64_t, std::shared_ptr<ResponseBufferEntry>> map_;
};

// Adapter from the engine's pollStreamOutput / pollRemoteOutput writer
// interface onto a ResponseBufferEntry. The writer signals success when the
// entry is not cancelled; it returns false on cancel so the producer side
// can terminate early like a closed gRPC stream would.
class ResponseBufferWriter: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    explicit ResponseBufferWriter(std::shared_ptr<ResponseBufferEntry> entry): entry_(std::move(entry)) {}

    bool Write(const GenerateOutputsPB& outputs, grpc::WriteOptions /*options*/) override;

private:
    std::shared_ptr<ResponseBufferEntry> entry_;
};

}  // namespace rtp_llm

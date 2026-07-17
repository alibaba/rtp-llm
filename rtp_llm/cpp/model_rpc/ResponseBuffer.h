#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include <grpcpp/grpcpp.h>
#include <grpcpp/impl/codegen/sync_stream.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

struct ResponseBufferEntry {
    static constexpr size_t kMaxQueueSize = 10240;
    static size_t           kMaxQueueBytes;  // 512MB per entry, defined in .cc (mutable for testability)

    struct DrainResult {
        std::deque<GenerateOutputsPB> outputs;
        grpc::Status                  terminal_status = grpc::Status::OK;
        bool                          terminal        = false;
    };

    void        installCancelProducer(std::function<void()> producer);
    bool        write(const GenerateOutputsPB& outputs);
    DrainResult waitAndDrain(std::chrono::milliseconds timeout);
    void        cancel();
    bool        isCancelled() const;
    size_t      droppedCount() const;

private:
    friend class ResponseBufferRegistry;

    void finish(const grpc::Status& status);
    bool producerDone() const;
    bool discardIfProducerDoneAndIdle(int64_t now_us, int64_t ttl_us);

    std::deque<GenerateOutputsPB> queue;
    std::atomic<bool>             done{false};
    std::atomic<bool>             cancelled{false};
    std::optional<grpc::Status>   error_status;
    std::function<void()>         cancel_producer;
    std::mutex                    mu;
    std::condition_variable       cv;
    std::atomic<int64_t>          last_activity_us{0};
    size_t                        queue_bytes_{0};
    std::atomic<size_t>           dropped_count_{0};
};

class ResponseBufferRegistry {
public:
    enum class ClaimStatus {
        SUCCESS,
        NOT_FOUND,
        ALREADY_CLAIMED,
    };

    struct ClaimResult {
        ClaimStatus                          status = ClaimStatus::NOT_FOUND;
        std::shared_ptr<ResponseBufferEntry> entry;
    };

    ResponseBufferRegistry() = default;

    std::shared_ptr<ResponseBufferEntry> reserve(int64_t request_id);
    void        publish(int64_t request_id, const std::shared_ptr<ResponseBufferEntry>& expected_entry);
    ClaimResult claim(int64_t request_id);
    void
    finish(int64_t request_id, const std::shared_ptr<ResponseBufferEntry>& expected_entry, const grpc::Status& status);
    void   releaseClaim(int64_t request_id, const std::shared_ptr<ResponseBufferEntry>& expected_entry);
    void   abort(int64_t request_id, const std::shared_ptr<ResponseBufferEntry>& expected_entry);
    void   cancelAll();
    size_t gc(std::chrono::microseconds ttl);
    size_t size() const;

private:
    enum class State {
        PENDING,
        READY,
        FETCH_CLAIMED,
    };

    struct Record {
        std::shared_ptr<ResponseBufferEntry> entry;
        State                                state          = State::PENDING;
        bool                                 fetch_released = false;
    };

    mutable std::mutex                  mu_;
    std::unordered_map<int64_t, Record> map_;
};

class ResponseBufferWriter: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    explicit ResponseBufferWriter(std::shared_ptr<ResponseBufferEntry> entry): entry_(std::move(entry)) {}

    bool Write(const GenerateOutputsPB& outputs, grpc::WriteOptions options) override;

private:
    std::shared_ptr<ResponseBufferEntry> entry_;
};

}  // namespace rtp_llm

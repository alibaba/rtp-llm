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
    std::deque<GenerateOutputsPB> queue;
    std::atomic<bool>             done{false};
    std::atomic<bool>             cancelled{false};
    std::optional<grpc::Status>   error_status;
    std::function<void()>          cancel_producer;
    std::mutex                    mu;
    std::condition_variable       cv;
    int64_t                       last_activity_us{0};
};

class ResponseBufferRegistry {
public:
    ResponseBufferRegistry() = default;

    std::shared_ptr<ResponseBufferEntry> createOrGet(int64_t request_id);
    std::shared_ptr<ResponseBufferEntry> reserve(int64_t request_id);
    std::shared_ptr<ResponseBufferEntry> get(int64_t request_id);
    void                                 erase(int64_t request_id);
    void                                 cancelAll();
    size_t                               gc(std::chrono::microseconds ttl);
    size_t                               size() const;

private:
    mutable std::mutex                                                mu_;
    std::unordered_map<int64_t, std::shared_ptr<ResponseBufferEntry>> map_;
};

class ResponseBufferWriter: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    explicit ResponseBufferWriter(std::shared_ptr<ResponseBufferEntry> entry): entry_(std::move(entry)) {}

    bool Write(const GenerateOutputsPB& outputs, grpc::WriteOptions options) override;

private:
    std::shared_ptr<ResponseBufferEntry> entry_;
};

}  // namespace rtp_llm

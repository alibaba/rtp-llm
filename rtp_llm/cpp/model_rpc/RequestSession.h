#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <vector>

#include <grpcpp/impl/codegen/sync_stream.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class GenerateStream;
struct AsyncProducerCancelState;

enum class SessionState {
    ADMITTED,
    RUNNING,
    FINISHED,
    CANCELLED,
    ERROR
};

enum class LookupResult {
    NOT_FOUND,
    RUNNING,
    FINISHED_IN_TTL,
    GONE,
    ALREADY_ATTACHED
};

enum class CancelReason {
    EXPLICIT_CANCEL,
    ATTACH_DEADLINE,
    EXECUTION_TIMEOUT,
    SLO_DEADLINE
};

// PD 路径 bounded 缓冲区，承载 finishStream 全程输出
class BoundedRelay {
public:
    explicit BoundedRelay(size_t cap = 1000);

    // cap 满时 cancel-aware 阻塞；closed 时返回 false
    bool push(GenerateOutputsPB output);

    bool tryPop(GenerateOutputsPB* out);

    size_t drainTo(std::vector<GenerateOutputsPB>* out);

    // 阻塞等数据或 close，返回 true 表示有数据
    bool waitForData(std::chrono::milliseconds timeout);

    void close();

    bool   empty() const;
    size_t size() const;
    bool   isClosed() const;

private:
    std::deque<GenerateOutputsPB> queue_;
    size_t                        cap_;
    mutable std::mutex            mu_;
    std::condition_variable       cv_;
    bool                          closed_{false};
};

// WriterInterface adapter: writes to BoundedRelay instead of gRPC ServerWriter
class RelayWriter: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    explicit RelayWriter(BoundedRelay* relay): relay_(relay) {}
    bool Write(const GenerateOutputsPB& outputs, grpc::WriteOptions /*options*/) override {
        return relay_->push(outputs);
    }
private:
    BoundedRelay* relay_;
};

class RequestSession {
public:
    RequestSession(int64_t request_id, int64_t batch_id, int64_t admitted_at_us, bool is_pd = false);

    void cancel(CancelReason reason);

    bool acquireLease();
    void releaseLease();

    SessionState deriveState();
    bool         isTerminal() const;

    BoundedRelay&                       getRelay() { return relay_; }
    std::shared_ptr<GenerateStream>     getStream() const;
    int64_t                             requestId() const { return request_id_; }
    int64_t                             batchId() const { return batch_id_; }
    int64_t                             admittedAtUs() const { return admitted_at_us_; }
    int64_t                             finishedAtUs() const { return finished_at_us_.load(); }
    SessionState                        state() const { return state_.load(); }
    CancelReason                        cancelReason() const { return cancel_reason_; }
    bool                                hasConsumer() const { return consumer_taken_.load(); }
    bool                                isPd() const { return is_pd_; }

    void setStream(std::shared_ptr<GenerateStream> stream);
    void setCancelState(std::shared_ptr<AsyncProducerCancelState> cancel_state);
    void markFinished();
    void markError(const std::string& error_msg);

private:
    bool trySetTerminal(SessionState target);

    int64_t                                    request_id_;
    int64_t                                    batch_id_;
    bool                                       is_pd_;
    std::shared_ptr<GenerateStream>            stream_;
    BoundedRelay                               relay_;
    std::atomic<bool>                          consumer_taken_{false};
    std::atomic<SessionState>                  state_{SessionState::ADMITTED};
    CancelReason                               cancel_reason_{CancelReason::EXPLICIT_CANCEL};
    int64_t                                    admitted_at_us_;
    std::atomic<int64_t>                       finished_at_us_{0};
    std::mutex                                 mu_;
    std::shared_ptr<AsyncProducerCancelState>  cancel_state_;
};

class SessionManager {
public:
    explicit SessionManager(int64_t terminal_ttl_us = 10LL * 60 * 1000 * 1000,
                            int64_t attach_deadline_us = 30LL * 1000 * 1000);
    ~SessionManager();

    bool registerSession(int64_t request_id, std::shared_ptr<RequestSession> session);

    std::pair<LookupResult, std::shared_ptr<RequestSession>> lookup(int64_t request_id);

    bool cancelSession(int64_t request_id, CancelReason reason);

    void removeSession(int64_t request_id);

    void   startGc();
    void   stopGc();
    size_t gcOnce();

    size_t reapAttachDeadline();

    size_t size() const;
    void   cancelAll();

private:
    std::unordered_map<int64_t, std::shared_ptr<RequestSession>> sessions_;
    std::unordered_map<int64_t, int64_t>                         tombstones_;
    mutable std::mutex                                           mu_;

    std::atomic<bool>       gc_stop_{false};
    std::thread             gc_thread_;
    std::mutex              gc_mu_;
    std::condition_variable gc_cv_;

    int64_t terminal_ttl_us_;
    int64_t attach_deadline_us_;
};

}  // namespace rtp_llm

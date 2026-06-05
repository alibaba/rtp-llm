#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <thread>
#include <vector>

#include <grpcpp/impl/codegen/sync_stream.h>
#include <grpcpp/support/status.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class GenerateStream;
struct AsyncProducerCancelState;

// ========================== Enums ==========================

enum class TerminalReason {
    FINISHED,
    ERROR,
    CANCELLED,
    TIMEOUT,
    RESOURCE_EXHAUSTED
};

enum class CancelReason {
    EXPLICIT_CANCEL,
    ATTACH_DEADLINE,
    EXECUTION_TIMEOUT,
    SLO_DEADLINE
};

enum class AttachState {
    LIVE,
    ALREADY_ATTACHED,
    FINISHED_IN_TTL,
    GONE,
    NOT_FOUND,
    EPOCH_MISMATCH
};

enum class BatchEnqueueStatus {
    ADMITTED,
    ALREADY_ADMITTED,
    REJECTED,
    CONFLICT_PAYLOAD
};

enum class PushResult {
    OK,
    CLOSED,
    BUDGET_EXCEEDED
};

// ========================== Value structs ==========================

struct TerminalInfo {
    TerminalReason    reason = TerminalReason::FINISHED;
    grpc::Status      status;
    int64_t           terminal_time_us = 0;
    int64_t           payload_expire_time_us = 0;
    int64_t           tombstone_expire_time_us = 0;
};

struct FrozenSnapshot {
    int64_t                          request_id = 0;
    int64_t                          session_epoch = 0;
    TerminalInfo                     terminal;
    std::vector<GenerateOutputsPB>   outputs;
};

struct TombstoneRecord {
    int64_t         request_id = 0;
    int64_t         session_epoch = 0;
    TerminalReason  terminal_reason = TerminalReason::FINISHED;
    grpc::Status    final_status;
    int64_t         terminal_time_us = 0;
    int64_t         tombstone_expire_time_us = 0;
};

struct SessionCreateOptions {
    int64_t request_id = 0;
    int64_t batch_id = 0;
    int32_t batch_index = 0;
    int64_t payload_hash = 0;
    int64_t create_time_us = 0;
    int64_t attach_deadline_us = 30LL * 1000 * 1000;
    int64_t payload_ttl_us = 10LL * 60 * 1000 * 1000;
    int64_t tombstone_ttl_us = 20LL * 60 * 1000 * 1000;
    size_t  max_buffer_bytes = 0;
};

struct CreateResult {
    BatchEnqueueStatus                status = BatchEnqueueStatus::ADMITTED;
    std::shared_ptr<class RequestSession> session;
    int64_t                           session_epoch = 0;
    grpc::Status                      error;
};

struct LookupResult {
    AttachState                                  state = AttachState::NOT_FOUND;
    std::shared_ptr<class RequestSession>        session;
    std::shared_ptr<const FrozenSnapshot>        snapshot;
    std::optional<TerminalInfo>                  terminal;
};

struct LeaseResult {
    AttachState state = AttachState::LIVE;
    uint64_t    lease_id = 0;
};

struct PopResult {
    enum Status { OUTPUT, WAIT_TIMEOUT, CLOSED };
    Status                           status = CLOSED;
    std::vector<GenerateOutputsPB>   outputs;
};

// ========================== RequestOutputBuffer ==========================

class RequestOutputBuffer {
public:
    explicit RequestOutputBuffer(size_t max_bytes = 0);

    PushResult push(GenerateOutputsPB&& output);

    PopResult popLive(uint64_t lease_id, int64_t wait_timeout_ms);

    std::shared_ptr<const FrozenSnapshot> freeze(int64_t request_id,
                                                  int64_t session_epoch,
                                                  TerminalInfo terminal);
    std::shared_ptr<const FrozenSnapshot> snapshot() const;

    void close();
    void dropSnapshot();

    size_t bytes() const;
    size_t pendingLiveCount() const;
    bool   isClosed() const;

private:
    mutable std::mutex              mu_;
    std::condition_variable         cv_;
    std::vector<GenerateOutputsPB>  outputs_;
    size_t                          next_live_seq_ = 0;
    bool                            closed_ = false;
    size_t                          bytes_ = 0;
    size_t                          max_bytes_;
    std::shared_ptr<const FrozenSnapshot> snapshot_;
};

// WriterInterface adapter: routes finishStream output through RequestSession
class SessionWriter: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    explicit SessionWriter(std::shared_ptr<class RequestSession> session): session_(std::move(session)) {}
    bool Write(const GenerateOutputsPB& outputs, grpc::WriteOptions options) override;
private:
    std::shared_ptr<class RequestSession> session_;
};

// ========================== RequestSession ==========================

class RequestSession {
public:
    RequestSession(const SessionCreateOptions& options, int64_t session_epoch);

    int64_t requestId() const { return request_id_; }
    int64_t sessionEpoch() const { return session_epoch_; }
    int64_t batchId() const { return batch_id_; }
    int64_t admittedAtUs() const { return create_time_us_; }
    int64_t payloadHash() const { return payload_hash_; }
    bool    samePayload(int64_t hash) const { return payload_hash_ == hash; }
    bool    isTerminal() const;
    bool    hasConsumer() const;

    bool bindStream(std::shared_ptr<GenerateStream> stream);
    void setCancelState(std::shared_ptr<AsyncProducerCancelState> cancel_state);
    std::shared_ptr<GenerateStream> getStream() const;

    PushResult pushOutput(GenerateOutputsPB&& output);

    LeaseResult acquireLiveLease();
    PopResult   popLive(uint64_t lease_id, int64_t wait_timeout_ms);
    void        releaseLiveLease(uint64_t lease_id);

    bool cancel(CancelReason reason, int64_t now_us);
    bool finalizeTerminal(TerminalReason reason, grpc::Status status, int64_t now_us);

    LookupResult buildLookup(int64_t now_us) const;
    bool         payloadExpired(int64_t now_us) const;
    TerminalInfo terminalInfo() const;
    std::shared_ptr<const FrozenSnapshot> snapshot() const;

    int64_t attachDeadlineUs() const { return attach_deadline_us_; }
    int64_t payloadTtlUs() const { return payload_ttl_us_; }
    int64_t tombstoneTtlUs() const { return tombstone_ttl_us_; }

private:
    enum class Phase { LIVE, FINALIZING, TERMINAL };

    RequestOutputBuffer            output_buffer_;
    std::atomic<Phase>             phase_{Phase::LIVE};

    const int64_t                  request_id_;
    const int64_t                  session_epoch_;
    const int64_t                  payload_hash_;
    const int64_t                  batch_id_;
    const int32_t                  batch_index_;
    const int64_t                  create_time_us_;
    const int64_t                  attach_deadline_us_;
    const int64_t                  payload_ttl_us_;
    const int64_t                  tombstone_ttl_us_;

    mutable std::mutex             lease_mu_;
    bool                           lease_active_ = false;
    uint64_t                       lease_id_ = 0;
    uint64_t                       next_lease_id_ = 1;

    mutable std::mutex             terminal_mu_;
    TerminalInfo                   terminal_;

    mutable std::mutex             resource_mu_;
    std::shared_ptr<GenerateStream>            stream_;
    std::shared_ptr<AsyncProducerCancelState>  cancel_state_;
};

// ========================== SessionManager ==========================

class SessionManager {
public:
    explicit SessionManager(int64_t default_payload_ttl_us = 10LL * 60 * 1000 * 1000,
                            int64_t default_attach_deadline_us = 30LL * 1000 * 1000,
                            int64_t default_tombstone_ttl_us = 20LL * 60 * 1000 * 1000);
    ~SessionManager();

    CreateResult create(const SessionCreateOptions& options);

    LookupResult lookup(int64_t request_id,
                        int64_t session_epoch,
                        int64_t now_us);

    bool cancelSession(int64_t request_id,
                       int64_t session_epoch,
                       CancelReason reason,
                       int64_t now_us);

    void startGc();
    void stopGc();
    size_t gcOnce();
    size_t reapTimeouts(int64_t now_us);
    void shutdown(int64_t now_us);

    size_t size() const;
    size_t tombstoneCount() const;

private:
    mutable std::mutex mu_;
    std::unordered_map<int64_t, std::shared_ptr<RequestSession>> sessions_;
    std::unordered_map<int64_t, TombstoneRecord>                 tombstones_;
    int64_t next_session_epoch_ = 1;

    int64_t default_payload_ttl_us_;
    int64_t default_attach_deadline_us_;
    int64_t default_tombstone_ttl_us_;

    std::atomic<bool>       gc_stop_{false};
    std::thread             gc_thread_;
    std::mutex              gc_mu_;
    std::condition_variable gc_cv_;
};

}  // namespace rtp_llm

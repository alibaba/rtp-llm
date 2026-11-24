#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockService.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockFormat.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockTimer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"
#include <list>
#include <google/protobuf/arena.h>
#include <arpa/inet.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <thread>
#include <string_view>
#include <unordered_map>

namespace rtp_llm {
namespace {
using Clock = std::chrono::steady_clock;
using Done  = google::protobuf::Closure;
// Non-payload representations coexist through validation and delayed encoding:
// reverse request/reply keys, protobuf objects, maps/sets, decoded descriptors,
// vector capacity, callbacks and per-operation arena overhead. Original request
// storage and serialization are measured separately rather than inferred from
// payload bytes. Unknown nested fields are not part of this bounded protocol.
constexpr uint64_t kMetadataPerBlock       = 8192;
constexpr uint64_t kMetadataPerOperation   = 8192;
constexpr uint64_t kMinimumRequestMetadata = 8192;
bool               unknownFields(const google::protobuf::Message& message) {
    const auto* reflection = message.GetReflection();
    if (reflection->GetUnknownFields(message).field_count())
        return true;
    const auto* descriptor = message.GetDescriptor();
    for (int i = 0; i < descriptor->field_count(); ++i) {
        const auto* field = descriptor->field(i);
        if (field->cpp_type() != google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE)
            continue;
        if (field->is_repeated()) {
            for (int j = 0; j < reflection->FieldSize(message, field); ++j)
                if (unknownFields(reflection->GetRepeatedMessage(message, field, j)))
                    return true;
        } else if (reflection->HasField(message, field) && unknownFields(reflection->GetMessage(message, field)))
            return true;
    }
    return false;
}
uint64_t operationEnvelope(uint64_t bytes, uint32_t blocks) {
    return bytes + uint64_t(blocks) * 512 + 4096;
}
uint64_t additionalMetadata(const google::protobuf::Message& request, uint32_t blocks) {
    // Called only after the global protobuf envelope and semantic shape checks.
    return request.SpaceUsedLong() + request.ByteSizeLong() + uint64_t(blocks) * kMetadataPerBlock
           + kMetadataPerOperation;
}

uint64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
RemoteMockSession sessionOf(const remote_mock::Session& s) {
    return {s.server_epoch(), s.session_id(), s.generation()};
}
void setSession(remote_mock::Session* out, const RemoteMockSession& s) {
    out->set_server_epoch(s.server_epoch);
    out->set_session_id(s.session_id);
    out->set_generation(s.generation);
}
KvCacheStoreServiceErrorCode errorCode(const absl::Status& status) {
    return status.ok() ? EC_SUCCESS :
                         (absl::IsResourceExhausted(status) ? EC_FAILED_LOAD_BUFFER : EC_FAILED_INVALID_REQ);
}
// Rejections are sent only after admission/lifecycle locks have unwound.
class DeferredDone: public Done {
public:
    explicit DeferredDone(Done* done): done_(done) {}
    ~DeferredDone() override {
        if (requested_ && done_)
            done_->Run();
    }
    void Run() override {
        requested_ = true;
    }
    Done* release() {
        auto result = done_;
        done_       = nullptr;
        return result;
    }

private:
    Done* done_;
    bool  requested_ = false;
};
class FunctionDone: public Done {
public:
    explicit FunctionDone(std::function<void()> fn): fn_(std::move(fn)) {}
    void Run() override {
        auto fn = std::move(fn_);
        delete this;
        fn();
    }

private:
    std::function<void()> fn_;
};
// Shared independently from the service: ANet owns response arenas until its
// delayed encoder finishes. Admission and source/slot ownership follow that
// arena, rather than ending at done->Run().
struct Admissions {
    std::mutex                                           mutex;
    std::map<std::string, std::pair<uint32_t, uint32_t>> peers;
};
struct Admission {
    std::shared_ptr<Admissions>                               owner;
    std::string                                               peer;
    bool                                                      write = false;
    std::shared_ptr<RemoteMockReadLease>                      read;
    std::shared_ptr<RemoteMockWriteLease>                     slot;
    std::vector<std::shared_ptr<const RemoteMockSourceLease>> sources;
    ~Admission() {
        std::lock_guard<std::mutex> lock(owner->mutex);
        auto                        it = owner->peers.find(peer);
        if (it != owner->peers.end()) {
            --(write ? it->second.second : it->second.first);
            if (!it->second.first && !it->second.second)
                owner->peers.erase(it);
        }
    }
};
std::string peerOf(google::protobuf::RpcController* controller) {
    auto c = dynamic_cast<arpc::ANetRPCController*>(controller);
    return c ? c->GetClientAddress() : "direct";
}
template<class Response>
void fail(Response* response, Done* done, KvCacheStoreServiceErrorCode code = EC_FAILED_INVALID_REQ) {
    response->set_error_code(code);
    done->Run();
}
void pin(google::protobuf::Message* response, const std::shared_ptr<Admission>& admission) {
    if (auto arena = response->GetArena())
        arena->Own(new std::shared_ptr<Admission>(admission));
}
}  // namespace

struct RemoteMockService::Impl {
    explicit Impl(RemoteMockServiceOptions opts): options(std::move(opts)), state(options.limits), control(*this) {}
    struct Control: public remote_mock::RemoteMockControlService {
        explicit Control(Impl& impl): impl(impl) {}
        void  initRank(google::protobuf::RpcController*,
                       const remote_mock::InitRankRequest* q,
                       remote_mock::ControlResponse*       r,
                       Done*                               d) override;
        void  ready(google::protobuf::RpcController*,
                    const remote_mock::SessionRequest* q,
                    remote_mock::ControlResponse*      r,
                    Done*                              d) override;
        void  close(google::protobuf::RpcController*,
                    const remote_mock::SessionRequest* q,
                    remote_mock::ControlResponse*      r,
                    Done*                              d) override;
        void  health(google::protobuf::RpcController*,
                     const remote_mock::HealthRequest*,
                     remote_mock::ControlResponse* r,
                     Done*                         d) override;
        Impl& impl;
    };
    struct Operation {
        std::list<std::function<void()>>      failure;
        uint64_t                              id = 0;
        RemoteMockSession                     session;
        Clock::time_point                     start   = Clock::now();
        uint32_t                              timeout = 0;
        uint32_t                              delay   = 0;
        uint64_t                              bytes   = 0;
        bool                                  write   = false;
        std::shared_ptr<Admission>            admission;
        Done*                                 done              = nullptr;
        const CacheLoadRequest*               load_request      = nullptr;
        CacheLoadResponse*                    load_response     = nullptr;
        const CacheTransferRequest*           transfer_request  = nullptr;
        CacheTransferResponse*                transfer_response = nullptr;
        BlockReadRequest                      reverse_request;
        BlockReadResponse                     reverse_response;
        arpc::ANetRPCController               controller;
        std::shared_ptr<arpc::RPCChannelBase> channel;
        std::vector<RemoteMockKey>            keys;
    };
    RemoteMockServiceOptions                                 options;
    RemoteMockState                                          state;
    Control                                                  control;
    TcpClient                                                client;
    std::shared_ptr<Admissions>                              admissions = std::make_shared<Admissions>();
    std::mutex                                               lifecycle;
    bool                                                     initialized = false;
    bool                                                     stopping    = false;
    bool                                                     closing     = false;
    std::unique_ptr<RemoteMockTimer>                         timer;
    std::set<std::string>                                    endpoints;
    std::mutex                                               mutex;
    std::condition_variable                                  condition;
    std::list<std::function<void()>>                         queue;
    std::list<std::function<void()>>                         writes;
    std::list<std::function<void()>>                         urgent;
    bool                                                     prefer_write = false;
    std::unordered_map<uint64_t, std::shared_ptr<Operation>> operations;
    uint64_t                                                 next_id        = 0;
    size_t                                                   sync_producers = 0;
    bool                                                     worker_stop    = false;
    std::thread                                              worker;
    mutable std::mutex                                       stats_mutex;
    RemoteMockServiceStats                                   counters;
    std::ofstream                                            events;
    uint64_t                                                 event_bytes           = 0;
    uint64_t                                                 event_count           = 0;
    static constexpr uint64_t                                kCaptureFooterReserve = 192;

    uint64_t envelope() const {
        return uint64_t(options.limits.max_rpc_bytes) + uint64_t(options.limits.max_blocks_per_rpc) * 512 + 4096;
    }
    bool shape(const google::protobuf::Message& message) const {
        return message.ByteSizeLong() <= envelope();
    }
    bool timeoutValid(uint32_t timeout) const {
        return timeout > 0 && timeout <= options.max_timeout_ms;
    }
    void event(const char* name, const Operation& op, uint64_t deadline = 0, uint64_t payload = UINT64_MAX) noexcept
        try {
        std::lock_guard<std::mutex> lock(stats_mutex);
        if (!events.is_open() || counters.event_capture_failed)
            return;
        auto line = std::string("{\"event\":\"") + name + "\",\"operation\":" + std::to_string(op.id)
                    + ",\"generation\":" + std::to_string(op.session.generation)
                    + ",\"steady_ns\":" + std::to_string(nowNs()) + ",\"deadline_ns\":" + std::to_string(deadline)
                    + ",\"payload_bytes\":" + std::to_string(payload == UINT64_MAX ? op.bytes : payload)
                    + (deadline ? ",\"boundary_ns\":" + std::to_string(deadline - uint64_t(op.delay) * 1000000) : "")
                    + "}\n";
        if (options.max_event_bytes < kCaptureFooterReserve
            || line.size() > options.max_event_bytes - kCaptureFooterReserve - event_bytes) {
            counters.event_capture_failed = true;
            return;
        }
        events << line;
        events.flush();
        if (!events) {
            counters.event_capture_failed = true;
            return;
        }
        event_bytes += line.size();
        ++event_count;
    } catch (...) {
        // Optional diagnostics cannot strand an accepted RPC on its failure path.
        std::lock_guard<std::mutex> lock(stats_mutex);
        counters.event_capture_failed = true;
    }
    void finishCapture() {
        // Called after the timer and service worker have joined: no producer can
        // append behind this certificate. Missing/failed footer is incomplete.
        std::lock_guard<std::mutex> lock(stats_mutex);
        if (!events.is_open())
            return;
        if (options.max_event_bytes < kCaptureFooterReserve || !events) {
            counters.event_capture_failed = true;
            return;
        }
        const auto footer = std::string("{\"capture_complete\":") + (counters.event_capture_failed ? "false" : "true")
                            + ",\"event_count\":" + std::to_string(event_count)
                            + ",\"prefix_bytes\":" + std::to_string(event_bytes) + "}\n";
        if (footer.size() > options.max_event_bytes - event_bytes) {
            counters.event_capture_failed = true;
            return;
        }
        events << footer;
        events.flush();
        if (!events)
            counters.event_capture_failed = true;
        events.close();
    }
    void enqueue(std::function<void()> fn, bool write = false, bool priority = false) {
        // Each admitted operation owns at most one continuation. Cancellation
        // runs first; alternating data queues preserve progress in both directions.
        std::lock_guard<std::mutex> lock(mutex);
        (priority ? urgent : (write ? writes : queue)).push_back(std::move(fn));
        condition.notify_all();
    }
    bool emptyQueue() const {
        return queue.empty() && writes.empty() && urgent.empty();
    }
    void work() {
        for (;;) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [&] { return worker_stop || !emptyQueue(); });
                if (emptyQueue() && worker_stop)
                    return;
                auto& selected =
                    !urgent.empty() ? urgent : ((prefer_write && !writes.empty()) || queue.empty() ? writes : queue);
                fn = std::move(selected.front());
                selected.pop_front();
                prefer_write = !prefer_write;
            }
            fn();
        }
    }
    std::shared_ptr<Admission> admit(google::protobuf::RpcController* controller, bool write) {
        auto                        peer = peerOf(controller);
        std::lock_guard<std::mutex> lock(admissions->mutex);
        auto&                       counts = admissions->peers[peer];
        auto&                       count  = write ? counts.second : counts.first;
        if (count >= (write ? options.writes_per_connection : options.reads_per_connection))
            return {};
        ++count;
        auto result   = std::make_shared<Admission>();
        result->owner = admissions;
        result->peer  = std::move(peer);
        result->write = write;
        return result;
    }
    // Reserve a terminal work item before admission. A failed handoff never
    // executes the RPC reply on the timer or network callback thread.
    void failHandoff(const std::shared_ptr<Operation>& op) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        urgent.splice(urgent.end(), op->failure);
        condition.notify_all();
    }
    void track(const std::shared_ptr<Operation>& op) {
        op->failure.emplace_back([this, weak = std::weak_ptr<Operation>(op)] {
            if (auto operation = weak.lock())
                finish(operation, EC_FAILED_LOAD_BUFFER);
        });
        std::lock_guard<std::mutex> lock(mutex);
        op->id = ++next_id;
        operations.emplace(op->id, op);
    }
    bool expired(const Operation& op) const {
        return Clock::now() >= op.start + std::chrono::milliseconds(op.timeout);
    }
    void finish(const std::shared_ptr<Operation>& op, KvCacheStoreServiceErrorCode code) {
        // All terminal decisions run on the one bounded worker. Reverse callbacks
        // and timer callbacks only enqueue their single next continuation.
        if (op->write) {
            op->transfer_response->set_error_code(code);
            if (code != EC_SUCCESS) {
                try {
                    op->transfer_response->set_error_msg("remote mock transfer rejected or expired");
                } catch (const std::bad_alloc&) {}  // The wire error code is already terminal.
            }
        } else {
            op->load_response->set_error_code(code);
            if (code != EC_SUCCESS)
                op->load_response->clear_blocks();
        }
        if (code != EC_SUCCESS) {
            std::lock_guard<std::mutex> lock(stats_mutex);
            ++counters.failed_operations;
        }
        event(code == EC_SUCCESS ? "complete" : "failed", *op);
        auto done = op->done;
        op->done  = nullptr;
        try {
            done->Run();
        } catch (...) {
            // Keep the service worker and its admitted-operation accounting alive.
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            operations.erase(op->id);
            condition.notify_all();
        }
    }
    template<class Continuation>
    void delay(const std::shared_ptr<Operation>& op, Clock::time_point boundary, Continuation fn) try {
        const auto deadline = boundary + std::chrono::milliseconds(op->delay);
        const auto timeout  = op->start + std::chrono::milliseconds(op->timeout);
        event("delay", *op, std::chrono::duration_cast<std::chrono::nanoseconds>(deadline.time_since_epoch()).count());
        std::lock_guard<std::mutex> lock(lifecycle);
        if (stopping || closing || !state.ready(op->session).ok()) {
            failHandoff(op);
            return;
        }
        bool ok = timer->schedule(std::min(deadline, timeout), [this, op, fn = std::move(fn)](bool cancelled) mutable {
            try {
                enqueue(
                    [this, op, cancelled, fn = std::move(fn)] {
                        if (cancelled || expired(*op) || !state.ready(op->session).ok())
                            finish(op, EC_FAILED_LOAD_BUFFER);
                        else
                            fn();
                    },
                    op->write,
                    cancelled);
            } catch (...) {
                failHandoff(op);
            }
        });
        if (!ok)
            failHandoff(op);
    } catch (...) {
        failHandoff(op);
    }
    void fillControl(remote_mock::ControlResponse* r, const absl::Status& status) {
        r->set_error_code(errorCode(status));
        r->set_error_message(std::string(status.message()));
        r->set_server_epoch(state.serverEpoch());
        r->set_last_generation(state.lastGeneration());
        r->set_allocated_bytes(state.allocatedBytes());
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            r->set_read_payload_bytes(counters.read_payload_bytes);
            r->set_write_payload_bytes(counters.write_payload_bytes);
            r->set_reverse_rpcs(counters.reverse_rpcs);
            r->set_failed_operations(counters.failed_operations);
        }
        r->set_reads_per_connection(options.reads_per_connection);
        r->set_writes_per_connection(options.writes_per_connection);
        r->set_max_callback_endpoints(options.max_callback_endpoints);
        r->set_max_timeout_ms(options.max_timeout_ms);
        auto        out = r->mutable_limits();
        const auto& in  = options.limits;
#define COPY_LIMIT(name) out->set_##name(in.name)
        COPY_LIMIT(total_bytes);
        COPY_LIMIT(source_bytes);
        COPY_LIMIT(write_slot_bytes);
        COPY_LIMIT(read_staging_bytes);
        COPY_LIMIT(write_staging_bytes);
        COPY_LIMIT(write_slots);
        COPY_LIMIT(max_rpc_bytes);
        COPY_LIMIT(max_blocks_per_rpc);
        COPY_LIMIT(max_read_operations);
        COPY_LIMIT(max_ranks);
        COPY_LIMIT(max_groups_per_rank);
        COPY_LIMIT(max_sources_per_group);
        COPY_LIMIT(max_members_per_group);
        COPY_LIMIT(replay_window);
        COPY_LIMIT(max_delay_ms);
#undef COPY_LIMIT
        auto info = state.sessionInfo();
        if (info.ok()) {
            setSession(r->mutable_session(), info->session);
            r->set_ready(info->ready);
            r->set_read_delay_ms(info->read_delay_ms);
            r->set_write_delay_ms(info->write_delay_ms);
            r->set_expected_ranks(info->expected_ranks);
            r->set_registered_ranks(info->registered_ranks);
        }
    }
    absl::Status closeSession(const RemoteMockSession& session) {
        std::unique_ptr<RemoteMockTimer> old;
        {
            std::lock_guard<std::mutex> lock(lifecycle);
            if (closing)
                return absl::FailedPreconditionError("close in progress");
            auto status = state.close(session);
            if (!status.ok())
                return status;
            closing = true;
            old     = std::move(timer);
        }
        if (old)
            old->stop();
        {
            std::lock_guard<std::mutex> lock(lifecycle);
            if (!stopping)
                timer = std::make_unique<RemoteMockTimer>(uint64_t(options.limits.max_read_operations)
                                                          + options.limits.write_slots);
            closing = false;
        }
        return absl::OkStatus();
    }
    void pulled(const std::shared_ptr<Operation>& op);
};

void RemoteMockService::Impl::Control::health(google::protobuf::RpcController*,
                                              const remote_mock::HealthRequest* q,
                                              remote_mock::ControlResponse*     r,
                                              Done*                             d) {
    impl.fillControl(
        r, q->ByteSizeLong() <= 4096 ? absl::OkStatus() : absl::InvalidArgumentError("control envelope exceeds limit"));
    d->Run();
}
void RemoteMockService::Impl::Control::ready(google::protobuf::RpcController*,
                                             const remote_mock::SessionRequest* q,
                                             remote_mock::ControlResponse*      r,
                                             Done*                              d) {
    impl.fillControl(r,
                     q->ByteSizeLong() <= 4096 ? impl.state.ready(sessionOf(q->session())) :
                                                 absl::InvalidArgumentError("control envelope exceeds limit"));
    d->Run();
}
void RemoteMockService::Impl::Control::close(google::protobuf::RpcController*,
                                             const remote_mock::SessionRequest* q,
                                             remote_mock::ControlResponse*      r,
                                             Done*                              d) {
    impl.fillControl(r,
                     q->ByteSizeLong() <= 4096 ? impl.closeSession(sessionOf(q->session())) :
                                                 absl::InvalidArgumentError("control envelope exceeds limit"));
    d->Run();
}
void RemoteMockService::Impl::Control::initRank(google::protobuf::RpcController*,
                                                const remote_mock::InitRankRequest* q,
                                                remote_mock::ControlResponse*       r,
                                                Done*                               d) {
    absl::Status status = absl::InvalidArgumentError("invalid manifest");
    {
        std::lock_guard<std::mutex> lock(impl.lifecycle);
        if (!impl.stopping && !impl.closing && q->ByteSizeLong() <= (8U << 20) && q->has_session() && q->has_rank()
            && q->has_expected_ranks() && q->has_layout_hash() && q->groups_size() > 0
            && uint32_t(q->groups_size()) <= impl.options.limits.max_groups_per_rank) {
            RemoteMockManifest m;
            m.session        = sessionOf(q->session());
            m.rank           = q->rank();
            m.expected_ranks = q->expected_ranks();
            m.layout_hash    = q->layout_hash();
            m.read_delay_ms  = q->read_delay_ms();
            m.write_delay_ms = q->write_delay_ms();
            bool valid       = true;
            for (const auto& g : q->groups()) {
                if (!g.has_group() || !g.has_pool() || !g.has_payload_bytes() || !g.has_storage_bytes()
                    || !g.has_stride_bytes() || !g.has_dtype() || !g.has_tokens_per_block() || !g.has_cp_world_size()
                    || !g.has_cp_rank()
                    || uint32_t(g.member_offsets_size()) > impl.options.limits.max_members_per_group) {
                    valid = false;
                    break;
                }
                RemoteMockLayout l;
                l.group              = g.group();
                l.pool               = g.pool();
                l.payload_bytes      = g.payload_bytes();
                l.storage_bytes      = g.storage_bytes();
                l.stride_bytes       = g.stride_bytes();
                l.crc_enabled        = g.crc_enabled();
                l.crc_format_version = g.crc_format_version();
                l.crc_footer_offset  = g.crc_footer_offset();
                l.source_count       = g.source_count();
                l.alignment          = g.alignment();
                l.dtype              = g.dtype();
                l.member_offsets.assign(g.member_offsets().begin(), g.member_offsets().end());
                l.tokens_per_block = g.tokens_per_block();
                l.cp_world_size    = g.cp_world_size();
                l.cp_rank          = g.cp_rank();
                l.attention_window = g.attention_window();
                m.groups.push_back(std::move(l));
            }
            if (valid) {
                auto registered = impl.state.registerRank(m);
                status          = registered.status();
                if (registered.ok()) {
                    auto sources = impl.state.describeRank(*registered, m.rank);
                    if (sources.ok())
                        for (const auto& s : *sources) {
                            auto out = r->add_sources();
                            out->set_rank(s.rank);
                            out->set_group(s.group);
                            out->set_pool(s.pool);
                            out->set_source_count(s.source_count);
                            out->set_layout_hash(s.layout_hash);
                            out->set_record_crc32c(s.record_checksum);
                        }
                }
            }
        }
        // Semantic rejection must roll back this same incomplete collective,
        // just like typed-core validation. Foreign/stale requests and a READY
        // session cannot be torn down by a malformed registration.
        auto current = impl.state.sessionInfo();
        if (!status.ok() && current.ok() && !current->ready && current->session == sessionOf(q->session()))
            impl.state.close(current->session);
        impl.fillControl(r, status);
    }
    d->Run();
}

RemoteMockService::RemoteMockService(RemoteMockServiceOptions options):
    impl_(std::make_unique<Impl>(std::move(options))) {}
RemoteMockService::~RemoteMockService() {
    shutdown();
}
bool RemoteMockService::init() {
    auto&                       p = *impl_;
    std::lock_guard<std::mutex> lock(p.lifecycle);
    if (p.initialized)
        return !p.stopping;
    if (!p.options.reads_per_connection || !p.options.writes_per_connection || !p.options.max_callback_endpoints
        || !p.options.max_timeout_ms || p.options.max_timeout_ms > 60000 || !p.options.limits.max_read_operations
        || p.options.limits.max_read_operations > 65536 || p.options.limits.write_slots > 65536
        || p.envelope() > INT32_MAX)
        return false;
    if (!p.options.event_path.empty()) {
        p.events.open(p.options.event_path, std::ios::out | std::ios::trunc);
        if (!p.events)
            return false;
    }
    auto minimum = remoteMockStagingCharge(
        1, 1, p.options.limits, kMetadataPerBlock + kMetadataPerOperation + kMinimumRequestMetadata);
    if (!minimum.ok() || p.options.limits.read_staging_bytes < *minimum
        || p.options.limits.write_staging_bytes < *minimum)
        return false;
    if (!p.client.init(1))
        return false;
    p.timer       = std::make_unique<RemoteMockTimer>(uint64_t(p.options.limits.max_read_operations)
                                                + p.options.limits.write_slots);
    p.worker      = std::thread([&p] { p.work(); });
    p.initialized = true;
    return true;
}
void RemoteMockService::shutdown() {
    auto&                            p = *impl_;
    std::unique_ptr<RemoteMockTimer> timer;
    {
        std::lock_guard<std::mutex> lock(p.lifecycle);
        if (!p.initialized || p.stopping)
            return;
        p.stopping = true;
        auto info  = p.state.sessionInfo();
        if (info.ok())
            p.state.close(info->session);
        timer = std::move(p.timer);
    }
    if (timer)
        timer->stop();
    {
        std::unique_lock<std::mutex> lock(p.mutex);
        p.condition.wait(lock, [&] { return p.operations.empty() && p.emptyQueue() && p.sync_producers == 0; });
        p.worker_stop = true;
        p.condition.notify_all();
    }
    if (p.worker.joinable())
        p.worker.join();
    p.finishCapture();
}
remote_mock::RemoteMockControlService* RemoteMockService::controlService() {
    return &impl_->control;
}
RemoteMockServiceStats RemoteMockService::stats() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    return impl_->counters;
}

void RemoteMockService::load(google::protobuf::RpcController* controller,
                             const CacheLoadRequest*          q,
                             CacheLoadResponse*               r,
                             Done*                            d) {
    DeferredDone reply(d);
    d        = &reply;
    auto& p  = *impl_;
    auto  op = std::make_shared<Impl::Operation>();
    {
        std::lock_guard<std::mutex> lock(p.lifecycle);
        if (p.stopping || p.closing || !p.initialized || !p.shape(*q) || !p.timeoutValid(q->timeout_ms())
            || q->blocks_size() <= 0 || uint32_t(q->blocks_size()) > p.options.limits.max_blocks_per_rpc
            || q->requestid().size() > 1024 || q->rdma_connections_size() != 0 || unknownFields(*q)) {
            fail(r, d);
            return;
        }
        auto info = p.state.sessionInfo();
        if (!info.ok() || !info->ready) {
            fail(r, d);
            return;
        }
        op->session   = info->session;
        op->timeout   = q->timeout_ms();
        op->delay     = info->read_delay_ms;
        op->admission = p.admit(controller, false);
        if (!op->admission) {
            fail(r, d, EC_FAILED_LOAD_BUFFER);
            return;
        }
        for (const auto& b : q->blocks()) {
            if (!b.has_key() || b.key().size() > kRemoteMockMaxKeyBytes || !b.has_len() || !b.len() || b.has_content()
                || b.has_rdma_info()) {
                fail(r, d);
                return;
            }
            op->bytes += b.len();
        }
        auto lease =
            p.state.reserveRead(op->session, op->bytes, q->blocks_size(), additionalMetadata(*q, q->blocks_size()));
        if (!lease.ok()) {
            fail(r, d, errorCode(lease.status()));
            return;
        }
        op->admission->read = *lease;
        op->admission->sources.reserve(q->blocks_size());
        std::set<std::string_view> keys;
        for (const auto& b : q->blocks()) {
            if (!keys.insert(b.key()).second) {
                fail(r, d);
                return;
            }
            auto source = p.state.resolve(b.key(), b.len());
            if (!source.ok()) {
                fail(r, d);
                return;
            }
            op->admission->sources.push_back(*source);
        }
        op->load_request  = q;
        op->load_response = r;
        try {
            pin(r, op->admission);
            p.track(op);
        } catch (const std::bad_alloc&) {
            fail(r, d, EC_FAILED_LOAD_BUFFER);
            return;
        }
        op->done = reply.release();
        {
            std::lock_guard<std::mutex> stats(p.stats_mutex);
            p.counters.read_admitted_ns = nowNs();
        }
    }
    p.event("read_admitted", *op);
    p.delay(op, Clock::now(), [&p, op] {
        p.event("read_copy_start", *op);
        auto r = op->load_response;
        for (int i = 0; i < op->load_request->blocks_size(); ++i) {
            const auto& b   = op->load_request->blocks(i);
            auto        out = r->add_blocks();
            out->set_key(b.key());
            out->set_len(b.len());
            out->set_content(op->admission->sources[i]->data(), b.len());
        }
        r->set_direct_write_response(false);
        if (r->ByteSizeLong() > operationEnvelope(op->bytes, op->load_request->blocks_size()) || p.expired(*op)
            || !p.state.ready(op->session).ok()) {
            p.finish(op, EC_FAILED_LOAD_BUFFER);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(p.stats_mutex);
            p.counters.read_payload_bytes += op->bytes;
            p.counters.read_payload_ns = nowNs();
        }
        p.event("read_payload", *op);
        p.finish(op, EC_SUCCESS);
    });
}

void RemoteMockService::transfer(google::protobuf::RpcController* controller,
                                 const CacheTransferRequest*      q,
                                 CacheTransferResponse*           r,
                                 Done*                            d) {
    DeferredDone reply(d);
    d        = &reply;
    auto& p  = *impl_;
    auto  op = std::make_shared<Impl::Operation>();
    {
        std::lock_guard<std::mutex> lock(p.lifecycle);
        const auto&                 t = q->transfer_info();
        in_addr                     address{};
        if (p.stopping || p.closing || !p.initialized || !p.shape(*q) || !p.timeoutValid(q->timeout_ms())
            || !q->has_transfer_info() || q->blocks_size() <= 0
            || uint32_t(q->blocks_size()) > p.options.limits.max_blocks_per_rpc
            || t.local_keys_size() != q->blocks_size() || t.remote_keys_size() != q->blocks_size()
            || t.partition_count() != 1 || t.partition_id() != 0 || q->request_id().size() > 1024 || !q->client_port()
            || q->client_port() > 65535 || unknownFields(*q)
            || inet_pton(AF_INET, q->client_ip().c_str(), &address) != 1) {
            fail(r, d);
            return;
        }
        auto info = p.state.sessionInfo();
        if (!info.ok() || !info->ready) {
            fail(r, d);
            return;
        }
        op->session   = info->session;
        op->timeout   = q->timeout_ms();
        op->delay     = info->write_delay_ms;
        op->write     = true;
        op->admission = p.admit(controller, true);
        if (!op->admission) {
            fail(r, d, EC_FAILED_LOAD_BUFFER);
            return;
        }
        // Validate borrowed protobuf fields without retaining N-key scratch.
        // Pairwise scans are bounded by max_blocks_per_rpc (default 256). This
        // preserves malformed/replay semantics without allocating a mapping or
        // consuming staging credit before reserveWrite checks remaining quota.
        for (int i = 0; i < q->blocks_size(); ++i) {
            const auto& b = q->blocks(i);
            if (!b.has_key() || b.key().empty() || b.key().size() > kRemoteMockMaxKeyBytes || !b.has_len() || !b.len()
                || !b.has_addr() || b.addr() < 0 || b.has_content() || b.has_rdma_info()) {
                fail(r, d);
                return;
            }
            for (int j = 0; j < i; ++j)
                if (q->blocks(j).key() == b.key()) {
                    fail(r, d);
                    return;
                }
            op->bytes += b.len();
        }
        auto findBlock = [q](const std::string& key) -> const BlockBufferInfo* {
            for (const auto& block : q->blocks())
                if (block.key() == key)
                    return &block;
            return nullptr;
        };
        RemoteMockWriteOperation operation;
        for (int i = 0; i < t.local_keys_size(); ++i) {
            const auto* block = findBlock(t.local_keys(i));
            auto        key   = decodeRemoteMockKey(t.remote_keys(i));
            if (!block || !key.ok() || !key->write) {
                fail(r, d);
                return;
            }
            for (int j = 0; j < i; ++j)
                if (t.local_keys(j) == t.local_keys(i) || t.remote_keys(j) == t.remote_keys(i)) {
                    fail(r, d);
                    return;
                }
            if (i == 0)
                operation = {key->session, key->layout_hash, key->rank, key->index, key->operation};
            if (!p.state.validateWriteKey(operation, t.remote_keys(i), block->len()).ok()) {
                fail(r, d);
                return;
            }
        }
        auto lease =
            p.state.reserveWrite(operation, op->bytes, q->blocks_size(), additionalMetadata(*q, q->blocks_size()));
        if (!lease.ok()) {
            fail(r, d, errorCode(lease.status()));
            return;
        }
        op->admission->slot = *lease;
        op->keys.reserve(q->blocks_size());
        op->reverse_request.mutable_blocks()->Reserve(q->blocks_size());
        for (int i = 0; i < t.local_keys_size(); ++i) {
            op->keys.push_back(*decodeRemoteMockKey(t.remote_keys(i)));
            op->reverse_request.add_blocks()->CopyFrom(*findBlock(t.local_keys(i)));
        }
        // Bound TcpClient's persistent endpoint map before asking it for a channel.
        auto endpoint = q->client_ip() + ":" + std::to_string(q->client_port());
        if (!p.endpoints.count(endpoint) && p.endpoints.size() >= p.options.max_callback_endpoints) {
            fail(r, d, EC_FAILED_LOAD_BUFFER);
            return;
        }
        p.endpoints.insert(endpoint);
        op->transfer_request  = q;
        op->transfer_response = r;
        try {
            pin(r, op->admission);
            p.track(op);
        } catch (const std::bad_alloc&) {
            fail(r, d, EC_FAILED_LOAD_BUFFER);
            return;
        }
        op->done = reply.release();
    }
    p.event("write_admitted", *op);
    try {
        p.enqueue(
            [&p, op] {
                if (p.expired(*op) || !p.state.ready(op->session).ok()) {
                    p.finish(op, EC_FAILED_LOAD_BUFFER);
                    return;
                }
                op->channel =
                    p.client.getChannel(op->transfer_request->client_ip(), op->transfer_request->client_port());
                if (!op->channel) {
                    p.finish(op, EC_FAILED_INTERNAL);
                    return;
                }
                auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     op->start + std::chrono::milliseconds(op->timeout) - Clock::now())
                                     .count();
                if (remaining <= 0) {
                    p.finish(op, EC_FAILED_LOAD_BUFFER);
                    return;
                }
                if (op->reverse_request.ByteSizeLong()
                    > operationEnvelope(op->bytes, op->reverse_request.blocks_size())) {
                    p.finish(op, EC_FAILED_INVALID_REQ);
                    return;
                }
                op->controller.SetExpireTime(remaining);
                {
                    std::lock_guard<std::mutex> lock(p.stats_mutex);
                    ++p.counters.reverse_rpcs;
                }
                p.event("reverse_pull", *op);
                KvCacheStoreService_Stub stub(op->channel.get());
                stub.blockRead(&op->controller, &op->reverse_request, &op->reverse_response, new FunctionDone([&p, op] {
                    try {
                        p.enqueue([&p, op] { p.pulled(op); }, true);
                    } catch (...) {
                        p.failHandoff(op);
                    }
                }));
            },
            true);
    } catch (...) {
        p.failHandoff(op);
    }
}

void RemoteMockService::Impl::pulled(const std::shared_ptr<Operation>& op) {
    auto&    response = op->reverse_response;
    uint64_t received = 0;
    for (const auto& block : response.blocks())
        received += block.content().size();
    {
        std::lock_guard<std::mutex> lock(stats_mutex);
        counters.write_payload_bytes += received;
    }
    event("write_received", *op, 0, received);
    if (op->controller.Failed() || !response.has_error_code() || response.error_code() != EC_SUCCESS
        || response.ByteSizeLong() > operationEnvelope(op->bytes, op->reverse_request.blocks_size())
        || unknownFields(response) || response.blocks_size() != op->reverse_request.blocks_size() || expired(*op)
        || !state.ready(op->session).ok()) {
        finish(op, EC_FAILED_LOAD_BUFFER);
        return;
    }
    event("write_validate_start", *op);
    std::map<std::string_view, const BlockBufferInfo*> returned;
    for (const auto& block : response.blocks())
        if (!block.has_key() || !block.has_len() || !block.has_content() || block.has_rdma_info()
            || block.len() != block.content().size() || !returned.emplace(block.key(), &block).second) {
            finish(op, EC_FAILED_INVALID_REQ);
            return;
        }
    // Validate the complete key set and all lengths, then all CRCs, before any
    // slot mutation. A valid first block never masks a corrupt later block.
    for (int i = 0; i < op->reverse_request.blocks_size(); ++i) {
        const auto& expected = op->reverse_request.blocks(i);
        auto        it       = returned.find(expected.key());
        if (it == returned.end() || it->second->len() != expected.len()) {
            finish(op, EC_FAILED_INVALID_REQ);
            return;
        }
    }
    for (int i = 0; i < op->reverse_request.blocks_size(); ++i) {
        const auto& block = *returned.at(op->reverse_request.blocks(i).key());
        if (remoteMockCrc32c(reinterpret_cast<const uint8_t*>(block.content().data()), block.len())
            != op->keys[i].checksum) {
            finish(op, EC_FAILED_INVALID_REQ);
            return;
        }
    }
    event("write_copy_start", *op);
    size_t offset = 0;
    for (const auto& expected : op->reverse_request.blocks()) {
        const auto& block = *returned.at(expected.key());
        std::memcpy(op->admission->slot->data() + offset, block.content().data(), block.len());
        offset += block.len();
    }
    offset = 0;
    for (size_t i = 0; i < op->keys.size(); ++i) {
        const auto length = op->reverse_request.blocks(i).len();
        if (remoteMockCrc32c(op->admission->slot->data() + offset, length) != op->keys[i].checksum) {
            finish(op, EC_FAILED_INTERNAL);
            return;
        }
        offset += length;
    }
    const auto stored = Clock::now();
    {
        std::lock_guard<std::mutex> lock(stats_mutex);
        counters.write_stored_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(stored.time_since_epoch()).count();
    }
    event("write_stored_crc", *op);
    response.Clear();
    delay(op, stored, [this, op] {
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            counters.write_ack_ns = nowNs();
        }
        event("write_ack", *op);
        finish(op, EC_SUCCESS);
    });
}

void RemoteMockService::blockRead(google::protobuf::RpcController* controller,
                                  const BlockReadRequest*          q,
                                  BlockReadResponse*               r,
                                  Done*                            d) {
    // Source reads through this existing method are immediate, never write-delay
    // or read-delay injected. Addresses are not dereferenced on the mock server.
    auto& p = *impl_;
    // Destroy after DeferredDone: shutdown includes copying, byte accounting
    // and the actual reply callback. Arena-owned payload leases may live longer.
    struct Producer {
        Impl& service;
        bool  admitted = false;
        ~Producer() {
            if (admitted) {
                std::lock_guard<std::mutex> lock(service.mutex);
                --service.sync_producers;
                service.condition.notify_all();
            }
        }
    } producer{p};
    DeferredDone reply(d);
    d = &reply;
    std::shared_ptr<Admission> admission;
    {
        std::lock_guard<std::mutex> lock(p.lifecycle);
        if (p.stopping || p.closing || !p.shape(*q) || q->blocks_size() <= 0
            || uint32_t(q->blocks_size()) > p.options.limits.max_blocks_per_rpc || unknownFields(*q)) {
            fail(r, d);
            return;
        }
        auto info = p.state.sessionInfo();
        if (!info.ok() || !info->ready) {
            fail(r, d);
            return;
        }
        admission = p.admit(controller, false);
        if (!admission) {
            fail(r, d, EC_FAILED_LOAD_BUFFER);
            return;
        }
        uint64_t bytes = 0;
        for (const auto& block : q->blocks()) {
            if (!block.has_key() || block.key().size() > kRemoteMockMaxKeyBytes || !block.has_len() || !block.len()
                || block.has_content() || block.has_rdma_info()) {
                fail(r, d);
                return;
            }
            bytes += block.len();
        }
        auto lease =
            p.state.reserveRead(info->session, bytes, q->blocks_size(), additionalMetadata(*q, q->blocks_size()));
        if (!lease.ok()) {
            fail(r, d, errorCode(lease.status()));
            return;
        }
        admission->read = *lease;
        admission->sources.reserve(q->blocks_size());
        std::set<std::string_view> keys;
        for (const auto& block : q->blocks()) {
            if (!keys.insert(block.key()).second) {
                fail(r, d);
                return;
            }
            auto source = p.state.resolve(block.key(), block.len());
            if (!source.ok()) {
                fail(r, d);
                return;
            }
            admission->sources.push_back(*source);
        }
        pin(r, admission);
        std::lock_guard<std::mutex> work(p.mutex);
        ++p.sync_producers;
        producer.admitted = true;
    }
    for (int i = 0; i < q->blocks_size(); ++i) {
        auto out = r->add_blocks();
        out->set_key(q->blocks(i).key());
        out->set_len(q->blocks(i).len());
        out->set_content(admission->sources[i]->data(), q->blocks(i).len());
    }
    uint64_t bytes = 0;
    for (const auto& block : r->blocks())
        bytes += block.len();
    if (r->ByteSizeLong() > operationEnvelope(bytes, q->blocks_size())) {
        r->clear_blocks();
        fail(r, d);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(p.stats_mutex);
        for (const auto& block : r->blocks())
            p.counters.read_payload_bytes += block.content().size();
    }
    r->set_error_code(EC_SUCCESS);
    d->Run();
}
}  // namespace rtp_llm

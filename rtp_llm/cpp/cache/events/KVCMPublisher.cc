#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/events/CurlKVCacheEventReporter.h"
#include "rtp_llm/cpp/cache/events/KVCMLogicalMirror.h"
#include "rtp_llm/cpp/cache/events/KVCMRequestBuilder.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventAdmissionGate.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventQueue.h"
#include "rtp_llm/cpp/cache/events/KVCMPublisherUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

constexpr int kShutdownRequestTimeoutMs = 500;

using detail::buildControlReport;
using detail::buildMutationReport;
using detail::buildRegisterInstanceRequest;
using detail::buildSnapshotReport;
using detail::coalesceMutations;
using detail::ControlEventType;
using detail::JsonPayloadLimitExceeded;
using detail::SnapshotBuildCancelled;

}  // namespace

class KVCMPublisher::Impl {
    enum class CircuitReason {
        NONE,
        INVALID_CONFIGURATION,
        INITIAL_SNAPSHOT_FAILURE,
        RECOVERY_SNAPSHOT_FAILURE,
        MIRROR_KEY_LIMIT,
        REPORT_PAYLOAD_LIMIT,
        SNAPSHOT_PAYLOAD_LIMIT,
        PERMANENT_PROTOCOL_ERROR,
        WORKER_EXCEPTION,
    };

    struct PendingSnapshotReport {
        uint64_t             generation = 0;
        std::vector<int64_t> block_keys;
        std::string          request;
    };

    struct SnapshotBuildResult {
        bool                       payload_limit_exceeded = false;
        bool                       cancelled              = false;
        std::optional<std::string> request;
    };

    struct PostResult {
        bool                     transport_ok = false;
        detail::KVCMResponseInfo response;

        bool ok() const noexcept {
            return transport_ok && response.ok();
        }

        bool requiresRegistration() const noexcept {
            return transport_ok && response.parsed && response.requiresRegistration();
        }

        uint64_t serverRetryAfterMs() const noexcept {
            return transport_ok && response.parsed ? response.retry_after_ms : 0;
        }
    };

    enum class HeartbeatAction {
        NO_ACTION,
        SENT,
        TRANSIENT_FAILURE,
        REREGISTER,
        CIRCUIT_OPEN,
    };

    struct HeartbeatOutcome {
        HeartbeatAction action             = HeartbeatAction::NO_ACTION;
        uint64_t        retry_after_ms     = 0;
        bool            snapshot_requested = false;
    };

    struct HeartbeatSchedule {
        std::chrono::steady_clock::time_point next;
        std::chrono::milliseconds             retry_interval{1};
        bool                                  degraded = false;
    };

    struct SnapshotUploadResult {
        PostResult       result;
        HeartbeatOutcome heartbeat;
    };

    struct RegistrationOutcome {
        bool     registered     = false;
        uint64_t retry_after_ms = 0;
    };

    enum class ReconcileAction {
        COMMITTED,
        RETRY,
        REREGISTER,
        CIRCUIT_OPEN,
    };

    struct ReconcileOutcome {
        ReconcileAction action         = ReconcileAction::RETRY;
        uint64_t        retry_after_ms = 0;
    };

public:
    Impl(KVCacheEventPublisherConfig           config,
         KVCacheEventPublisherContext          context,
         KVCacheSnapshotProvider               snapshot_provider,
         std::shared_ptr<KVCacheEventReporter> reporter):
        config_(std::move(config)),
        context_(std::move(context)),
        snapshot_provider_(std::move(snapshot_provider)),
        reporter_(std::move(reporter)),
        queue_(config_.queue_capacity),
        logical_mirror_(config_.snapshot_max_keys) {
        config_.manager_endpoint = detail::normalizeKVCacheEventEndpoint(std::move(config_.manager_endpoint));
        reporter_injected_       = static_cast<bool>(reporter_);
        if (reporter_) {
            snapshot_reporter_ = reporter_;
            shutdown_reporter_ = reporter_;
        } else if (!config_.manager_endpoint.empty()) {
            reporter_ = detail::makeCurlKVCacheEventReporter(config_.manager_endpoint, config_.request_timeout_ms);
            snapshot_reporter_ =
                detail::makeCurlKVCacheEventReporter(config_.manager_endpoint, config_.snapshot_timeout_ms);
            // Shutdown must stay bounded even when the regular timeout was
            // deliberately configured for a slow control plane. Keep a
            // separate reporter so cancelling active traffic does not also
            // suppress the terminal best-effort HOST_DOWN.
            shutdown_reporter_ = detail::makeCurlKVCacheEventReporter(
                config_.manager_endpoint, std::min(config_.request_timeout_ms, kShutdownRequestTimeoutMs));
        }
    }

    ~Impl() {
        stop();
    }

    bool start() noexcept {
        std::lock_guard<std::mutex> lock(lifecycle_mu_);
        if (circuit_open_.load(std::memory_order_acquire)) {
            RTP_LLM_LOG_WARNING("KVCMPublisher cannot restart after its circuit opened");
            return false;
        }
        if (started_.load(std::memory_order_acquire)) {
            return true;
        }
        if (stopped_permanently_) {
            RTP_LLM_LOG_WARNING("KVCMPublisher cannot restart after stop");
            return false;
        }
        if (const char* invalid_field = invalidConfigField()) {
            tripCircuit(CircuitReason::INVALID_CONFIGURATION);
            stopped_permanently_ = true;
            RTP_LLM_LOG_ERROR("KVCMPublisher is disabled by invalid config field=%s instance_id=%s dp_rank=%d",
                              invalid_field,
                              context_.instance_id.c_str(),
                              context_.dp_rank);
            return false;
        }
        stopping_.store(false, std::memory_order_release);
        state_.store(PublisherState::STARTING, std::memory_order_relaxed);
        // The publisher is already installed on SharedBlockCache at this
        // point. Accept mutations before taking the initial snapshot: changes
        // before the cache lock is captured are present in both places and
        // therefore idempotent, while later changes remain queued as deltas.
        accepting_.store(true, std::memory_order_release);

        KVCacheSnapshot initial_snapshot;
        try {
            initial_snapshot = snapshot_provider_();
            if (!logical_mirror_.seed(initial_snapshot)) {
                tripCircuit(CircuitReason::MIRROR_KEY_LIMIT);
                quiesceFailedStart();
                RTP_LLM_LOG_WARNING("KVCMPublisher initial snapshot exceeds key limit, keys=%zu limit=%zu",
                                    initial_snapshot.block_keys.size(),
                                    config_.snapshot_max_keys);
                return false;
            }
        } catch (const std::exception& e) {
            tripCircuit(CircuitReason::INITIAL_SNAPSHOT_FAILURE);
            quiesceFailedStart();
            RTP_LLM_LOG_WARNING("KVCMPublisher initial snapshot failed: %s", e.what());
            return false;
        } catch (...) {
            tripCircuit(CircuitReason::INITIAL_SNAPSHOT_FAILURE);
            quiesceFailedStart();
            RTP_LLM_LOG_WARNING("KVCMPublisher initial snapshot failed with unknown exception");
            return false;
        }

        if (circuit_open_.load(std::memory_order_acquire)) {
            quiesceFailedStart();
            RTP_LLM_LOG_WARNING("KVCMPublisher circuit opened while taking the initial snapshot, reason=%s",
                                circuitReasonName(circuit_reason_.load(std::memory_order_relaxed)));
            return false;
        }

        try {
            worker_ = std::thread(&Impl::workerLoop, this);
        } catch (const std::exception& e) {
            tripCircuit(CircuitReason::WORKER_EXCEPTION);
            quiesceFailedStart();
            RTP_LLM_LOG_WARNING("start KVCMPublisher failed: %s", e.what());
            return false;
        }
        started_.store(true, std::memory_order_release);
        return true;
    }

    PublishResult tryPublish(KVCacheEvent event) noexcept {
        auto publication = publication_gate_.tryEnter();
        if (!publication || !accepting_.load(std::memory_order_acquire)) {
            return PublishResult::NOT_RUNNING;
        }
        auto recovery_admission = recovery_gate_.tryEnter();
        if (!recovery_admission || !accepting_.load(std::memory_order_acquire)) {
            return PublishResult::NOT_RUNNING;
        }
        const auto result = queue_.tryPush(std::move(event));
        if (result == detail::QueuePushResult::ACCEPTED) {
            accepted_count_.fetch_add(1, std::memory_order_relaxed);
        } else if (result == detail::QueuePushResult::FULL) {
            // A producer can enter immediately before stop() closes the
            // admission gate. Do not turn that shutdown race into a spurious
            // overflow circuit or dropped-event metric.
            if (!accepting_.load(std::memory_order_acquire)) {
                return PublishResult::NOT_RUNNING;
            }
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            // Close the reusable recovery epoch before publishing the request.
            // The worker can quiesce every producer that entered this epoch,
            // discard the now-unordered backlog, and reopen a fresh epoch.
            // Recovery, cache snapshotting, and transport all remain on the
            // worker; the producer path performs atomics only.
            recovery_gate_.close();
            recovery_requested_.store(true, std::memory_order_release);
            state_.store(PublisherState::RESYNCING, std::memory_order_relaxed);
            return PublishResult::DROPPED_RECOVERABLE;
        }
        return detail::toPublishResult(result);
    }

    void stop() noexcept {
        std::lock_guard<std::mutex> lock(lifecycle_mu_);
        if (!started_.load(std::memory_order_relaxed) && !worker_.joinable()) {
            accepting_.store(false, std::memory_order_release);
            available_.store(false, std::memory_order_release);
            stopping_.store(true, std::memory_order_release);
            stopped_permanently_ = true;
            publication_gate_.close();
            recovery_gate_.close();
            queue_.stop();
            publication_gate_.quiesce();
            recovery_gate_.quiesce();
            queue_.quiescePushes();
            state_.store(PublisherState::STOPPED, std::memory_order_relaxed);
            return;
        }
        accepting_.store(false, std::memory_order_release);
        available_.store(false, std::memory_order_release);
        stopping_.store(true, std::memory_order_release);
        snapshot_build_cancelled_.store(true, std::memory_order_release);
        publication_gate_.close();
        recovery_gate_.close();
        if (reporter_) {
            reporter_->cancel();
        }
        if (snapshot_reporter_ && snapshot_reporter_ != reporter_) {
            snapshot_reporter_->cancel();
        }
        RTP_LLM_LOG_INFO("KVCMPublisher stopping; cancelling in-flight requests and waiting for the worker");
        queue_.stop();
        publication_gate_.quiesce();
        recovery_gate_.quiesce();
        queue_.quiescePushes();
        if (worker_.joinable()) {
            worker_.join();
        }
        started_.store(false, std::memory_order_release);
        stopped_permanently_ = true;
        state_.store(PublisherState::STOPPED, std::memory_order_relaxed);
    }

    PublisherStatus status() const noexcept {
        // Terminal flags are authoritative. A producer can open the circuit
        // between a worker's final flag check and its next diagnostic state
        // store; never let that narrow race hide a permanent export failure
        // from metrics. An explicit stop takes precedence once teardown has
        // completed.
        auto       state        = state_.load(std::memory_order_relaxed);
        const bool circuit_open = circuit_open_.load(std::memory_order_acquire);
        if (circuit_open && state != PublisherState::STOPPED) {
            state = PublisherState::CIRCUIT_OPEN;
        } else if (recoveryRequested() && state != PublisherState::STOPPED) {
            // A control request already in flight may briefly store DEGRADED
            // after a producer requested overflow recovery. The admission
            // epoch is the authoritative lifecycle fact for diagnostics.
            state = PublisherState::RESYNCING;
        }
        PublisherStatus status;
        status.state = state;
        // A circuit makes pending cells permanently unactionable; report zero
        // immediately even before the worker finishes physical cleanup.
        status.queue_size              = circuit_open ? 0 : queue_.size();
        status.accepted_count          = accepted_count_.load(std::memory_order_relaxed);
        status.dropped_count           = dropped_count_.load(std::memory_order_relaxed);
        status.queue_high_watermark    = queue_.highWatermark();
        status.request_failure_count   = request_failure_count_.load(std::memory_order_relaxed);
        status.overflow_recovery_count = overflow_recovery_count_.load(std::memory_order_relaxed);
        status.snapshot_attempt_count  = snapshot_attempt_count_.load(std::memory_order_relaxed);
        status.snapshot_commit_count   = snapshot_commit_count_.load(std::memory_order_relaxed);
        return status;
    }

    bool enabled() const noexcept {
        return available_.load(std::memory_order_acquire);
    }

private:
    void quiesceFailedStart() noexcept {
        // No worker exists to own terminal cleanup on these paths. The
        // circuit has already closed both admission gates, so wait for calls
        // admitted before that close and then release every published cell.
        // This keeps a failed start from retaining or reporting an orphaned
        // backlog until process teardown.
        publication_gate_.quiesce();
        queue_.discardPending();
        stopped_permanently_ = true;
    }

    bool recoveryRequested() const noexcept {
        return recovery_requested_.load(std::memory_order_acquire);
    }

    bool recoverFromQueueOverflow() noexcept {
        while (recoveryRequested()) {
            // The producer closed this reusable epoch before publishing the
            // request. Close defensively as well: several producers may have
            // entered the same epoch before the first one observed FULL, and
            // their later stores must not be mistaken for a new open epoch.
            // Quiescing closes the final ring-cell reservation race without
            // closing the separate one-way lifetime gate.
            recovery_gate_.close();
            recovery_gate_.quiesce();
            if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)) {
                return false;
            }
            // Every producer admitted to this closed epoch has now returned,
            // so none can overwrite this clear. A request published after the
            // gate is reopened necessarily belongs to the next epoch and also
            // closes that epoch before setting the flag.
            recovery_requested_.store(false, std::memory_order_release);
            queue_.discardAvailable();
            pending_snapshot_report_.reset();

            // Reopen admission before acquiring SharedBlockCache's snapshot
            // mutex. Mutations before that lock are represented by both the
            // snapshot and an idempotent queued delta; mutations after it are
            // represented by the delta. Mutations made while admission was
            // paused are necessarily visible in the later snapshot.
            if (!recovery_gate_.reopenAfterQuiesce()) {
                tripCircuit(CircuitReason::WORKER_EXCEPTION);
                RTP_LLM_LOG_WARNING("KVCMPublisher could not reopen its recovery admission gate");
                return false;
            }
            if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)) {
                recovery_gate_.close();
                return false;
            }
            KVCacheSnapshot authoritative_snapshot;
            try {
                authoritative_snapshot = snapshot_provider_();
                if (!logical_mirror_.seed(authoritative_snapshot)) {
                    tripCircuit(CircuitReason::MIRROR_KEY_LIMIT);
                    RTP_LLM_LOG_WARNING("KVCMPublisher overflow recovery snapshot exceeds key limit, keys=%zu "
                                        "limit=%zu",
                                        authoritative_snapshot.block_keys.size(),
                                        config_.snapshot_max_keys);
                    return false;
                }
            } catch (const std::exception& e) {
                tripCircuit(CircuitReason::RECOVERY_SNAPSHOT_FAILURE);
                RTP_LLM_LOG_WARNING("KVCMPublisher overflow recovery snapshot failed: %s", e.what());
                return false;
            } catch (...) {
                tripCircuit(CircuitReason::RECOVERY_SNAPSHOT_FAILURE);
                RTP_LLM_LOG_WARNING("KVCMPublisher overflow recovery snapshot failed with unknown exception");
                return false;
            }

            markDirty();
            overflow_recovery_count_.fetch_add(1, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING(
                "KVCMPublisher recovered queue admission from authoritative cache state, "
                "keys=%zu recoveries=%llu",
                authoritative_snapshot.block_keys.size(),
                static_cast<unsigned long long>(overflow_recovery_count_.load(std::memory_order_relaxed)));
            // Another overflow can occur while the snapshot provider owns the
            // cache lock or immediately after it returns. Its producer has
            // already paused admission again; loop to establish a newer
            // authoritative handoff before sending anything remotely.
        }
        return !stopping_.load(std::memory_order_acquire) && !circuit_open_.load(std::memory_order_acquire);
    }

    const char* invalidConfigField() const noexcept {
        if (!reporter_) {
            return "reporter";
        }
        if (!snapshot_reporter_) {
            return "snapshot_reporter";
        }
        if (!shutdown_reporter_) {
            return "shutdown_reporter";
        }
        if (!snapshot_provider_) {
            return "snapshot_provider";
        }
        if (!reporter_injected_ && !detail::isValidKVCacheEventEndpoint(config_.manager_endpoint)) {
            return "manager_endpoint";
        }
        if (!detail::isValidKVCacheEventIdentity(context_.instance_group)) {
            return "instance_group";
        }
        if (!detail::isValidKVCacheEventIdentity(context_.instance_id)) {
            return "instance_id";
        }
        if (!detail::isValidKVCacheEventHostIpPort(context_.host_ip_port)) {
            return "host_ip_port";
        }
        if (context_.model_name.empty()) {
            return "model_name";
        }
        if (context_.dtype.empty()) {
            return "dtype";
        }
        if (context_.spec_name.empty()) {
            return "spec_name";
        }
        if (context_.location_uri.empty()) {
            return "location_uri";
        }
        if (context_.block_size_tokens <= 0) {
            return "block_size_tokens";
        }
        if (context_.spec_size_bytes <= 0) {
            return "spec_size_bytes";
        }
        if (context_.tp_size <= 0) {
            return "tp_size";
        }
        if (context_.dp_size <= 0) {
            return "dp_size";
        }
        if (context_.pp_size != 1) {
            return "pp_size";
        }
        if (context_.dp_rank < 0 || context_.dp_rank >= context_.dp_size) {
            return "dp_rank";
        }
        if (config_.queue_capacity == 0 || config_.queue_capacity > kKVCacheEventMaxQueueCapacity) {
            return "queue_capacity";
        }
        if (config_.report_batch_size == 0 || config_.report_batch_size > kKVCacheEventMaxReportBatchSize) {
            return "report_batch_size";
        }
        if (config_.flush_interval_ms <= 0) {
            return "flush_interval_ms";
        }
        if (config_.heartbeat_interval_ms <= 0) {
            return "heartbeat_interval_ms";
        }
        if (config_.request_timeout_ms <= 0) {
            return "request_timeout_ms";
        }
        if (config_.snapshot_timeout_ms <= 0) {
            return "snapshot_timeout_ms";
        }
        if (config_.retry_interval_ms <= 0) {
            return "retry_interval_ms";
        }
        if (config_.snapshot_interval_ms <= 0) {
            return "snapshot_interval_ms";
        }
        if (config_.snapshot_max_keys == 0 || config_.snapshot_max_keys > kKVCacheEventMaxSnapshotKeys) {
            return "snapshot_max_keys";
        }
        if (config_.snapshot_max_bytes == 0 || config_.snapshot_max_bytes > kKVCacheEventMaxSnapshotBytes) {
            return "snapshot_max_bytes";
        }
        if (config_.report_max_bytes == 0 || config_.report_max_bytes > kKVCacheEventMaxReportBytes) {
            return "report_max_bytes";
        }
        return nullptr;
    }

    std::string nextTraceId(const char* operation) {
        return "rtp-kv-event-" + std::to_string(context_.dp_rank) + '-' + operation + '-'
               + std::to_string(next_request_id_++);
    }

    PostResult postWith(const std::shared_ptr<KVCacheEventReporter>& target,
                        const std::string&                           route,
                        const std::string&                           request,
                        std::optional<size_t>                        expected_item_count = std::nullopt) {
        std::string response;
        PostResult  result;
        result.transport_ok = target->post(route, request, response);
        if (result.transport_ok) {
            result.response = detail::parseKVCMResponse(response);
            if (result.response.parsed && expected_item_count && !result.response.item_results.empty()
                && result.response.item_results.size() != *expected_item_count) {
                RTP_LLM_LOG_WARNING("KVCM event response item count mismatch, route=%s expected=%zu actual=%zu",
                                    route.c_str(),
                                    *expected_item_count,
                                    result.response.item_results.size());
                result.response.parsed = false;
            }
            if (!result.response.parsed) {
                RTP_LLM_LOG_WARNING("KVCM event request returned a malformed response, route=%s bytes=%zu",
                                    route.c_str(),
                                    response.size());
            } else if (result.response.has_unrecognized_code) {
                RTP_LLM_LOG_WARNING("KVCM event request returned an unrecognized protocol code, route=%s "
                                    "snapshot_required=%d retry_after_ms=%llu",
                                    route.c_str(),
                                    static_cast<int>(result.response.snapshot_required),
                                    static_cast<unsigned long long>(result.response.retry_after_ms));
            } else if (!result.response.ok()) {
                RTP_LLM_LOG_WARNING("KVCM event request returned failure, route=%s code=%d item_count=%zu "
                                    "retry_after_ms=%llu snapshot_required=%d",
                                    route.c_str(),
                                    static_cast<int>(result.response.firstFailure()),
                                    result.response.item_results.size(),
                                    static_cast<unsigned long long>(result.response.retry_after_ms),
                                    static_cast<int>(result.response.snapshot_required));
            }
        }
        if (!result.ok()) {
            request_failure_count_.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }

    static bool isPermanentProtocolFailure(const PostResult& result) noexcept {
        return result.transport_ok && result.response.parsed && result.response.hasPermanentFailure();
    }

    bool tripOnPermanentProtocolFailure(const PostResult& result) noexcept {
        if (!isPermanentProtocolFailure(result)) {
            return false;
        }
        tripCircuit(CircuitReason::PERMANENT_PROTOCOL_ERROR);
        return true;
    }

    void markDirty() noexcept {
        dirty_generation_.fetch_add(1, std::memory_order_relaxed);
    }

    bool observeSnapshotAdvisory(const PostResult& result) noexcept {
        const bool requested = result.transport_ok && result.response.parsed && result.response.requestsSnapshot();
        if (requested) {
            markDirty();
        }
        return requested;
    }

    std::chrono::milliseconds retryDelay(std::chrono::milliseconds& retry_interval,
                                         uint64_t                   server_retry_after_ms) const noexcept {
        constexpr uint64_t kMaxServerRetryAfterMs = 300000;
        const auto base_retry_interval = std::chrono::milliseconds(std::max<int64_t>(config_.retry_interval_ms, 1));
        const auto max_retry_interval =
            std::chrono::milliseconds(std::max<int64_t>(base_retry_interval.count(), 30000));
        const auto server_delay =
            std::chrono::milliseconds(static_cast<int64_t>(std::min(server_retry_after_ms, kMaxServerRetryAfterMs)));
        const auto delay = std::max(retry_interval, server_delay);
        if (retry_interval < max_retry_interval) {
            const auto remaining = max_retry_interval - retry_interval;
            retry_interval += std::min(retry_interval, remaining);
        }
        return delay;
    }

    HeartbeatOutcome sendHeartbeatIfDue(HeartbeatSchedule& schedule, bool force = false) {
        if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)) {
            return {HeartbeatAction::CIRCUIT_OPEN, 0};
        }
        const auto now = std::chrono::steady_clock::now();
        if (!force && now < schedule.next) {
            return {};
        }

        const auto result = heartbeat();
        if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)) {
            return {HeartbeatAction::CIRCUIT_OPEN, 0};
        }
        // Preserve a server recovery advisory even when the accompanying
        // operation failed (including with a newer, unknown error code). The
        // failed heartbeat is still retried; dirtying the mirror additionally
        // schedules the requested authoritative snapshot without pretending
        // that the response was successful.
        const bool snapshot_requested = observeSnapshotAdvisory(result);
        if (!result.ok()) {
            if (result.requiresRegistration()) {
                return {HeartbeatAction::REREGISTER, result.response.retry_after_ms};
            }
            if (tripOnPermanentProtocolFailure(result)) {
                return {HeartbeatAction::CIRCUIT_OPEN, 0};
            }
            schedule.degraded = true;
            const auto delay  = retryDelay(schedule.retry_interval, result.serverRetryAfterMs());
            schedule.next     = std::chrono::steady_clock::now() + delay;
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("KVCMPublisher heartbeat failed; retrying in %lld ms",
                                static_cast<long long>(delay.count()));
            return {HeartbeatAction::TRANSIENT_FAILURE, 0, snapshot_requested};
        }

        schedule.degraded       = false;
        schedule.retry_interval = std::chrono::milliseconds(std::max<int64_t>(config_.retry_interval_ms, 1));
        schedule.next =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(config_.heartbeat_interval_ms, 1));
        return {HeartbeatAction::SENT, 0, snapshot_requested};
    }

    void updateStateAfterRealtimeSuccess(bool heartbeat_degraded = false) noexcept {
        if (recoveryRequested()) {
            state_.store(PublisherState::RESYNCING, std::memory_order_relaxed);
            return;
        }
        const auto dirty = dirty_generation_.load(std::memory_order_relaxed);
        state_.store(reconciled_generation_ != dirty ? PublisherState::RESYNCING :
                     heartbeat_degraded              ? PublisherState::DEGRADED :
                                                       PublisherState::READY,
                     std::memory_order_relaxed);
    }

    RegistrationOutcome registerNode() {
        state_.store(PublisherState::REGISTERING, std::memory_order_relaxed);
        auto        trace_id = nextTraceId("register");
        std::string request;
        try {
            request = buildRegisterInstanceRequest(context_, trace_id, config_.report_max_bytes);
        } catch (const JsonPayloadLimitExceeded&) {
            tripCircuit(CircuitReason::REPORT_PAYLOAD_LIMIT);
            RTP_LLM_LOG_WARNING("KVCMPublisher registration exceeds payload limit, limit_bytes=%zu",
                                config_.report_max_bytes);
            return {};
        }
        auto result = postWith(reporter_, "/api/registerInstance", request);
        if (!result.ok()) {
            (void)tripOnPermanentProtocolFailure(result);
            return {false, result.serverRetryAfterMs()};
        }
        trace_id = nextTraceId("node-register");
        try {
            request = buildControlReport(context_, trace_id, ControlEventType::NODE_REGISTER, config_.report_max_bytes);
        } catch (const JsonPayloadLimitExceeded&) {
            tripCircuit(CircuitReason::REPORT_PAYLOAD_LIMIT);
            RTP_LLM_LOG_WARNING("KVCMPublisher node registration exceeds payload limit, limit_bytes=%zu",
                                config_.report_max_bytes);
            return {};
        }
        result = postWith(reporter_, "/api/reportEvent", request, /*expected_item_count=*/1);
        if (!result.ok()) {
            // This is ReportEvent, not RegisterInstance. If the instance was
            // concurrently lost after the first call succeeded, repeat the
            // full idempotent registration sequence instead of treating the
            // response as a permanent configuration error.
            if (!result.requiresRegistration()) {
                (void)tripOnPermanentProtocolFailure(result);
            }
            return {false, result.serverRetryAfterMs()};
        }
        // A successful registration establishes a new reporter lifecycle and
        // therefore always needs one authoritative snapshot. Any payload
        // retained from the previous lifecycle must be rebuilt: registration
        // can be requested by heartbeat or mutation failure while a snapshot
        // retry is pending, and mutations absorbed during that retry window
        // need to be included in the first commit of the new lifecycle.
        pending_snapshot_report_.reset();
        markDirty();
        return {true, 0};
    }

    bool applyToMirror(const std::vector<KVCacheEvent>& batch) {
        if (!logical_mirror_.apply(batch)) {
            tripCircuit(CircuitReason::MIRROR_KEY_LIMIT);
            RTP_LLM_LOG_WARNING("KVCMPublisher mirror exceeds key limit, limit=%zu", config_.snapshot_max_keys);
            return false;
        }
        return true;
    }

    bool drainOneAvailableBatchToMirror() {
        auto batch = queue_.waitPop(config_.report_batch_size, std::chrono::milliseconds::zero());
        if (batch.empty()) {
            return true;
        }
        if (!applyToMirror(batch)) {
            return false;
        }
        markDirty();
        return true;
    }

    std::chrono::milliseconds snapshotWorkPollInterval() const noexcept {
        return std::min(std::chrono::milliseconds(std::max(config_.flush_interval_ms, 1)),
                        std::chrono::milliseconds(10));
    }

    bool drainSnapshotIngress(std::chrono::milliseconds poll_interval) {
        auto batch = queue_.waitPop(config_.report_batch_size, poll_interval);
        if (!batch.empty()) {
            if (!applyToMirror(batch)) {
                return false;
            }
            markDirty();
        }
        return !stopping_.load(std::memory_order_acquire) && !circuit_open_.load(std::memory_order_acquire);
    }

    SnapshotUploadResult postSnapshotWhileDrainingQueue(const std::string& request, HeartbeatSchedule& heartbeat) {
        // A full snapshot can legitimately occupy its longer HTTP timeout.
        // Run only that upload on a short-lived helper thread so the sole
        // mirror owner can continue draining the bounded producer queue. No
        // mutation request is sent concurrently. Once the immutable payload
        // commits, its captured key set is diffed against the current mirror
        // and the final state is replayed in ordinary mutation batches.
        // `request` belongs to pending_snapshot_report_ and is not replaced
        // until this function joins the upload, so capture it by reference to
        // avoid copying a payload that may be hundreds of megabytes.
        auto             upload = std::async(std::launch::async, [this, &request] {
            snapshot_attempt_count_.fetch_add(1, std::memory_order_relaxed);
            return postWith(snapshot_reporter_, "/api/reportEvent", request, /*expected_item_count=*/1);
        });
        HeartbeatOutcome heartbeat_outcome;
        const auto       poll_interval = snapshotWorkPollInterval();
        try {
            while (upload.wait_for(std::chrono::milliseconds::zero()) != std::future_status::ready) {
                if (!drainSnapshotIngress(poll_interval)) {
                    // Producers never call transport cancellation. If stop()
                    // or a terminal mirror failure ends this upload, the
                    // worker owns cancellation and joins it promptly. A
                    // recoverable overflow is handled after this bounded
                    // upload returns because reporter cancellation is a
                    // terminal operation for some injected implementations.
                    snapshot_reporter_->cancel();
                    break;
                }
                const auto heartbeat_result = heartbeat_outcome.action == HeartbeatAction::REREGISTER ?
                                                  HeartbeatOutcome{} :
                                                  sendHeartbeatIfDue(heartbeat);
                heartbeat_outcome.snapshot_requested |= heartbeat_result.snapshot_requested;
                if (heartbeat_result.action == HeartbeatAction::REREGISTER) {
                    heartbeat_outcome.action         = heartbeat_result.action;
                    heartbeat_outcome.retry_after_ms = heartbeat_result.retry_after_ms;
                    // The public reporter seam predates recoverable
                    // cancellation, so an injected reporter may implement
                    // cancel() as terminal. Keep draining until this stale
                    // snapshot returns; production I/O is still bounded by
                    // snapshot_timeout_ms.
                } else if (heartbeat_result.action == HeartbeatAction::CIRCUIT_OPEN) {
                    heartbeat_outcome.action = heartbeat_result.action;
                    snapshot_reporter_->cancel();
                    break;
                }
            }
        } catch (...) {
            // Avoid blocking in std::future's destructor for the full snapshot
            // timeout if mirror ingestion itself fails (for example, an
            // allocation failure). The outer worker guard opens the circuit.
            snapshot_reporter_->cancel();
            upload.wait();
            throw;
        }
        return {upload.get(), heartbeat_outcome};
    }

    SnapshotBuildResult buildSnapshotWhileDrainingQueue(const std::string&               trace_id,
                                                        const KVCacheSnapshot&           snapshot,
                                                        HeartbeatSchedule&               heartbeat,
                                                        std::optional<HeartbeatOutcome>& control_outcome) {
        // Serialization of a maximum-sized snapshot can be comparable to its
        // HTTP upload. Keep the sole mirror owner available during both
        // phases so a valid, bounded snapshot never opens the ingress circuit
        // merely because JSON construction occupied the worker.
        auto       build         = std::async(std::launch::async, [this, trace_id, &snapshot] {
            SnapshotBuildResult result;
            try {
                result.request = buildSnapshotReport(
                    context_, trace_id, snapshot, config_.snapshot_max_bytes, &snapshot_build_cancelled_);
            } catch (const JsonPayloadLimitExceeded&) {
                result.payload_limit_exceeded = true;
            } catch (const SnapshotBuildCancelled&) {
                result.cancelled = true;
            }
            return result;
        });
        const auto poll_interval = snapshotWorkPollInterval();
        try {
            while (build.wait_for(std::chrono::milliseconds::zero()) != std::future_status::ready) {
                if (!drainSnapshotIngress(poll_interval)) {
                    break;
                }
                const auto heartbeat_result =
                    control_outcome && control_outcome->action == HeartbeatAction::REREGISTER ?
                        HeartbeatOutcome{} :
                        sendHeartbeatIfDue(heartbeat);
                if (heartbeat_result.action == HeartbeatAction::REREGISTER) {
                    control_outcome = heartbeat_result;
                    // A recoverable registration loss must not cancel this
                    // otherwise reusable payload. Keep draining ingress until
                    // it joins, then discard it before re-registering.
                } else if (heartbeat_result.action == HeartbeatAction::CIRCUIT_OPEN) {
                    control_outcome = heartbeat_result;
                    break;
                }
                // sendHeartbeatIfDue() already folds a snapshot advisory into
                // dirty_generation_. The snapshot being built started after
                // that server state and will satisfy the advisory when sent;
                // a newer local generation is replayed after it commits.
            }
        } catch (...) {
            snapshot_build_cancelled_.store(true, std::memory_order_release);
            build.wait();
            throw;
        }
        return build.get();
    }

    ReconcileOutcome reconcile(uint64_t generation, HeartbeatSchedule& heartbeat) {
        state_.store(PublisherState::RESYNCING, std::memory_order_relaxed);
        if (recoveryRequested()) {
            return {ReconcileAction::RETRY, 0};
        }
        if (!pending_snapshot_report_) {
            auto                            snapshot = logical_mirror_.snapshot();
            std::optional<HeartbeatOutcome> build_control_outcome;
            const auto                      trace_id = nextTraceId("snapshot");
            auto build_result = buildSnapshotWhileDrainingQueue(trace_id, snapshot, heartbeat, build_control_outcome);
            if (recoveryRequested()) {
                return {ReconcileAction::RETRY, 0};
            }
            if (build_control_outcome) {
                if (build_control_outcome->action == HeartbeatAction::REREGISTER) {
                    return {ReconcileAction::REREGISTER, build_control_outcome->retry_after_ms};
                }
                if (build_control_outcome->action == HeartbeatAction::CIRCUIT_OPEN) {
                    return {ReconcileAction::CIRCUIT_OPEN, 0};
                }
            }
            if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)) {
                return {ReconcileAction::CIRCUIT_OPEN, 0};
            }
            if (build_result.cancelled) {
                // The cancellation token is terminal and is published only by
                // stop(), tripCircuit(), or an exception escaping the sole
                // mirror owner.
                return {ReconcileAction::CIRCUIT_OPEN, 0};
            }
            if (build_result.payload_limit_exceeded || !build_result.request) {
                tripCircuit(CircuitReason::SNAPSHOT_PAYLOAD_LIMIT);
                RTP_LLM_LOG_WARNING("KVCMPublisher snapshot exceeds payload limit, keys=%zu limit_bytes=%zu",
                                    snapshot.block_keys.size(),
                                    config_.snapshot_max_bytes);
                return {ReconcileAction::CIRCUIT_OPEN, 0};
            }
            pending_snapshot_report_ =
                PendingSnapshotReport{generation, std::move(snapshot.block_keys), std::move(*build_result.request)};
        }

        // Avoid starting a potentially long upload when ingress was lost in
        // the narrow window after the snapshot build completed. Once an
        // upload has started it remains bounded by snapshot_timeout_ms, but a
        // not-yet-started stale payload can be discarded immediately.
        if (recoveryRequested()) {
            pending_snapshot_report_.reset();
            return {ReconcileAction::RETRY, 0};
        }
        const auto upload = postSnapshotWhileDrainingQueue(pending_snapshot_report_->request, heartbeat);
        if (recoveryRequested()) {
            pending_snapshot_report_.reset();
            return {ReconcileAction::RETRY, 0};
        }
        if (upload.heartbeat.action == HeartbeatAction::REREGISTER) {
            pending_snapshot_report_.reset();
            return {ReconcileAction::REREGISTER, upload.heartbeat.retry_after_ms};
        }
        if (upload.heartbeat.action == HeartbeatAction::CIRCUIT_OPEN) {
            return {ReconcileAction::CIRCUIT_OPEN, 0};
        }
        const auto& result = upload.result;
        if (!result.ok()) {
            if (result.requiresRegistration()) {
                // Drop eagerly as well as at successful registration so a
                // long registration outage cannot retain a large stale body.
                pending_snapshot_report_.reset();
                return {ReconcileAction::REREGISTER, result.response.retry_after_ms};
            }
            if (tripOnPermanentProtocolFailure(result)) {
                return {ReconcileAction::CIRCUIT_OPEN, 0};
            }
            return {ReconcileAction::RETRY, result.serverRetryAfterMs()};
        }
        if (!detail::isValidSnapshotVersionToken(result.response.committed_snapshot_version)) {
            // A successful KVCM snapshot creates a 128-bit reconciliation
            // generation. Without that token there is no evidence that query
            // visibility advanced to this payload, so retain and retry the
            // immutable request rather than acknowledging an unknown baseline.
            request_failure_count_.fetch_add(1, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("KVCMPublisher snapshot response has an invalid committed generation, bytes=%zu",
                                result.response.committed_snapshot_version.size());
            return {ReconcileAction::RETRY, 0};
        }

        const uint64_t committed_generation = pending_snapshot_report_->generation;
        snapshot_commit_count_.fetch_add(1, std::memory_order_relaxed);
        auto remote_keys = std::move(pending_snapshot_report_->block_keys);
        // The upload is joined and no retry can reuse this immutable JSON.
        // Destroy it before building the target snapshot and replay batches;
        // otherwise a payload up to snapshot_max_bytes remains resident for
        // the entire catch-up pass despite never being read again.
        pending_snapshot_report_.reset();
        if (observeSnapshotAdvisory(result)) {
            // A successful snapshot response should normally observe the
            // generation that request just committed. If KVCM still asks for
            // another snapshot, do not send any delta against a baseline the
            // server explicitly says is incomplete.
            reconciled_generation_ = committed_generation;
            return {ReconcileAction::COMMITTED, 0};
        }
        if (upload.heartbeat.snapshot_requested) {
            // A heartbeat may have observed the pre-commit generation while
            // this long request was in flight. Verify once after joining the
            // upload: a current OK response proves the committed generation
            // is visible, while a request from a restarted KVCM preserves the
            // advisory and causes a fresh authoritative snapshot.
            const auto verification = sendHeartbeatIfDue(heartbeat, /*force=*/true);
            if (verification.action == HeartbeatAction::REREGISTER) {
                return {ReconcileAction::REREGISTER, verification.retry_after_ms};
            }
            if (verification.action == HeartbeatAction::CIRCUIT_OPEN) {
                return {ReconcileAction::CIRCUIT_OPEN, 0};
            }
            if (verification.action == HeartbeatAction::TRANSIENT_FAILURE) {
                return {ReconcileAction::RETRY, 0};
            }
            if (verification.snapshot_requested) {
                reconciled_generation_ = committed_generation;
                return {ReconcileAction::COMMITTED, 0};
            }
        }
        // This generation includes every mutation absorbed while the
        // immutable payload was in flight and any pre-commit advisory proven
        // stale by the verification heartbeat above.
        auto replayed_generation = dirty_generation_.load(std::memory_order_relaxed);
        RTP_LLM_LOG_INFO("KVCMPublisher snapshot committed, instance_id=%s host=%s dp_rank=%d "
                         "server_version=%s keys=%zu generation=%llu",
                         context_.instance_id.c_str(),
                         context_.host_ip_port.c_str(),
                         context_.dp_rank,
                         result.response.committed_snapshot_version.c_str(),
                         remote_keys.size(),
                         static_cast<unsigned long long>(committed_generation));

        for (;;) {
            auto   target       = logical_mirror_.snapshot().block_keys;
            size_t source_index = 0;
            size_t target_index = 0;
            while (source_index < remote_keys.size() || target_index < target.size()) {
                if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)
                    || recoveryRequested()) {
                    if (recoveryRequested()) {
                        return {ReconcileAction::RETRY, 0};
                    }
                    return {ReconcileAction::CIRCUIT_OPEN, 0};
                }
                auto batch = detail::KVCMLogicalMirror::nextMutationBatch(
                    remote_keys, target, source_index, target_index, config_.report_batch_size);
                if (batch.empty()) {
                    continue;
                }
                const auto replay_result = reportMutations(batch);
                if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)
                    || recoveryRequested()) {
                    if (recoveryRequested()) {
                        return {ReconcileAction::RETRY, 0};
                    }
                    return {ReconcileAction::CIRCUIT_OPEN, 0};
                }
                if (!replay_result.ok()) {
                    if (replay_result.requiresRegistration()) {
                        return {ReconcileAction::REREGISTER, replay_result.response.retry_after_ms};
                    }
                    if (tripOnPermanentProtocolFailure(replay_result)) {
                        return {ReconcileAction::CIRCUIT_OPEN, 0};
                    }
                    return {ReconcileAction::RETRY, replay_result.serverRetryAfterMs()};
                }
                if (observeSnapshotAdvisory(replay_result)) {
                    // KVCM may have restarted after the preceding snapshot.
                    // Even though this delta committed, it cannot reconstruct
                    // keys outside the current batch. Resnapshot immediately
                    // instead of continuing from an invalid remote baseline.
                    reconciled_generation_ = replayed_generation;
                    return {ReconcileAction::COMMITTED, 0};
                }
                // A large final-state diff may itself take many network
                // round-trips. Drain one bounded ingress batch after every
                // request so replay cannot become a second queue bottleneck,
                // while continuous producers cannot starve replay forever.
                // These later transitions are intentionally outside the
                // current replayed_generation and are reconciled by the next
                // final-state diff pass below.
                if (!drainOneAvailableBatchToMirror()) {
                    return {ReconcileAction::CIRCUIT_OPEN, 0};
                }
                const auto heartbeat_result = sendHeartbeatIfDue(heartbeat);
                if (heartbeat_result.action == HeartbeatAction::REREGISTER) {
                    return {ReconcileAction::REREGISTER, heartbeat_result.retry_after_ms};
                }
                if (heartbeat_result.action == HeartbeatAction::CIRCUIT_OPEN) {
                    return {ReconcileAction::CIRCUIT_OPEN, 0};
                }
                if (heartbeat_result.snapshot_requested) {
                    reconciled_generation_ = replayed_generation;
                    return {ReconcileAction::COMMITTED, 0};
                }
            }

            // This acknowledges only the target captured before processing
            // response advisories. Even a non-empty stream can fold back to
            // the known remote state, in which case an empty diff is still a
            // complete reconciliation.
            if (recoveryRequested()) {
                return {ReconcileAction::RETRY, 0};
            }
            reconciled_generation_        = replayed_generation;
            remote_keys                   = std::move(target);
            const auto current_generation = dirty_generation_.load(std::memory_order_relaxed);
            if (current_generation == replayed_generation) {
                return {ReconcileAction::COMMITTED, 0};
            }

            // New ingress was drained while the previous diff was crossing the
            // network. Since remote_keys is now an exact committed baseline,
            // catch up with another final-state diff instead of issuing a new
            // full snapshot. Under sustained traffic this behaves like normal
            // bounded batching; shutdown, heartbeats, and overflow remain
            // observable between every request.
            replayed_generation = current_generation;
        }
    }

    PostResult reportMutations(const std::vector<KVCacheEvent>& batch) {
        if (batch.empty()) {
            return {};
        }
        const auto coalesced = coalesceMutations(batch);
        const auto trace_id  = nextTraceId("mutation");
        try {
            return postWith(reporter_,
                            "/api/reportEvent",
                            buildMutationReport(context_, trace_id, coalesced, config_.report_max_bytes),
                            coalesced.size());
        } catch (const JsonPayloadLimitExceeded&) {
            tripCircuit(CircuitReason::REPORT_PAYLOAD_LIMIT);
            RTP_LLM_LOG_WARNING("KVCMPublisher mutation batch exceeds payload limit, events=%zu limit_bytes=%zu",
                                coalesced.size(),
                                config_.report_max_bytes);
            return {};
        }
    }

    PostResult heartbeat() {
        const auto trace_id = nextTraceId("heartbeat");
        try {
            return postWith(
                reporter_,
                "/api/reportEvent",
                buildControlReport(context_, trace_id, ControlEventType::HEARTBEAT, config_.report_max_bytes),
                /*expected_item_count=*/1);
        } catch (const JsonPayloadLimitExceeded&) {
            tripCircuit(CircuitReason::REPORT_PAYLOAD_LIMIT);
            RTP_LLM_LOG_WARNING("KVCMPublisher heartbeat exceeds payload limit, limit_bytes=%zu",
                                config_.report_max_bytes);
            return {};
        }
    }

    void tripCircuit(CircuitReason reason, bool cancel_in_flight = true) noexcept {
        auto expected = CircuitReason::NONE;
        if (!circuit_reason_.compare_exchange_strong(
                expected, reason, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return;
        }
        circuit_open_.store(true, std::memory_order_release);
        accepting_.store(false, std::memory_order_release);
        available_.store(false, std::memory_order_release);
        snapshot_build_cancelled_.store(true, std::memory_order_release);
        publication_gate_.close();
        recovery_gate_.close();
        state_.store(PublisherState::CIRCUIT_OPEN, std::memory_order_relaxed);
        // A circuit is terminal for this publisher instance. Interrupt any
        // synchronous request immediately so the worker can report HOST_DOWN
        // through its separately bounded shutdown reporter and exit.
        if (cancel_in_flight) {
            if (reporter_) {
                reporter_->cancel();
            }
            if (snapshot_reporter_ && snapshot_reporter_ != reporter_) {
                snapshot_reporter_->cancel();
            }
        }
        queue_.stop();
    }

    static const char* circuitReasonName(CircuitReason reason) noexcept {
        switch (reason) {
            case CircuitReason::NONE:
                return "none";
            case CircuitReason::INVALID_CONFIGURATION:
                return "invalid_configuration";
            case CircuitReason::INITIAL_SNAPSHOT_FAILURE:
                return "initial_snapshot_failure";
            case CircuitReason::RECOVERY_SNAPSHOT_FAILURE:
                return "recovery_snapshot_failure";
            case CircuitReason::MIRROR_KEY_LIMIT:
                return "mirror_key_limit";
            case CircuitReason::REPORT_PAYLOAD_LIMIT:
                return "report_payload_limit";
            case CircuitReason::SNAPSHOT_PAYLOAD_LIMIT:
                return "snapshot_payload_limit";
            case CircuitReason::PERMANENT_PROTOCOL_ERROR:
                return "permanent_protocol_error";
            case CircuitReason::WORKER_EXCEPTION:
                return "worker_exception";
        }
        return "unknown";
    }

    void bestEffortHostDown(bool registered) noexcept {
        if (!registered) {
            return;
        }
        try {
            const auto trace_id = nextTraceId("shutdown");
            const auto result =
                postWith(shutdown_reporter_,
                         "/api/reportEvent",
                         buildControlReport(context_, trace_id, ControlEventType::HOST_DOWN, config_.report_max_bytes),
                         /*expected_item_count=*/1);
            if (!result.ok()) {
                RTP_LLM_LOG_WARNING("KVCMPublisher failed to report HOST_DOWN; KVCM lease cleanup remains active");
            }
        } catch (...) {
            RTP_LLM_LOG_WARNING("KVCMPublisher failed to build or report HOST_DOWN");
        }
    }

    bool absorbQueuedEventsUntil(std::chrono::steady_clock::time_point deadline) {
        while (!stopping_.load(std::memory_order_acquire) && !circuit_open_.load(std::memory_order_acquire)
               && !recoveryRequested()) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                break;
            }
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            const auto timeout =
                std::max(std::chrono::milliseconds(1),
                         std::min(std::chrono::milliseconds(std::max(config_.flush_interval_ms, 1)), remaining));
            auto batch = queue_.waitPop(config_.report_batch_size, timeout);
            if (!batch.empty() && !applyToMirror(batch)) {
                return false;
            } else if (!batch.empty()) {
                markDirty();
            }
        }
        return !circuit_open_.load(std::memory_order_acquire);
    }

    bool waitBeforeRetry(std::chrono::milliseconds delay) {
        return absorbQueuedEventsUntil(std::chrono::steady_clock::now() + delay);
    }

    void waitUntil(std::chrono::steady_clock::time_point deadline) {
        (void)absorbQueuedEventsUntil(deadline);
    }

    void workerLoop() noexcept {
        const int64_t base_retry_ms               = std::max<int64_t>(config_.retry_interval_ms, 1);
        const auto    base_retry_interval         = std::chrono::milliseconds(base_retry_ms);
        auto          registration_retry_interval = base_retry_interval;
        auto          reconcile_retry_interval    = base_retry_interval;

        const auto        heartbeat_interval = std::chrono::milliseconds(std::max(config_.heartbeat_interval_ms, 1));
        const auto        snapshot_interval  = std::chrono::milliseconds(std::max(config_.snapshot_interval_ms, 1));
        bool              registered         = false;
        HeartbeatSchedule heartbeat{std::chrono::steady_clock::now(), base_retry_interval};
        auto              next_reconcile = std::chrono::steady_clock::now();
        auto              next_snapshot  = std::chrono::steady_clock::now() + snapshot_interval;
        try {
            while (!stopping_.load(std::memory_order_acquire) && !circuit_open_.load(std::memory_order_acquire)) {
                if (recoveryRequested() && !recoverFromQueueOverflow()) {
                    break;
                }
                if (!registered) {
                    const auto registration = registerNode();
                    if (!registration.registered) {
                        if (stopping_.load(std::memory_order_acquire)
                            || circuit_open_.load(std::memory_order_acquire)) {
                            break;
                        }
                        const auto delay = retryDelay(registration_retry_interval, registration.retry_after_ms);
                        state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                        RTP_LLM_LOG_WARNING("KVCMPublisher registration failed; retrying in %lld ms",
                                            static_cast<long long>(delay.count()));
                        waitBeforeRetry(delay);
                        continue;
                    }
                    // Registration starts a new reporter lifecycle. Do not
                    // inherit a snapshot deadline or exponential backoff from
                    // the previous lifecycle: a server retry_after may be
                    // minutes long, but the newly registered node needs its
                    // authoritative snapshot immediately.
                    registered                  = true;
                    registration_retry_interval = base_retry_interval;
                    reconcile_retry_interval    = base_retry_interval;
                    const auto now              = std::chrono::steady_clock::now();
                    heartbeat                   = {now + heartbeat_interval, base_retry_interval};
                    next_reconcile              = now;
                    next_snapshot               = now + snapshot_interval;
                }

                const auto now = std::chrono::steady_clock::now();
                if (now >= next_snapshot) {
                    markDirty();
                    // The next deadline is set only after reconciliation
                    // commits; retries of this generation must not create
                    // additional dirty generations or recapture payloads.
                    next_snapshot = std::chrono::steady_clock::time_point::max();
                }

                const auto heartbeat_outcome = sendHeartbeatIfDue(heartbeat);
                if (heartbeat_outcome.action == HeartbeatAction::CIRCUIT_OPEN) {
                    break;
                }
                if (heartbeat_outcome.action == HeartbeatAction::REREGISTER) {
                    const auto delay = retryDelay(registration_retry_interval, heartbeat_outcome.retry_after_ms);
                    state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                    RTP_LLM_LOG_WARNING("KVCMPublisher heartbeat lost registration; retrying in %lld ms",
                                        static_cast<long long>(delay.count()));
                    registered = false;
                    waitBeforeRetry(delay);
                    continue;
                }
                if (heartbeat_outcome.action == HeartbeatAction::SENT) {
                    updateStateAfterRealtimeSuccess(heartbeat.degraded);
                }

                if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)) {
                    break;
                }

                const uint64_t dirty_generation = dirty_generation_.load(std::memory_order_relaxed);
                if (reconciled_generation_ != dirty_generation) {
                    const auto reconcile_now = std::chrono::steady_clock::now();
                    if (reconcile_now < next_reconcile) {
                        waitUntil(std::min(next_reconcile, heartbeat.next));
                        continue;
                    }

                    const auto outcome = reconcile(dirty_generation, heartbeat);
                    if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)) {
                        break;
                    }
                    if (recoveryRequested()) {
                        continue;
                    }
                    if (outcome.action == ReconcileAction::CIRCUIT_OPEN) {
                        break;
                    }
                    if (outcome.action != ReconcileAction::COMMITTED) {
                        auto&      retry_interval = outcome.action == ReconcileAction::REREGISTER ?
                                                        registration_retry_interval :
                                                        reconcile_retry_interval;
                        const auto delay          = retryDelay(retry_interval, outcome.retry_after_ms);
                        state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                        RTP_LLM_LOG_WARNING("KVCMPublisher reconciliation failed; retrying in %lld ms",
                                            static_cast<long long>(delay.count()));
                        if (outcome.action == ReconcileAction::REREGISTER) {
                            registered = false;
                            waitBeforeRetry(delay);
                        } else {
                            next_reconcile = std::chrono::steady_clock::now() + delay;
                        }
                        continue;
                    }
                    reconcile_retry_interval = base_retry_interval;

                    const uint64_t current_generation = dirty_generation_.load(std::memory_order_relaxed);
                    if (reconciled_generation_ != current_generation) {
                        state_.store(PublisherState::RESYNCING, std::memory_order_relaxed);
                        next_reconcile = std::chrono::steady_clock::now() + base_retry_interval;
                    } else {
                        next_reconcile = std::chrono::steady_clock::now();
                        // A long upload may cross its old periodic deadline.
                        // Schedule from the completed authoritative commit so
                        // success never causes an immediate redundant snapshot.
                        next_snapshot = std::chrono::steady_clock::now() + snapshot_interval;
                        state_.store(heartbeat.degraded ? PublisherState::DEGRADED : PublisherState::READY,
                                     std::memory_order_relaxed);
                    }
                    continue;
                }

                // Batch latency must never postpone control-plane deadlines.
                // In particular, a deliberately large flush interval cannot
                // make KVCM expire this node's heartbeat lease or suppress a
                // periodic authoritative snapshot while the queue is idle.
                const auto wait_now         = std::chrono::steady_clock::now();
                const auto control_deadline = std::min(heartbeat.next, next_snapshot);
                auto control_wait = std::chrono::duration_cast<std::chrono::milliseconds>(control_deadline - wait_now);
                if (control_wait <= std::chrono::milliseconds::zero()) {
                    control_wait = std::chrono::milliseconds(1);
                }
                const auto batch_wait =
                    std::min(std::chrono::milliseconds(std::max(config_.flush_interval_ms, 1)), control_wait);
                auto batch = queue_.waitPop(config_.report_batch_size, batch_wait);
                if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)) {
                    break;
                }
                if (recoveryRequested()) {
                    continue;
                }
                if (batch.empty()) {
                    continue;
                }
                if (!applyToMirror(batch)) {
                    break;
                }

                const auto result = reportMutations(batch);
                if (stopping_.load(std::memory_order_acquire) || circuit_open_.load(std::memory_order_acquire)) {
                    break;
                }
                if (recoveryRequested()) {
                    continue;
                }
                if (!result.ok()) {
                    markDirty();
                    if (result.requiresRegistration()) {
                        const auto delay = retryDelay(registration_retry_interval, result.response.retry_after_ms);
                        state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                        RTP_LLM_LOG_WARNING("KVCMPublisher mutation report lost registration; retrying in %lld ms",
                                            static_cast<long long>(delay.count()));
                        registered = false;
                        waitBeforeRetry(delay);
                    } else if (tripOnPermanentProtocolFailure(result)) {
                        break;
                    } else {
                        const auto delay = retryDelay(reconcile_retry_interval, result.serverRetryAfterMs());
                        state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
                        RTP_LLM_LOG_WARNING("KVCMPublisher mutation report failed; reconciling in %lld ms",
                                            static_cast<long long>(delay.count()));
                        next_reconcile = std::chrono::steady_clock::now() + delay;
                    }
                    continue;
                }
                observeSnapshotAdvisory(result);
                updateStateAfterRealtimeSuccess(heartbeat.degraded);
            }
        } catch (const std::exception& e) {
            tripCircuit(CircuitReason::WORKER_EXCEPTION);
            RTP_LLM_LOG_WARNING("KVCMPublisher worker stopped after exception: %s", e.what());
        } catch (...) {
            tripCircuit(CircuitReason::WORKER_EXCEPTION);
            RTP_LLM_LOG_WARNING("KVCMPublisher worker stopped after unknown exception");
        }
        accepting_.store(false, std::memory_order_release);
        // A stopped/circuit-open queue has no actionable backlog. Physically
        // release pending entries so queue_size converges to zero instead of
        // looking permanently stuck, while the high-water and cumulative
        // counters retain the failure evidence.
        queue_.discardPending();
        bestEffortHostDown(registered);
        // A terminal exporter stays attached to the manager for diagnostics.
        // Release its potentially large retry payload and mirror nodes here so
        // a control-plane failure cannot pin snapshot-scale memory until
        // process shutdown. Queue cells are fixed-capacity, trivial objects.
        pending_snapshot_report_.reset();
        logical_mirror_.release();
        if (circuit_open_.load(std::memory_order_acquire)) {
            RTP_LLM_LOG_WARNING(
                "KVCMPublisher circuit opened; cache-event export is disabled for this process, "
                "reason=%s accepted=%llu dropped=%llu request_failures=%llu",
                circuitReasonName(circuit_reason_.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(accepted_count_.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(dropped_count_.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(request_failure_count_.load(std::memory_order_relaxed)));
        }
    }

private:
    KVCacheEventPublisherConfig           config_;
    KVCacheEventPublisherContext          context_;
    KVCacheSnapshotProvider               snapshot_provider_;
    std::shared_ptr<KVCacheEventReporter> reporter_;
    std::shared_ptr<KVCacheEventReporter> snapshot_reporter_;
    std::shared_ptr<KVCacheEventReporter> shutdown_reporter_;
    bool                                  reporter_injected_{false};
    detail::KVCacheEventQueue             queue_;
    detail::KVCMLogicalMirror             logical_mirror_;
    std::thread                           worker_;
    std::mutex                            lifecycle_mu_;
    std::atomic<bool>                     started_{false};
    std::atomic<bool>                     accepting_{false};
    std::atomic<bool>                     stopping_{false};
    std::atomic<bool>                     available_{true};
    std::atomic<bool>                     circuit_open_{false};
    std::atomic<bool>                     recovery_requested_{false};
    std::atomic<bool>                     snapshot_build_cancelled_{false};
    std::atomic<CircuitReason>            circuit_reason_{CircuitReason::NONE};
    detail::KVCacheEventAdmissionGate     publication_gate_;
    detail::KVCacheEventAdmissionGate     recovery_gate_;
    bool                                  stopped_permanently_{false};
    std::atomic<PublisherState>           state_{PublisherState::DISABLED};
    std::atomic<uint64_t>                 accepted_count_{0};
    std::atomic<uint64_t>                 dropped_count_{0};
    std::atomic<uint64_t>                 request_failure_count_{0};
    std::atomic<uint64_t>                 overflow_recovery_count_{0};
    std::atomic<uint64_t>                 snapshot_attempt_count_{0};
    std::atomic<uint64_t>                 snapshot_commit_count_{0};
    std::atomic<uint64_t>                 dirty_generation_{1};
    uint64_t                              reconciled_generation_ = 0;
    uint64_t                              next_request_id_       = 1;
    std::optional<PendingSnapshotReport>  pending_snapshot_report_;
};

KVCMPublisher::KVCMPublisher(KVCacheEventPublisherConfig           config,
                             KVCacheEventPublisherContext          context,
                             KVCacheSnapshotProvider               snapshot_provider,
                             std::shared_ptr<KVCacheEventReporter> reporter):
    impl_(std::make_unique<Impl>(
        std::move(config), std::move(context), std::move(snapshot_provider), std::move(reporter))) {}

KVCMPublisher::~KVCMPublisher() = default;

bool KVCMPublisher::start() noexcept {
    return impl_->start();
}

PublishResult KVCMPublisher::tryPublish(KVCacheEvent event) noexcept {
    return impl_->tryPublish(std::move(event));
}

void KVCMPublisher::stop() noexcept {
    impl_->stop();
}

PublisherStatus KVCMPublisher::status() const noexcept {
    return impl_->status();
}

bool KVCMPublisher::enabled() const noexcept {
    return impl_->enabled();
}

}  // namespace rtp_llm

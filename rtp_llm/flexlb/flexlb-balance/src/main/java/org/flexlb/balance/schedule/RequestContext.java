package org.flexlb.balance.schedule;

import io.opentelemetry.context.Context;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.balance.requestlog.RoutingDecision;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.RequestEventType;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicLong;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Owns one request's inputs, telemetry and lifecycle facts under the request monitor.
 * Selection, cancellation and worker reports update this object; callbacks and resource cleanup run
 * outside its lock.
 */
public final class RequestContext implements org.flexlb.worker.endpoint.WorkerRequest {
    private static final AtomicLong SELECTION_IDS = new AtomicLong();

    @Getter private final FlexlbConfig config;

    @Getter private volatile Request request;

    @Getter private long originalNumericRequestId;

    @Getter
    private org.flexlb.balance.schedule.RequestPhase phase =
            org.flexlb.balance.schedule.RequestPhase.GENERATION;

    @Getter
    private java.util.List<org.flexlb.config.service.RoleType> requestedRoles = java.util.List.of();

    @Getter private volatile SchedulingMetadata schedulingMetadata;
    @Getter private volatile CompletableFuture<Response> future;
    @Getter private volatile AbstractRequestScheduler requestScheduler;
    @Getter @Setter private Context traceContext;

    @Getter
    private volatile Timing timing =
            new Timing(System.currentTimeMillis(), System.nanoTime(), 0L, 0L, 0L, 0L, 0L, null);

    @Getter
    private volatile PreemptionStatistics preemptionStatistics = new PreemptionStatistics(0L, 0);

    @Getter @Setter private volatile long placementSequence;

    @Getter @Setter private volatile Map<String, Object> schedulingDiagnostics;
    @Getter @Setter private Long requestMessageBytes;
    private String cacheSource;

    private long cacheQueryUs;
    private int cacheQueryCount;

    private final EnumMap<RoleType, CacheMatchSelection> cacheSelections =
            new EnumMap<>(RoleType.class);
    private final EnumMap<RoleType, RoutingDecision> routingDecisions =
            new EnumMap<>(RoleType.class);
    private final EnumMap<RoleType, Integer> routingAttempts = new EnumMap<>(RoleType.class);

    // Set by lifecycle changes; cleared after updating the last-change timestamp.
    private boolean stateChanged;
    private long updatedAtMs;

    private String registeredRequestId;
    private RequestResult route;

    private long activeSelectionId;
    // Negative until the selected route and its resources are committed.
    private long routeCommittedAtMs = -1L;
    private boolean prefillWorkerReleaseConfirmed;
    private boolean decodeWorkerReleaseConfirmed;
    private long prefillCompletedAtMs;
    private boolean decodeAccepted;
    private long visibilityAtMs;
    private boolean visibilityOverdue;

    private CancelReason cancellationReason;
    private StrategyErrorType cancellationError;
    private String cancellationDetail;
    private RequestResult requestEndResult;
    private RequestEndEvent pendingEndEvent;
    private RequestResult responseForFuture;

    private boolean prefillCleanupCompleted = true;
    private boolean decodeCleanupCompleted = true;
    private boolean notificationsDone;
    private boolean expired;

    private ExpirationTimer.Plan deadlines = new ExpirationTimer.Plan(0L, null, null, null);
    private RequestResult requestRecordForArchive;

    private long lastWorkerAtMs;
    private boolean requestDeadlineConsumed;
    private long inactivityTimeoutMs;

    private StrategyErrorType timeoutError;

    public RequestContext(FlexlbConfig config) {
        this.config = Objects.requireNonNull(config, "config");
    }

    /**
     * Captures a read-only view; the mutable facts exist only on this request.
     */
    public synchronized Snapshot snapshot() {
        long inactivityAtMs =
                saturatedAdd(
                        Math.max(lastWorkerAtMs, Math.max(0L, routeCommittedAtMs)),
                        inactivityTimeoutMs);
        return new Snapshot(
                registeredRequestId,
                originalNumericRequestId,
                phase,
                timing.registeredAtMs(),
                route,
                activeSelectionId,
                routeCommittedAtMs,
                prefillWorkerReleaseConfirmed,
                decodeWorkerReleaseConfirmed,
                decodeAccepted,
                visibilityOverdue,
                cancellationReason,
                cancellationDetail,
                requestEndResult,
                responseForFuture,
                prefillCleanupCompleted,
                decodeCleanupCompleted,
                notificationsDone,
                deadlines,
                requestRecordForArchive,
                updatedAtMs,
                inactivityAtMs);
    }

    public synchronized void setOriginalNumericRequestId(long id) {
        checkState(requestScheduler == null, "registered request identity is immutable");
        originalNumericRequestId = id;
    }

    public synchronized void setPhase(org.flexlb.balance.schedule.RequestPhase value) {
        checkState(requestScheduler == null, "registered request phase is immutable");
        phase = Objects.requireNonNull(value);
    }

    public synchronized void setRequestedRoles(java.util.List<RoleType> roles) {
        checkState(requestScheduler == null, "registered requested roles are immutable");
        requestedRoles = java.util.List.copyOf(roles);
    }

    public synchronized void setRequest(Request value) {
        checkState(
                !(future instanceof RequestFuture) || request == value,
                "registered request inputs are immutable");
        request = Objects.requireNonNull(value, "request");
    }

    public synchronized void setSchedulingMetadata(SchedulingMetadata value) {
        checkState(
                !(future instanceof RequestFuture) || schedulingMetadata == value,
                "registered scheduling metadata is immutable");
        schedulingMetadata = value;
    }

    public synchronized void setFuture(CompletableFuture<Response> value) {
        checkState(
                !(future instanceof RequestFuture) && !(value instanceof RequestFuture)
                        || future == value,
                "registered request future cannot be replaced or rebound");
        future = value;
    }

    public String getRequestId() {
        return request.getRequestId();
    }

    public int getPriority() {
        return schedulingMetadata == null
                ? (request == null ? 0 : request.getRawPriority())
                : schedulingMetadata.priority();
    }

    public long getRequestExpiresAtMs() {
        return Objects.requireNonNull(schedulingMetadata, "scheduling metadata required")
                .expiresAtMs();
    }

    public Response getResponse() {
        var response = future;
        if (response == null) {
            return null;
        }
        try {
            return response.getNow(null);
        } catch (CompletionException | CancellationException failed) {
            return null;
        }
    }

    public synchronized void recordRequestArrival(long clientSentAtMs) {
        if (clientSentAtMs > 0L) {
            timing = timing.withArrivalDelayMs(timing.startedAtMs() - clientSentAtMs);
        }
    }

    public synchronized void recordGrpcArrival(long receivedAtMs, long receivedNanos) {
        timing = timing.withGrpcReceivedAtMs(receivedAtMs).withGrpcReceivedNanos(receivedNanos);
    }

    public synchronized void finishRequestTiming() {
        timing = timing.withTotalElapsedUs((System.nanoTime() - timing.startedNanos()) / 1_000L);
    }

    public void recordPreemptionStatistics(long cost, int requestCount) {
        preemptionStatistics = new PreemptionStatistics(cost, requestCount);
    }

    public synchronized void recordCacheQuery(String source, long queryTimeUs) {
        cacheSource = source;
        cacheQueryUs += Math.max(0L, queryTimeUs);
        cacheQueryCount++;
    }

    public synchronized void recordCacheSelection(
            RoleType role, String selectedIp, long hitCacheTokens) {
        cacheSelections.put(role, new CacheMatchSelection(role, selectedIp, hitCacheTokens));
    }

    public synchronized void beginRoutingAttempt(RoleType role) {
        routingAttempts.merge(role, 1, Integer::sum);
        cacheSelections.remove(role);
        routingDecisions.remove(role);
    }

    public synchronized void recordRoutingDecision(RoutingDecision decision) {
        Objects.requireNonNull(decision.role(), "role");
        Objects.requireNonNull(decision.selectionReason(), "selectionReason");
        routingDecisions.put(
                decision.role(),
                new RoutingDecision(
                        decision.role(),
                        decision.group(),
                        decision.strategy(),
                        decision.selectionReason(),
                        decision.decisionTimeMs(),
                        routingAttempts.getOrDefault(decision.role(), 0),
                        decision.selectedEndpoint(),
                        decision.totalWorkerCount(),
                        decision.candidateWorkerCount(),
                        decision.snapshotTruncated(),
                        decision.rejections(),
                        decision.candidates()));
    }

    public synchronized String selectionReason(RoleType role) {
        var decision = routingDecisions.get(role);
        return decision == null ? null : decision.selectionReason();
    }

    public synchronized RoutingTelemetry getRoutingTelemetry() {
        return new RoutingTelemetry(
                cacheSource,
                cacheQueryUs,
                cacheQueryCount,
                cacheSelections,
                routingDecisions,
                routingAttempts);
    }

    /**
     * Returns the current selection identity without storing another request context.
     */
    public synchronized SelectionAttempt currentSelectionAttempt() {
        return activeSelectionId == 0L ? null : new SelectionAttempt(this, activeSelectionId);
    }

    /**
     * Starts worker selection for a request that has no route and no other routing work in progress.
     */
    synchronized SelectionAttempt beginRouteSelection() {
        if (registeredRequestId == null
                || requestEndResult != null
                || cancellationReason != null
                || pendingEndEvent != null
                || activeSelectionId != 0L
                || route != null) {
            return null;
        }
        long selectionId = SELECTION_IDS.incrementAndGet();
        checkState(selectionId > 0L, "selection id exhausted");
        stateChanged = true;
        activeSelectionId = selectionId;
        this.setSchedulingDiagnostics(null);
        refreshState(updatedAtMs);
        return new SelectionAttempt(this, activeSelectionId);
    }

    /**
     * Ends an uncommitted selection after rollback and applies any cancellation or failure received
     * while selecting.
     */
    synchronized Snapshot finishSelectionWithoutCommit(
            SelectionAttempt exact, Response failure, long nowMs) {
        if (!matchesSelection(exact)) {
            return null;
        }
        stateChanged = true;
        activeSelectionId = 0L;
        boolean rolledBack = route == null && prefillCleanupCompleted && decodeCleanupCompleted;
        boolean admissionFailure =
                failure != null && requestEndResult == null && cancellationReason == null;
        if (admissionFailure
                && pendingEndEvent != null
                && pendingEndEvent.kind() == EndKind.SHUTDOWN) {
            stateChanged |= !Objects.equals(pendingEndEvent, null);
            pendingEndEvent = null;
        }
        // WorkerCleanup of a rolled-back preparation must not end the request: no route
        // was delivered, so the scheduler can select another endpoint. Keep all other endings.
        boolean unpublishedRollback = rolledBack && routeCommittedAtMs < 0L && !decodeAccepted;
        if (unpublishedRollback
                && pendingEndEvent != null
                && pendingEndEvent.kind() == EndKind.ENDPOINT_CLOSED) {
            stateChanged |= !Objects.equals(pendingEndEvent, null);
            pendingEndEvent = null;
        }
        if (failure != null
                && (pendingEndEvent == null
                        || pendingEndEvent.kind() == EndKind.SHUTDOWN
                                && cancellationReason == null)) {
            retainEnd(RequestEndEvent.publicationFailure(route, failure));
        } else if (pendingEndEvent != null) {
            retainEnd(pendingEndEvent);
        }
        return commitSnapshot(nowMs);
    }

    /**
     * Records a route event under the request lock. Returns whether the scheduler must update
     * timers or advance cleanup.
     */
    synchronized boolean recordEvent(RequestEvent event) {
        boolean remoteCleanup =
                event.type() == RequestEventType.REMOTE_CLEANED
                        || event.type() == RequestEventType.REMOTE_FENCED;
        if (requestRecordForArchive != null) return false;
        boolean endpointClosed = event.type() == RequestEventType.ENDPOINT_CLOSED;
        if (remoteCleanup) {
            if (routeCommittedAtMs < 0L || route != event.route()) return false;
        } else {
            var role = event.role();
            if (route == null
                    || route != event.route()
                    || role == null
                    || role != RoleType.DECODE
                            && role != RoleType.ENCODER
                            && !role.supportsPrefill()
                    || role == RoleType.DECODE
                            && !Objects.equals(route.decodeRegistration(), event.reservation())) {
                return false;
            }
            if (endpointClosed) {
                WorkerEndpoint source =
                        switch (role) {
                            case DECODE -> route.decodeEndpoint();
                            case ENCODER -> route.encoderEndpoint();
                            case PREFILL, PDFUSION -> route.prefillEndpoint();
                            default -> null;
                        };
                if (source == null || source != event.endpoint()) return false;
            }
        }
        var previousDeadlines = deadlines;
        if (remoteCleanup) {
            recordWorkerReleaseConfirmed(true, event.type() == RequestEventType.REMOTE_CLEANED);
        } else if (endpointClosed) {
            recordEndpointClose(event);
        } else {
            boolean accepted = decodeAccepted || event.role() == RoleType.DECODE;
            long completed = prefillCompletedAtMs;
            if (event.role() != RoleType.DECODE
                    && event.type() == RequestEventType.COMPLETED
                    && completed == 0L) {
                completed = event.nowMs();
            }
            long visibility = visibilityAtMs;
            if (accepted) {
                visibility = 0L;
            } else if (completed > 0L && route.decodeEndpoint() != null) {
                visibility = saturatedAdd(completed, 10_000L);
            }
            if (accepted != decodeAccepted
                    || completed != prefillCompletedAtMs
                    || visibility != visibilityAtMs) {
                decodeAccepted = accepted;
                prefillCompletedAtMs = completed;
                visibilityAtMs = visibility;
                visibilityOverdue = false;
                stateChanged = true;
            }
            long latestActivity = Math.max(lastWorkerAtMs, event.nowMs());
            stateChanged |= lastWorkerAtMs != latestActivity;
            lastWorkerAtMs = latestActivity;
            if (event.terminal()) {
                recordWorkerResourceRelease(event.role());
                boolean ending =
                        event.role() == RoleType.DECODE
                                || event.role() == RoleType.PDFUSION
                                || event.role() == RoleType.ENCODER
                                || event.type() == RequestEventType.FAILED
                                || event.type() == RequestEventType.PRIORITY_CANCELLED;
                if (requestEndResult == null && ending) retainEnd(event.end());
            }
        }
        boolean changed = stateChanged;
        refreshState(event.nowMs());
        boolean timersChanged = previousDeadlines != deadlines;
        return timersChanged || changed && (pendingEndEvent != null || requestEndResult != null);
    }

    synchronized boolean requestCancel(
            CancelReason reason,
            boolean frontend,
            long nowMs,
            StrategyErrorType errorType,
            String detail,
            RequestResult responseCandidate) {
        if (registeredRequestId == null) {
            return false;
        }
        Objects.requireNonNull(reason);
        if (requestEndResult != null) {
            return false;
        }
        if (cancellationReason != null) {
            return false;
        }
        if (reason == CancelReason.DEADLINE_EXCEEDED && routeCommittedAtMs >= 0L) {
            return false;
        }
        boolean beforeSend = routeCommittedAtMs < 0L;
        if (frontend && (responseForFuture != null || !beforeSend)) {
            return false;
        }
        StrategyErrorType error = Objects.requireNonNull(errorType, "cancellation error required");
        recordCancellation(reason, error, detail == null ? reason.getMessage() : detail);
        retainEnd(
                RequestEndEvent.failure(
                        route, EndKind.FRONTEND, cancellationError, cancellationDetail));
        selectResponse(responseCandidate);
        refreshState(nowMs);
        return true;
    }

    synchronized Snapshot recordDeadline(
            ExpirationTimer.Token exact, long nowMs, String diagnosis) {
        if (exact.owner() != this
                || deadlines.token(exact.kind()) != exact
                || requestRecordForArchive != null) {
            return null;
        }
        if (nowMs < exact.atMs()) {
            var nextDeadlines = deadlines.without(exact.kind());
            stateChanged |= !Objects.equals(deadlines, nextDeadlines);
            deadlines = nextDeadlines;
            return commitSnapshot(nowMs);
        }
        if (exact.kind() == ExpirationTimer.Kind.VISIBILITY) {
            stateChanged |= visibilityAtMs != 0L;
            visibilityAtMs = 0L;
            stateChanged |= visibilityOverdue != true;
            visibilityOverdue = true;
            return commitSnapshot(nowMs);
        }
        boolean inactive = exact.kind() == ExpirationTimer.Kind.INACTIVITY;
        if (inactive) {
            var nextDeadlines = deadlines.without(exact.kind());
            stateChanged |= !Objects.equals(deadlines, nextDeadlines);
            deadlines = nextDeadlines;
            long activity = Math.max(lastWorkerAtMs, Math.max(0L, routeCommittedAtMs));
            if (nowMs - activity < inactivityTimeoutMs) return commitSnapshot(nowMs);
            stateChanged |= expired != true;
            expired = true;
        } else {
            stateChanged |= requestDeadlineConsumed != true;
            requestDeadlineConsumed = true;
            if (requestEndResult != null || routeCommittedAtMs >= 0L && activeSelectionId == 0L)
                return commitSnapshot(nowMs);
        }
        String detail = inactive ? "request inactive" : "request scheduling deadline exceeded";
        if (cancellationReason == null && requestEndResult == null) {
            recordCancellation(
                    CancelReason.DEADLINE_EXCEEDED,
                    timeoutError,
                    diagnosis == null ? detail : diagnosis);
        }
        if (inactive || activeSelectionId != 0L || routeCommittedAtMs < 0L && !decodeAccepted) {
            retainEnd(RequestEndEvent.failure(route, EndKind.INACTIVITY, timeoutError, detail));
        }
        return commitSnapshot(nowMs);
    }

    /**
     * Records an ending and selects its reserved response together; a missing candidate leaves
     * response publication pending.
     */
    synchronized Snapshot endRequest(
            RequestEndEvent evidence, long nowMs, RequestResult responseCandidate) {
        if (registeredRequestId == null
                || requestRecordForArchive != null
                || evidence != null && evidence.route() != null && evidence.route() != route) {
            return snapshot();
        }
        if (evidence != null) {
            retainEnd(evidence);
        }
        selectResponse(responseCandidate);
        return commitSnapshot(nowMs);
    }

    synchronized WorkerEndpoint.ReleaseReason releaseReason() {
        if (requestEndResult == null
                || requestRecordForArchive != null
                || activeSelectionId != 0L) {
            return null;
        }
        return routeCommittedAtMs >= 0L
                        && prefillWorkerReleaseConfirmed
                        && decodeWorkerReleaseConfirmed
                ? WorkerEndpoint.ReleaseReason.REMOTE_CLEANUP
                : expired
                        ? WorkerEndpoint.ReleaseReason.EXPIRED
                        : WorkerEndpoint.ReleaseReason.COUNTERPART_FINISHED;
    }

    synchronized void recordRouteCommitStart(long startedNanos) {
        timing = timing.withRouteCommitStartedNanos(startedNanos);
    }

    /**
     * Associates acquired resources with this attempt before submission or rollback.
     * Cancellation waits for the active attempt to return its resources.
     */
    synchronized void registerSelectedRoute(SelectionAttempt attempt, RequestResult acquired) {
        checkState(matchesSelection(attempt), "route requires its current selection");
        checkState(
                acquired.ctx() == this
                        && Objects.equals(acquired.requestId(), registeredRequestId)
                        && (route == null || route == acquired),
                "foreign route");
        if (route == acquired) return;
        stateChanged = true;
        route = acquired;
        var nextPrefillCleanupCompleted =
                acquired.prefillEndpoint() == null && acquired.encoderEndpoint() == null;
        stateChanged |= prefillCleanupCompleted != nextPrefillCleanupCompleted;
        prefillCleanupCompleted = nextPrefillCleanupCompleted;
        var nextDecodeCleanupCompleted = acquired.decodeRegistration() == null;
        stateChanged |= decodeCleanupCompleted != nextDecodeCleanupCompleted;
        decodeCleanupCompleted = nextDecodeCleanupCompleted;
        refreshState(updatedAtMs);
    }

    /**
     * Commits the route and selects its success response once, under the request monitor.
     * Scheduler has checked endpoint capacity and reserved response execution before this call.
     */
    synchronized boolean tryCommitRoute(
            SelectionAttempt attempt, RequestResult acquired, RequestResult response, long nowMs) {
        if (!canPublishRoute(attempt) || route != acquired || responseForFuture != null)
            return false;
        long schedulingDeadlineAtMs = getRequestExpiresAtMs();
        if (schedulingDeadlineAtMs > 0L && nowMs >= schedulingDeadlineAtMs
                || nowMs - lastWorkerAtMs >= inactivityTimeoutMs) {
            recordCancellation(
                    CancelReason.DEADLINE_EXCEEDED,
                    timeoutError,
                    "request expired before route commit");
            retainEnd(
                    RequestEndEvent.failure(
                            route, EndKind.INACTIVITY, timeoutError, cancellationDetail));
            refreshState(nowMs);
            return false;
        }
        stateChanged = true;
        routeCommittedAtMs = nowMs;
        activeSelectionId = 0L;
        responseForFuture = response;
        var nextDecodeWorkerReleaseConfirmed = acquired.decodeRegistration() == null;
        stateChanged |= decodeWorkerReleaseConfirmed != nextDecodeWorkerReleaseConfirmed;
        decodeWorkerReleaseConfirmed = nextDecodeWorkerReleaseConfirmed;
        refreshState(nowMs);
        return true;
    }

    synchronized boolean canPublishRoute(SelectionAttempt exact) {
        return registeredRequestId != null
                && requestEndResult == null
                && cancellationReason == null
                && pendingEndEvent == null
                && routeCommittedAtMs < 0L
                && matchesSelection(exact);
    }

    /**
     * Records one cleanup pass, including partial resource release and submitted cancellation notifications.
     * Only the exact route may advance cleanup; archiving is checked once after recording all progress.
     */
    synchronized void recordCleanupProgress(
            RequestResult exact,
            boolean prefillReleased,
            boolean decodeReleased,
            boolean notificationsSubmitted,
            long nowMs) {
        if (route != exact || requestRecordForArchive != null) return;
        recordCleanupCompleted(prefillReleased, decodeReleased);
        if (requestEndResult != null) {
            stateChanged |= notificationsSubmitted && !notificationsDone;
            notificationsDone |= notificationsSubmitted;
        } else {
            if (routeCommittedAtMs < 0L && prefillCleanupCompleted && decodeCleanupCompleted) {
                stateChanged |= !Objects.equals(route, null);
                route = null;
            }
        }
        refreshState(nowMs);
    }

    synchronized void recordRoutingDiagnostics(
            SelectionAttempt exact, java.util.Map<String, Object> diagnostics) {
        if (matchesSelection(exact) && cancellationReason == null) {
            this.setSchedulingDiagnostics(
                    diagnostics == null ? null : java.util.Map.copyOf(diagnostics));
        }
    }

    /**
     * Binds the scheduler and response, then initializes request deadlines under the Repository ->
     * Context lock.
     */
    synchronized void register(
            AbstractRequestScheduler owner,
            RequestFuture response,
            long nowMs,
            long inactivityTimeoutMs,
            StrategyErrorType timeoutError) {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(schedulingMetadata, "scheduling metadata required");
        if (inactivityTimeoutMs <= 0L) {
            throw new IllegalArgumentException("positive inactivity timeout required");
        }
        String requestId = Objects.requireNonNull(getRequestId(), "requestId");
        Objects.requireNonNull(timeoutError, "timeout error required");
        checkState(registeredRequestId == null, "request already registered");
        requestScheduler = owner;
        future = response;
        registeredRequestId = requestId;
        this.timeoutError = timeoutError;
        timing = timing.withRegisteredAtMs(nowMs);
        lastWorkerAtMs = nowMs;
        this.inactivityTimeoutMs = inactivityTimeoutMs;
        stateChanged = true;
        refreshState(nowMs);
    }

    /**
     * Release scheduling inputs after the route response is completed.
     */
    synchronized void releaseSchedulingInput() {
        if (request != null
                && (!request.getBlockCacheKeys().isEmpty()
                        || !request.getCacheAffinityKeys().isEmpty())) {
            request =
                    request.toBuilder()
                            .blockCacheKeys(java.util.List.of())
                            .cacheAffinityKeys(java.util.List.of())
                            .build();
        }
    }

    private void selectResponse(RequestResult candidate) {
        if (candidate == null || candidate.ctx() != this || responseForFuture != null) return;
        if (candidate.responseKind() == ResponseKind.ACK) {
            return;
        } else if (candidate.responseKind() == ResponseKind.TERMINAL && requestEndResult == null
                || candidate.responseKind() == ResponseKind.CANCELLATION
                        && cancellationReason == null) {
            return;
        }
        stateChanged = true;
        responseForFuture = candidate;
    }

    private void recordEndpointClose(RequestContext.RequestEvent endpointClosed) {
        boolean isDecode = endpointClosed.role() == RoleType.DECODE;
        recordWorkerResourceRelease(endpointClosed.role());
        boolean ending =
                isDecode
                        || endpointClosed.role() == RoleType.PDFUSION
                        || endpointClosed.role() == RoleType.ENCODER
                        || routeCommittedAtMs < 0L && !decodeAccepted;
        if (activeSelectionId != 0L
                && ending
                && (!isDecode || routeCommittedAtMs < 0L && !decodeAccepted)) {
            // An unsent route can still roll back and retry. Keep cleanup as the
            // pending end, below any other end already accepted for this routing work.
            if (requestEndResult == null && pendingEndEvent == null) {
                var nextPendingEndEvent = endpointClosed.end();
                stateChanged |= !Objects.equals(pendingEndEvent, nextPendingEndEvent);
                pendingEndEvent = nextPendingEndEvent;
            }
            return;
        }
        if (requestEndResult == null && ending) retainEnd(endpointClosed.end());
    }

    private void recordWorkerResourceRelease(RoleType role) {
        boolean isDecode = role == RoleType.DECODE;
        if (routeCommittedAtMs >= 0L) {
            recordWorkerReleaseConfirmed(!isDecode, isDecode);
        }
        recordCleanupCompleted(!isDecode, isDecode);
    }

    private void recordWorkerReleaseConfirmed(boolean prefillReleased, boolean decodeReleased) {
        var nextPrefillWorkerReleaseConfirmed = prefillWorkerReleaseConfirmed || prefillReleased;
        stateChanged |= prefillWorkerReleaseConfirmed != nextPrefillWorkerReleaseConfirmed;
        prefillWorkerReleaseConfirmed = nextPrefillWorkerReleaseConfirmed;
        var nextDecodeWorkerReleaseConfirmed = decodeWorkerReleaseConfirmed || decodeReleased;
        stateChanged |= decodeWorkerReleaseConfirmed != nextDecodeWorkerReleaseConfirmed;
        decodeWorkerReleaseConfirmed = nextDecodeWorkerReleaseConfirmed;
    }

    private void recordCleanupCompleted(boolean prefillReleased, boolean decodeReleased) {
        var nextPrefillCleanupCompleted = prefillCleanupCompleted || prefillReleased;
        stateChanged |= prefillCleanupCompleted != nextPrefillCleanupCompleted;
        prefillCleanupCompleted = nextPrefillCleanupCompleted;
        var nextDecodeCleanupCompleted = decodeCleanupCompleted || decodeReleased;
        stateChanged |= decodeCleanupCompleted != nextDecodeCleanupCompleted;
        decodeCleanupCompleted = nextDecodeCleanupCompleted;
    }

    private void retainEnd(RequestEndEvent evidence) {
        if (requestEndResult != null) {
            var nextExpired = expired || evidence.kind() == EndKind.INACTIVITY;
            stateChanged |= expired != nextExpired;
            expired = nextExpired;
            return;
        }
        if (activeSelectionId != 0L) {
            if (pendingEndEvent == null
                    || pendingEndEvent.kind() == EndKind.ENDPOINT_CLOSED
                    || evidence.kind() == EndKind.WORKER && pendingEndEvent.kind() != EndKind.WORKER
                    || evidence.kind() == EndKind.ENDPOINT_CLOSED
                            && pendingEndEvent.kind() != EndKind.WORKER) {
                stateChanged |= !Objects.equals(pendingEndEvent, evidence);
                pendingEndEvent = evidence;
            }
            return;
        }
        if (pendingEndEvent == null
                || evidence.kind() == EndKind.WORKER && pendingEndEvent.kind() != EndKind.WORKER) {
            pendingEndEvent = evidence;
        }
        // An end result moves OPEN to ENDING; resource cleanup and notifications gate CLOSED.
        requestEndResult = buildResult(pendingEndEvent);
        expired |= pendingEndEvent.kind() == EndKind.INACTIVITY;
        pendingEndEvent = null;
        stateChanged = true;
    }

    private RequestResult buildResult(RequestEndEvent evidence) {
        // Preserve the first cancellation cause, except when a committed route reports FAILURE.
        // That failure keeps its own result instead of being replaced by the cancellation.
        boolean failureAfterRouteCommitted =
                evidence.kind() == EndKind.FAILURE && routeCommittedAtMs >= 0L;
        if (cancellationReason != null && !failureAfterRouteCommitted) {
            var state = RequestResult.State.CANCELLED;
            if (cancellationReason == CancelReason.DEADLINE_EXCEEDED) {
                state = RequestResult.State.TIMED_OUT;
            }
            return new RequestResult(evidence.kind(), state, cancellationError, cancellationDetail);
        }

        // Inactivity without an earlier cancellation ends the request as TIMED_OUT.
        if (evidence.kind() == EndKind.INACTIVITY) {
            return new RequestResult(
                    evidence.kind(),
                    RequestResult.State.TIMED_OUT,
                    timeoutError,
                    evidence.detail());
        }

        // Worker/frontend success produces COMPLETED when no overriding cancellation applies.
        boolean completionReported =
                evidence.kind() == EndKind.WORKER || evidence.kind() == EndKind.FRONTEND;
        if (completionReported && evidence.success()) {
            String detail =
                    evidence.detail() == null || evidence.detail().isBlank()
                            ? "request completed"
                            : evidence.detail();
            return new RequestResult(evidence.kind(), RequestResult.State.COMPLETED, null, detail);
        }
        // Worker-reported preemption produces CANCELLED; other unsuccessful endings produce FAILED.
        var state = RequestResult.State.FAILED;
        if (evidence.kind() == EndKind.PRIORITY) {
            state = RequestResult.State.CANCELLED;
        }
        var error = evidence.errorType();
        if (error == null) {
            error =
                    switch (evidence.kind()) {
                        case PRIORITY -> StrategyErrorType.PRIORITY_PREEMPTED;
                        case WORKER -> StrategyErrorType.WORKER_EXECUTION_FAILED;
                        default -> StrategyErrorType.DISPATCH_FAILED;
                    };
        }
        String detail = evidence.detail();
        if (detail == null || detail.isBlank()) {
            detail =
                    switch (evidence.kind()) {
                        case WORKER -> "worker error code " + evidence.errorCode();
                        case PRIORITY -> "priority victim canceled by worker";
                        default -> "request ended";
                    };
        }
        return new RequestResult(evidence.kind(), state, error, detail);
    }

    private Snapshot commitSnapshot(long nowMs) {
        refreshState(nowMs);
        return snapshot();
    }

    /**
     * Archives a finished request when cleanup and response selection allow it, then updates timers
     * and change time.
     */
    private void refreshState(long nowMs) {
        if (requestEndResult != null
                && requestRecordForArchive == null
                && prefillCleanupCompleted
                && decodeCleanupCompleted
                && notificationsDone
                && (responseForFuture != null || future.isDone())
                && activeSelectionId == 0L) {
            var nextRequestRecordForArchive =
                    new RequestResult(
                            registeredRequestId,
                            originalNumericRequestId,
                            phase,
                            requestEndResult.state(),
                            timing.registeredAtMs(),
                            nowMs,
                            requestEndResult.detail(),
                            getRequestScheduler());
            stateChanged = true;
            requestRecordForArchive = nextRequestRecordForArchive;
            route = null;
        }
        long inactivityAtMs =
                saturatedAdd(
                        Math.max(lastWorkerAtMs, Math.max(0L, routeCommittedAtMs)),
                        inactivityTimeoutMs);
        if (registeredRequestId != null) {
            long requestAt =
                    requestEndResult == null && !requestDeadlineConsumed && routeCommittedAtMs < 0L
                            ? getRequestExpiresAtMs()
                            : 0L;
            long visibilityAt = requestEndResult == null ? visibilityAtMs : 0L;
            long inactivityAt = requestRecordForArchive != null || expired ? 0L : inactivityAtMs;
            var nextDeadlines = deadlines.schedule(this, requestAt, visibilityAt, inactivityAt);
            stateChanged |= !Objects.equals(deadlines, nextDeadlines);
            deadlines = nextDeadlines;
        }
        if (stateChanged) {
            updatedAtMs = Math.max(updatedAtMs, nowMs);
            stateChanged = false;
        }
    }

    private boolean matchesSelection(SelectionAttempt attempt) {
        return attempt != null
                && attempt.requestContext() == this
                && activeSelectionId != 0L
                && activeSelectionId == attempt.id();
    }

    private void recordCancellation(CancelReason reason, StrategyErrorType error, String detail) {
        stateChanged = true;
        cancellationReason = reason;
        cancellationError = error;
        cancellationDetail = detail;
    }

    /**
     * Request observation times: AtMs values use Unix milliseconds, Nanos values use the monotonic clock.
     * ElapsedUs and DelayMs are durations; zero timestamps mean that the stage has not been recorded.
     */
    @lombok.With
    public record Timing(
            long startedAtMs,
            long startedNanos,
            long grpcReceivedAtMs,
            long grpcReceivedNanos,
            long routeCommitStartedNanos,
            long registeredAtMs,
            long totalElapsedUs,
            Long arrivalDelayMs) {}

    /**
     * Diagnostic score and request count of the committed preemption plan; cost is not a duration.
     */
    public record PreemptionStatistics(long cost, int requestCount) {}

    /**
     * Captures routing diagnostics with copied maps so later decisions cannot change the captured view.
     */
    public record RoutingTelemetry(
            String cacheSource,
            long cacheQueryUs,
            int cacheQueryCount,
            Map<RoleType, CacheMatchSelection> cacheSelections,
            Map<RoleType, RoutingDecision> routingDecisions,
            Map<RoleType, Integer> routingAttempts) {
        public RoutingTelemetry {
            cacheSelections = Map.copyOf(cacheSelections);
            routingDecisions = Map.copyOf(routingDecisions);
            routingAttempts = Map.copyOf(routingAttempts);
        }
    }

    public record CacheMatchSelection(RoleType role, String selectedIp, long hitCacheTokens) {}

    public enum Phase {
        OPEN,
        ENDING,
        CLOSED
    }

    /**
     * Tracks selection and completion of the current routing work.
     */
    public enum RoutingPhase {
        NONE,
        SELECTING
    }

    public enum ResponseKind {
        ACK,
        TERMINAL,
        CANCELLATION
    }

    public enum EndKind {
        FAILURE,
        WORKER,
        ENDPOINT_CLOSED,
        PRIORITY,
        INACTIVITY,
        SHUTDOWN,
        FRONTEND
    }

    /**
     * Carries an ending cause for one exact route; resource cleanup is recorded separately on the context.
     */
    public record RequestEndEvent(
            RequestResult route,
            EndKind kind,
            boolean success,
            long errorCode,
            StrategyErrorType errorType,
            String detail) {
        public static RequestEndEvent publicationFailure(RequestResult route, Response response) {
            var error = StrategyErrorType.DISPATCH_FAILED;
            if (response != null) {
                for (var candidate : StrategyErrorType.values()) {
                    if (candidate.getErrorCode() == response.getCode()) {
                        error = candidate;
                        break;
                    }
                }
            }
            return failure(
                    route,
                    EndKind.FAILURE,
                    error,
                    response == null
                            ? "route publication did not commit"
                            : response.getErrorMessage());
        }

        public static RequestEndEvent failure(
                RequestResult route, EndKind kind, StrategyErrorType errorType, String detail) {
            return new RequestEndEvent(route, kind, false, 0L, errorType, detail);
        }

        public static RequestEndEvent externalFailure(
                RequestResult route, Response response, Throwable error) {
            String detail =
                    error != null
                            ? "external future failure: " + error.getMessage()
                            : response != null && response.getErrorMessage() != null
                                    ? response.getErrorMessage()
                                    : "external future completion";
            return failure(route, EndKind.FRONTEND, StrategyErrorType.DISPATCH_FAILED, detail);
        }
    }

    /**
     * Identifies a report or cleanup for one route. DiscoveryEndpoint closure requires its exact source endpoint.
     * Remote cleanup applies to the committed route as a whole and does not use role, endpoint or
     * reservation.
     */
    public record RequestEvent(
            RequestResult route,
            DecodeEndpoint.RequestRegistration reservation,
            RoleType role,
            WorkerEndpoint endpoint,
            RequestEventType type,
            long errorCode,
            String detail,
            long nowMs) {
        boolean terminal() {
            return type == RequestEventType.COMPLETED
                    || type == RequestEventType.FAILED
                    || type == RequestEventType.PRIORITY_CANCELLED;
        }

        RequestEndEvent end() {
            EndKind end =
                    switch (type) {
                        case PRIORITY_CANCELLED -> EndKind.PRIORITY;
                        case ENDPOINT_CLOSED -> EndKind.ENDPOINT_CLOSED;
                        default -> EndKind.WORKER;
                    };
            StrategyErrorType error =
                    switch (end) {
                        case PRIORITY -> StrategyErrorType.PRIORITY_PREEMPTED;
                        case ENDPOINT_CLOSED -> StrategyErrorType.DISPATCH_FAILED;
                        default -> StrategyErrorType.WORKER_EXECUTION_FAILED;
                    };
            return new RequestEndEvent(
                    route, end, type == RequestEventType.COMPLETED, errorCode, error, detail);
        }
    }

    /**
     * Request facts read together for closing, response selection and diagnostics.
     * Worker observations and state-change revisions remain private to the context.
     */
    public record Snapshot(
            String requestId,
            long originalNumericRequestId,
            org.flexlb.balance.schedule.RequestPhase requestPhase,
            long createdAtMs,
            RequestResult route,
            long selectionId,
            long routeCommittedAtMs,
            boolean prefillWorkerReleaseConfirmed,
            boolean decodeWorkerReleaseConfirmed,
            boolean decodeAccepted,
            boolean visibilityOverdue,
            CancelReason cancellationReason,
            String cancellationDetail,
            RequestResult requestEndResult,
            RequestResult responseForFuture,
            boolean prefillCleanupCompleted,
            boolean decodeCleanupCompleted,
            boolean notificationsDone,
            ExpirationTimer.Plan deadlines,
            RequestResult requestRecordForArchive,
            long updatedAtMs,
            long inactivityExpiresAtMs)
            implements org.flexlb.worker.endpoint.WorkerRequestSnapshot {
        public RoutingPhase routingPhase() {
            return selectionId == 0L ? RoutingPhase.NONE : RoutingPhase.SELECTING;
        }

        public RequestResult committedRoute() {
            return routeCommittedAtMs >= 0L ? route : null;
        }

        public Phase phase() {
            return requestRecordForArchive != null
                    ? Phase.CLOSED
                    : requestEndResult != null ? Phase.ENDING : Phase.OPEN;
        }
    }

    /**
     * A read-only request reference and selection number; only RequestContext owns selection state.
     */
    public record SelectionAttempt(RequestContext requestContext, long id) {
        public SelectionAttempt {
            Objects.requireNonNull(requestContext, "requestContext");
            if (id <= 0L) throw new IllegalArgumentException("positive selection id required");
        }
    }
}

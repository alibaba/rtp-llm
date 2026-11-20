package org.flexlb.balance.schedule;

import io.opentelemetry.context.Context;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.balance.request.Request;
import org.flexlb.balance.request.Response;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.RequestEventType;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerRequestEvent;

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

    @Getter(lombok.AccessLevel.PACKAGE)
    @Setter(lombok.AccessLevel.PACKAGE)
    private volatile String routingGroup;

    @Getter @Setter private volatile Map<String, Object> schedulingDiagnostics;
    @Getter @Setter private Long requestMessageBytes;

    private final EnumMap<RoleType, CacheMatchSelection> cacheSelections =
            new EnumMap<>(RoleType.class);
    private final EnumMap<RoleType, Integer> routingAttempts = new EnumMap<>(RoleType.class);

    private long updatedAtMs;

    private String registeredRequestId;
    private RequestRoute route;

    private long activeSelectionId;

    private volatile DecodeEndpoint.ResourceWait currentResourceWait;

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
    private RequestState.State endState;
    private EndCause endCause;
    private StrategyErrorType endError;
    private String endDetail;
    private RequestEndEvent pendingEndEvent;
    // Set when an endpoint closes during selection; cleared after rollback and execution recheck.
    private String selectionEndpointCloseDetail;
    private ResponseCompletionExecutor.Registration responseForFuture;

    private boolean prefillCleanupCompleted = true;
    private boolean decodeCleanupCompleted = true;
    private boolean notificationsDone;

    private CleanupExecution cleanupExecution = CleanupExecution.IDLE;
    private ResponseCompletionExecutor.Registration cleanupResponse;

    private boolean expired;

    private ExpirationTimer.Plan deadlines = new ExpirationTimer.Plan(0L, null, null, null);
    private RequestState requestRecordForArchive;

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
        long inactivityAtMs = inactivityExpiresAtMs();
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
                expired,
                cancellationReason,
                cancellationDetail,
                endState,
                endCause,
                endError,
                endDetail,
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

    public synchronized void recordCacheSelection(
            RoleType role, String selectedIp, long hitCacheTokens) {
        cacheSelections.put(role, new CacheMatchSelection(role, selectedIp, hitCacheTokens));
    }

    public synchronized void beginRoutingAttempt(RoleType role) {
        routingAttempts.merge(role, 1, Integer::sum);
        cacheSelections.remove(role);
    }

    public synchronized RoutingTelemetry getRoutingTelemetry() {
        return new RoutingTelemetry(cacheSelections, routingAttempts);
    }

    public synchronized SelectionAttempt currentSelectionAttempt() {
        return activeSelectionId == 0 ? null : new SelectionAttempt(this, activeSelectionId);
    }

    public boolean isResponseDone() {
        return future != null && future.isDone();
    }

    void recordPreemptionStatistics(long cost, int requestCount) {
        preemptionStatistics = new PreemptionStatistics(cost, requestCount);
    }

    /**
     * Starts selection only while this registered request is still waiting for its first route.
     */
    synchronized SelectionAttempt beginRouteSelection() {
        if (!canBeginRouteSelection()) return null;
        long id = SELECTION_IDS.incrementAndGet();
        checkState(id > 0, "selection id exhausted");
        activeSelectionId = id;
        schedulingDiagnostics = null;
        return new SelectionAttempt(this, id);
    }

    synchronized boolean canBeginRouteSelection() {
        return registeredRequestId != null
                && endState == null
                && cancellationReason == null
                && pendingEndEvent == null
                && activeSelectionId == 0
                && route == null;
    }

    synchronized boolean bindResourceWait(
            SelectionAttempt exact, DecodeEndpoint.ResourceWait wait) {
        if (!canPublishRoute(exact) || isResponseDone() || currentResourceWait != null)
            return false;
        checkState(
                wait.request() == this && wait.selectionId() == exact.id(),
                "foreign resource wait");
        currentResourceWait = wait;
        return true;
    }

    synchronized boolean canPublishRoute(SelectionAttempt exact) {
        return registeredRequestId != null
                && endState == null
                && cancellationReason == null
                && pendingEndEvent == null
                && selectionEndpointCloseDetail == null
                && routeCommittedAtMs < 0
                && matchesSelection(exact);
    }

    synchronized boolean clearResourceWait(DecodeEndpoint.ResourceWait exact) {
        if (currentResourceWait != exact) return false;
        revokeResourceWait();
        return true;
    }

    /**
     * Ends selection after rollback. An endpoint disappearing before publication permits a retry;
     * cancellation and worker failures remain request endings.
     */
    synchronized Snapshot finishSelectionWithoutCommit(
            SelectionAttempt exact, Response failure, long nowMs) {
        if (!matchesSelection(exact)) return null;
        activeSelectionId = 0;
        revokeResourceWait();
        boolean rolledBackBeforePublication =
                route == null
                        && prefillCleanupCompleted
                        && decodeCleanupCompleted
                        && routeCommittedAtMs < 0
                        && !decodeAccepted;
        RequestEndEvent ending =
                failure == null ? null : RequestEndEvent.publicationFailure(route, failure);
        if (failure != null
                && endState == null
                && cancellationReason == null
                && pendingEndEvent != null
                && pendingEndEvent.cause() == EndCause.SCHEDULER_SHUTDOWN) pendingEndEvent = null;
        if (ending == null && selectionEndpointCloseDetail != null && !rolledBackBeforePublication)
            ending =
                    RequestEndEvent.failure(
                            route,
                            EndCause.ENDPOINT_CLOSED,
                            StrategyErrorType.DISPATCH_FAILED,
                            selectionEndpointCloseDetail);
        selectionEndpointCloseDetail = null;
        return endRequest(ending, nowMs, null);
    }

    /**
     * Accepts one request ending and determines the result after selection exits. Prior cancellation
     * still determines the result, except for failure of an already committed response.
     */
    synchronized Snapshot endRequest(
            RequestEndEvent evidence,
            long nowMs,
            ResponseCompletionExecutor.Registration responseCandidate) {
        if (registeredRequestId == null
                || requestRecordForArchive != null
                || evidence != null && evidence.route() != null && evidence.route() != route)
            return snapshot();
        if (evidence != null) revokeResourceWait();
        if (endState != null) {
            expired |= evidence != null && evidence.cause().isTimeout();
        } else {
            var accepted = pendingEndEvent;
            if (evidence != null
                    && (accepted == null
                            || evidence.cause().precedence() > accepted.cause().precedence()
                            || evidence.cause() == EndCause.ROUTE_PUBLICATION_FAILED
                                    && routeCommittedAtMs >= 0L)) accepted = evidence;
            if (activeSelectionId != 0) {
                pendingEndEvent = accepted;
            } else if (accepted != null) {
                endCause = accepted.cause();
                boolean committedResponseFailed =
                        endCause == EndCause.ROUTE_PUBLICATION_FAILED && routeCommittedAtMs >= 0L;
                if (cancellationReason != null && !committedResponseFailed) {
                    endState =
                            cancellationReason == CancelReason.DEADLINE_EXCEEDED
                                    ? RequestState.State.TIMED_OUT
                                    : RequestState.State.CANCELLED;
                    endError = cancellationError;
                    endDetail = cancellationDetail;
                } else {
                    endState =
                            switch (endCause) {
                                case WORKER_COMPLETED -> RequestState.State.COMPLETED;
                                case PRIORITY_PREEMPTED -> RequestState.State.CANCELLED;
                                case WORKER_REPORT_TIMEOUT, REQUEST_DEADLINE_EXCEEDED ->
                                        RequestState.State.TIMED_OUT;
                                default -> RequestState.State.FAILED;
                            };
                    endError = endCause == EndCause.WORKER_COMPLETED ? null : accepted.errorType();
                    if (endError == null && endState != RequestState.State.COMPLETED)
                        endError =
                                switch (endCause) {
                                    case WORKER_FAILED -> StrategyErrorType.WORKER_EXECUTION_FAILED;
                                    case PRIORITY_PREEMPTED -> StrategyErrorType.PRIORITY_PREEMPTED;
                                    case WORKER_REPORT_TIMEOUT, REQUEST_DEADLINE_EXCEEDED ->
                                            timeoutError;
                                    default -> StrategyErrorType.DISPATCH_FAILED;
                                };
                    endDetail = accepted.detail();
                    if (endDetail == null || endDetail.isBlank())
                        endDetail =
                                switch (endCause) {
                                    case WORKER_COMPLETED -> "request completed";
                                    case PRIORITY_PREEMPTED -> "priority victim canceled by worker";
                                    default -> "request ended";
                                };
                }
                expired |= endCause.isTimeout();
                pendingEndEvent = null;
            }
        }
        selectResponse(responseCandidate);
        refreshState(nowMs);
        return snapshot();
    }

    /**
     * Invalidates selection without ending the request. Rollback rechecks execution before retrying.
     */
    synchronized void invalidateSelection(String detail) {
        if (selectionEndpointCloseDetail == null)
            selectionEndpointCloseDetail =
                    detail == null || detail.isBlank() ? "endpoint closed" : detail;
        revokeResourceWait();
    }

    /**
     * Validates a report and updates execution times and resource-release facts. The scheduler decides
     * whether the observed role ends the request while holding the same request monitor.
     */
    synchronized boolean updateWorkerStatus(RequestRoute expected, WorkerRequestEvent event) {
        if (requestRecordForArchive != null
                || route != expected
                || route == null
                || event.role() == null) return false;
        var selected = route.role(event.role());
        if (selected == null
                || selected.registration() != event.registration()
                || selected.endpoint() != event.endpoint()) return false;
        var type = event.type();
        if (type == RequestEventType.REMOTE_CLEANED || type == RequestEventType.REMOTE_FENCED)
            return false;
        var role = event.role();
        var nowMs = event.nowMs();
        boolean decode = role == RoleType.DECODE;
        boolean changed = false;
        if (type != RequestEventType.ENDPOINT_CLOSED) {
            if (decode && !decodeAccepted) {
                decodeAccepted = true;
                visibilityAtMs = 0;
                visibilityOverdue = false;
                changed = true;
            }
            if (!decode && type == RequestEventType.COMPLETED && prefillCompletedAtMs == 0) {
                prefillCompletedAtMs = nowMs;
                if (!decodeAccepted && route.decodeEndpoint() != null)
                    visibilityAtMs = saturatedAdd(nowMs, 10_000L);
                visibilityOverdue = false;
                changed = true;
            }
            if (nowMs > lastWorkerAtMs) {
                lastWorkerAtMs = nowMs;
                changed = true;
            }
        }
        if (type == RequestEventType.COMPLETED
                || type == RequestEventType.FAILED
                || type == RequestEventType.PRIORITY_CANCELLED
                || type == RequestEventType.ENDPOINT_CLOSED) {
            changed |= decode ? !decodeCleanupCompleted : !prefillCleanupCompleted;
            if (routeCommittedAtMs >= 0) changed |= confirmWorkerRelease(!decode, decode);
            if (decode) decodeCleanupCompleted = true;
            else prefillCleanupCompleted = true;
        }
        if (changed) refreshState(nowMs);
        return true;
    }

    synchronized boolean recordRemoteCleanup(RequestRoute expected, boolean cleaned, long nowMs) {
        if (requestRecordForArchive != null
                || route == null
                || route != expected
                || routeCommittedAtMs < 0) return false;
        var previousDeadlines = deadlines;
        boolean changed = confirmWorkerRelease(true, cleaned);
        if (changed) refreshState(nowMs);
        return previousDeadlines != deadlines
                || changed && (pendingEndEvent != null || endState != null);
    }

    synchronized boolean requestCancel(
            CancelReason reason,
            boolean frontend,
            long nowMs,
            StrategyErrorType errorType,
            String detail,
            ResponseCompletionExecutor.Registration responseCandidate) {
        Objects.requireNonNull(reason);
        if (registeredRequestId == null || endState != null || cancellationReason != null)
            return false;
        if (reason == CancelReason.DEADLINE_EXCEEDED && routeCommittedAtMs >= 0) return false;
        if (frontend && (responseForFuture != null || routeCommittedAtMs >= 0)) return false;
        recordCancellation(
                reason,
                Objects.requireNonNull(errorType),
                detail == null ? reason.getMessage() : detail);
        endRequest(
                RequestEndEvent.failure(
                        route,
                        EndCause.CANCELLATION_REQUESTED,
                        cancellationError,
                        cancellationDetail),
                nowMs,
                responseCandidate);
        return true;
    }

    synchronized Snapshot recordDeadline(
            ExpirationTimer.Token exact, long nowMs, String diagnosis) {
        if (exact.owner() != this
                || deadlines.token(exact.kind()) != exact
                || requestRecordForArchive != null) return null;
        deadlines = deadlines.without(exact.kind());
        if (nowMs < exact.atMs()) {
            refreshState(nowMs);
            return snapshot();
        }
        if (exact.kind() == ExpirationTimer.Kind.VISIBILITY) {
            visibilityAtMs = 0;
            visibilityOverdue = true;
        } else {
            boolean inactive = exact.kind() == ExpirationTimer.Kind.INACTIVITY;
            if (inactive && nowMs < inactivityExpiresAtMs()) {
                refreshState(nowMs);
                return snapshot();
            }
            if (inactive) expired = true;
            else requestDeadlineConsumed = true;
            boolean requestAlreadyCommitted =
                    !inactive
                            && (endState != null
                                    || routeCommittedAtMs >= 0 && activeSelectionId == 0);
            if (!requestAlreadyCommitted) {
                String detail =
                        inactive ? "request inactive" : "request scheduling deadline exceeded";
                if (cancellationReason == null && (endState == null || inactive)) {
                    // A successful response can still be waiting for the other role to release.
                    // Its inactivity deadline starts cancellation without changing that response.
                    notificationsDone = false;
                    recordCancellation(
                            CancelReason.DEADLINE_EXCEEDED,
                            timeoutError,
                            diagnosis == null ? detail : diagnosis);
                }
                if (inactive || activeSelectionId != 0 || routeCommittedAtMs < 0 && !decodeAccepted)
                    return endRequest(
                            RequestEndEvent.failure(
                                    route,
                                    inactive
                                            ? EndCause.WORKER_REPORT_TIMEOUT
                                            : EndCause.REQUEST_DEADLINE_EXCEEDED,
                                    timeoutError,
                                    detail),
                            nowMs,
                            null);
            }
        }
        refreshState(nowMs);
        return snapshot();
    }

    synchronized void recordRouteCommitStart(long startedNanos) {
        timing = timing.withRouteCommitStartedNanos(startedNanos);
    }

    synchronized void registerSelectedRoute(SelectionAttempt exact, RequestRoute acquired) {
        checkState(matchesSelection(exact), "route requires its current selection");
        checkState(
                acquired.ctx() == this
                        && Objects.equals(acquired.requestId(), registeredRequestId)
                        && (route == null || route == acquired),
                "foreign route");
        if (route == acquired) return;
        route = acquired;
        prefillCleanupCompleted =
                acquired.prefillEndpoint() == null && acquired.encoderEndpoint() == null;
        decodeCleanupCompleted = acquired.decodeRegistration() == null;
    }

    synchronized boolean tryCommitRoute(
            SelectionAttempt exact,
            RequestRoute acquired,
            ResponseCompletionExecutor.Registration response,
            long nowMs,
            long predictedVisibilityAtMs) {
        if (!canPublishRoute(exact) || route != acquired || responseForFuture != null) return false;
        if (getRequestExpiresAtMs() > 0 && nowMs >= getRequestExpiresAtMs()) {
            recordCancellation(
                    CancelReason.DEADLINE_EXCEEDED,
                    timeoutError,
                    "request expired before route commit");
            endRequest(
                    RequestEndEvent.failure(
                            route,
                            EndCause.REQUEST_DEADLINE_EXCEEDED,
                            timeoutError,
                            cancellationDetail),
                    nowMs,
                    null);
            return false;
        }
        routeCommittedAtMs = nowMs;
        visibilityAtMs = predictedVisibilityAtMs;
        activeSelectionId = 0;
        revokeResourceWait();
        responseForFuture = response;
        decodeWorkerReleaseConfirmed = acquired.decodeRegistration() == null;
        refreshState(nowMs);
        return true;
    }

    /**
     * Claims cleanup execution or asks the current executor to check the latest facts again.
     * The response has already been prepared and belongs to this cleanup path.
     */
    synchronized boolean requestCleanup(ResponseCompletionExecutor.Registration response) {
        if (response != null) cleanupResponse = response;
        if (cleanupExecution != CleanupExecution.IDLE) {
            cleanupExecution = CleanupExecution.RUN_AGAIN;
            return false;
        }
        cleanupExecution = CleanupExecution.RUNNING;
        return true;
    }

    synchronized boolean finishCleanupPass() {
        if (cleanupExecution == CleanupExecution.RUN_AGAIN) {
            cleanupExecution = CleanupExecution.RUNNING;
            return false;
        }
        cleanupExecution = CleanupExecution.IDLE;
        return true;
    }

    synchronized ResponseCompletionExecutor.Registration rejectCleanup() {
        cleanupExecution = CleanupExecution.IDLE;
        return takeCleanupResponse();
    }

    synchronized ResponseCompletionExecutor.Registration takeCleanupResponse() {
        var response = cleanupResponse;
        cleanupResponse = null;
        return response;
    }

    synchronized void recordCleanupProgress(
            RequestRoute exact,
            boolean prefillReleased,
            boolean decodeReleased,
            boolean notificationsSubmitted,
            long nowMs) {
        if (route != exact || requestRecordForArchive != null) return;
        prefillCleanupCompleted |= prefillReleased;
        decodeCleanupCompleted |= decodeReleased;
        if (endState != null) notificationsDone |= notificationsSubmitted;
        else if (routeCommittedAtMs < 0 && prefillCleanupCompleted && decodeCleanupCompleted)
            route = null;
        refreshState(nowMs);
    }

    synchronized void recordRoutingDiagnostics(
            SelectionAttempt exact, Map<String, Object> diagnostics) {
        if (matchesSelection(exact) && cancellationReason == null)
            schedulingDiagnostics = diagnostics == null ? null : Map.copyOf(diagnostics);
    }

    synchronized void register(
            AbstractRequestScheduler owner,
            RequestFuture response,
            long nowMs,
            long inactivityTimeoutMs,
            StrategyErrorType timeoutError) {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(schedulingMetadata, "scheduling metadata required");
        if (inactivityTimeoutMs <= 0)
            throw new IllegalArgumentException("positive inactivity timeout required");
        String id = Objects.requireNonNull(getRequestId(), "requestId");
        Objects.requireNonNull(timeoutError, "timeout error required");
        checkState(registeredRequestId == null, "request already registered");
        requestScheduler = owner;
        future = response;
        registeredRequestId = id;
        this.timeoutError = timeoutError;
        timing = timing.withRegisteredAtMs(nowMs);
        lastWorkerAtMs = nowMs;
        this.inactivityTimeoutMs = inactivityTimeoutMs;
        refreshState(nowMs);
    }

    synchronized void releaseSchedulingInput() {
        if (request != null
                && (!request.getBlockCacheKeys().isEmpty()
                        || !request.getCacheAffinityKeys().isEmpty()))
            request =
                    request.toBuilder()
                            .blockCacheKeys(java.util.List.of())
                            .cacheAffinityKeys(java.util.List.of())
                            .build();
    }

    private boolean matchesSelection(SelectionAttempt exact) {
        return exact != null
                && exact.requestContext() == this
                && activeSelectionId != 0
                && activeSelectionId == exact.id();
    }

    private boolean confirmWorkerRelease(boolean prefillReleased, boolean decodeReleased) {
        boolean changed =
                prefillReleased && !prefillWorkerReleaseConfirmed
                        || decodeReleased && !decodeWorkerReleaseConfirmed;
        prefillWorkerReleaseConfirmed |= prefillReleased;
        decodeWorkerReleaseConfirmed |= decodeReleased;
        return changed;
    }

    private void selectResponse(ResponseCompletionExecutor.Registration candidate) {
        if (candidate == null || candidate.requestContext() != this || responseForFuture != null)
            return;
        if (candidate.responseKind() == ResponseKind.ACK
                || candidate.responseKind() == ResponseKind.TERMINAL && endState == null
                || candidate.responseKind() == ResponseKind.CANCELLATION
                        && cancellationReason == null) return;
        responseForFuture = candidate;
    }

    private void recordCancellation(CancelReason reason, StrategyErrorType error, String detail) {
        revokeResourceWait();
        cancellationReason = reason;
        cancellationError = error;
        cancellationDetail = detail;
    }

    private void revokeResourceWait() {
        if (currentResourceWait != null) {
            currentResourceWait.revoke();
            currentResourceWait = null;
        }
    }

    /**
     * Archives only after selection, local cleanup, cancellation submission and response selection finish.
     * Deadline tokens are replaced only when their scheduled time changes.
     */
    private void refreshState(long nowMs) {
        if (endState != null
                && requestRecordForArchive == null
                && prefillCleanupCompleted
                && decodeCleanupCompleted
                && notificationsDone
                && (responseForFuture != null || future.isDone())
                && activeSelectionId == 0) {
            requestRecordForArchive =
                    new RequestState(
                            registeredRequestId,
                            originalNumericRequestId,
                            phase,
                            endState,
                            timing.registeredAtMs(),
                            nowMs,
                            endDetail);
            route = null;
        }
        if (registeredRequestId != null) {
            long requestAt =
                    endState == null && !requestDeadlineConsumed && routeCommittedAtMs < 0
                            ? getRequestExpiresAtMs()
                            : 0;
            long visibilityAt = endState == null ? visibilityAtMs : 0;
            long inactivityAt =
                    requestRecordForArchive != null || expired || routeCommittedAtMs < 0
                            ? 0
                            : inactivityExpiresAtMs();
            deadlines = deadlines.schedule(this, requestAt, visibilityAt, inactivityAt);
        }
        updatedAtMs = Math.max(updatedAtMs, nowMs);
    }

    private long inactivityExpiresAtMs() {
        if (routeCommittedAtMs < 0) return Long.MAX_VALUE;
        return saturatedAdd(
                Math.max(lastWorkerAtMs, Math.max(0L, routeCommittedAtMs)), inactivityTimeoutMs);
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
            Map<RoleType, CacheMatchSelection> cacheSelections,
            Map<RoleType, Integer> routingAttempts) {
        public RoutingTelemetry {
            cacheSelections = Map.copyOf(cacheSelections);
            routingAttempts = Map.copyOf(routingAttempts);
        }
    }

    public record CacheMatchSelection(RoleType role, String selectedIp, long hitCacheTokens) {}

    public enum Phase {
        OPEN,
        ENDING,
        CLOSED
    }

    public enum ResponseKind {
        ACK,
        TERMINAL,
        CANCELLATION
    }

    /**
     * Identifies why the request ends after role-specific reports have been interpreted.
     */
    public enum EndCause {
        ROUTE_PUBLICATION_FAILED,
        WORKER_COMPLETED,
        WORKER_FAILED,
        PRIORITY_PREEMPTED,
        ENDPOINT_CLOSED,
        WORKER_REPORT_TIMEOUT,
        REQUEST_DEADLINE_EXCEEDED,
        SCHEDULER_SHUTDOWN,
        CANCELLATION_REQUESTED,
        EXTERNAL_RESPONSE_FAILED;

        boolean isTimeout() {
            return this == WORKER_REPORT_TIMEOUT || this == REQUEST_DEADLINE_EXCEEDED;
        }

        // Only pending endings compete. Execution results outrank failures; closure is the
        // fallback.
        private int precedence() {
            return switch (this) {
                case WORKER_COMPLETED, WORKER_FAILED -> 3;
                case PRIORITY_PREEMPTED -> 2;
                case ENDPOINT_CLOSED -> 0;
                default -> 1;
            };
        }
    }

    /**
     * Carries a request ending cause for one exact route; resource cleanup is recorded separately.
     */
    public record RequestEndEvent(
            RequestRoute route, EndCause cause, StrategyErrorType errorType, String detail) {
        public static RequestEndEvent publicationFailure(RequestRoute route, Response response) {
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
                    EndCause.ROUTE_PUBLICATION_FAILED,
                    error,
                    response == null
                            ? "route publication did not commit"
                            : response.getErrorMessage());
        }

        public static RequestEndEvent failure(
                RequestRoute route, EndCause cause, StrategyErrorType errorType, String detail) {
            return new RequestEndEvent(route, cause, errorType, detail);
        }

        public static RequestEndEvent externalFailure(
                RequestRoute route, Response response, Throwable error) {
            String detail =
                    error != null
                            ? "external future failure: " + error.getMessage()
                            : response != null && response.getErrorMessage() != null
                                    ? response.getErrorMessage()
                                    : "external future completion";
            return failure(
                    route,
                    EndCause.EXTERNAL_RESPONSE_FAILED,
                    StrategyErrorType.DISPATCH_FAILED,
                    detail);
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
            RequestRoute route,
            long selectionId,
            long routeCommittedAtMs,
            boolean prefillWorkerReleaseConfirmed,
            boolean decodeWorkerReleaseConfirmed,
            boolean decodeAccepted,
            boolean visibilityOverdue,
            boolean inactivityExpired,
            CancelReason cancellationReason,
            String cancellationDetail,
            RequestState.State endState,
            EndCause endCause,
            StrategyErrorType endError,
            String endDetail,
            ResponseCompletionExecutor.Registration responseForFuture,
            boolean prefillCleanupCompleted,
            boolean decodeCleanupCompleted,
            boolean notificationsDone,
            ExpirationTimer.Plan deadlines,
            RequestState requestRecordForArchive,
            long updatedAtMs,
            long inactivityExpiresAtMs) {
        public boolean isSelectingRoute() {
            return selectionId != 0L;
        }

        public RequestRoute committedRoute() {
            return routeCommittedAtMs >= 0L ? route : null;
        }

        public Phase phase() {
            return requestRecordForArchive != null
                    ? Phase.CLOSED
                    : endState != null ? Phase.ENDING : Phase.OPEN;
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

    private enum CleanupExecution {
        IDLE,
        RUNNING,
        RUN_AGAIN
    }
}

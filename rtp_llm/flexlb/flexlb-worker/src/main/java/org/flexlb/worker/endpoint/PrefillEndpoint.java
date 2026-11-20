package org.flexlb.worker.endpoint;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.Failures;
import org.flexlb.util.formula.PrefillTimePredictor;
import org.flexlb.worker.status.PriorityPreemptionProgress;
import org.flexlb.worker.status.WorkerStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;
import java.util.function.LongSupplier;
import java.util.function.Predicate;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Maintains per-request compute tokens and resource capacity for one worker generation.
 * Reconciles worker reports under the state lock, then publishes scheduler notifications outside it.
 */
public class PrefillEndpoint extends WorkerEndpoint
        implements EndpointStats.Prefill, SelectionRollback {
    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final Map<String, RequestRegistration> requests = new HashMap<>();
    private final LongSupplier clock = System::currentTimeMillis;

    private long registeredComputeTokens;
    private long reportedUnknownComputeTokens;
    private long reportedExtraComputeTokens;
    private long resourceVersion;
    private long reportedUnknownRequests;
    private long reportedUnknownDurationMs;
    private long unobservedRequests;

    private final int maxInflightRequests;

    private final Long engineWaitingThreshold;

    private long reportedWaitingRequests = -1L;

    private final PrefillTimePredictor predictor = new PrefillTimePredictor();
    private final CapacityNotifier capacityNotifier;

    private SelectionSnapshot selectionSnapshotCache;

    PrefillEndpoint(WorkerStatus status, FlexlbConfig config, CapacityNotifier capacityNotifier) {
        super(status);
        this.capacityNotifier = Objects.requireNonNull(capacityNotifier);
        var admission = config.getRouter().getRoles().getPrefill().getAdmission();
        maxInflightRequests = admission.getMaxInflightRequests();
        var queue = config.getRouter().getRoles().getPrefill().engineQueueAdmission();
        engineWaitingThreshold = queue == null ? null : queue.getWaitingThreshold();
    }

    public WorkerAdmissionCapacity requestCapacity() {
        stateLock().lock();
        try {
            var status = getStatus();
            return new WorkerAdmissionCapacity(
                    CapacityKey.exact(status.getRole(), status.getGroup(), ipPort()),
                    status.getGenerationId(),
                    resourceVersion,
                    acceptsNewRequests() ? availableRequestSlotsLocked() : 0L);
        } finally {
            stateLock().unlock();
        }
    }

    public PrefillEndpoint.SelectionSnapshot selectionSnapshot() {
        stateLock().lock();
        try {
            long generationId = getStatus().getGenerationId();
            if (selectionSnapshotCache == null
                    || selectionSnapshotCache.generationId() != generationId
                    || selectionSnapshotCache.resourceVersion() != resourceVersion) {
                selectionSnapshotCache =
                        new SelectionSnapshot(
                                generationId,
                                resourceVersion,
                                currentComputeTokensLocked(),
                                requests.size() + reportedUnknownRequests,
                                maxInflightRequests,
                                engineQueueBlocksAdmissionLocked());
            }
            return selectionSnapshotCache;
        } finally {
            stateLock().unlock();
        }
    }

    public Registration<RequestRegistration> tryRegisterSelection(
            String requestId,
            SelectionSnapshot expected,
            long computeTokens,
            long predictedDurationMs) {
        checkArgument(predictedDurationMs >= -1L, "invalid predicted duration");
        Objects.requireNonNull(requestId, "requestId");
        Objects.requireNonNull(expected, "work snapshot");
        checkArgument(computeTokens >= 0L, "computeTokens must be non-negative");
        stateLock().lock();
        try {
            if (!acceptsNewRequests())
                return new Registration<>(WorkerEndpoint.RegistrationStatus.UNAVAILABLE, null);
            if (getStatus().getGenerationId() != expected.generationId())
                return new Registration<>(WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT, null);
            if (requests.size() + reportedUnknownRequests >= maxInflightRequests
                    || engineQueueBlocksAdmissionLocked())
                return Registration.rejected(RegistrationStatus.CAPACITY_FULL);
            long currentComputeTokens = currentComputeTokensLocked();
            // A registration followed by rollback can leave the ranking inputs unchanged.
            if (currentComputeTokens != expected.computeTokens())
                return Registration.rejected(RegistrationStatus.VERSION_CONFLICT);
            if (requests.containsKey(requestId))
                return Registration.rejected(RegistrationStatus.ALREADY_REGISTERED);
            var record =
                    new RequestRegistration(
                            expected.generationId(),
                            requestId,
                            RegistrationPhase.REGISTERED,
                            clock.getAsLong());
            record.computeTokens = computeTokens;
            record.predictedDurationMs = predictedDurationMs;
            record.registrationOrder = resourceVersion;
            requests.put(requestId, record);
            unobservedRequests++;
            registeredComputeTokens += computeTokens;
            resourceVersion++;
            return new Registration<>(WorkerEndpoint.RegistrationStatus.REGISTERED, record);
        } finally {
            stateLock().unlock();
        }
    }

    public CommitStatus confirmSelection(
            RequestRegistration entry, WorkerEventSink events, long sequence) {
        stateLock().lock();
        try {
            if (!acceptsNewRequests()) return CommitStatus.UNAVAILABLE;
            if (entry == null
                    || requests.get(entry.requestId()) != entry
                    || entry.phase != RegistrationPhase.REGISTERED)
                return CommitStatus.INVALID_REGISTRATION;
            entry.events = Objects.requireNonNull(events, "events");
            entry.sequence = sequence;
            entry.commit(clock.getAsLong());
            return CommitStatus.COMMITTED;
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Estimates remaining duration from preceding registered work and the selected request.
     * Missing Engine work or an unknown duration suppresses the estimate.
     */
    public OptionalLong estimatedCompletionDurationMs(RequestRegistration exact, long nowMs) {
        stateLock().lock();
        try {
            if (requests.get(exact.requestId()) != exact || reportedUnknownDurationMs < 0)
                return OptionalLong.empty();
            long duration = reportedUnknownDurationMs;
            for (var entry : requests.values()) {
                if (entry.registrationOrder > exact.registrationOrder) continue;
                if (entry.predictedDurationMs < 0) return OptionalLong.empty();
                long remaining =
                        entry.runningObservedAtMs > 0
                                ? Math.max(
                                        0L,
                                        entry.predictedDurationMs
                                                - entry.observedExecutionMs
                                                - Math.max(0L, nowMs - entry.runningObservedAtMs))
                                : entry.predictedDurationMs;
                duration = saturatedAdd(duration, remaining);
            }
            return OptionalLong.of(duration);
        } finally {
            stateLock().unlock();
        }
    }

    @Override
    public OptionalLong tryGetLoadMetric() {
        if (!stateLock().tryLock()) return OptionalLong.empty();
        try {
            return OptionalLong.of(currentComputeTokensLocked());
        } finally {
            stateLock().unlock();
        }
    }

    @Override
    public int evictExpiredRequests(long ttlMs, Predicate<String> retained) {
        var candidates = new HashSet<RequestRegistration>();
        for (var route : captureCommittedRegistrations())
            if (!retained.test(route.requestId())) candidates.add(route);
        return removeExpiredOrphanRequests(ttlMs, candidates);
    }

    @Override
    public boolean rollbackSelection(WorkerEndpoint.RequestRegistration registration) {
        checkArgument(
                registration.endpointGenerationId() == getStatus().getGenerationId(),
                "request belongs to another endpoint");
        stateLock().lock();
        try {
            var request = requests.get(registration.requestId());
            if (request != registration || request.phase != RegistrationPhase.REGISTERED)
                return false;
            removeRequestRecordLocked(request);
        } finally {
            stateLock().unlock();
        }
        return true;
    }

    public boolean releaseRequest(RequestRegistration registration) {
        Objects.requireNonNull(registration, "registration");
        stateLock().lock();
        try {
            RequestRegistration entry = requests.get(registration.requestId());
            if (entry != registration) return false;

            removeRequestRecordLocked(entry);
        } finally {
            stateLock().unlock();
        }
        notifyCapacityChanged();
        return true;
    }

    @Override
    public void publishStatus(StatusChange change) {
        if (change instanceof StatusChange.None) {
            return;
        }
        var reconciliation = (StatusChange.Prefill) change;
        if (reconciliation.capacityReleased() || reconciliation.schedulingInputsChanged()) {
            notifyCapacityChanged();
        }
        for (var status : reconciliation.requestStatuses()) {
            var registration = status.registration();
            publishWorkerRequest(
                    registration,
                    registration.events,
                    getStatus().getRole(),
                    status.eventType(),
                    status.errorCode(),
                    "");
        }
    }

    public EndpointStats.PrefillCounts stats() {
        int locallyOwned = 0;
        long maxAgeMs = 0L;
        long nowMs = clock.getAsLong();
        stateLock().lock();
        try {
            for (RequestRegistration entry : requests.values()) {
                locallyOwned++;
                maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - entry.lastObservedAtMs));
            }

            return new EndpointStats.PrefillCounts(locallyOwned, maxAgeMs);
        } finally {
            stateLock().unlock();
        }
    }

    @Override
    protected StatusChange updateStatusLocked(WorkerStatus.StatusObservation observation) {
        var changes = updateRequestStatusLocked(observation);
        return changes;
    }

    /**
     * Clears request records once and publishes cleanup notifications outside the state lock.
     */
    @Override
    protected void closeEndpoint() {
        Throwable cleanupFailure = null;
        try {
            for (var registration : removeRequestsOnClose()) {
                try {
                    publishWorkerRequest(
                            registration,
                            registration.events,
                            getStatus().getRole(),
                            RequestEventType.ENDPOINT_CLOSED,
                            0L,
                            "Prefill generation retired: " + ipPort());
                } catch (Throwable notificationFailure) {
                    cleanupFailure = Failures.append(cleanupFailure, notificationFailure);
                }
            }

        } catch (Throwable committedCleanupFailure) {
            cleanupFailure = Failures.append(cleanupFailure, committedCleanupFailure);
        }
        notifyCapacityChanged();
        Failures.rethrow(cleanupFailure, "Prefill endpoint retirement failed");
    }

    /**
     * Captures cleanup notifications and clears all records in one Endpoint lock.
     */
    private List<RequestRegistration> removeRequestsOnClose() {
        List<RequestRegistration> ownedItems = new ArrayList<>();

        stateLock().lock();
        try {
            for (var entry : requests.values()) {
                if (entry.events != null) ownedItems.add(entry);
            }

            requests.clear();

            registeredComputeTokens = 0L;
            reportedUnknownComputeTokens = 0L;
            reportedExtraComputeTokens = 0L;
            reportedUnknownRequests = 0L;
            reportedUnknownDurationMs = 0L;
            unobservedRequests = 0L;
            resourceVersion++;
        } finally {
            stateLock().unlock();
        }
        ownedItems.sort(
                Comparator.comparingLong((RequestRegistration entry) -> entry.sequence)
                        .thenComparing(RequestRegistration::requestId));

        return ownedItems;
    }

    /**
     * Captures committed registrations before querying Scheduler retention outside the state lock.
     */
    private List<RequestRegistration> captureCommittedRegistrations() {
        stateLock().lock();
        try {
            return requests.values().stream()
                    .filter(entry -> entry.phase == RegistrationPhase.COMMITTED)
                    .toList();
        } finally {
            stateLock().unlock();
        }
    }

    private int removeExpiredOrphanRequests(long ttlMs, Set<RequestRegistration> candidates) {
        int removed = 0;
        stateLock().lock();
        try {
            for (var entry : List.copyOf(requests.values())) {
                if (entry.phase == RegistrationPhase.COMMITTED
                        && candidates.contains(entry)
                        && clock.getAsLong() - entry.lastObservedAtMs >= Math.max(0L, ttlMs)) {
                    removeRequestRecordLocked(entry);
                    removed++;
                }
            }
        } finally {
            stateLock().unlock();
        }
        if (removed > 0) notifyCapacityChanged();
        return removed;
    }

    /**
     * Removes the exact request and releases its capacity while advancing the resource version.
     */
    private void removeRequestRecordLocked(RequestRegistration entry) {
        if (!requests.remove(entry.requestId, entry)) return;
        checkState(
                registeredComputeTokens >= entry.computeTokens,
                "Prefill compute-token count underflow");
        registeredComputeTokens -= entry.computeTokens;
        reportedExtraComputeTokens -= entry.reportedExtraComputeTokens;
        if (!entry.observedByEngine) unobservedRequests--;
        resourceVersion++;
    }

    private void notifyCapacityChanged() {
        try {
            capacityNotifier.notifyChanged(this);
        } catch (RuntimeException failure) {
            logger.error("Prefill capacity notification failed", failure);
        }
    }

    private long currentComputeTokensLocked() {
        return saturatedAdd(
                saturatedAdd(registeredComputeTokens, reportedExtraComputeTokens),
                reportedUnknownComputeTokens);
    }

    /**
     * Matches Worker reports to registered requests and removes completed resource records.
     */
    private StatusChange.Prefill updateRequestStatusLocked(
            WorkerStatus.StatusObservation observation) {
        long nowMs = clock.getAsLong();
        var finishedRequests = matchFinishedRequests(observation.finishedTasks());
        var engine = observation.engine();
        long availableBefore = availableRequestSlotsLocked();
        reportedWaitingRequests = engine.waitingQueryLen();
        var statuses =
                new ArrayList<PrefillRequestStatus>(
                        finishedRequests.size() + engine.runningTaskList().size());

        for (var finished : finishedRequests.values()) {
            RequestEventType kind;
            if (finished.errorCode() == 0L) {
                kind = RequestEventType.COMPLETED;
            } else if (finished.preemptionProgress() == PriorityPreemptionProgress.CANCELED
                    && finished.errorCode()
                            == StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode()) {
                kind = RequestEventType.PRIORITY_CANCELLED;
            } else {
                kind = RequestEventType.FAILED;
            }
            var entry = finished.owner();
            statuses.add(new PrefillRequestStatus(entry, kind, finished.errorCode()));
        }

        boolean changed = !finishedRequests.isEmpty();
        long previousExtraComputeTokens = reportedExtraComputeTokens;
        reportedExtraComputeTokens = 0L;
        for (var entry : requests.values()) {
            entry.reportedExtraComputeTokens = 0L;
            if (entry.observedByEngine) {
                entry.observedByEngine = false;
                unobservedRequests++;
            }
        }
        Set<String> unknown = new HashSet<>(), known = new HashSet<>();
        long unknownComputeTokens = 0L;
        long unknownDurationMs = 0L;
        boolean unknownDuration = false;
        for (var task : engine.runningTaskList().values()) {
            if (finishedRequests.containsKey(task.requestId())) continue;
            var entry = requests.get(task.requestId());
            if (entry == null) {
                if (!task.isPriorityCancelOverlayOnly()) {
                    unknown.add(task.requestId());
                    long tokens =
                            task.inputLength() <= 0
                                    ? Long.MAX_VALUE
                                    : predictor.computeTokens(
                                            task.inputLength(),
                                            task.telemetry().prefixLengthValid()
                                                    ? task.prefixLength()
                                                    : 0L);
                    unknownComputeTokens = saturatedAdd(unknownComputeTokens, tokens);
                    if (task.inputLength() <= 0) unknownDuration = true;
                    else {
                        long duration =
                                PrefillTimePredictor.estimateDurationMs(
                                        task.inputLength(),
                                        task.telemetry().prefixLengthValid()
                                                ? task.prefixLength()
                                                : 0);
                        unknownDurationMs =
                                saturatedAdd(
                                        unknownDurationMs,
                                        Math.max(
                                                0L,
                                                duration - Math.max(0L, task.executionTimeMs())));
                    }
                }
                continue;
            }
            if (!task.isPriorityCancelOverlayOnly()) {
                known.add(task.requestId());
                if (task.inputLength() > 0) {
                    long actual =
                            predictor.computeTokens(
                                    task.inputLength(),
                                    task.telemetry().prefixLengthValid()
                                            ? task.prefixLength()
                                            : 0L);
                    entry.reportedExtraComputeTokens = Math.max(0L, actual - entry.computeTokens);
                    reportedExtraComputeTokens += entry.reportedExtraComputeTokens;
                }
                if (entry.phase == RegistrationPhase.COMMITTED) {
                    if (!entry.observedByEngine) {
                        entry.observedByEngine = true;
                        unobservedRequests--;
                    }
                    statuses.add(new PrefillRequestStatus(entry, RequestEventType.ACTIVE, 0L));
                }
            }
            if (entry.phase != RegistrationPhase.COMMITTED) continue;
            entry.lastObservedAtMs = Math.max(entry.lastObservedAtMs, nowMs);
            if (task.phase() == org.flexlb.worker.status.TaskPhase.RUNNING) {
                entry.runningObservedAtMs = nowMs;
                entry.observedExecutionMs = Math.max(0L, task.executionTimeMs());
            }
        }

        for (var finished : finishedRequests.values()) {
            removeRequestRecordLocked(finished.owner());
        }
        long total =
                saturatedAdd(
                        Math.max(0L, engine.waitingQueryLen()),
                        Math.max(0L, engine.runningQueryLen()));
        long unknownCount = Math.max(unknown.size(), Math.max(0L, total - known.size()));
        if (unknownCount > unknown.size()) {
            unknownComputeTokens = Long.MAX_VALUE;
            unknownDuration = true;
        }
        reportedUnknownDurationMs = unknownDuration ? -1L : unknownDurationMs;
        boolean released =
                !finishedRequests.isEmpty()
                        || unknownComputeTokens < reportedUnknownComputeTokens
                        || reportedExtraComputeTokens < previousExtraComputeTokens;
        changed |=
                unknownComputeTokens != reportedUnknownComputeTokens
                        || reportedExtraComputeTokens != previousExtraComputeTokens
                        || unknownCount != reportedUnknownRequests;
        released |= unknownCount < reportedUnknownRequests;
        reportedUnknownRequests = unknownCount;
        reportedUnknownComputeTokens = unknownComputeTokens;
        changed |= availableBefore != availableRequestSlotsLocked();
        if (changed) {
            resourceVersion++;
        }
        return new StatusChange.Prefill(statuses, changed, released);
    }

    private Map<String, TerminalObservation> matchFinishedRequests(
            Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        // Both maps are keyed by request ID; worker history may outnumber local requests.
        var candidateIds =
                requests.size() < finishedTasks.size() ? requests.keySet() : finishedTasks.keySet();
        Map<String, TerminalObservation> matched = new HashMap<>();
        for (String requestId : candidateIds) {
            var entry = requests.get(requestId);
            var task = finishedTasks.get(requestId);
            if (entry == null || task == null || entry.phase != RegistrationPhase.COMMITTED) {
                continue;
            }

            var next =
                    new TerminalObservation(
                            entry, task.errorCode(), task.priorityPreemptionProgress());
            matched.put(requestId, next);
        }
        return matched;
    }

    private long availableRequestSlotsLocked() {
        long occupied = saturatedAdd(requests.size(), reportedUnknownRequests);
        long slots = Math.max(0L, maxInflightRequests - occupied);
        if (engineWaitingThreshold != null && reportedWaitingRequests >= 0) {
            long headroom = engineWaitingThreshold - unobservedRequests;
            if (reportedWaitingRequests > headroom) return 0L;
            // Registration checks the current watermark before adding the next request.
            slots = Math.min(slots, saturatedAdd(headroom - reportedWaitingRequests, 1L));
        }
        return slots;
    }

    private boolean engineQueueBlocksAdmissionLocked() {
        return engineWaitingThreshold != null
                && reportedWaitingRequests >= 0
                && reportedWaitingRequests > engineWaitingThreshold - unobservedRequests;
    }

    /**
     * Request status matched to its exact Prefill registration, published after ledger reconciliation.
     */
    public record PrefillRequestStatus(
            RequestRegistration registration, RequestEventType eventType, long errorCode) {
        public PrefillRequestStatus {
            Objects.requireNonNull(registration, "registration");
            Objects.requireNonNull(eventType, "eventType");
            checkArgument(
                    eventType != RequestEventType.ALLOCATED,
                    "Prefill status must describe execution or completion");
            checkArgument(
                    eventType != RequestEventType.ACTIVE || errorCode == 0L,
                    "an active Prefill request status cannot carry an error code");
        }
    }

    /**
     * Freezes work estimates, request capacity and the exact registration version for selection.
     */
    public record SelectionSnapshot(
            long generationId,
            long resourceVersion,
            long computeTokens,
            long outstandingRequests,
            int maxInflightRequests,
            boolean engineQueueBlocked) {
        public boolean admissionBlocked() {
            return outstandingRequests >= maxInflightRequests || engineQueueBlocked;
        }
    }

    /**
     * Retains one Prefill selection and its compute-token charge; event association and updates
     * require the endpoint lock.
     */
    public static final class RequestRegistration implements WorkerEndpoint.RequestRegistration {
        private final long endpointGenerationId;

        RegistrationPhase phase;

        final String requestId;

        private volatile WorkerEventSink events;
        // Keeps retry notifications in submission order when this endpoint closes.
        private long sequence;

        /**
         * Input tokens requiring computation after cache reuse; charged until this record is removed.
         */
        long computeTokens;

        // Engine work exceeding the cache estimate, replaced on each full status report.
        private long reportedExtraComputeTokens;

        long lastObservedAtMs;
        private boolean observedByEngine;
        private long predictedDurationMs = -1L;
        private long registrationOrder;
        private long runningObservedAtMs;
        private long observedExecutionMs;

        private RequestRegistration(
                long endpointGenerationId, String requestId, RegistrationPhase phase, long nowMs) {
            this.endpointGenerationId = endpointGenerationId;
            this.phase = phase;
            this.requestId = Objects.requireNonNull(requestId);
            this.lastObservedAtMs = nowMs;
        }

        @Override
        public String requestId() {
            return requestId;
        }

        @Override
        public long endpointGenerationId() {
            return endpointGenerationId;
        }

        private void commit(long nowMs) {
            lastObservedAtMs = nowMs;
            phase = RegistrationPhase.COMMITTED;
        }
    }

    record TerminalObservation(
            RequestRegistration owner,
            long errorCode,
            PriorityPreemptionProgress preemptionProgress) {}
}

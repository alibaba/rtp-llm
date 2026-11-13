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
    private long resourceVersion;

    private final PrefillTimePredictor predictor = new PrefillTimePredictor();

    private final long maxComputeTokens;

    private final CapacityNotifier capacityNotifier;

    PrefillEndpoint(WorkerStatus status, FlexlbConfig config, CapacityNotifier capacityNotifier) {
        super(status);
        this.capacityNotifier = Objects.requireNonNull(capacityNotifier);
        this.maxComputeTokens = config.getRouter().getRoles().getPrefill().getMaxComputeTokens();
    }

    public PrefillEndpoint.SelectionSnapshot selectionSnapshot() {
        stateLock().lock();
        try {
            return new PrefillEndpoint.SelectionSnapshot(
                    getStatus().getGenerationId(),
                    resourceVersion,
                    currentComputeTokensLocked(),
                    maxComputeTokens);
        } finally {
            stateLock().unlock();
        }
    }

    public Registration<RequestRegistration> tryRegisterSelection(
            String requestId, PrefillEndpoint.SelectionSnapshot expected, long computeTokens) {
        Objects.requireNonNull(requestId, "requestId");
        Objects.requireNonNull(expected, "work snapshot");
        checkArgument(computeTokens >= 0L, "computeTokens must be non-negative");
        stateLock().lock();
        try {
            if (!acceptsNewRequests())
                return new Registration<>(WorkerEndpoint.RegistrationStatus.UNAVAILABLE, null);
            if (getStatus().getGenerationId() != expected.generationId()
                    || resourceVersion != expected.resourceVersion())
                return new Registration<>(WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT, null);
            if (requests.containsKey(requestId))
                return Registration.rejected(RegistrationStatus.ALREADY_REGISTERED);
            if (computeTokens > maxComputeTokens
                    || currentComputeTokensLocked() > maxComputeTokens - computeTokens)
                return new Registration<>(WorkerEndpoint.RegistrationStatus.CAPACITY_FULL, null);
            var record = new RequestRegistration(this, requestId, RegistrationPhase.REGISTERED);
            record.computeTokens = computeTokens;
            requests.put(requestId, record);
            registeredComputeTokens += computeTokens;
            resourceVersion++;
            return new Registration<>(WorkerEndpoint.RegistrationStatus.REGISTERED, record);
        } finally {
            stateLock().unlock();
        }
    }

    public boolean commitSelection(WorkerRoute route) {
        stateLock().lock();
        try {
            var entry = route.prefillRegistration();
            if (!acceptsNewRequests()
                    || entry == null
                    || requests.get(route.requestId()) != entry
                    || entry.route != route
                    || entry.phase != RegistrationPhase.REGISTERED) return false;
            entry.commit(clock.getAsLong());
            return true;
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
        var candidates = new HashSet<WorkerRoute>();
        for (var route : captureCommittedRoutes())
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
        notifyCapacityChanged();
        return true;
    }

    public boolean releaseRequest(WorkerRoute route) {
        Objects.requireNonNull(route, "route");
        stateLock().lock();
        try {
            RequestRegistration entry = requests.get(route.requestId());
            if (entry == null || entry.route != route) return false;

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
            var route = status.route();
            if (route.prefillEndpoint() != this) {
                continue;
            }
            publishWorkerRequest(
                    route, null, getStatus().getRole(), status.eventType(), status.errorCode(), "");
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
            RemovedRequests cleanup = removeRequestsOnClose();
            for (WorkerRoute route : cleanup.ownedItems()) {
                try {
                    var context = route.ctx();
                    var event =
                            new org.flexlb.worker.endpoint.WorkerRequestEvent(
                                    route,
                                    null,
                                    getStatus().getRole(),
                                    this,
                                    org.flexlb.worker.endpoint.RequestEventType.ENDPOINT_CLOSED,
                                    0L,
                                    "Prefill generation retired: " + ipPort(),
                                    System.currentTimeMillis());
                    context.getRequestScheduler().onWorkerRequestEvent(context, event);
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
    RemovedRequests removeRequestsOnClose() {
        List<WorkerRoute> ownedItems = new ArrayList<>();

        stateLock().lock();
        try {
            for (var entry : requests.values()) {
                if (entry.route != null) ownedItems.add(entry.route);
            }

            requests.clear();

            registeredComputeTokens = 0L;
            reportedUnknownComputeTokens = 0L;
            resourceVersion++;
        } finally {
            stateLock().unlock();
        }
        ownedItems.sort(
                Comparator.comparingLong((WorkerRoute route) -> route.ctx().getPlacementSequence())
                        .thenComparing(WorkerRoute::requestId));

        return new RemovedRequests(ownedItems);
    }

    /**
     * Captures committed routes before querying Scheduler retention outside the state lock.
     */
    private List<WorkerRoute> captureCommittedRoutes() {
        stateLock().lock();
        try {
            return requests.values().stream()
                    .filter(entry -> entry.phase == RegistrationPhase.COMMITTED)
                    .map(entry -> entry.route)
                    .toList();
        } finally {
            stateLock().unlock();
        }
    }

    private int removeExpiredOrphanRequests(long ttlMs, Set<WorkerRoute> candidates) {
        int removed = 0;
        stateLock().lock();
        try {
            for (var entry : List.copyOf(requests.values())) {
                if (entry.phase == RegistrationPhase.COMMITTED
                        && candidates.contains(entry.route)
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
        return saturatedAdd(registeredComputeTokens, reportedUnknownComputeTokens);
    }

    /**
     * Matches Worker reports to registered requests and removes completed resource records.
     */
    private StatusChange.Prefill updateRequestStatusLocked(
            WorkerStatus.StatusObservation observation) {
        long nowMs = clock.getAsLong();
        var finishedRequests = matchFinishedRequests(observation.finishedTasks());
        var engine = observation.engine();
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
            statuses.add(new PrefillRequestStatus(entry.route, kind, finished.errorCode()));
        }

        boolean changed = !finishedRequests.isEmpty();
        Set<String> unknown = new HashSet<>(), known = new HashSet<>();
        long unknownComputeTokens = 0L;
        for (var task : engine.runningTaskList().values()) {
            if (finishedRequests.containsKey(task.requestId())) continue;
            var entry = requests.get(task.requestId());
            if (entry == null) {
                if (!task.isPriorityCancelOverlayOnly()) {
                    unknown.add(task.requestId());
                    long tokens =
                            task.inputLength() <= 0
                                    ? maxComputeTokens
                                    : predictor.computeTokens(
                                            task.inputLength(),
                                            task.telemetry().prefixLengthValid()
                                                    ? task.prefixLength()
                                                    : 0L);
                    unknownComputeTokens = saturatedAdd(unknownComputeTokens, tokens);
                }
                continue;
            }
            if (!task.isPriorityCancelOverlayOnly()) {
                known.add(task.requestId());
                if (entry.phase == RegistrationPhase.COMMITTED) {
                    statuses.add(
                            new PrefillRequestStatus(entry.route, RequestEventType.ACTIVE, 0L));
                }
            }
            if (entry.phase != RegistrationPhase.COMMITTED) continue;
            entry.lastObservedAtMs = Math.max(entry.lastObservedAtMs, nowMs);
        }

        for (var finished : finishedRequests.values()) {
            removeRequestRecordLocked(finished.owner());
        }
        long total =
                saturatedAdd(
                        Math.max(0L, engine.waitingQueryLen()),
                        Math.max(0L, engine.runningQueryLen()));
        long unknownCount = Math.max(unknown.size(), Math.max(0L, total - known.size()));
        if (unknownCount > unknown.size()) unknownComputeTokens = maxComputeTokens;
        boolean released =
                !finishedRequests.isEmpty() || unknownComputeTokens < reportedUnknownComputeTokens;
        changed |= unknownComputeTokens != reportedUnknownComputeTokens;
        reportedUnknownComputeTokens = unknownComputeTokens;
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

    /**
     * Request status matched to its exact Prefill route, published after ledger reconciliation.
     */
    public record PrefillRequestStatus(
            WorkerRoute route, RequestEventType eventType, long errorCode) {
        public PrefillRequestStatus {
            Objects.requireNonNull(route, "route");
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
     * Captures an immutable list of routes removed from a Prefill endpoint.
     */
    public record RemovedRequests(List<WorkerRoute> ownedItems) {
        public RemovedRequests {
            ownedItems = List.copyOf(ownedItems);
        }
    }

    /**
     * Freezes compute-token usage, its limit and the exact registration version for selection.
     */
    public record SelectionSnapshot(
            long generationId, long resourceVersion, long computeTokens, long maxComputeTokens) {}

    /**
     * Retains one Prefill selection and its compute-token charge; route association and updates
     * require the endpoint lock.
     */
    public static final class RequestRegistration implements WorkerEndpoint.RequestRegistration {
        final PrefillEndpoint owner;

        RegistrationPhase phase;

        final String requestId;

        WorkerRoute route;

        /**
         * Input tokens requiring computation after cache reuse; charged until this record is removed.
         */
        long computeTokens;

        long lastObservedAtMs;

        private RequestRegistration(
                PrefillEndpoint owner, String requestId, RegistrationPhase phase) {
            this.owner = owner;
            this.phase = phase;
            this.requestId = Objects.requireNonNull(requestId);
            this.lastObservedAtMs = owner.clock.getAsLong();
        }

        @Override
        public String requestId() {
            return requestId;
        }

        @Override
        public long endpointGenerationId() {
            return owner.getStatus().getGenerationId();
        }

        /**
         * Associate the complete route with its exact selection record before publication.
         */
        public void bind(WorkerRoute completed) {
            owner.stateLock().lock();
            try {
                checkArgument(
                        requestId.equals(completed.requestId()),
                        "Prefill registration belongs to another selection");
                checkState(route == null || route == completed, "registration already bound");
                route = completed;
            } finally {
                owner.stateLock().unlock();
            }
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

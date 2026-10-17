package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * One service-discovery generation of an Engine worker.
 *
 * <p>The object identity is the generation identity. Its independently updated
 * concurrency domains are published explicitly:
 *
 * <ul>
 *   <li>{@link TopologySnapshot}: discovery-owned address and placement data;</li>
 *   <li>{@link CommittedWorkerStatus}: stable Engine fields and both cursors
 *       from one successfully reduced, strictly newer status response;</li>
 *   <li>{@link PollHealth}: transport health from status polling, independent
 *       from Engine state and endpoint routability;</li>
 *   <li>cache status and poll coordination, which have independent writers.</li>
 * </ul>
 *
 * <p>A status response is first converted into an opaque {@link PreparedStatus}.
 * Endpoint and scheduler reducers consume its frozen observation, and only
 * then does {@link #publishPreparedStatus(PreparedStatus)} replace the one
 * committed holder. Readers therefore cannot observe fields from a response
 * whose cursor is still uncommitted.</p>
 */
public class WorkerStatus {

    private static final AtomicLong NEXT_GENERATION_ID = new AtomicLong();

    /** Serializes lifecycle transitions for this generation. */
    public final transient ReentrantLock lock = new ReentrantLock();

    /** Serializes address-scoped cache writes and final cleanup; independent of the status lock. */
    public final transient ReentrantLock cacheMutationLock = new ReentrantLock();

    private final long generationId = NEXT_GENERATION_ID.incrementAndGet();

    private final EndpointTrace trace = new EndpointTrace();

    @JsonIgnore
    public EndpointTrace trace() {
        return trace;
    }

    public record TopologySnapshot(
            String group,
            String ip,
            int port,
            int grpcPort,
            String site) {
    }

    /** Deeply immutable copy of the task fields reported by one status RPC. */
    public record TaskObservation(
            long requestId,
            long prefixLength,
            long prefillTime,
            long inputLength,
            long waitingTime,
            long iterateCount,
            long endTimeMs,
            long dpRank,
            long errorCode,
            String errorMessage,
            long batchId,
            TaskPhase phase,
            long executionTimeMs,
            PriorityPreemptionProgress priorityPreemptionProgress) {

        /** A pending observation that reports only priority cancellation progress. */
        @JsonIgnore
        public boolean isPriorityCancelOverlayOnly() {
            return (priorityPreemptionProgress == PriorityPreemptionProgress.CANCELING
                    || priorityPreemptionProgress == PriorityPreemptionProgress.CANCELED)
                    && phase == TaskPhase.PENDING;
        }

        private static TaskObservation copyOf(TaskInfo task) {
            return new TaskObservation(
                    task.getRequestId(),
                    task.getPrefixLength(),
                    task.getPrefillTime(),
                    task.getInputLength(),
                    task.getWaitingTime(),
                    task.getIterateCount(),
                    task.getEndTimeMs(),
                    task.getDpRank(),
                    task.getErrorCode(),
                    task.getErrorMessage(),
                    task.getBatchId(),
                    task.getPhase(),
                    task.getExecutionTimeMs(),
                    task.getPriorityPreemptionProgress());
        }
    }

    /**
     * Immutable view of one atomically published Engine status observation.
     * {@code runningTaskList} is the exact immutable map carried by the
     * transaction observation that produced this value.
     */
    public record EngineObservation(
            RoleType role,
            Long availableConcurrency,
            long availableKvCacheTokens,
            long totalKvCacheTokens,
            Map<String, TaskObservation> runningTaskList,
            double stepLatencyMs,
            long iterateCount,
            long dpSize,
            long tpSize,
            long dpRank,
            long maxSeqLen,
            long maxBatchTokensSize,
            long runningQueryLen,
            long waitingQueryLen) {

        public EngineObservation {
            Objects.requireNonNull(role, "role");
            runningTaskList = Map.copyOf(runningTaskList);
        }
    }

    /**
     * The two cursors acknowledged by one successful status transaction.
     * Keeping them in one immutable value prevents readers from observing a
     * committed status version with an older finished-task cursor.
     */
    public record AppliedStatusCursor(
            long statusVersion,
            long latestFinishedTaskVersion) {
    }

    /** The single public value for one successfully committed status round. */
    public record CommittedWorkerStatus(
            EngineObservation fields,
            AppliedStatusCursor cursor,
            long snapshotReceivedUs) {
    }

    public enum StatusContact { UNKNOWN, ALIVE, DEAD }

    /**
     * Status-poll transport health. Scheduling also requires a fresh applied
     * snapshot and an open endpoint generation gate.
     */
    public record PollHealth(
            long lastSuccessfulPollUs,
            long successfulPollIntervalUs,
            long consecutiveTransportFailures,
            StatusContact contact) {

        public PollHealth(long lastSuccessfulPollUs, long successfulPollIntervalUs,
                          long consecutiveTransportFailures, boolean reportedAlive) {
            this(lastSuccessfulPollUs, successfulPollIntervalUs, consecutiveTransportFailures,
                    reportedAlive ? StatusContact.ALIVE : StatusContact.DEAD);
        }

        public boolean reportedAlive() { return contact == StatusContact.ALIVE; }

        public PollHealth {
            Objects.requireNonNull(contact, "contact");
            if (lastSuccessfulPollUs < 0L
                    || successfulPollIntervalUs < 0L
                    || consecutiveTransportFailures < 0L) {
                throw new IllegalArgumentException(
                        "poll health counters must be non-negative");
            }
        }
    }

    public record CacheIndexSnapshot(
            CacheStatus cacheStatus,
            boolean indexInitialized,
            long indexedVersion) {
    }

    /** One deeply frozen RPC observation shared by every status reducer. */
    public static final class StatusObservation {
        private final WorkerStatus owner;
        private final EngineObservation engine;
        private final boolean reportedAlive;
        private final Long statusVersion;
        private final Long latestFinishedVersion;
        private final Map<String, TaskObservation> finishedTasks;

        private StatusObservation(
                WorkerStatus owner,
                EngineObservation engine,
                boolean reportedAlive,
                Long statusVersion,
                Long latestFinishedVersion,
                Map<String, TaskObservation> finishedTasks) {
            this.owner = owner;
            this.engine = engine;
            this.reportedAlive = reportedAlive;
            this.statusVersion = statusVersion;
            this.latestFinishedVersion = latestFinishedVersion;
            this.finishedTasks = finishedTasks;
        }

        /** Exact service-discovery generation which issued this observation. */
        public WorkerStatus owner() {
            return owner;
        }

        public EngineObservation engine() {
            return engine;
        }

        public RoleType role() {
            return engine.role();
        }

        public boolean alive() {
            return reportedAlive;
        }

        public Long statusVersion() {
            return statusVersion;
        }

        public Long latestFinishedVersion() {
            return latestFinishedVersion;
        }

        public Map<String, TaskObservation> runningTasks() {
            return engine.runningTaskList();
        }

        public Map<String, TaskObservation> finishedTasks() {
            return finishedTasks;
        }

        public long runningQueryLen() {
            return engine.runningQueryLen();
        }

        public long waitingQueryLen() {
            return engine.waitingQueryLen();
        }
    }

    /**
     * Worker-issued transaction token. Its private owner/base identities make
     * it impossible to forge or replay against another committed generation.
     */
    public static final class PreparedStatus {
        private final CommittedWorkerStatus baseCommitted;
        private final StatusObservation observation;
        private final CommittedWorkerStatus nextCommitted;
        private final EndpointTrace.Poll pollTrace;

        private PreparedStatus(
                CommittedWorkerStatus baseCommitted,
                StatusObservation observation,
                CommittedWorkerStatus nextCommitted,
                EndpointTrace.Poll pollTrace) {
            this.baseCommitted = baseCommitted;
            this.observation = observation;
            this.nextCommitted = nextCommitted;
            this.pollTrace = pollTrace;
        }

        public StatusObservation observation() {
            return observation;
        }
    }

    private final AtomicReference<TopologySnapshot> topology;

    private final AtomicReference<CommittedWorkerStatus> committedStatus;

    /** Cache polling is independent from worker-status polling. */
    private final AtomicReference<CacheIndexSnapshot> cacheIndexSnapshot =
            new AtomicReference<>(new CacheIndexSnapshot(null, false, -1L));
    private final AtomicLong cacheLastUpdateTime = new AtomicLong(-1L);

    private static final long DEFAULT_STATUS_FRESHNESS_US = 10_000_000L;

    /** Owns transport, application ownership and lifecycle transitions for this generation. */
    private final StatusProgress statusProgress;

    private enum StatusLifecycle { ACTIVE, RETIREMENT_CLAIMED, RETIRING }

    /**
     * Only constant-time progress transitions take this monitor. Never hold it
     * across conversion, state reduction, endpoint withdrawal, logging or cleanup.
     */
    private final class StatusProgress {
        private final long maximumAgeUs;
        private volatile PollHealth health;
        private volatile StatusLifecycle lifecycle = StatusLifecycle.ACTIVE;
        // Its receipt time lives on the same lease; both are published under this monitor.
        private PollLease application;

        private StatusProgress(long maximumAgeUs) {
            if (maximumAgeUs <= 0L) { throw new IllegalArgumentException("status freshness must be positive"); }
            this.maximumAgeUs = maximumAgeUs;
            health = new PollHealth(System.nanoTime()/1000, 0L, 0L, StatusContact.UNKNOWN);
        }

        private synchronized boolean claimRetirement(PollHealth expected) {
            if (lifecycle != StatusLifecycle.ACTIVE || health != expected) { return false; }
            lifecycle = StatusLifecycle.RETIREMENT_CLAIMED;
            return true;
        }

        private synchronized boolean beginRetirement() {
            if (lifecycle == StatusLifecycle.RETIRING) { return false; }
            lifecycle = StatusLifecycle.RETIRING;
            return true;
        }

        private synchronized void rpcStarted() {
            if (lifecycle != StatusLifecycle.ACTIVE) { return; }
            // Change identity so a prior expiration scan cannot claim this generation.
            health = new PollHealth(health.lastSuccessfulPollUs(), health.successfulPollIntervalUs(),
                    health.consecutiveTransportFailures(), health.contact());
        }

        private synchronized Stage rpcResult(PollLease lease, boolean valid, boolean alive, Throwable failure) {
            if (statusPollLease.get() != lease || lifecycle != StatusLifecycle.ACTIVE) { return null; }
            if (failure != null) {
                failedPoll();
                return Stage.TRANSPORT_FAILURE_RECORDED;
            }
            if (valid) {
                successfulPoll(alive, System.nanoTime()/1000);
                return Stage.STATUS_HEALTH_UPDATED;
            }
            return null;
        }

        private synchronized PollHealth successfulPoll(boolean alive, long receivedUs) {
            if (lifecycle != StatusLifecycle.ACTIVE) { return health; }
            health = new PollHealth(receivedUs, Math.max(0L, receivedUs-health.lastSuccessfulPollUs()),
                    0L, alive ? StatusContact.ALIVE : StatusContact.DEAD);
            return health;
        }

        private synchronized PollHealth failedPoll() {
            if (lifecycle != StatusLifecycle.ACTIVE) { return health; }
            health = new PollHealth(health.lastSuccessfulPollUs(), health.successfulPollIntervalUs(),
                    health.consecutiveTransportFailures() == Long.MAX_VALUE ? Long.MAX_VALUE
                            : health.consecutiveTransportFailures()+1L, health.contact());
            return health;
        }

        private synchronized boolean beginApplication(PollLease lease, long receivedUs) {
            if (lease == null || lease.owner != WorkerStatus.this || lease.kind != PollKind.STATUS) {
                throw new IllegalArgumentException("status application requires this generation's status lease");
            }
            if (lifecycle != StatusLifecycle.ACTIVE || application != null || statusPollLease.get() != lease) { return false; }
            lease.applicationReceivedUs = receivedUs;
            application = lease;
            return true;
        }

        private synchronized void finishApplication(PollLease lease) {
            if (application == lease) { application = null; }
        }

        private synchronized long applicationReceivedUs() {
            return application == null ? System.nanoTime()/1000 : application.applicationReceivedUs;
        }

        private synchronized boolean schedulingFresh(long nowUs) {
            long received = committedStatus.get().snapshotReceivedUs();
            return lifecycle == StatusLifecycle.ACTIVE && health.reportedAlive()
                    && health.consecutiveTransportFailures() < 3L && received > 0L
                    && nowUs >= received && nowUs-received <= maximumAgeUs;
        }
    }

    public enum HealthRetirementReason {
        CONSECUTIVE_STATUS_FAILURES,
        WORKER_REPORTED_NOT_ALIVE,
        STATUS_EXPIRED
    }

    /**
     * Claim health retirement and return its reason, or null while healthy,
     * protected by an in-flight poll, or superseded by concurrent RPC progress.
     * The caller must hold the generation lock.
     */
    public HealthRetirementReason tryClaimHealthRetirement(
            long nowUs, long workerTimeoutUs, long statusCommitGraceUs) {
        requireGenerationLock();
        PollHealth health = pollHealth();
        HealthRetirementReason reason;
        if (health.consecutiveTransportFailures() >= 3L) {
            reason = HealthRetirementReason.CONSECUTIVE_STATUS_FAILURES;
        } else if (health.contact() == StatusContact.DEAD) {
            reason = HealthRetirementReason.WORKER_REPORTED_NOT_ALIVE;
        } else if (nowUs > health.lastSuccessfulPollUs() + workerTimeoutUs
                && !isStatusPollProtected(nowUs, statusCommitGraceUs)) {
            reason = HealthRetirementReason.STATUS_EXPIRED;
        } else {
            return null;
        }
        return statusProgress.claimRetirement(health) ? reason : null;
    }

    /** Fence a stale-health decision against an RPC start/result which raced the scan. */
    public boolean tryClaimHealthRetirement(PollHealth expected) {
        requireGenerationLock();
        return statusProgress.claimRetirement(expected);
    }

    private void noteStatusRpcStarted() { statusProgress.rpcStarted(); }

    /** Original receipt time of the snapshot whose state was successfully applied. */
    public long appliedSnapshotReceivedUs() { return committedStatus.get().snapshotReceivedUs(); }

    /** @deprecated Use appliedSnapshotReceivedUs; this is not the application completion time. */
    @Deprecated
    public long lastStatusAppliedUs() { return appliedSnapshotReceivedUs(); }

    public boolean hasReceivedStatusRpc() { return statusProgress.health.contact() != StatusContact.UNKNOWN; }

    /** Advisory admission guard; existing request cleanup remains allowed. */
    public boolean isSchedulingStatusFresh(long nowUs) { return statusProgress.schedulingFresh(nowUs); }

    /** One retained response per generation; no additional per-response application object. */
    public boolean tryBeginStatusApplication(PollLease lease, long receivedUs) {
        return statusProgress.beginApplication(lease, receivedUs);
    }

    public void finishStatusApplication(PollLease lease) { statusProgress.finishApplication(lease); }

    /** Runs at wire completion without taking the state/application lock. */
    public void recordStatusRpcResult(PollLease lease, boolean validResponse, boolean alive, Throwable failure) {
        Stage stage = statusProgress.rpcResult(lease, validResponse, alive, failure);
        if (stage != null) { lease.pollTrace().record(stage); }
    }

    public void recordStatusApplied(EndpointTrace.Poll trace) {
        requireGenerationLock();
        requireActiveGeneration();
        refreshAppliedReceipt(statusProgress.applicationReceivedUs());
        if (trace != null) { trace.record(Stage.STATUS_APPLIED); }
    }

    private void refreshAppliedReceipt(long receivedUs) {
        CommittedWorkerStatus current = committedStatus.get();
        if (current.snapshotReceivedUs() != receivedUs) {
            committedStatus.set(new CommittedWorkerStatus(current.fields(), current.cursor(), receivedUs));
        }
    }

    private final AtomicReference<PollLease> statusPollLease =
            new AtomicReference<>();
    private final AtomicReference<PollLease> cachePollLease =
            new AtomicReference<>();

    private enum PollKind {
        STATUS,
        CACHE
    }

    /**
     * Exact ownership of one generation-local poll slot. Closing is
     * token-fenced and idempotent: a delayed or duplicate close cannot release
     * a later poll which acquired the same logical slot.
     */
    public static final class PollLease implements AutoCloseable {
        private final WorkerStatus owner;
        private final PollKind kind;
        private final AtomicReference<PollLease> slot;
        private final EndpointTrace.Poll pollTrace;

        private volatile StatusPollProgress statusProgress;
        private long applicationReceivedUs;

        /** Called when the RPC actually starts, never while merely queued. */
        public void statusRpcStarted(long timeoutMs) {
            owner.requireStatusPollLease(this);
            statusProgress = new StatusPollProgress(
                    System.nanoTime() / 1000, timeoutMs * 1000L, false);
            owner.noteStatusRpcStarted();
        }

        /** Transport completion does not depend on the callback executor or Worker lock. */
        public void statusRpcCompleted(boolean validAliveResponse) {
            statusProgress = validAliveResponse
                    ? new StatusPollProgress(System.nanoTime() / 1000, 0L, true)
                    : null;
        }

        private PollLease(
                WorkerStatus owner,
                PollKind kind,
                AtomicReference<PollLease> slot, long syncCycleId) {
            this.owner = owner;
            this.kind = kind;
            this.slot = slot;
            this.pollTrace = owner.trace.beginPoll(kind.name(), syncCycleId);
        }

        public EndpointTrace.Poll pollTrace() {
            return pollTrace;
        }

        @Override
        public void close() {
            if (slot.compareAndSet(this, null)) {
                pollTrace.record(Stage.LEASE_RELEASED);
            }
        }
    }

    private record StatusPollProgress(long sinceUs, long rpcTimeoutUs, boolean awaitingCommit) { }

    /** Bounded transport/local-commit grace; this never refreshes scheduling capacity. */
    public boolean isStatusPollProtected(long nowUs, long commitGraceUs) {
        PollLease lease = statusPollLease.get();
        StatusPollProgress progress = lease == null ? null : lease.statusProgress;
        if (progress == null || !isActiveGeneration()) {
            return false;
        }
        long budgetUs = progress.awaitingCommit() ? commitGraceUs : progress.rpcTimeoutUs();
        long ageUs = nowUs - progress.sinceUs();
        return ageUs >= 0L && ageUs <= budgetUs;
    }

    private WorkerStatus(
            TopologySnapshot initialTopology,
            EngineObservation initialStatus, long maximumStatusAgeUs) {
        topology = new AtomicReference<>(initialTopology);
        committedStatus = new AtomicReference<>(new CommittedWorkerStatus(
                initialStatus,
                new AppliedStatusCursor(-1L, -1L), 0L));
        statusProgress = new StatusProgress(maximumStatusAgeUs);
    }

    /** Create one fully initialized service-discovery generation. */
    public static WorkerStatus createDiscovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site) {
        return createDiscovered(role, group, ip, port, grpcPort, site, DEFAULT_STATUS_FRESHNESS_US);
    }

    public static WorkerStatus createDiscovered(
            RoleType role, String group, String ip, int port, int grpcPort, String site,
            long maximumStatusAgeUs) {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(ip, "ip");
        if (port <= 0 || grpcPort <= 0) {
            throw new IllegalArgumentException(
                    "worker ports must be positive");
        }
        return new WorkerStatus(
                new TopologySnapshot(group, ip, port, grpcPort, site),
                new EngineObservation(
                        role,
                        null,
                        0L,
                        0L,
                        Map.of(),
                        0.0,
                        0L,
                        0L,
                        0L,
                        0L,
                        0L,
                        0L,
                        0L,
                        0L), maximumStatusAgeUs);
    }

    @JsonIgnore
    public long getGenerationId() {
        return generationId;
    }

    public TopologySnapshot topologySnapshot() {
        return topology.get();
    }

    public CommittedWorkerStatus committedWorkerStatus() {
        return committedStatus.get();
    }

    public EngineObservation committedEngineObservation() {
        return committedStatus.get().fields();
    }

    public AppliedStatusCursor appliedStatusCursor() {
        return committedStatus.get().cursor();
    }

    /**
     * Deep-freeze one RPC response. Finished tasks remain response-local and
     * are never copied into the committed Engine observation.
     */
    public StatusObservation freezeStatusResponse(
            WorkerStatusResponse response) {
        Objects.requireNonNull(response, "response");
        Map<String, TaskObservation> runningTasks = freezeTaskMap(
                response.getRunningTaskInfo());
        Map<String, TaskObservation> finishedTasks = freezeTaskMap(
                response.getFinishedTaskInfo());
        EngineObservation engine = new EngineObservation(
                response.getRole(),
                response.getAvailableConcurrency(),
                response.getAvailableKvCacheTokens(),
                response.getTotalKvCacheTokens(),
                runningTasks,
                response.getStepLatencyMs(),
                response.getIterateCount(),
                response.getDpSize(),
                response.getTpSize(),
                response.getDpRank(),
                response.getMaxSeqLen(),
                response.getMaxBatchTokensSize(),
                response.getRunningQueryLen(),
                response.getWaitingQueryLen());
        return new StatusObservation(
                this,
                engine,
                response.isAlive(),
                response.getStatusVersion(),
                response.getLatestFinishedVersion(),
                finishedTasks);
    }

    /** Bind an already immutable protocol conversion to this exact generation. */
    public StatusObservation bindStatusObservation(
            EngineObservation engine,
            boolean reportedAlive,
            Long statusVersion,
            Long latestFinishedVersion,
            Map<String, TaskObservation> finishedTasks) {
        return new StatusObservation(
                this,
                Objects.requireNonNull(engine, "engine"),
                reportedAlive,
                statusVersion,
                latestFinishedVersion,
                finishedTasks == null || finishedTasks.isEmpty()
                        ? Map.of() : Map.copyOf(finishedTasks));
    }

    /** Carry poll correlation through the role reducer to the actual commit boundary. */
    public PreparedStatus prepareNewStatus(StatusObservation observation, EndpointTrace.Poll pollTrace) {
        requireGenerationLock();
        requireActiveGeneration();
        if (observation.owner != this) {
            throw new IllegalArgumentException(
                    "status observation belongs to another worker generation");
        }
        CommittedWorkerStatus current = committedStatus.get();
        Long responseVersion = observation.statusVersion();
        if (responseVersion == null || responseVersion <= 0L
                || responseVersion <= current.cursor().statusVersion()) {
            throw new IllegalArgumentException(
                    "new status version must strictly advance committed version: committed="
                            + current.cursor().statusVersion()
                            + ", response=" + responseVersion);
        }
        AppliedStatusCursor cursor = new AppliedStatusCursor(
                responseVersion,
                mergedFinishedVersion(current.cursor(),
                        observation.latestFinishedVersion()));
        return new PreparedStatus(
                current,
                observation,
                new CommittedWorkerStatus(observation.engine(), cursor, statusProgress.applicationReceivedUs()),
                pollTrace);
    }

    /**
     * Publish fields and both cursors at one linearization point after every
     * response reducer has succeeded.
     */
    public void publishPreparedStatus(PreparedStatus prepared) {
        requireGenerationLock();
        requireActiveGeneration();
        if (prepared.observation.owner != this) {
            throw new IllegalArgumentException(
                    "prepared status belongs to another worker generation");
        }
        Long responseVersion = prepared.observation.statusVersion();
        if (responseVersion == null || responseVersion <= 0L
                || responseVersion
                        <= prepared.baseCommitted.cursor().statusVersion()) {
            throw new IllegalStateException(
                    "invalid prepared status version: committed="
                            + prepared.baseCommitted.cursor().statusVersion()
                            + ", response=" + responseVersion);
        }
        if (!committedStatus.compareAndSet(
                prepared.baseCommitted, prepared.nextCommitted)) {
            throw new IllegalStateException(
                    "prepared status is stale or has already been published");
        }
        if (prepared.pollTrace != null) {
            prepared.pollTrace.record(Stage.STATUS_COMMITTED);
        }
    }

    /** Record one validated status response independently from Engine state. */
    public PollHealth recordSuccessfulPoll(boolean reportedAlive, EndpointTrace.Poll pollTrace) {
        requireGenerationLock();
        requireActiveGeneration();
        long nowUs = System.nanoTime() / 1000;
        refreshAppliedReceipt(nowUs);
        PollHealth updated = statusProgress.successfulPoll(reportedAlive, nowUs);
        if (pollTrace != null) {
            pollTrace.record(Stage.STATUS_HEALTH_UPDATED);
        }
        return updated;
    }

    /** Record a transport failure without rewriting reported Engine state. */
    public PollHealth recordTransportFailure() {
        requireGenerationLock();
        requireActiveGeneration();
        return statusProgress.failedPoll();
    }

    public void requireGenerationLock() {
        if (!lock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "status transaction requires generation lock");
        }
    }

    /** Reject any new publication or ownership after retirement starts. */
    public void requireActiveGeneration() {
        requireGenerationLock();
        if (!isActiveGeneration()) {
            throw new IllegalStateException(
                    "WorkerStatus generation is retiring: " + generationId);
        }
    }

    /** Require an operation to target the one retiring generation. */
    public void requireRetiringGeneration() {
        requireGenerationLock();
        if (isActiveGeneration()) {
            throw new IllegalStateException(
                    "WorkerStatus generation is not retiring: " + generationId);
        }
    }

    /**
     * Publish this generation as RETIRING after its exact endpoint admission
     * gate has closed. The status-map identity remains until endpoint drain
     * succeeds.
     *
     * @return {@code true} only for the caller which starts retirement
     */
    public boolean beginRetirementAfterEndpointGateClosed() {
        requireGenerationLock();
        if (!statusProgress.beginRetirement()) { return false; }
        trace.record(Stage.GENERATION_RETIRING);
        return true;
    }

    @JsonIgnore
    public boolean isActiveGeneration() {
        return statusProgress.lifecycle != StatusLifecycle.RETIRING;
    }

    private static long mergedFinishedVersion(
            AppliedStatusCursor current,
            Long responseFinishedVersion) {
        return responseFinishedVersion == null
                ? current.latestFinishedTaskVersion()
                : Math.max(current.latestFinishedTaskVersion(),
                        responseFinishedVersion);
    }

    /** Atomically refresh discovery-owned placement labels for this generation. */
    public void updateDiscoveryLabels(String site, String group) {
        requireGenerationLock();
        requireActiveGeneration();
        topology.updateAndGet(current -> new TopologySnapshot(
                group, current.ip(), current.port(), current.grpcPort(), site));
    }

    public RoleType getRole() {
        return committedStatus.get().fields().role();
    }

    public String getGroup() {
        return topology.get().group();
    }

    public String getIp() {
        return topology.get().ip();
    }

    public int getPort() {
        return topology.get().port();
    }

    public int getGrpcPort() {
        return topology.get().grpcPort();
    }

    public String getSite() {
        return topology.get().site();
    }

    /**
     * Compatibility-only mirror of WorkerStatusPB.available_concurrency.
     * LocalRpcServer currently leaves that protobuf field unset, so this value
     * must not participate in routing, admission control, or batch sizing.
     */
    public Long getAvailableConcurrency() {
        return committedStatus.get().fields().availableConcurrency();
    }

    public long getAvailableKvCacheTokens() {
        return committedStatus.get().fields().availableKvCacheTokens();
    }

    public long getTotalKvCacheTokens() {
        return committedStatus.get().fields().totalKvCacheTokens();
    }

    public CacheStatus getCacheStatus() {
        return cacheIndexSnapshot.get().cacheStatus();
    }

    public CacheIndexSnapshot cacheIndexSnapshot() {
        return cacheIndexSnapshot.get();
    }

    public void publishCacheStatus(CacheStatus cacheStatus) {
        requireCacheMutationLock();
        CacheStatus next = Objects.requireNonNull(cacheStatus, "cacheStatus");
        cacheIndexSnapshot.updateAndGet(current -> new CacheIndexSnapshot(
                next, current.indexInitialized(), current.indexedVersion()));
    }

    public long getCacheIndexedVersion() {
        return cacheIndexSnapshot.get().indexedVersion();
    }

    public void publishCacheIndexedVersion(long version) {
        requireCacheMutationLock();
        CacheIndexSnapshot current = cacheIndexSnapshot.get();
        if (current.cacheStatus() == null
                || current.cacheStatus().getVersion() != version) {
            throw new IllegalStateException(
                    "indexed cache version must match current cache status");
        }
        cacheIndexSnapshot.set(new CacheIndexSnapshot(
                current.cacheStatus(), true, version));
    }

    public long recordSuccessfulCachePoll() {
        long nowUs = System.nanoTime() / 1000;
        long previousUs = cacheLastUpdateTime.getAndSet(nowUs);
        return previousUs <= 0L ? 0L : Math.max(0L, nowUs - previousUs);
    }

    public double getStepLatencyMs() {
        return committedStatus.get().fields().stepLatencyMs();
    }

    public long getIterateCount() {
        return committedStatus.get().fields().iterateCount();
    }

    public long getDpSize() {
        return committedStatus.get().fields().dpSize();
    }

    public long getTpSize() {
        return committedStatus.get().fields().tpSize();
    }

    public long getDpRank() {
        return committedStatus.get().fields().dpRank();
    }

    /** Model-level maximum sequence length reported by the Engine. */
    public long getMaxSeqLen() {
        return committedStatus.get().fields().maxSeqLen();
    }

    /** Strict aggregate context-token limit for an Engine batch/group. */
    public long getMaxBatchTokensSize() {
        return committedStatus.get().fields().maxBatchTokensSize();
    }

    public PollHealth pollHealth() {
        return statusProgress.health;
    }

    /** Acquire the one in-flight status-poll slot for this generation. */
    public PollLease tryBeginStatusPoll(long syncCycleId) {
        if (statusProgress.lifecycle != StatusLifecycle.ACTIVE) { return null; }
        PollLease active = statusPollLease.get();
        if (active != null) { active.pollTrace.skippedBusy(); return null; }
        PollLease lease = new PollLease(this, PollKind.STATUS, statusPollLease, syncCycleId);
        if (!statusPollLease.compareAndSet(null, lease)) { return null; }
        if (statusProgress.lifecycle != StatusLifecycle.ACTIVE) { lease.close(); return null; }
        return lease;
    }

    public void requireStatusPollLease(PollLease lease) {
        requirePollLease(lease, PollKind.STATUS, statusPollLease);
    }

    /** Acquire the independent cache-poll slot for this generation. */
    public PollLease tryBeginCachePoll(long syncCycleId) {
        return tryBeginPoll(PollKind.CACHE, cachePollLease, syncCycleId);
    }

    public void requireCachePollLease(PollLease lease) {
        requirePollLease(lease, PollKind.CACHE, cachePollLease);
    }

    private PollLease tryBeginPoll(PollKind kind, AtomicReference<PollLease> slot,
                                   long syncCycleId) {
        if (!isActiveGeneration()) { return null; }
        PollLease active = slot.get();
        if (active != null) { active.pollTrace.skippedBusy(); return null; }
        PollLease lease = new PollLease(this, kind, slot, syncCycleId);
        if (!slot.compareAndSet(null, lease)) { return null; }
        if (!isActiveGeneration()) { lease.close(); return null; }
        return lease;
    }

    private void requireCacheMutationLock() {
        if (!cacheMutationLock.isHeldByCurrentThread()) {
            throw new IllegalStateException("cache publication requires cache mutation lock");
        }
    }

    private void requirePollLease(
            PollLease lease,
            PollKind expectedKind,
            AtomicReference<PollLease> slot) {
        if (lease == null
                || lease.owner != this
                || lease.kind != expectedKind
                || slot.get() != lease) {
            throw new IllegalArgumentException(
                    "poll lease is not the active "
                            + expectedKind.name().toLowerCase()
                            + " lease for this worker generation");
        }
    }

    /** Get the HTTP IP:PORT address. */
    public String getIpPort() {
        TopologySnapshot current = topology.get();
        if (current.ip() == null) {
            return null;
        }
        return current.ip() + ":" + current.port();
    }

    private static Map<String, TaskObservation> freezeTaskMap(
            Map<String, TaskInfo> tasks) {
        if (tasks == null || tasks.isEmpty()) {
            return Map.of();
        }
        Map<String, TaskObservation> frozen = new HashMap<>(tasks.size());
        for (Map.Entry<String, TaskInfo> entry : tasks.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                frozen.put(entry.getKey(), TaskObservation.copyOf(
                        entry.getValue()));
            }
        }
        return Map.copyOf(frozen);
    }

    @Override
    public String toString() {
        return "WorkerStatus{" +
                "generationId=" + generationId +
                ", retiring=" + !isActiveGeneration() +
                ", topology=" + topology.get() +
                ", committedStatus=" + committedStatus.get() +
                ", pollHealth=" + pollHealth() +
                '}';
    }
}

package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Represents one discovered worker generation; object identity distinguishes replacements at the
 * same address. Status reduction precedes publication of one committed observation, while topology,
 * poll health and cache polling retain independent update paths.
 */
public class WorkerStatus {
    private static final AtomicLong NEXT_GENERATION_ID = new AtomicLong();

    /** Serializes lifecycle transitions for this generation. */
    public final transient ReentrantLock lock = new ReentrantLock();

    private final long generationId = NEXT_GENERATION_ID.incrementAndGet();

    /** Set under {@link #lock}; read lock-free by poll gates. */
    private volatile boolean retiring;

    private final AtomicReference<TopologySnapshot> topology;
    private final WorkerIdentity workerIdentity;
    private final AtomicReference<CommittedWorkerStatus> committedStatus;
    private final AtomicReference<PollHealth> pollHealth;
    private final Object transportHealthLock = new Object();

    private volatile boolean healthRetirementClaimed;
    private volatile boolean reportedDead;
    private volatile long lastStatusAppliedUs;
    private volatile long statusFreshnessUs;

    private final AtomicReference<PollLease> statusApplicationLease = new AtomicReference<>();

    /** Cache polling is independent from worker-status polling. */
    private final AtomicReference<CacheIndexSnapshot> cacheIndexSnapshot =
            new AtomicReference<>(new CacheIndexSnapshot(null, false, -1L));

    private final AtomicLong cacheLastUpdateTime = new AtomicLong(-1L);
    private final AtomicReference<PollLease> statusPollLease = new AtomicReference<>();
    private final AtomicReference<PollLease> cachePollLease = new AtomicReference<>();
    private final AtomicLong vitRoutedInputs = new AtomicLong();

    private volatile VitLoad vitLoad;

    private WorkerStatus(TopologySnapshot initialTopology, EngineObservation initialStatus) {
        topology = new AtomicReference<>(initialTopology);
        workerIdentity =
                new WorkerIdentity(
                        initialTopology.ip(),
                        initialTopology.port(),
                        initialTopology.engineIndex());
        committedStatus =
                new AtomicReference<>(
                        new CommittedWorkerStatus(
                                initialStatus, new AppliedStatusCursor(-1L, -1L)));
        long discoveredAtUs = System.nanoTime() / 1000;
        pollHealth = new AtomicReference<>(new PollHealth(discoveredAtUs, 0L, 0L, false));
    }

    /** Create one fully initialized service-discovery generation. */
    public static WorkerStatus createDiscovered(
            RoleType role, String group, String ip, int port, int grpcPort, String site) {
        return createDiscovered(role, group, ip, port, grpcPort, site, null);
    }

    public static WorkerStatus createDiscovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site,
            String deploymentName) {
        return createDiscovered(role, group, ip, port, grpcPort, site, deploymentName, 0, 1);
    }

    public static WorkerStatus createDiscovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site,
            String deploymentName,
            int engineIndex,
            int multiEngineNum) {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(ip, "ip");
        if (port <= 0 || grpcPort <= 0) {
            throw new IllegalArgumentException("worker ports must be positive");
        }
        if (engineIndex < 0 || multiEngineNum <= 0 || engineIndex >= multiEngineNum) {
            throw new IllegalArgumentException("invalid logical worker index");
        }
        return new WorkerStatus(
                new TopologySnapshot(
                        group,
                        ip,
                        port,
                        grpcPort,
                        site,
                        deploymentName,
                        engineIndex,
                        multiEngineNum),
                new EngineObservation(
                        role, null, 0L, 0L, Map.of(), 0.0, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L));
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

    /** Observe assignments before submitting the ViT status RPC. */
    public long vitRoutedInputs() {
        return vitRoutedInputs.get();
    }

    /** Charge the selected generation for the media inputs routed to it. */
    public void recordVitRoute(int mediaCount) {
        vitRoutedInputs.addAndGet(mediaCount);
    }

    /**
     * Replace the accepted ViT sample with the receive time and the RPC submission cursor.
     */
    public void recordVitLoad(
            OptionalLong reportedInputs, long routedAtPollStart, long receivedNanos) {
        vitLoad =
                reportedInputs.isEmpty() || reportedInputs.getAsLong() < 0
                        ? null
                        : new VitLoad(reportedInputs.getAsLong(), routedAtPollStart, receivedNanos);
    }

    /** Estimate inflight media from a recent sample and the subsequent assignments. */
    public long vitLoadEstimate(long nowNanos, long maxAgeNanos) {
        VitLoad sample = vitLoad;
        if (sample == null || nowNanos - sample.receivedNanos() > maxAgeNanos) {
            return Long.MAX_VALUE;
        }
        long unreported = vitRoutedInputs.get() - sample.routedAtPollStart();
        return Math.min(sample.reportedInputs(), Long.MAX_VALUE - unreported) + unreported;
    }

    /**
     * Deep-freeze one RPC response. Finished tasks remain response-local and
     * are never copied into the committed Engine observation.
     */
    public StatusObservation freezeStatusResponse(WorkerStatusResponse response) {
        Objects.requireNonNull(response, "response");
        Map<String, TaskObservation> runningTasks = freezeTaskMap(response.getRunningTaskInfo());
        Map<String, TaskObservation> finishedTasks = freezeTaskMap(response.getFinishedTaskInfo());
        EngineObservation engine =
                new EngineObservation(
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
                        response.getCacheStatus() == null
                                ? 0L
                                : response.getCacheStatus().getBlockSize(),
                        response.getBlockHashLookaheadTokens(),
                        response.getCacheMatchRollbackBlocks(),
                        response.getKvCacheGroupMode(),
                        response.getMaxSeqLen(),
                        response.getMaxBatchTokensSize(),
                        response.getRunningQueryLen(),
                        response.getWaitingQueryLen(),
                        response.getLastStepMetrics());
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
                        ? Map.of()
                        : Map.copyOf(finishedTasks));
    }

    /** Bind a strictly newer frozen observation to the current commit holder. */
    public PreparedStatus prepareNewStatus(StatusObservation observation) {
        requireGenerationLock();
        requireActiveGeneration();
        if (observation.owner != this) {
            throw new IllegalArgumentException(
                    "status observation belongs to another worker generation");
        }
        CommittedWorkerStatus current = committedStatus.get();
        Long responseVersion = observation.statusVersion();
        if (responseVersion == null
                || responseVersion <= 0L
                || responseVersion <= current.cursor().statusVersion()) {
            throw new IllegalArgumentException(
                    "new status version must strictly advance committed version: committed="
                            + current.cursor().statusVersion()
                            + ", response="
                            + responseVersion);
        }
        AppliedStatusCursor cursor =
                new AppliedStatusCursor(
                        responseVersion,
                        mergedFinishedVersion(
                                current.cursor(), observation.latestFinishedVersion()));
        return new PreparedStatus(
                current, observation, new CommittedWorkerStatus(observation.engine(), cursor));
    }

    public void requireGenerationLock() {
        if (!lock.isHeldByCurrentThread()) {
            throw new IllegalStateException("status transaction requires generation lock");
        }
    }

    /** Reject any new publication or ownership after retirement starts. */
    public void requireActiveGeneration() {
        requireGenerationLock();
        if (retiring) {
            throw new IllegalStateException("WorkerStatus generation is retiring: " + generationId);
        }
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
        if (responseVersion == null
                || responseVersion <= 0L
                || responseVersion <= prepared.baseCommitted.cursor().statusVersion()) {
            throw new IllegalStateException(
                    "invalid prepared status version: committed="
                            + prepared.baseCommitted.cursor().statusVersion()
                            + ", response="
                            + responseVersion);
        }
        if (!committedStatus.compareAndSet(prepared.baseCommitted, prepared.nextCommitted)) {
            throw new IllegalStateException(
                    "prepared status is stale or has already been published");
        }
    }

    /** Record contact while the RPC lease is still held, independently of local application. */
    public void recordStatusRpcResult(
            PollLease lease,
            long receivedUs,
            boolean valid,
            boolean alive,
            boolean transportFailure) {
        synchronized (transportHealthLock) {
            if (statusPollLease.get() != lease
                    || retiring
                    || healthRetirementClaimed
                    || reportedDead) {
                return;
            }
            if (valid) {
                recordSuccessfulPoll(alive, receivedUs);
                reportedDead = !alive;
            } else if (transportFailure) {
                PollHealth current = pollHealth.get();
                long failures = current.consecutiveTransportFailures();
                pollHealth.set(
                        new PollHealth(
                                current.lastSuccessfulPollUs(),
                                current.successfulPollIntervalUs(),
                                failures == Long.MAX_VALUE ? failures : failures + 1L,
                                current.reportedAlive()));
            }
        }
    }

    /** Record one validated status response independently from Engine state. */
    public PollHealth recordSuccessfulPoll(boolean reportedAlive) {
        requireGenerationLock();
        requireActiveGeneration();
        synchronized (transportHealthLock) {
            return recordSuccessfulPoll(reportedAlive, System.nanoTime() / 1000);
        }
    }

    /** Atomically fence retirement against a successful contact since the scanner read health. */
    public boolean tryClaimHealthRetirement(long currentUs, long maximumAgeUs) {
        requireGenerationLock();
        synchronized (transportHealthLock) {
            if (retiring
                    || healthRetirementClaimed
                    || (!reportedDead
                            && currentUs - pollHealth.get().lastSuccessfulPollUs()
                                    <= maximumAgeUs)) {
                return false;
            }
            healthRetirementClaimed = true;
            return true;
        }
    }

    /** Enable admission freshness for a generation managed by status polling. */
    public void configureStatusFreshness(long maximumAgeUs) {
        statusFreshnessUs = maximumAgeUs;
    }

    /** Check whether this generation can accept new work with its current applied status. */
    public boolean hasFreshSchedulingStatus(long nowUs) {
        long maximumAgeUs = statusFreshnessUs;
        long appliedUs = lastStatusAppliedUs;
        return !healthRetirementClaimed
                && !reportedDead
                && (maximumAgeUs == 0L || (appliedUs != 0L && nowUs - appliedUs <= maximumAgeUs));
    }

    /** Keep at most one queued or running state application per generation. */
    public boolean tryBeginStatusApplication(PollLease lease) {
        return !retiring
                && !healthRetirementClaimed
                && statusApplicationLease.compareAndSet(null, lease);
    }

    public void finishStatusApplication(PollLease lease) {
        statusApplicationLease.compareAndSet(lease, null);
    }

    /** The age of applied state starts when it was received, including time spent queued. */
    public void recordStatusApplied(long receivedUs) {
        requireGenerationLock();
        lastStatusAppliedUs = receivedUs;
    }

    /** Require an operation to target the one retiring generation. */
    public void requireRetiringGeneration() {
        requireGenerationLock();
        if (!retiring) {
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
        if (retiring) {
            return false;
        }
        retiring = true;
        return true;
    }

    @JsonIgnore
    public boolean isActiveGeneration() {
        return !retiring;
    }

    /** Atomically refresh discovery-owned placement labels for this generation. */
    public void updateDiscoveryLabels(String site, String group) {
        updateDiscoveryLabels(site, group, topology.get().deploymentName());
    }

    public void updateDiscoveryLabels(String site, String group, String deploymentName) {
        requireGenerationLock();
        requireActiveGeneration();
        topology.updateAndGet(
                current ->
                        new TopologySnapshot(
                                group,
                                current.ip(),
                                current.port(),
                                current.grpcPort(),
                                site,
                                deploymentName,
                                current.engineIndex(),
                                current.multiEngineNum()));
    }

    public String routingAddress() {
        return getRole() == RoleType.VIT
                ? workerIdentity.getPhysicalIpPort()
                : workerIdentity.getLogicalIpPort();
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

    public String getDeploymentName() {
        return topology.get().deploymentName();
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
        requireGenerationLock();
        requireActiveGeneration();
        CacheStatus next = Objects.requireNonNull(cacheStatus, "cacheStatus");
        cacheIndexSnapshot.updateAndGet(
                current ->
                        new CacheIndexSnapshot(
                                next, current.indexInitialized(), current.indexedVersion()));
    }

    public long getCacheIndexedVersion() {
        return cacheIndexSnapshot.get().indexedVersion();
    }

    public void publishCacheIndexedVersion(long version) {
        requireGenerationLock();
        requireActiveGeneration();
        CacheIndexSnapshot current = cacheIndexSnapshot.get();
        if (current.cacheStatus() == null || current.cacheStatus().getVersion() != version) {
            throw new IllegalStateException(
                    "indexed cache version must match current cache status");
        }
        cacheIndexSnapshot.set(new CacheIndexSnapshot(current.cacheStatus(), true, version));
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

    public int getBlockHashLookaheadTokens() {
        return committedStatus.get().fields().blockHashLookaheadTokens();
    }

    public int getCacheMatchRollbackBlocks() {
        return committedStatus.get().fields().cacheMatchRollbackBlocks();
    }

    public KvCacheGroupMode getKvCacheGroupMode() {
        return committedStatus.get().fields().kvCacheGroupMode();
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
        return pollHealth.get();
    }

    public boolean isAlive() {
        return pollHealth.get().reportedAlive();
    }

    /** Acquire the one in-flight status-poll slot for this generation. */
    public PollLease tryBeginStatusPoll() {
        return reportedDead ? null : tryBeginPoll(PollKind.STATUS, statusPollLease);
    }

    public void requireStatusPollLease(PollLease lease) {
        requirePollLease(lease, PollKind.STATUS, statusPollLease);
    }

    /** Acquire the independent cache-poll slot for this generation. */
    public PollLease tryBeginCachePoll() {
        return tryBeginPoll(PollKind.CACHE, cachePollLease);
    }

    public void requireCachePollLease(PollLease lease) {
        requirePollLease(lease, PollKind.CACHE, cachePollLease);
    }

    /** Get the HTTP IP:PORT address. */
    public String getIpPort() {
        return workerIdentity.getPhysicalIpPort();
    }

    public WorkerIdentity getWorkerIdentity() {
        return workerIdentity;
    }

    public String getPhysicalIpPort() {
        return workerIdentity.getPhysicalIpPort();
    }

    public String getLogicalIpPort() {
        return workerIdentity.getLogicalIpPort();
    }

    @JsonIgnore
    public String getMetricIpPort() {
        return workerIdentity.getMetricIpPort(getMultiEngineNum());
    }

    public int getMultiEngineNum() {
        return topology.get().multiEngineNum();
    }

    public int getEngineIndex() {
        return workerIdentity.getEngineIndex();
    }

    @Override
    public String toString() {
        return "WorkerStatus{"
                + "generationId="
                + generationId
                + ", retiring="
                + retiring
                + ", topology="
                + topology.get()
                + ", committedStatus="
                + committedStatus.get()
                + ", pollHealth="
                + pollHealth.get()
                + '}';
    }

    private PollHealth recordSuccessfulPoll(boolean reportedAlive, long nowUs) {
        return pollHealth.updateAndGet(
                current ->
                        new PollHealth(
                                nowUs, nowUs - current.lastSuccessfulPollUs(), 0L, reportedAlive));
    }

    private PollLease tryBeginPoll(PollKind kind, AtomicReference<PollLease> slot) {
        if (retiring || healthRetirementClaimed || slot.get() != null) {
            return null;
        }
        PollLease lease = new PollLease(this, kind, slot);
        if (!slot.compareAndSet(null, lease)) {
            return null;
        }
        if (retiring || healthRetirementClaimed) {
            lease.close();
            return null;
        }
        return lease;
    }

    private static long mergedFinishedVersion(
            AppliedStatusCursor current, Long responseFinishedVersion) {
        return responseFinishedVersion == null
                ? current.latestFinishedTaskVersion()
                : Math.max(current.latestFinishedTaskVersion(), responseFinishedVersion);
    }

    private void requirePollLease(
            PollLease lease, PollKind expectedKind, AtomicReference<PollLease> slot) {
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

    private static Map<String, TaskObservation> freezeTaskMap(Map<String, TaskInfo> tasks) {
        if (tasks == null || tasks.isEmpty()) {
            return Map.of();
        }
        Map<String, TaskObservation> frozen = new HashMap<>(tasks.size());
        for (Map.Entry<String, TaskInfo> entry : tasks.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                frozen.put(entry.getKey(), TaskObservation.copyOf(entry.getValue()));
            }
        }
        return Map.copyOf(frozen);
    }

    /**
     * Freezes a worker generation's discovery labels separately from its engine observations.
     */
    public record TopologySnapshot(
            String group,
            String ip,
            int port,
            int grpcPort,
            String site,
            String deploymentName,
            int engineIndex,
            int multiEngineNum) {

        public TopologySnapshot(String group, String ip, int port, int grpcPort, String site) {
            this(group, ip, port, grpcPort, site, null, 0, 1);
        }
    }

    /**
     * Deeply immutable copy of the task fields reported by one status RPC.
     */
    public record TaskObservation(
            String requestId,
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
            PriorityPreemptionProgress priorityPreemptionProgress,
            TaskTelemetry telemetry) {

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
                    task.getPriorityPreemptionProgress(),
                    new TaskTelemetry(
                            task.isPrefixLengthValid(),
                            task.getRequestReceivedTimeMs(),
                            task.getInputQueueEnqueueTimeMs(),
                            task.getInputQueueDrainTimeMs(),
                            task.getWaitingEnteredTimeMs(),
                            task.getRunningEnteredTimeMs(),
                            task.getRemoteKvWaitMs(),
                            task.getFirstTokenTimeMs(),
                            task.getHbmLocalMatchTokens(),
                            task.getRemoteKvAddedMatchTokens(),
                            task.getFirstPrefillStepId(),
                            task.getLastPrefillStepId(),
                            task.getPrefillStepCount(),
                            task.getPrefillNonfinalChunkTokensMin(),
                            task.getPrefillNonfinalChunkTokensMax(),
                            task.getCompletedPrefillTokens(),
                            task.getRemainingPrefillTokens() < 0
                                    ? null
                                    : task.getRemainingPrefillTokens(),
                            task.getLastCompletedPrefillStepId()));
        }
    }

    /**
     * Immutable request telemetry retained from the Engine status RPC.
     * Nullable prefill progress values preserve omitted protobuf fields.
     */
    public record TaskTelemetry(
            boolean prefixLengthValid,
            long requestReceivedTimeMs,
            long inputQueueEnqueueTimeMs,
            long inputQueueDrainTimeMs,
            long waitingEnteredTimeMs,
            long runningEnteredTimeMs,
            long remoteKvWaitMs,
            long firstTokenTimeMs,
            long hbmLocalMatchTokens,
            long remoteKvAddedMatchTokens,
            long firstPrefillStepId,
            long lastPrefillStepId,
            long prefillStepCount,
            long prefillNonfinalChunkTokensMin,
            long prefillNonfinalChunkTokensMax,
            Long completedPrefillTokens,
            Long remainingPrefillTokens,
            Long lastCompletedPrefillStepId) {

        public TaskTelemetry(
                boolean prefixLengthValid,
                long requestReceivedTimeMs,
                long inputQueueEnqueueTimeMs,
                long inputQueueDrainTimeMs,
                long waitingEnteredTimeMs,
                long runningEnteredTimeMs,
                long remoteKvWaitMs,
                long firstTokenTimeMs,
                long hbmLocalMatchTokens,
                long remoteKvAddedMatchTokens,
                long firstPrefillStepId,
                long lastPrefillStepId,
                long prefillStepCount,
                long prefillNonfinalChunkTokensMin,
                long prefillNonfinalChunkTokensMax) {
            this(
                    prefixLengthValid,
                    requestReceivedTimeMs,
                    inputQueueEnqueueTimeMs,
                    inputQueueDrainTimeMs,
                    waitingEnteredTimeMs,
                    runningEnteredTimeMs,
                    remoteKvWaitMs,
                    firstTokenTimeMs,
                    hbmLocalMatchTokens,
                    remoteKvAddedMatchTokens,
                    firstPrefillStepId,
                    lastPrefillStepId,
                    prefillStepCount,
                    prefillNonfinalChunkTokensMin,
                    prefillNonfinalChunkTokensMax,
                    null,
                    null,
                    null);
        }
    }

    public record StepMetrics(
            long stepId,
            long completedTimeMs,
            long totalScheduledTokens,
            long prefillRequestCount,
            long prefillTokens,
            long tokenBudget,
            double budgetFillRatio) {}

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
            long blockSize,
            int blockHashLookaheadTokens,
            int cacheMatchRollbackBlocks,
            KvCacheGroupMode kvCacheGroupMode,
            long maxSeqLen,
            long maxBatchTokensSize,
            long runningQueryLen,
            long waitingQueryLen,
            StepMetrics lastStepMetrics) {

        public EngineObservation {
            Objects.requireNonNull(role, "role");
            kvCacheGroupMode =
                    kvCacheGroupMode == null ? KvCacheGroupMode.UNSPECIFIED : kvCacheGroupMode;
            runningTaskList = Map.copyOf(runningTaskList);
        }

        public EngineObservation(
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
            this(
                    role,
                    availableConcurrency,
                    availableKvCacheTokens,
                    totalKvCacheTokens,
                    runningTaskList,
                    stepLatencyMs,
                    iterateCount,
                    dpSize,
                    tpSize,
                    dpRank,
                    0L,
                    0,
                    0,
                    KvCacheGroupMode.UNSPECIFIED,
                    maxSeqLen,
                    maxBatchTokensSize,
                    runningQueryLen,
                    waitingQueryLen,
                    null);
        }
    }

    /**
     * The two cursors acknowledged by one successful status transaction.
     * Keeping them in one immutable value prevents readers from observing a
     * committed status version with an older finished-task cursor.
     */
    public record AppliedStatusCursor(long statusVersion, long latestFinishedTaskVersion) {}

    /**
     * The single public value for one successfully committed status round.
     */
    public record CommittedWorkerStatus(EngineObservation fields, AppliedStatusCursor cursor) {}

    /**
     * Status-poll transport health. It is never consulted for routing: the
     * endpoint registry and the endpoint generation gate own routability.
     */
    public record PollHealth(
            long lastSuccessfulPollUs,
            long successfulPollIntervalUs,
            long consecutiveTransportFailures,
            boolean reportedAlive) {

        public PollHealth {
            if (lastSuccessfulPollUs < 0L
                    || successfulPollIntervalUs < 0L
                    || consecutiveTransportFailures < 0L) {
                throw new IllegalArgumentException("poll health counters must be non-negative");
            }
        }
    }

    public record CacheIndexSnapshot(
            CacheStatus cacheStatus, boolean indexInitialized, long indexedVersion) {}

    /**
     * One deeply frozen RPC observation shared by every status reducer.
     */
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

        /**
         * Returns all active task phases reported in running_task_info.
         */
        public Map<String, TaskObservation> activeTasks() {
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

        private PreparedStatus(
                CommittedWorkerStatus baseCommitted,
                StatusObservation observation,
                CommittedWorkerStatus nextCommitted) {
            this.baseCommitted = baseCommitted;
            this.observation = observation;
            this.nextCommitted = nextCommitted;
        }

        public StatusObservation observation() {
            return observation;
        }
    }

    private enum PollKind {
        STATUS,
        CACHE
    }

    private record VitLoad(long reportedInputs, long routedAtPollStart, long receivedNanos) {}

    /**
     * Exact ownership of one generation-local poll slot. Closing is
     * token-fenced and idempotent: a delayed or duplicate close cannot release
     * a later poll which acquired the same logical slot.
     */
    public static final class PollLease implements AutoCloseable {
        private final WorkerStatus owner;
        private final PollKind kind;
        private final AtomicReference<PollLease> slot;

        private PollLease(WorkerStatus owner, PollKind kind, AtomicReference<PollLease> slot) {
            this.owner = owner;
            this.kind = kind;
            this.slot = slot;
        }

        @Override
        public void close() {
            slot.compareAndSet(this, null);
        }
    }
}

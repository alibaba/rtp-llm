package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.priority.ReleaseTracker;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerLifecycleState;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.ObjectFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

@Component
public class EndpointRegistry {

    private final ConcurrentHashMap<String, PrefillEndpoint> prefillEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, PrefillEndpoint> pdFusionEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, SimpleWorkerEndpoint> vitEndpoints = new ConcurrentHashMap<>();
    private final ConfigService configService;
    private final ObjectFactory<FlexlbBatchScheduler> batchSchedulerFactory;
    private final BatchSchedulerReporter reporter;
    private final ConcurrentHashMap<EndpointKey, EndpointSlot> endpointSlots = new ConcurrentHashMap<>();
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private record EndpointKey(RoleType role, String ipPort) {
    }

    /** Serialized publication/retirement state for one role+address slot. */
    private static final class EndpointSlot {
        long lastGeneration;
        CompletableFuture<Void> retirementBarrier;
        RetirementRecord retirement;
    }

    /**
     * Durable state for a generation that has been unpublished but is not yet
     * safe to replace.  The record deliberately outlives an individual
     * retirement attempt: every cleanup operation is idempotent, so a later
     * reconciliation pass can finish a transiently failed stage without ever
     * admitting a new generation across an incomplete settlement.
     */
    private static final class RetirementRecord {
        final WorkerEndpoint endpoint;
        final WorkerStatus status;
        final EndpointRetireCause cause;
        final CompletableFuture<Void> barrier;
        List<BatchItem> drainedItems = List.of();
        boolean queueDrained;
        boolean attemptInProgress;
        Throwable lastFailure;

        RetirementRecord(WorkerEndpoint endpoint,
                         WorkerStatus status,
                         EndpointRetireCause cause,
                         CompletableFuture<Void> barrier) {
            this.endpoint = endpoint;
            this.status = status;
            this.cause = cause;
            this.barrier = barrier;
        }
    }

    public EndpointRegistry(ConfigService configService,
                            ObjectFactory<FlexlbBatchScheduler> batchSchedulerFactory,
                            BatchSchedulerReporter reporter) {
        this.configService = configService;
        this.batchSchedulerFactory = batchSchedulerFactory;
        this.reporter = reporter;
    }

    public WorkerEndpoint get(RoleType roleType, String ipPort) {
        WorkerEndpoint endpoint;
        if (roleType == RoleType.PREFILL) {
            endpoint = prefillEndpoints.get(ipPort);
        } else if (roleType == RoleType.DECODE) {
            endpoint = decodeEndpoints.get(ipPort);
        } else if (roleType == RoleType.PDFUSION) {
            endpoint = pdFusionEndpoints.get(ipPort);
        } else if (roleType == RoleType.VIT) {
            endpoint = vitEndpoints.get(ipPort);
        } else {
            endpoint = null;
        }
        return endpoint != null && endpoint.isReady() ? endpoint : null;
    }

    /**
     * Resolve exactly the endpoint generation selected during routing.
     *
     * <p>Address-only resolution is valid for making a fresh routing choice,
     * but not after a strategy has reserved/accounted a request. A replacement
     * endpoint at the same address must not inherit that request.
     */
    public WorkerEndpoint get(RoleType roleType, String ipPort, long generation) {
        WorkerEndpoint endpoint = get(roleType, ipPort);
        return endpoint != null && endpoint.getEndpointId().generation() == generation
                ? endpoint : null;
    }

    public Map<String, ? extends WorkerEndpoint> getEndpoints(RoleType roleType) {
        if (roleType == RoleType.PREFILL) {
            return prefillEndpoints;
        }
        if (roleType == RoleType.DECODE) {
            return decodeEndpoints;
        }
        if (roleType == RoleType.PDFUSION) {
            return pdFusionEndpoints;
        }
        if (roleType == RoleType.VIT) {
            return vitEndpoints;
        }
        return Map.of();
    }

    public PrefillEndpoint getPrefill(String ipPort) {
        PrefillEndpoint endpoint = prefillEndpoints.get(ipPort);
        return endpoint != null && endpoint.isReady() ? endpoint : null;
    }

    /** Resolve the exact prefill generation selected during routing. */
    public PrefillEndpoint getPrefill(String ipPort, long generation) {
        PrefillEndpoint endpoint = getPrefill(ipPort);
        return endpoint != null && endpoint.getEndpointId().generation() == generation
                ? endpoint : null;
    }

    public DecodeEndpoint getDecode(String ipPort) {
        DecodeEndpoint endpoint = decodeEndpoints.get(ipPort);
        return endpoint != null && endpoint.isReady() ? endpoint : null;
    }

    /** Resolve the exact decode generation selected during routing. */
    public DecodeEndpoint getDecode(String ipPort, long generation) {
        DecodeEndpoint endpoint = getDecode(ipPort);
        return endpoint != null && endpoint.getEndpointId().generation() == generation
                ? endpoint : null;
    }

    private PrefillEndpoint getPdFusion(String ipPort) {
        PrefillEndpoint endpoint = pdFusionEndpoints.get(ipPort);
        return endpoint != null && endpoint.isReady() ? endpoint : null;
    }

    private SimpleWorkerEndpoint getVit(String ipPort) {
        SimpleWorkerEndpoint endpoint = vitEndpoints.get(ipPort);
        return endpoint != null && endpoint.isReady() ? endpoint : null;
    }

    /**
     * Trusted compatibility publication path for manually assembled endpoints.
     *
     * <p>This preserves the pre-lifecycle API contract for tests and bootstrap
     * callers by explicitly promoting a PROBING status. Discovery/status sync
     * must never call this method: production publication goes exclusively
     * through {@link #publishValidatedEndpoint} after a validated live response.
     */
    @Deprecated
    public WorkerEndpoint ensureEndpoint(RoleType roleType, String ipPort, WorkerStatus status) {
        if (status == null) {
            return null;
        }
        if (status.getLifecycleState() == WorkerLifecycleState.PROBING
                && !status.tryMarkReady()) {
            return null;
        }
        if (!status.isReady()) {
            return null;
        }
        return publish(roleType, ipPort, status, null);
    }

    /**
     * Publish a PROBING worker only after its first validated live response has
     * initialized endpoint-local accounting. No uncalibrated endpoint is ever
     * visible from the READY maps.
     */
    public WorkerEndpoint publishValidatedEndpoint(RoleType roleType,
                                                   String ipPort,
                                                   WorkerStatus status,
                                                   WorkerStatusResponse response) {
        if (status == null || response == null
                || response.getRole() != roleType || !response.isAlive()
                || !ipPort.equals(status.getIpPort())
                || (status.getLifecycleState() != WorkerLifecycleState.PROBING
                && status.getLifecycleState() != WorkerLifecycleState.READY)) {
            return null;
        }
        return publish(roleType, ipPort, status, response);
    }

    private WorkerEndpoint publish(RoleType roleType,
                                   String ipPort,
                                   WorkerStatus status,
                                   WorkerStatusResponse response) {
        if (closed.get()) {
            return null;
        }
        EndpointKey key = new EndpointKey(roleType, ipPort);
        EndpointSlot slot = endpointSlots.computeIfAbsent(key, ignored -> new EndpointSlot());
        synchronized (slot) {
            if (closed.get() || slot.retirementBarrier != null) {
                return null;
            }
            WorkerEndpoint current = currentEndpoint(roleType, ipPort);
            if (current != null) {
                if (current.getStatus() != status || !current.isReady()) {
                    return null;
                }
                if (response != null) {
                    current.tryOnWorkerStatusUpdate(status, response);
                }
                return current;
            }

            EndpointId endpointId = new EndpointId(roleType, ipPort, ++slot.lastGeneration);
            if (response != null) {
                status.updateFromResponse(response);
            }
            WorkerEndpoint candidate = createEndpoint(endpointId, status);
            boolean published = false;
            try {
                if (response != null && !candidate.tryOnWorkerStatusUpdate(status, response)) {
                    return null;
                }
                if (status.getLifecycleState() == WorkerLifecycleState.PROBING
                        && !status.tryMarkReady()) {
                    return null;
                }
                if (!status.isReady()) {
                    return null;
                }
                putEndpoint(roleType, ipPort, candidate);
                published = true;
                return candidate;
            } finally {
                if (!published) {
                    candidate.close();
                }
            }
        }
    }

    /** Compatibility removal path. Prefer {@link #retire} with an explicit cause. */
    public boolean remove(RoleType roleType, String ipPort, WorkerStatus expectedStatus) {
        return retire(roleType, ipPort, expectedStatus, EndpointRetireCause.HEALTH_CHECK_FAILED);
    }

    /**
     * Retire exactly one status/endpoint generation through a publication
     * barrier and the scheduler's single settlement path.
     */
    public boolean retire(RoleType roleType,
                          String ipPort,
                          WorkerStatus expectedStatus,
                          EndpointRetireCause cause) {
        if (expectedStatus == null || cause == null) {
            return false;
        }
        EndpointKey key = new EndpointKey(roleType, ipPort);
        EndpointSlot slot = endpointSlots.computeIfAbsent(key, ignored -> new EndpointSlot());
        RetirementRecord retirement;

        synchronized (slot) {
            if (slot.retirementBarrier != null) {
                retirement = slot.retirement;
                // A later health/discovery pass may safely drive an earlier
                // failed retirement only for the exact same status object.
                // A different status must remain fenced behind the barrier.
                if (retirement == null || retirement.status != expectedStatus
                        || retirement.attemptInProgress) {
                    return false;
                }
                retirement.attemptInProgress = true;
            } else {
                WorkerEndpoint endpoint = currentEndpoint(roleType, ipPort);
                if (endpoint == null || endpoint.getStatus() != expectedStatus
                        || !endpoint.beginRetirement()) {
                    return false;
                }
                CompletableFuture<Void> barrier = new CompletableFuture<>();
                retirement = new RetirementRecord(endpoint, expectedStatus, cause, barrier);
                retirement.attemptInProgress = true;
                slot.retirementBarrier = barrier;
                slot.retirement = retirement;
                removeEndpoint(roleType, ipPort, endpoint);
                expectedStatus.tryBeginRetirement();
            }
        }
        return driveRetirement(slot, retirement);
    }

    /**
     * Re-drive every unfinished retirement.  This is both the explicit
     * recovery hook for callers/tests and the implementation behind the
     * scheduled reconciliation pass below.
     *
     * @return number of records whose cleanup completed during this pass
     */
    public int reconcilePendingRetirements() {
        int completed = 0;
        for (EndpointSlot slot : endpointSlots.values()) {
            RetirementRecord retirement;
            synchronized (slot) {
                retirement = slot.retirement;
                if (retirement == null || retirement.attemptInProgress) {
                    continue;
                }
                retirement.attemptInProgress = true;
            }
            if (driveRetirement(slot, retirement)) {
                completed++;
            }
        }
        return completed;
    }

    /**
     * A failed cleanup remains fail-closed, but never becomes permanently
     * poisoned: retry its idempotent stages until the generation is fully
     * settled and the publication barrier can be released safely.
     */
    @Scheduled(fixedDelayString = "${flexlb.endpoint.retirement-reconcile-ms:1000}")
    public void reconcileRetirements() {
        reconcilePendingRetirements();
    }

    private boolean driveRetirement(EndpointSlot slot, RetirementRecord retirement) {
        WorkerEndpoint endpoint = retirement.endpoint;
        Throwable failure = null;

        failure = runRetirementStage(failure, endpoint, retirement.cause,
                "signal", endpoint::signalRetirement);
        failure = runRetirementStage(failure, endpoint, retirement.cause,
                "quiesce", endpoint::awaitOperationQuiescence);
        if (!retirement.queueDrained) {
            try {
                List<BatchItem> drainedItems = endpoint.drainForRetirement();
                retirement.drainedItems = drainedItems == null ? List.of() : List.copyOf(drainedItems);
                retirement.queueDrained = true;
            } catch (Throwable t) {
                failure = addFailure(failure, t);
                Logger.error("Endpoint retirement stage failed: endpoint={} cause={} stage=drain",
                        endpoint.getEndpointId(), retirement.cause, t);
            }
        }
        try {
            FlexlbBatchScheduler scheduler = batchScheduler();
            if (scheduler != null) {
                scheduler.retireEndpoint(endpoint, retirement.cause, retirement.drainedItems);
            }
        } catch (Throwable t) {
            failure = addFailure(failure, t);
            Logger.error("Endpoint retirement stage failed: endpoint={} cause={} stage=scheduler",
                    endpoint.getEndpointId(), retirement.cause, t);
        }
        if (endpoint instanceof DecodeEndpoint) {
            // ReleaseTracker keys are currently ip:port scoped.  Drain every
            // waiter/cache entry before this old Decode generation can release
            // its publication barrier, otherwise a replacement at the same
            // address could consume stale release proof or leave accepted
            // eviction waiters sleeping until their deadline.
            failure = runRetirementStage(failure, endpoint, retirement.cause,
                    "release-tracker", () -> ReleaseTracker.global().onWorkerUnhealthy(endpoint.ipPort()));
        }
        failure = runRetirementStage(failure, endpoint, retirement.cause,
                "local-clear", endpoint::clearLocalStateForRetirement);

        synchronized (slot) {
            retirement.attemptInProgress = false;
            if (failure == null) {
                endpoint.completeRetirement();
                retirement.status.markClosed();
                retirement.barrier.complete(null);
                if (slot.retirementBarrier == retirement.barrier) {
                    slot.retirementBarrier = null;
                    slot.retirement = null;
                }
            } else {
                // Keep the marker pending (rather than completing it
                // exceptionally): a later reconciliation pass retries every
                // idempotent stage.  Publishing remains blocked until one pass
                // establishes complete settlement.
                retirement.lastFailure = failure;
            }
        }
        return failure == null;
    }

    private static Throwable runRetirementStage(Throwable aggregate,
                                                 WorkerEndpoint endpoint,
                                                 EndpointRetireCause cause,
                                                 String stage,
                                                 Runnable operation) {
        try {
            operation.run();
        } catch (Throwable t) {
            Logger.error("Endpoint retirement stage failed: endpoint={} cause={} stage={}",
                    endpoint.getEndpointId(), cause, stage, t);
            return addFailure(aggregate, t);
        }
        return aggregate;
    }

    private static Throwable addFailure(Throwable aggregate, Throwable failure) {
        if (aggregate == null) {
            return failure;
        }
        aggregate.addSuppressed(failure);
        return aggregate;
    }

    public CompletableFuture<Void> getRetirementBarrier(RoleType roleType, String ipPort) {
        EndpointSlot slot = endpointSlots.get(new EndpointKey(roleType, ipPort));
        if (slot == null) {
            return CompletableFuture.completedFuture(null);
        }
        synchronized (slot) {
            return slot.retirementBarrier == null
                    ? CompletableFuture.completedFuture(null) : slot.retirementBarrier;
        }
    }

    private FlexlbBatchScheduler batchScheduler() {
        return batchSchedulerFactory.getObject();
    }

    private WorkerEndpoint createEndpoint(EndpointId endpointId, WorkerStatus status) {
        return switch (endpointId.role()) {
            case PREFILL, PDFUSION -> createPrefillEndpoint(endpointId, status);
            case DECODE -> createDecodeEndpoint(endpointId, status);
            case VIT -> createSimpleEndpoint(endpointId, status);
            default -> throw new IllegalArgumentException("Unsupported role: " + endpointId.role());
        };
    }

    private PrefillEndpoint createPrefillEndpoint(EndpointId endpointId, WorkerStatus status) {
        FlexlbConfig config = configService.loadBalanceConfig();
        prepareEndpointMetrics(endpointId.role(), status);
        return new PrefillEndpoint(endpointId, status, config, batchScheduler(), reporter);
    }

    private DecodeEndpoint createDecodeEndpoint(EndpointId endpointId, WorkerStatus status) {
        prepareEndpointMetrics(RoleType.DECODE, status);
        return new DecodeEndpoint(endpointId, status);
    }

    private SimpleWorkerEndpoint createSimpleEndpoint(EndpointId endpointId, WorkerStatus status) {
        prepareEndpointMetrics(endpointId.role(), status);
        return new SimpleWorkerEndpoint(endpointId, status);
    }

    private void prepareEndpointMetrics(RoleType roleType, WorkerStatus status) {
        reporter.prepareEndpointMetrics(roleType.name(), status.getIp());
    }

    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        List<WorkerEndpoint> endpoints = new ArrayList<>();
        endpoints.addAll(prefillEndpoints.values());
        endpoints.addAll(decodeEndpoints.values());
        endpoints.addAll(pdFusionEndpoints.values());
        endpoints.addAll(vitEndpoints.values());
        for (WorkerEndpoint endpoint : endpoints) {
            retire(endpoint.getEndpointId().role(), endpoint.ipPort(), endpoint.getStatus(),
                    EndpointRetireCause.REGISTRY_CLOSED);
        }
    }

    /**
     * Expose all prefill endpoints for per-worker metrics reporting.
     */
    public ConcurrentHashMap<String, PrefillEndpoint> getPrefillEndpoints() {
        return prefillEndpoints;
    }

    /**
     * Expose all decode endpoints for per-worker metrics reporting.
     */
    public ConcurrentHashMap<String, DecodeEndpoint> getDecodeEndpoints() {
        return decodeEndpoints;
    }

    public int getEndpointCount(RoleType roleType) {
        return getEndpoints(roleType).size();
    }

    private WorkerEndpoint currentEndpoint(RoleType roleType, String ipPort) {
        if (roleType == RoleType.PREFILL) {
            return prefillEndpoints.get(ipPort);
        }
        if (roleType == RoleType.DECODE) {
            return decodeEndpoints.get(ipPort);
        }
        if (roleType == RoleType.PDFUSION) {
            return pdFusionEndpoints.get(ipPort);
        }
        if (roleType == RoleType.VIT) {
            return vitEndpoints.get(ipPort);
        }
        throw new IllegalArgumentException("Unsupported role: " + roleType);
    }

    private void putEndpoint(RoleType roleType, String ipPort, WorkerEndpoint endpoint) {
        if (roleType == RoleType.PREFILL) {
            prefillEndpoints.put(ipPort, (PrefillEndpoint) endpoint);
        } else if (roleType == RoleType.DECODE) {
            decodeEndpoints.put(ipPort, (DecodeEndpoint) endpoint);
        } else if (roleType == RoleType.PDFUSION) {
            pdFusionEndpoints.put(ipPort, (PrefillEndpoint) endpoint);
        } else if (roleType == RoleType.VIT) {
            vitEndpoints.put(ipPort, (SimpleWorkerEndpoint) endpoint);
        } else {
            throw new IllegalArgumentException("Unsupported role: " + roleType);
        }
    }

    private boolean removeEndpoint(RoleType roleType, String ipPort, WorkerEndpoint endpoint) {
        if (roleType == RoleType.PREFILL) {
            return prefillEndpoints.remove(ipPort, endpoint);
        }
        if (roleType == RoleType.DECODE) {
            return decodeEndpoints.remove(ipPort, endpoint);
        }
        if (roleType == RoleType.PDFUSION) {
            return pdFusionEndpoints.remove(ipPort, endpoint);
        }
        if (roleType == RoleType.VIT) {
            return vitEndpoints.remove(ipPort, endpoint);
        }
        return false;
    }

    /**
     * Trigger TTL eviction on all prefill and decode endpoints.
     *
     * @param ttlMs max age before eviction
     */
    private void evictExpiredAll(long ttlMs) {
        prefillEndpoints.values().forEach(ep -> ep.evictExpiredBatches(ttlMs));
        decodeEndpoints.values().forEach(ep -> ep.evictExpiredRequests(ttlMs));
        pdFusionEndpoints.values().forEach(ep -> ep.evictExpiredBatches(ttlMs));
    }

    /**
     * Periodic TTL eviction for all endpoints.
     * <p>Each endpoint is responsible for its own inflight lifecycle.
     * This scheduled method provides a safety-net fallback for entries
     * that were not cleaned up by {@code calibrate()} (e.g., engine crash,
     * network partition, status report delay).
     */
    @Scheduled(fixedRate = 60000L)
    public void scheduledEviction() {
        long ttlMs = configService.loadBalanceConfig().getFlexlbInflightTtlMs();
        evictExpiredAll(ttlMs);
    }
}

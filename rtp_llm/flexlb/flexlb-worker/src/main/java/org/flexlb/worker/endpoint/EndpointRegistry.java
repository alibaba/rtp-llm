package org.flexlb.worker.endpoint;

import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.worker.cache.WorkerCaches;
import org.flexlb.worker.status.EndpointTrace.Stage;
import org.flexlb.worker.status.WorkerStatus;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Owns discovered workers, publishes routing endpoints and runs each worker retirement to completion.
 * Admission closes before routing withdrawal; endpoint and cache cleanup run outside worker locks.
 */
@Component
public class EndpointRegistry
        implements EndpointStats, org.flexlb.worker.status.WorkerStatusProvider {
    private static final org.slf4j.Logger logger =
            org.slf4j.LoggerFactory.getLogger(EndpointRegistry.class);

    private final EnumMap<RoleType, ConcurrentHashMap<String, WorkerStatus>> statusesByRole =
            roleMaps(RoleType.values());
    private final EnumMap<RoleType, ConcurrentHashMap<String, WorkerEndpoint>> endpointsByRole =
            roleMaps(
                    RoleType.PREFILL,
                    RoleType.DECODE,
                    RoleType.PDFUSION,
                    RoleType.ENCODER,
                    RoleType.VIT);

    // Immutable, address-sorted lists are published together after each endpoint map change.
    private volatile Map<RoleType, List<WorkerEndpoint>> routingDirectories = Map.of();

    private final ConfigService configService;
    private final WorkerRequests requests;
    private final EndpointMetrics reporter;
    private final CapacityNotifier capacityNotifier;
    private final WorkerCaches cacheAwareService;
    private final ConcurrentHashMap<WorkerStatus, WorkerCleanup> workerCleanups =
            new ConcurrentHashMap<>();
    private final Object lifecycleGate = new Object();

    /**
     * Null while open; every closing caller observes the same completed cleanup result.
     */
    private volatile CompletableFuture<Throwable> closeCompletion;

    private int inflightPublications;
    private int pendingWorkerCleanups;

    private final long closeTimeoutMs =
            Long.getLong("flexlb.endpoint.retirement.timeout.ms", 30_000L);

    @Autowired
    public EndpointRegistry(
            ConfigService configService,
            WorkerRequests requests,
            EndpointMetrics reporter,
            CapacityNotifier capacityNotifier,
            @org.springframework.context.annotation.Lazy WorkerCaches cacheAwareService) {
        this.cacheAwareService = Objects.requireNonNull(cacheAwareService, "cacheAwareService");
        checkArgument(closeTimeoutMs > 0L, "endpoint retirement timeout must be positive");
        this.configService = java.util.Objects.requireNonNull(configService, "configService");
        this.requests = java.util.Objects.requireNonNull(requests, "requests");
        this.reporter = java.util.Objects.requireNonNull(reporter, "reporter");
        this.capacityNotifier =
                java.util.Objects.requireNonNull(capacityNotifier, "capacityNotifier");
    }

    @org.springframework.scheduling.annotation.Scheduled(fixedRate = 60_000L)
    public void maintainExpiration() {
        if (!requests.isClosed()) evictExpiredOrphans(10_000L, requests::retainsIdentity);
    }

    /**
     * Ask every registered endpoint to remove expired requests no longer retained by the scheduler.
     *
     * @param ttlMs max age before eviction
     * @param retainForSchedulerCleanup current directory lookup, called outside endpoint locks;
     *                                  candidates are revalidated in an DiscoveryEndpoint lock before eviction
     */
    public void evictExpiredOrphans(long ttlMs, Predicate<String> retainForSchedulerCleanup) {
        for (var roleEntry : endpointsByRole.entrySet()) {
            var role = roleEntry.getKey();
            for (var endpointEntry : roleEntry.getValue().entrySet()) {
                var endpoint = endpointEntry.getValue();
                int evicted = endpoint.evictExpiredRequests(ttlMs, retainForSchedulerCleanup);
                if (evicted <= 0) continue;
                var address = endpointEntry.getKey();
                reporter.reportEndpointInflightTtlExpired(role.name(), address, "ttl", evicted);
                Logger.info(
                        "event=endpoint_inflight_ttl_eviction role={} endpoint={} evicted={} ttl_ms={}",
                        role,
                        address,
                        evicted,
                        ttlMs);
            }
        }
    }

    /**
     * Finds or creates the DiscoveryEndpoint, applies the report, then registers a new DiscoveryEndpoint for routing.
     * Failed candidates are closed; notifications from the update are published by the caller after unlocking.
     */
    public EndpointUpdate updateEndpoint(
            String address, WorkerStatus status, WorkerStatus.PreparedStatus prepared) {
        status.requireActiveGeneration();
        var existing = getEndpoint(status.getRole(), address);
        if (existing != null && existing.getStatus() != status) existing = null;
        if (existing == null) beginCandidatePublication();
        WorkerEndpoint candidate = existing;
        boolean publicationAttempted = false;
        Throwable failure = null;
        try {
            WorkerStatus.StatusObservation observation = prepared.observation();
            checkArgument(
                    observation.owner() == status,
                    "staged status belongs to another worker generation");
            RoleType role = observation.role();
            checkArgument(
                    role == status.getRole(),
                    "staged status role does not match its worker generation");
            java.util.Objects.requireNonNull(address, "address");
            if (candidate == null) {
                checkState(
                        status.appliedStatusCursor().statusVersion() < 0L,
                        "Committed WorkerStatus has no exact endpoint generation: %s",
                        address);
                candidate = createEndpoint(status, observation.engine());
            }
            var change = candidate.applyWorkerStatus(status, prepared);
            if (existing != null) return new EndpointUpdate(candidate, change);
            status.trace().record(Stage.ENDPOINT_PUBLISHED);
            WorkerEndpoint exact = candidate;
            publicationAttempted = true;
            mutateEndpointMap(
                    role,
                    address,
                    current -> {
                        checkState(
                                current == null,
                                "Existing endpoint generation must be withdrawn before publication for %s",
                                address);
                        return exact;
                    });
            capacityNotifier.notifyChanged(exact);
            return new EndpointUpdate(candidate, change);
        } catch (Throwable publicationFailure) {
            failure = publicationFailure;
            throw Failures.propagate(failure, "DiscoveryEndpoint update failed for " + address);
        } finally {
            try {
                if (failure != null && candidate != null && existing == null) {
                    WorkerEndpoint exact = candidate;
                    try {
                        if (publicationAttempted) {
                            RoleType role = status.getRole();
                            mutateEndpointMap(
                                    role, address, current -> current == exact ? null : current);
                        }
                    } catch (Throwable withdrawalFailure) {
                        Failures.append(failure, withdrawalFailure);
                    } finally {
                        Failures.run(failure, exact::closeAsync);
                    }
                }
            } finally {
                if (existing == null) endCandidatePublication();
            }
        }
    }

    public WorkerEndpoint getEndpoint(RoleType roleType, String ipPort) {
        Map<String, WorkerEndpoint> endpoints = endpointsByRole.get(roleType);
        return endpoints == null ? null : endpoints.get(ipPort);
    }

    /**
     * Starts retirement for an unavailable worker without waiting for status or admission locks.
     */
    public java.util.concurrent.CompletionStage<Void> closeUnavailableWorker(
            RoleType role, String address, WorkerStatus status) {
        if (status.closeReason() == null || !status.lock.tryLock()) return null;
        try {
            var currentEndpoint = getEndpoint(role, address);
            var endpoint =
                    currentEndpoint != null && currentEndpoint.getStatus() == status
                            ? currentEndpoint
                            : null;
            if (endpoint != null && !endpoint.stateLock().tryLock()) return null;
            try {
                return closeWorker(role, address, status);
            } finally {
                if (endpoint != null) endpoint.stateLock().unlock();
            }
        } finally {
            status.lock.unlock();
        }
    }

    /**
     * Stops admission and withdraws an exact worker, then automatically runs asynchronous cleanup.
     * Repeated calls share completion; callers never wait for cleanup while holding worker locks.
     */
    public java.util.concurrent.CompletionStage<Void> closeWorker(
            RoleType role, String address, WorkerStatus status) {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(address, "address");
        Objects.requireNonNull(status, "status");
        WorkerCleanup cleanup;
        status.lock.lock();
        try {
            var previous = workerCleanups.get(status);
            if (previous != null) return previous.result;
            cleanup = prepareWorkerClose(role, address, status);
        } finally {
            status.lock.unlock();
        }
        if (cleanup == null) return CompletableFuture.completedFuture(null);
        startWorkerCleanup(cleanup);
        return cleanup.result;
    }

    /**
     * Accept updates only for an exact current generation while this registry is open.
     */
    public boolean isCurrentStatus(RoleType role, String address, WorkerStatus expected) {
        return closeCompletion == null
                && expected != null
                && role != null
                && address != null
                && statusesByRole.get(role).get(address) == expected;
    }

    /**
     * Withdraws unavailable workers before starting their cleanup notifications.
     * Busy status transactions or admission locks are left for the next scan.
     */
    public void closeUnavailableWorkers() {
        List<WorkerCleanup> cleanups = new ArrayList<>();
        Throwable failure = null;
        try {
            for (var roleStatuses : statusesByRole.entrySet()) {
                for (var item : roleStatuses.getValue().entrySet()) {
                    var status = item.getValue();
                    if (status.closeReason() == null || !status.lock.tryLock()) continue;
                    try {
                        var endpoint = getEndpoint(roleStatuses.getKey(), item.getKey());
                        // A busy admission must not delay retirement of other workers in this scan.
                        if (endpoint != null && !endpoint.stateLock().tryLock()) continue;
                        try {
                            var cleanup =
                                    prepareWorkerClose(
                                            roleStatuses.getKey(), item.getKey(), status);
                            if (cleanup != null) cleanups.add(cleanup);
                        } finally {
                            if (endpoint != null) endpoint.stateLock().unlock();
                        }
                    } catch (Throwable problem) {
                        failure = Failures.append(failure, problem);
                    } finally {
                        status.lock.unlock();
                    }
                }
            }
        } finally {
            for (var cleanup : cleanups) {
                try {
                    startWorkerCleanup(cleanup);
                    var status = cleanup.status;
                    logger.warn(
                            "Retiring unavailable worker: {}, role: {}, generation={}, trace={}",
                            status.getIpPort(),
                            status.getRole(),
                            status.getGenerationId(),
                            status.trace().snapshot().compact());
                } catch (Throwable problem) {
                    failure = Failures.append(failure, problem);
                }
            }
        }
        Failures.rethrow(failure, "Worker retirement scan failed");
    }

    public void close() {
        boolean closeOwner;
        CompletableFuture<Throwable> completion;
        synchronized (lifecycleGate) {
            closeOwner = closeCompletion == null;
            if (closeOwner) {
                closeCompletion = new CompletableFuture<>();
                routingDirectories = Map.of();
            }
            completion = closeCompletion;
        }
        if (!closeOwner) {
            // Join outside the gate; concurrent closers retain their interruption status.
            Failures.rethrow(completion.join(), "EndpointRegistry close failed");
            return;
        }

        boolean interrupted = false;
        synchronized (lifecycleGate) {
            while (inflightPublications != 0) {
                try {
                    lifecycleGate.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }

        Throwable failure;
        try {
            failure = closeEndpointGenerations();
        } catch (RuntimeException | Error unexpectedFailure) {
            failure = unexpectedFailure;
        }
        try {
            awaitWorkerCleanups();
        } catch (RuntimeException | Error detachedBarrierFailure) {
            failure = Failures.append(failure, detachedBarrierFailure);
        } finally {
            for (var cleanup : workerCleanups.values()) {
                if (cleanup.completion.isCompletedExceptionally()) {
                    try {
                        cleanup.completion.join();
                    } catch (java.util.concurrent.CompletionException problem) {
                        failure = Failures.append(failure, problem.getCause());
                    }
                }
            }
            completion.complete(failure);
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        Failures.rethrow(failure, "EndpointRegistry close failed");
    }

    @Override
    public long blockSize(RoleType role) {
        for (var worker : getWorkerStatuses(role, null)) {
            long size = worker.committedEngineObservation().blockSize();
            if (size > 0) return size;
        }
        return 0L;
    }

    /**
     * Returns available discovered workers matching the requested role and group.
     */
    @Override
    public java.util.Collection<WorkerStatus> getWorkerStatuses(RoleType role, String group) {
        if (role == null) return List.of();
        return statusesByRole.get(role).values().stream()
                .filter(
                        worker ->
                                worker.isAvailableForScheduling()
                                        && (group == null || group.equals(worker.getGroup())))
                .toList();
    }

    /**
     * Cheap role counts; no endpoint or request snapshots are constructed.
     */
    public Counts counts(RoleType role) {
        Objects.requireNonNull(role, "role");
        Map<String, WorkerEndpoint> endpoints = endpointsByRole.get(role);
        return new Counts(
                statusesByRole.get(role).size(), endpoints == null ? 0 : endpoints.size());
    }

    /**
     * Immutable point-in-time view of discovered generations for one role.
     */
    public Map<String, WorkerStatus> statusSnapshot(RoleType role) {
        if (role == null) {
            return Map.of();
        }
        return Map.copyOf(statusesByRole.get(role));
    }

    public WorkerStatus currentOrDiscover(
            RoleType role, String address, Supplier<WorkerStatus> discoveredFactory) {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(address, "address");
        Objects.requireNonNull(discoveredFactory, "discoveredFactory");
        checkState(closeCompletion == null, "EndpointRegistry is closing");
        ConcurrentHashMap<String, WorkerStatus> statuses = statusesByRole.get(role);
        WorkerStatus current = statuses.get(address);
        if (current != null) {
            return current;
        }
        // Construct private discovery state outside the gate: factories must not
        // introduce a lifecycle-gate -> generation-lock edge.
        WorkerStatus discovered =
                Objects.requireNonNull(discoveredFactory.get(), "discovered status");
        checkArgument(
                discovered.getRole() == role && address.equals(discovered.getLogicalIpPort()),
                "Discovered WorkerStatus identity does not match directory key");
        synchronized (lifecycleGate) {
            checkState(closeCompletion == null, "EndpointRegistry is closing");
            WorkerStatus raced = statuses.putIfAbsent(address, discovered);
            return raced == null ? discovered : raced;
        }
    }

    public int discoveredCount() {
        int total = 0;
        for (Map<String, WorkerStatus> statuses : statusesByRole.values()) {
            total += statuses.size();
        }
        return total;
    }

    /**
     * Returns immutable endpoints for a role, optionally filtered by group; null includes all groups.
     * The type parameter must match the role: PrefillEndpoint for PREFILL/PDFUSION,
     * DecodeEndpoint for DECODE, EncoderEndpoint for ENCODER and WorkerEndpoint for VIT.
     * DiscoveryEndpoint creation fixes these types before publishing each directory.
     */
    @SuppressWarnings("unchecked")
    public <T extends WorkerEndpoint> List<T> endpoints(RoleType role, String group) {
        List<T> directory = (List<T>) (List<?>) routingDirectories.getOrDefault(role, List.of());
        return group == null
                ? directory
                : directory.stream()
                        .filter(endpoint -> group.equals(endpoint.getStatus().getGroup()))
                        .toList();
    }

    /**
     * Registers cleanup before withdrawing a worker so shutdown can see all accepted retirements.
     * The caller holds only this worker's status lock; endpoint lock waits stay outside lifecycleGate.
     */
    private WorkerCleanup prepareWorkerClose(RoleType role, String address, WorkerStatus status) {
        if (!isCurrentStatus(role, address, status) || workerCleanups.containsKey(status))
            return null;
        var currentEndpoint = getEndpoint(role, address);
        var endpoint =
                currentEndpoint != null && currentEndpoint.getStatus() == status
                        ? currentEndpoint
                        : null;
        var cleanup = new WorkerCleanup(status, endpoint);
        synchronized (lifecycleGate) {
            checkState(closeCompletion == null, "EndpointRegistry is closing");
            workerCleanups.put(status, cleanup);
            pendingWorkerCleanups++;
        }
        try {
            if (endpoint != null) endpoint.stopAcceptingRequests();
            else status.stopUpdatesAfterAdmissionClosed();
            if (endpoint != null) {
                mutateEndpointMap(role, address, current -> current == endpoint ? null : current);
            }
        } catch (Throwable failure) {
            cleanup.fail(failure);
        }
        return cleanup;
    }

    /**
     * Changes one endpoint mapping and publishes its address-sorted role directory under the lifecycle lock.
     */
    private void mutateEndpointMap(
            RoleType role, String address, UnaryOperator<WorkerEndpoint> mutation) {
        synchronized (lifecycleGate) {
            var endpoints = endpointsByRole.get(role);
            WorkerEndpoint current = endpoints.get(address);
            WorkerEndpoint next = mutation.apply(current);
            if (next == current) return;
            var directory =
                    directoryAfterMutation(
                            routingDirectories.getOrDefault(role, List.of()), address, next);
            if (next == null) endpoints.remove(address);
            else endpoints.put(address, next);
            var directories = new EnumMap<RoleType, List<WorkerEndpoint>>(RoleType.class);
            directories.putAll(routingDirectories);
            directories.put(role, directory);
            routingDirectories = Map.copyOf(directories);
        }
    }

    /**
     * Keep the immutable directory in address order; request routing then needs no full sort.
     */
    private List<WorkerEndpoint> directoryAfterMutation(
            List<WorkerEndpoint> previous, String address, WorkerEndpoint next) {
        if (closeCompletion != null) {
            return List.of();
        }
        List<WorkerEndpoint> updated = new ArrayList<>(previous.size() + (next == null ? 0 : 1));
        boolean inserted = next == null;
        for (WorkerEndpoint entry : previous) {
            String currentAddress = entry.ipPort();
            if (!inserted && address.compareTo(currentAddress) <= 0) {
                updated.add(next);
                inserted = true;
            }
            if (!address.equals(currentAddress)) {
                updated.add(entry);
            }
        }
        if (!inserted) {
            updated.add(next);
        }
        return List.copyOf(updated);
    }

    private void startWorkerCleanup(WorkerCleanup cleanup) {
        if (cleanup.completion.isDone()) return;
        try {
            Thread.ofVirtual().name("worker-retirement").start(cleanup);
        } catch (Throwable failure) {
            cleanup.fail(failure);
        }
    }

    private static <T> EnumMap<RoleType, ConcurrentHashMap<String, T>> roleMaps(RoleType... roles) {
        EnumMap<RoleType, ConcurrentHashMap<String, T>> maps = new EnumMap<>(RoleType.class);
        for (RoleType role : roles) {
            maps.put(role, new ConcurrentHashMap<>());
        }
        return maps;
    }

    private void beginCandidatePublication() {
        synchronized (lifecycleGate) {
            checkState(closeCompletion == null, "EndpointRegistry is closing");
            inflightPublications++;
        }
    }

    private void endCandidatePublication() {
        synchronized (lifecycleGate) {
            checkState(inflightPublications > 0, "EndpointRegistry publication count underflow");
            inflightPublications--;
            if (inflightPublications == 0) {
                lifecycleGate.notifyAll();
            }
        }
    }

    /**
     * Ends bookkeeping for a worker retirement after success or a reported failure.
     */
    private void decrementPendingWorkerCleanups() {
        synchronized (lifecycleGate) {
            checkState(pendingWorkerCleanups > 0, "Detached retirement count underflow");
            pendingWorkerCleanups--;
            if (pendingWorkerCleanups == 0) {
                lifecycleGate.notifyAll();
            }
        }
    }

    private WorkerEndpoint createEndpoint(
            WorkerStatus status, WorkerStatus.EngineObservation engineStatus) {
        var role = status.getRole();
        checkArgument(role != RoleType.FRONTEND, "Unsupported role: %s", role);
        boolean prefill = role == RoleType.PREFILL || role == RoleType.PDFUSION;
        if (prefill && engineStatus.dpSize() > 1) {
            throw new UnsupportedOperationException(
                    role
                            + " DP group endpoint not yet supported: ipPort="
                            + status.getIpPort()
                            + ", dp_size="
                            + engineStatus.dpSize());
        }
        return switch (role) {
            case PREFILL, PDFUSION ->
                    new PrefillEndpoint(
                            status, configService.loadBalanceConfig(), capacityNotifier);
            case DECODE -> new DecodeEndpoint(status, requests, capacityNotifier);
            case ENCODER -> new EncoderEndpoint(status, capacityNotifier);
            case VIT -> new WorkerEndpoint(status);
            case FRONTEND -> throw new AssertionError("validated above");
        };
    }

    /**
     * Wait for every worker retirement registered before shutdown.
     */
    private void awaitWorkerCleanups() {
        boolean interrupted = false;
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(closeTimeoutMs);
        try {
            synchronized (lifecycleGate) {
                while (pendingWorkerCleanups != 0) {
                    long remaining = deadlineNanos - System.nanoTime();
                    checkState(
                            remaining > 0L,
                            "Detached worker retirement timed out after %sms: pending=%s",
                            closeTimeoutMs,
                            pendingWorkerCleanups);
                    try {
                        TimeUnit.NANOSECONDS.timedWait(lifecycleGate, remaining);
                    } catch (InterruptedException interruption) {
                        interrupted = true;
                    }
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private Throwable closeEndpointGenerations() {
        Set<WorkerEndpoint> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<WorkerEndpoint> endpoints = new ArrayList<>();
        Throwable failure = null;
        for (Map<String, WorkerEndpoint> roleEndpoints : endpointsByRole.values()) {
            try {
                for (WorkerEndpoint endpoint : roleEndpoints.values()) {
                    if (seen.add(endpoint)) {
                        endpoints.add(endpoint);
                    }
                }
            } catch (RuntimeException | Error snapshotFailure) {
                failure = Failures.append(failure, snapshotFailure);
            }
        }
        Set<WorkerStatus> stoppedStatuses = Collections.newSetFromMap(new IdentityHashMap<>());
        // Stop every generation before any cleanup can notify scheduling listeners.
        for (WorkerEndpoint endpoint : endpoints) {
            failure =
                    Failures.run(
                            failure,
                            () -> {
                                endpoint.stopAcceptingRequests();
                                stoppedStatuses.add(endpoint.getStatus());
                            });
        }
        // Discovered generations may have an in-flight cache poll before their
        // first endpoint is published. Fence these callbacks under the same lock.
        for (Map<String, WorkerStatus> statuses : statusesByRole.values()) {
            for (WorkerStatus status : statuses.values()) {
                if (stoppedStatuses.contains(status)) continue;
                failure =
                        Failures.run(
                                failure,
                                () -> {
                                    status.lock.lock();
                                    try {
                                        status.stopUpdatesAfterAdmissionClosed();
                                    } finally {
                                        status.lock.unlock();
                                    }
                                });
            }
        }
        List<java.util.concurrent.CompletionStage<Void>> completions = new ArrayList<>();
        for (WorkerEndpoint endpoint : endpoints) {
            failure = Failures.run(failure, () -> completions.add(endpoint.closeAsync()));
        }
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(closeTimeoutMs);
        for (var completion : completions) {
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    awaitCloseCompletion(
                                            completion,
                                            Math.max(
                                                    1L,
                                                    TimeUnit.NANOSECONDS.toMillis(
                                                            deadlineNanos - System.nanoTime()))));
        }
        return failure;
    }

    private static void awaitCloseCompletion(
            java.util.concurrent.CompletionStage<Void> closing, long timeoutMs) {
        try {
            closing.toCompletableFuture().orTimeout(timeoutMs, TimeUnit.MILLISECONDS).join();
        } catch (java.util.concurrent.CompletionException failure) {
            throw Failures.propagate(
                    failure.getCause(), "DiscoveryEndpoint close timed out or failed");
        }
    }

    private boolean finishWorkerCleanup(WorkerStatus status) {
        var role = status.getRole();
        var address = status.getLogicalIpPort();
        Map<String, WorkerStatus> statuses = statusesByRole.get(role);
        status.trace().record(Stage.RETIREMENT_CACHE_LOCK_ATTEMPT);
        status.cacheMutationLock.lock();
        status.trace().record(Stage.RETIREMENT_CACHE_LOCK_ACQUIRED);
        try {
            checkState(
                    !status.isActiveGeneration(), "Cache cleanup requires a retiring generation");
            if (statuses.get(address) != status) {
                logger.error(
                        "Status identity changed before retirement finalized for {}#{}",
                        address,
                        status.getGenerationId());
                return true;
            }
            try {
                if (role.requiresCacheKeys()) {
                    status.trace().record(Stage.CACHE_CLEANUP_BEGIN);
                    cacheAwareService.removeEngineBlockCache(address);
                    status.trace().record(Stage.CACHE_CLEANUP_END);
                }
            } catch (Throwable cacheCleanupFailure) {
                status.trace().record(Stage.CACHE_CLEANUP_FAILED, cacheCleanupFailure);
                logger.error(
                        "Cache cleanup failed while retiring generation {} for {}",
                        status.getGenerationId(),
                        address,
                        cacheCleanupFailure);
                return false;
            }
            status.trace().record(Stage.RETIREMENT_CLEANUP_END);
            capacityNotifier.removeClosedWorker(status);
            if (!statuses.remove(address, status)) {
                logger.error(
                        "Exact status removal failed while its cache lock was held for {}#{}",
                        address,
                        status.getGenerationId());
            }
            status.trace().record(Stage.STATUS_REMOVED);
            return true;
        } catch (Throwable finalizationFailure) {
            logger.error(
                    "Status retirement finalization failed for {}#{}",
                    address,
                    status.getGenerationId(),
                    finalizationFailure);
            return false;
        } finally {
            status.cacheMutationLock.unlock();
            status.trace().record(Stage.RETIREMENT_CACHE_LOCK_RELEASED);
        }
    }

    /**
     * Notifications belong to the returned DiscoveryEndpoint and must be published after releasing the WorkerStatus lock.
     */
    public record EndpointUpdate(WorkerEndpoint endpoint, WorkerEndpoint.StatusChange change) {}

    /**
     * Drains one withdrawn worker and retries cache cleanup before removing its exact status.
     */
    private final class WorkerCleanup implements Runnable {
        private final WorkerStatus status;
        private final WorkerEndpoint endpoint;
        private final CompletableFuture<Void> completion = new CompletableFuture<>();
        private final java.util.concurrent.CompletionStage<Void> result =
                completion.minimalCompletionStage();

        private WorkerCleanup(WorkerStatus status, WorkerEndpoint endpoint) {
            this.status = status;
            this.endpoint = endpoint;
        }

        @Override
        public void run() {
            var address = status.getLogicalIpPort();
            // The caller may have entered closeWorker inside a larger status transaction.
            status.lock.lock();
            status.lock.unlock();
            status.trace().record(Stage.RETIREMENT_CLEANUP_BEGIN);
            try {
                if (endpoint != null) {
                    try {
                        capacityNotifier.notifyChanged(endpoint);
                    } catch (Throwable failure) {
                        logger.error(
                                "Worker withdrawal notification failed for {}", address, failure);
                    }
                    endpoint.closeAsync().toCompletableFuture().join();
                }
                while (!finishWorkerCleanup(status)) {
                    if (closeCompletion != null) {
                        throw new IllegalStateException(
                                "Worker cache cleanup failed during shutdown: " + address);
                    }
                    TimeUnit.SECONDS.sleep(1L);
                }
                decrementPendingWorkerCleanups();
                completion.complete(null);
                workerCleanups.remove(status, this);
            } catch (Throwable failure) {
                if (failure instanceof InterruptedException) Thread.currentThread().interrupt();
                fail(failure);
            }
        }

        private void fail(Throwable failure) {
            completion.completeExceptionally(failure);
            decrementPendingWorkerCleanups();
            logger.error("Worker retirement failed for {}", status.getLogicalIpPort(), failure);
        }
    }
}

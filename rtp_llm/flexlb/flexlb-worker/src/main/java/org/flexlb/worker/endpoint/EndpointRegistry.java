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
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Predicate;
import java.util.function.Supplier;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Publishes endpoints after their first worker report and withdraws them before resource cleanup.
 * Discovery retains each worker identity until its endpoint and cache cleanup finish; shutdown fences
 * publication, stops every worker, and waits for the accepted cleanup futures outside worker locks.
 */
@Component
public class EndpointRegistry
        implements EndpointStats, org.flexlb.worker.status.WorkerStatusProvider {
    private static final org.slf4j.Logger logger =
            org.slf4j.LoggerFactory.getLogger(EndpointRegistry.class);

    private final EnumMap<RoleType, ConcurrentHashMap<String, WorkerStatus>> statusesByRole =
            roleMaps(RoleType.values());
    private final EnumMap<RoleType, ConcurrentHashMap<String, WorkerEndpoint>> endpointsByRole =
            roleMaps(RoleType.values());
    private final ConcurrentHashMap<WorkerStatus, WorkerCleanup> workerCleanups =
            new ConcurrentHashMap<>();
    private final ConfigService configService;
    private final WorkerRequests requests;
    private final EndpointMetrics reporter;
    private final CapacityNotifier capacityNotifier;
    private final WorkerCaches cacheAwareService;
    // Read-side operations may hold a worker lock. Shutdown releases the write lock before taking
    // one.
    private final ReentrantReadWriteLock publications = new ReentrantReadWriteLock();
    private final Object directoryLock = new Object();

    private volatile Map<RoleType, List<WorkerEndpoint>> routingDirectories = Map.of();
    private volatile CompletableFuture<Throwable> closeCompletion;

    private final long closeTimeoutMs =
            Long.getLong("flexlb.endpoint.retirement.timeout.ms", 30_000L);

    @Autowired
    public EndpointRegistry(
            ConfigService configService,
            WorkerRequests requests,
            EndpointMetrics reporter,
            CapacityNotifier capacityNotifier,
            @org.springframework.context.annotation.Lazy WorkerCaches cacheAwareService) {
        this.configService = Objects.requireNonNull(configService);
        this.requests = Objects.requireNonNull(requests);
        this.reporter = Objects.requireNonNull(reporter);
        this.capacityNotifier = Objects.requireNonNull(capacityNotifier);
        this.cacheAwareService = Objects.requireNonNull(cacheAwareService);
        checkArgument(closeTimeoutMs > 0, "endpoint retirement timeout must be positive");
    }

    @org.springframework.scheduling.annotation.Scheduled(fixedRate = 60_000L)
    public void maintainExpiration() {
        if (!requests.isClosed()) evictExpiredOrphans(10_000L, requests::retainsIdentity);
    }

    public void evictExpiredOrphans(long ttlMs, Predicate<String> retainForSchedulerCleanup) {
        for (var endpoints : endpointsByRole.entrySet()) {
            for (var entry : endpoints.getValue().entrySet()) {
                int evicted =
                        entry.getValue().evictExpiredRequests(ttlMs, retainForSchedulerCleanup);
                if (evicted <= 0) continue;
                var role = endpoints.getKey();
                reporter.reportEndpointInflightTtlExpired(
                        role.name(), entry.getKey(), "ttl", evicted);
                Logger.info(
                        "event=endpoint_inflight_ttl_eviction role={} endpoint={} evicted={} ttl_ms={}",
                        role,
                        entry.getKey(),
                        evicted,
                        ttlMs);
            }
        }
    }

    /**
     * Applies a report under its worker lock. A new endpoint becomes routable only after applying it.
     */
    public EndpointUpdate updateEndpoint(
            String address, WorkerStatus status, WorkerStatus.PreparedStatus prepared) {
        status.requireActiveGeneration();
        var observation = prepared.observation();
        checkArgument(
                observation.owner() == status && observation.role() == status.getRole(),
                "staged status belongs to another worker generation");
        Objects.requireNonNull(address, "address");
        publications.readLock().lock();
        try {
            checkState(closeCompletion == null, "EndpointRegistry is closing");
            var current = getEndpoint(status.getRole(), address);
            if (current != null && current.getStatus() == status) {
                return new EndpointUpdate(current, current.applyWorkerStatus(status, prepared));
            }
            checkState(
                    current == null,
                    "Existing endpoint generation must be withdrawn before publication for %s",
                    address);
            checkState(
                    status.appliedStatusCursor().statusVersion() < 0,
                    "Committed WorkerStatus has no exact endpoint generation: %s",
                    address);
            var endpoint = createEndpoint(status, observation.engine());
            try {
                var change = endpoint.applyWorkerStatus(status, prepared);
                setEndpoint(status.getRole(), address, endpoint);
                status.trace().record(Stage.ENDPOINT_PUBLISHED);
                capacityNotifier.notifyChanged(endpoint);
                return new EndpointUpdate(endpoint, change);
            } catch (Throwable failure) {
                removeEndpoint(endpoint);
                Failures.run(failure, endpoint::closeAsync);
                throw Failures.propagate(failure, "Endpoint update failed for " + address);
            }
        } finally {
            publications.readLock().unlock();
        }
    }

    public WorkerEndpoint getEndpoint(RoleType role, String address) {
        var endpoints = endpointsByRole.get(role);
        return endpoints == null ? null : endpoints.get(address);
    }

    public WorkerStatus currentOrDiscover(
            RoleType role, String address, Supplier<WorkerStatus> factory) {
        Objects.requireNonNull(role);
        Objects.requireNonNull(address);
        Objects.requireNonNull(factory);
        checkState(closeCompletion == null, "EndpointRegistry is closing");
        var statuses = statusesByRole.get(role);
        var current = statuses.get(address);
        if (current != null) return current;
        var discovered = Objects.requireNonNull(factory.get(), "discovered status");
        checkArgument(
                discovered.getRole() == role && address.equals(discovered.getLogicalIpPort()),
                "Discovered WorkerStatus identity does not match directory key");
        publications.readLock().lock();
        try {
            checkState(closeCompletion == null, "EndpointRegistry is closing");
            var raced = statuses.putIfAbsent(address, discovered);
            return raced == null ? discovered : raced;
        } finally {
            publications.readLock().unlock();
        }
    }

    public void closeUnavailableWorkers() {
        Throwable failure = null;
        for (var role : statusesByRole.entrySet()) {
            for (var entry : role.getValue().entrySet()) {
                try {
                    var status = entry.getValue();
                    if (closeUnavailableWorker(role.getKey(), entry.getKey(), status) != null) {
                        logger.warn(
                                "Retiring unavailable worker: {}, role: {}, generation={}, trace={}",
                                status.getIpPort(),
                                status.getRole(),
                                status.getGenerationId(),
                                status.trace().snapshot().compact());
                    }
                } catch (Throwable problem) {
                    failure = Failures.append(failure, problem);
                }
            }
        }
        Failures.rethrow(failure, "Worker retirement scan failed");
    }

    /**
     * Busy workers are retried by the next scan without delaying other workers.
     */
    public java.util.concurrent.CompletionStage<Void> closeUnavailableWorker(
            RoleType role, String address, WorkerStatus status) {
        if (status.closeReason() == null || !status.lock.tryLock()) return null;
        try {
            var endpoint = getEndpoint(role, address);
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
     * Removes one exact worker from routing before starting its asynchronous endpoint and cache cleanup.
     */
    public java.util.concurrent.CompletionStage<Void> closeWorker(
            RoleType role, String address, WorkerStatus status) {
        Objects.requireNonNull(role);
        Objects.requireNonNull(address);
        Objects.requireNonNull(status);
        WorkerCleanup cleanup;
        status.lock.lock();
        try {
            var previous = workerCleanups.get(status);
            if (previous != null) return previous.result;
            publications.readLock().lock();
            try {
                if (!isCurrentStatus(role, address, status))
                    return CompletableFuture.completedFuture(null);
                var endpoint = getEndpoint(role, address);
                cleanup = new WorkerCleanup(status, endpoint);
                workerCleanups.put(status, cleanup);
                try {
                    stopWorker(status, endpoint);
                    if (endpoint != null) removeEndpoint(endpoint);
                } catch (Throwable failure) {
                    cleanup.completion.completeExceptionally(failure);
                }
            } finally {
                publications.readLock().unlock();
            }
        } finally {
            status.lock.unlock();
        }
        if (!cleanup.completion.isDone()) {
            try {
                Thread.ofVirtual().name("worker-retirement").start(cleanup);
            } catch (Throwable failure) {
                cleanup.completion.completeExceptionally(failure);
            }
        }
        return cleanup.result;
    }

    public boolean isCurrentStatus(RoleType role, String address, WorkerStatus expected) {
        return closeCompletion == null
                && expected != null
                && role != null
                && address != null
                && statusesByRole.get(role).get(address) == expected;
    }

    public void close() {
        CompletableFuture<Throwable> completion;
        synchronized (directoryLock) {
            completion = closeCompletion;
            if (completion == null) {
                closeCompletion = new CompletableFuture<>();
                routingDirectories = Map.of();
            }
        }
        if (completion != null) {
            Failures.rethrow(completion.join(), "EndpointRegistry close failed");
            return;
        }
        // Wait for admitted publication and withdrawal transactions, without holding a worker lock.
        publications.writeLock().lock();
        publications.writeLock().unlock();
        Throwable failure = null;
        try {
            Set<WorkerEndpoint> endpoints = Collections.newSetFromMap(new IdentityHashMap<>());
            for (var directory : endpointsByRole.values()) endpoints.addAll(directory.values());
            Set<WorkerStatus> stopped = Collections.newSetFromMap(new IdentityHashMap<>());
            for (var endpoint : endpoints) {
                failure = Failures.run(failure, endpoint::stopAcceptingRequests);
                stopped.add(endpoint.getStatus());
            }
            for (var statuses : statusesByRole.values()) {
                for (var status : statuses.values()) {
                    if (!stopped.contains(status))
                        failure = Failures.run(failure, () -> stopWorker(status, null));
                }
            }
            var completions = new ArrayList<java.util.concurrent.CompletionStage<Void>>();
            for (var endpoint : endpoints)
                failure = Failures.run(failure, () -> completions.add(endpoint.closeAsync()));
            for (var cleanup : workerCleanups.values()) completions.add(cleanup.result);
            long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(closeTimeoutMs);
            for (var result : completions)
                failure = Failures.run(failure, () -> awaitCleanup(result, deadline));
        } catch (Throwable problem) {
            failure = Failures.append(failure, problem);
        } finally {
            closeCompletion.complete(failure);
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

    public Counts counts(RoleType role) {
        Objects.requireNonNull(role);
        return new Counts(statusesByRole.get(role).size(), endpointsByRole.get(role).size());
    }

    public Map<String, WorkerStatus> statusSnapshot(RoleType role) {
        return role == null ? Map.of() : Map.copyOf(statusesByRole.get(role));
    }

    public int discoveredCount() {
        return statusesByRole.values().stream().mapToInt(Map::size).sum();
    }

    /**
     * The requested type must match the role's endpoint implementation.
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

    private void setEndpoint(RoleType role, String address, WorkerEndpoint endpoint) {
        synchronized (directoryLock) {
            endpointsByRole.get(role).put(address, endpoint);
            publishDirectory(role);
        }
    }

    private void publishDirectory(RoleType role) {
        if (closeCompletion != null) return;
        var directories = new EnumMap<RoleType, List<WorkerEndpoint>>(RoleType.class);
        directories.putAll(routingDirectories);
        directories.put(
                role,
                endpointsByRole.get(role).values().stream()
                        .sorted(java.util.Comparator.comparing(WorkerEndpoint::ipPort))
                        .toList());
        routingDirectories = Map.copyOf(directories);
    }

    private void removeEndpoint(WorkerEndpoint endpoint) {
        synchronized (directoryLock) {
            var role = endpoint.getStatus().getRole();
            if (endpointsByRole.get(role).remove(endpoint.ipPort(), endpoint))
                publishDirectory(role);
        }
    }

    private static void stopWorker(WorkerStatus status, WorkerEndpoint endpoint) {
        if (endpoint != null) endpoint.stopAcceptingRequests();
        else {
            status.lock.lock();
            try {
                status.stopUpdatesAfterAdmissionClosed();
            } finally {
                status.lock.unlock();
            }
        }
    }

    private static void awaitCleanup(
            java.util.concurrent.CompletionStage<Void> result, long deadline) {
        try {
            result.toCompletableFuture()
                    .orTimeout(Math.max(1L, deadline - System.nanoTime()), TimeUnit.NANOSECONDS)
                    .join();
        } catch (java.util.concurrent.CompletionException failure) {
            throw Failures.propagate(failure.getCause(), "Worker cleanup timed out or failed");
        }
    }

    private static <T> EnumMap<RoleType, ConcurrentHashMap<String, T>> roleMaps(RoleType... roles) {
        EnumMap<RoleType, ConcurrentHashMap<String, T>> maps = new EnumMap<>(RoleType.class);
        for (RoleType role : roles) {
            maps.put(role, new ConcurrentHashMap<>());
        }
        return maps;
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
            case DECODE -> new DecodeEndpoint(status, capacityNotifier);
            case ENCODER -> new EncoderEndpoint(status, capacityNotifier);
            case VIT -> new WorkerEndpoint(status);
            case FRONTEND -> throw new AssertionError("validated above");
        };
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

    public record EndpointUpdate(WorkerEndpoint endpoint, WorkerEndpoint.StatusChange change) {}

    /**
     * Retains the discovered identity until resource and cache cleanup finish. Failed futures remain
     * in the registry so every shutdown caller observes the failure.
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
            // closeWorker may be called inside an outer worker-status transaction.
            status.lock.lock();
            status.lock.unlock();
            status.trace().record(Stage.RETIREMENT_CLEANUP_BEGIN);
            try {
                if (endpoint != null) {
                    try {
                        capacityNotifier.notifyChanged(endpoint);
                    } catch (Throwable failure) {
                        logger.error(
                                "Worker withdrawal notification failed for {}",
                                status.getLogicalIpPort(),
                                failure);
                    }
                    endpoint.closeAsync().toCompletableFuture().join();
                }
                while (!finishWorkerCleanup(status)) {
                    checkState(
                            closeCompletion == null,
                            "Worker cache cleanup failed during shutdown: %s",
                            status.getLogicalIpPort());
                    TimeUnit.SECONDS.sleep(1);
                }
                completion.complete(null);
                workerCleanups.remove(status, this);
            } catch (Throwable failure) {
                if (failure instanceof InterruptedException) Thread.currentThread().interrupt();
                completion.completeExceptionally(failure);
                logger.error("Worker retirement failed for {}", status.getLogicalIpPort(), failure);
            }
        }
    }
}

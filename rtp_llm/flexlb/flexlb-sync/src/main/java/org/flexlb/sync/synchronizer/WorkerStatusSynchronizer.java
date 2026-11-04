package org.flexlb.sync.synchronizer;

import com.google.common.math.StatsAccumulator;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.match.localsync.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.GrpcCacheStatusCheckRunner;
import org.flexlb.sync.runner.GrpcWorkerStatusRunner;
import org.flexlb.util.Failures;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;

/**
 * Master engine status synchronizer
 */
@Component
public final class WorkerStatusSynchronizer {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String modelName;
    private final List<RoleType> requiredRoles;
    private final WorkerAddressService workerAddressService;
    private final EndpointRegistry endpointRegistry;
    private final EngineHealthReporter engineHealthReporter;
    private final FlexlbConfig flexlbConfig;
    private final WorkerStatusRpcClient workerStatusRpcClient;
    private final CacheAwareService cacheAwareService;
    private final DynamicCacheIntervalService cacheIntervalService;
    private final long syncRequestTimeoutMs;
    private final AtomicLong syncCount = new AtomicLong();
    private final EnumMap<RoleType, AtomicBoolean> pendingCycles = new EnumMap<>(RoleType.class);
    private final AtomicBoolean closed = new AtomicBoolean();
    private final Long syncEngineStatusInterval;
    private final ScheduledThreadPoolExecutor scheduler;
    private final ThreadPoolExecutor statusCheckExecutor;
    private final ThreadPoolExecutor cacheStatusExecutor;
    private final ThreadPoolExecutor statusApplicationExecutor;
    private final ThreadPoolExecutor engineSyncExecutor;

    public WorkerStatusSynchronizer(
            WorkerAddressService workerAddressService,
            EngineHealthReporter engineHealthReporter,
            EndpointRegistry endpointRegistry,
            WorkerStatusRpcClient workerStatusRpcClient,
            ModelMetaConfig modelMetaConfig,
            CacheAwareService cacheAwareService,
            DynamicCacheIntervalService cacheIntervalService,
            ConfigService configService) {

        this.workerAddressService = workerAddressService;
        this.engineHealthReporter = engineHealthReporter;
        this.endpointRegistry = endpointRegistry;
        this.flexlbConfig = configService.loadBalanceConfig();
        this.workerStatusRpcClient = workerStatusRpcClient;
        this.cacheAwareService = cacheAwareService;
        this.cacheIntervalService = cacheIntervalService;
        this.modelName = modelMetaConfig.modelName();
        this.requiredRoles = modelMetaConfig.requiredRoles();
        for (RoleType role : requiredRoles) {
            pendingCycles.put(role, new AtomicBoolean());
        }

        this.syncEngineStatusInterval =
                flexlbConfig.getWorkerRegistry().getHealth().getStatusPollIntervalMs();
        this.syncRequestTimeoutMs =
                flexlbConfig.getWorkerRegistry().getHealth().getStatusRpcTimeoutMs();
        int engineThreads = flexlbConfig.getInternalRuntime().getEngineSyncExecutorThreads();
        engineSyncExecutor = executor(engineThreads, "engine-sync-executor");
        int statusThreads = flexlbConfig.getInternalRuntime().getStatusCheckExecutorThreads();
        statusCheckExecutor = executor(statusThreads, "status-checker-executor");
        statusApplicationExecutor = executor(statusThreads, "status-application-executor");
        cacheStatusExecutor = executor(statusThreads, "cache-status-executor");
        this.scheduler =
                new ScheduledThreadPoolExecutor(
                        5,
                        new NamedThreadFactory("sync-status-scheduler"),
                        new ThreadPoolExecutor.AbortPolicy());
        this.scheduler.scheduleAtFixedRate(
                this::syncEngineStatus, 0, syncEngineStatusInterval, TimeUnit.MILLISECONDS);
        this.scheduler.scheduleAtFixedRate(this::reportExecutorMetrics, 2, 2, TimeUnit.SECONDS);
    }

    private static ThreadPoolExecutor executor(int threads, String name) {
        return new ThreadPoolExecutor(
                threads,
                threads,
                60L,
                TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(15_000),
                new NamedThreadFactory(name),
                new ThreadPoolExecutor.AbortPolicy());
    }

    private void reportExecutorMetrics() {
        try {
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    "engineSyncExecutor",
                    engineSyncExecutor.getActiveCount(),
                    engineSyncExecutor.getQueue().size(),
                    engineSyncExecutor.getCorePoolSize(),
                    engineSyncExecutor.getPoolSize());
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    "statusCheckExecutor",
                    statusCheckExecutor.getActiveCount(),
                    statusCheckExecutor.getQueue().size(),
                    statusCheckExecutor.getCorePoolSize(),
                    statusCheckExecutor.getPoolSize());
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    "statusApplicationExecutor",
                    statusApplicationExecutor.getActiveCount(),
                    statusApplicationExecutor.getQueue().size(),
                    statusApplicationExecutor.getCorePoolSize(),
                    statusApplicationExecutor.getPoolSize());
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    "cacheStatusExecutor",
                    cacheStatusExecutor.getActiveCount(),
                    cacheStatusExecutor.getQueue().size(),
                    cacheStatusExecutor.getCorePoolSize(),
                    cacheStatusExecutor.getPoolSize());
        } catch (Throwable failure) {
            logger.warn("Failed to report worker sync executor metrics", failure);
        }
    }

    public void syncEngineStatus() {
        if (closed.get()) {
            return;
        }
        long syncCycleId = syncCount.incrementAndGet();
        logger.debug("sync engine status start, times:{}, modelName:{}", syncCycleId, modelName);
        try {
            for (RoleType roleType : requiredRoles) {
                AtomicBoolean pending = pendingCycles.get(roleType);
                if (!pending.compareAndSet(false, true)) {
                    continue;
                }
                try {
                    engineSyncExecutor.submit(
                            () -> {
                                try {
                                    if (closed.get()) {
                                        return;
                                    }
                                    syncRole(roleType, syncCycleId);
                                } finally {
                                    pending.set(false);
                                }
                            });
                } catch (RuntimeException | Error rejected) {
                    pending.set(false);
                    throw rejected;
                }
            }
        } catch (Exception e) {
            logger.error("sync engine prefill status error", e);
        }
    }

    public boolean isReady() {
        return requiredRoles.stream()
                .allMatch(role -> endpointRegistry.counts(role).published() > 0);
    }

    @PreDestroy
    public void destroy() {
        closed.set(true);
        scheduler.shutdown();
        engineSyncExecutor.shutdown();
        statusCheckExecutor.shutdown();
        cacheStatusExecutor.shutdown();
        statusApplicationExecutor.shutdown();
    }

    public void syncRole(RoleType roleType, long syncCycleId) {
        logger.debug(
                "Worker status sync start for model: {}, role: {}", modelName, roleType.toString());
        try {
            Map<String, WorkerStatus> cachedWorkerStatuses =
                    endpointRegistry.statusSnapshot(roleType);
            List<WorkerHost> discovered;
            try {
                discovered =
                        Objects.requireNonNull(
                                workerAddressService.getEngineWorkerList(modelName, roleType),
                                "discovery result");
            } catch (RuntimeException discoveryFailure) {
                logger.warn(
                        "Worker discovery failed; polling cached generations, model={}, role={}",
                        modelName,
                        roleType,
                        discoveryFailure);
                for (WorkerStatus worker : cachedWorkerStatuses.values()) {
                    pollWorker(worker, syncCycleId);
                }
                return;
            }
            List<WorkerHost> latestEngineWorkerList = discovered;
            reportTelemetry(
                    () ->
                            engineHealthReporter.reportServiceDiscoveryResult(
                                    modelName, latestEngineWorkerList.size(), roleType.toString()));
            Set<String> latestValidIpPorts =
                    latestEngineWorkerList.stream()
                            .map(WorkerHost::getLogicalIpPort)
                            .collect(Collectors.toSet());
            for (var entry : cachedWorkerStatuses.entrySet()) {
                if (!latestValidIpPorts.contains(entry.getKey())) {
                    WorkerStatus status = entry.getValue();
                    status.markRemovedFromDiscovery();
                    var retirement =
                            endpointRegistry.beginRequestedRetirement(
                                    roleType, entry.getKey(), status);
                    if (retirement != null) {
                        retirement.completeAsync(cacheAwareService, logger);
                    }
                }
            }

            logger.debug(
                    "Submitting status check tasks for {} workers", latestEngineWorkerList.size());
            for (WorkerHost host : latestEngineWorkerList) {
                try {
                    WorkerStatus workerStatus = getOrCreateWorkerStatus(roleType, host);
                    if (workerStatus != null) {
                        pollWorker(workerStatus, syncCycleId);
                    }
                } catch (RuntimeException failure) {
                    logger.error(
                            "Worker sync failed, model={}, role={}, address={}",
                            modelName,
                            roleType,
                            host.getLogicalIpPort(),
                            failure);
                }
            }
            logger.debug(
                    "Finished submitting status check tasks for model: {}, role: {}, worker count: {}",
                    modelName,
                    roleType,
                    latestEngineWorkerList.size());

        } catch (Exception e) {
            logger.error(
                    "sync engine workers status exception, modelName:{}, error:{}",
                    modelName,
                    e.getMessage(),
                    e);
            reportTelemetry(
                    () ->
                            engineHealthReporter.reportStatusCheckerFail(
                                    BalanceStatusEnum.UNKNOWN_ERROR, null));
        } finally {
            reportTelemetry(() -> reportLoadVariance(roleType));
        }
    }

    private void pollWorker(WorkerStatus workerStatus, long syncCycleId) {
        if (!workerStatus.isAvailableForScheduling()) {
            return;
        }
        submitPoll(
                statusCheckExecutor,
                lease ->
                        new GrpcWorkerStatusRunner(
                                modelName,
                                workerStatus,
                                lease,
                                endpointRegistry,
                                engineHealthReporter,
                                workerStatusRpcClient,
                                syncRequestTimeoutMs,
                                cacheAwareService,
                                statusApplicationExecutor),
                workerStatus.tryBeginStatusPoll(syncCycleId));
        if (flexlbConfig.isKvcmCacheMatching()) return;
        long intervalCycles =
                Math.max(
                        1L, cacheIntervalService.getCurrentIntervalMs() / syncEngineStatusInterval);
        WorkerStatus.PollLease cacheLease = workerStatus.tryBeginCachePoll(syncCycleId);
        if (cacheLease == null) {
            return;
        }
        if (workerStatus.getRole().requiresCacheKeys()
                && !workerStatus.shouldPollCache(syncCycleId, intervalCycles)) {
            cacheLease.close();
            return;
        }
        submitPoll(
                cacheStatusExecutor,
                lease ->
                        new GrpcCacheStatusCheckRunner(
                                modelName,
                                workerStatus,
                                lease,
                                endpointRegistry,
                                engineHealthReporter,
                                workerStatusRpcClient,
                                cacheAwareService,
                                cacheIntervalService,
                                syncRequestTimeoutMs,
                                flexlbConfig
                                        .getWorkerRegistry()
                                        .getCacheStatus()
                                        .isFullSnapshotDebugMode(),
                                cacheStatusExecutor),
                cacheLease);
    }

    private void reportTelemetry(Runnable action) {
        Throwable failure = Failures.run(null, action);
        if (failure != null) {
            logger.warn("Worker sync telemetry failed for model: {}", modelName, failure);
        }
    }

    /** Build the factory before acquiring a lease; failed submissions release it here. */
    private void submitPoll(
            ExecutorService executor,
            Function<WorkerStatus.PollLease, Runnable> createRunner,
            WorkerStatus.PollLease lease) {
        if (lease == null) {
            return;
        }
        boolean handedOff = false;
        try {
            lease.trace().record(EndpointTrace.Stage.RUNNER_SUBMITTED);
            executor.submit(createRunner.apply(lease));
            handedOff = true;
        } catch (RejectedExecutionException rejected) {
            lease.trace().record(EndpointTrace.Stage.RUNNER_REJECTED, rejected);
            logger.debug("Worker poll rejected for model: {}", modelName);
        } finally {
            if (!handedOff) {
                lease.close();
            }
        }
    }

    private void reportLoadVariance(RoleType roleType) {
        StatsAccumulator stepLatency = new StatsAccumulator();
        StatsAccumulator runningLoad = new StatsAccumulator();
        for (var entry : endpointRegistry.statusSnapshot(roleType).entrySet()) {
            WorkerStatus worker = entry.getValue();
            if (!worker.isActiveGeneration()) {
                continue;
            }
            WorkerStatus.EngineObservation observation = worker.committedEngineObservation();
            if (!endpointRegistry.isCurrentStatus(roleType, entry.getKey(), worker)) {
                continue;
            }
            stepLatency.add(observation.stepLatencyMs());
            WorkerEndpoint endpoint = endpointRegistry.get(roleType, entry.getKey(), worker);
            if (endpoint != null) {
                endpoint.tryGetLoadMetric().ifPresent(runningLoad::add);
            }
        }
        if (stepLatency.count() >= 2) {
            engineHealthReporter.reportStepLatencyVariance(
                    modelName, roleType.name(), stepLatency.sampleVariance());
        }
        if (runningLoad.count() >= 2) {
            engineHealthReporter.reportRunningLoadVariance(
                    modelName, roleType.name(), runningLoad.sampleVariance());
        }
    }

    private WorkerStatus getOrCreateWorkerStatus(RoleType roleType, WorkerHost host) {
        if (host.getIp() == null || host.getIp().isBlank()) {
            throw new IllegalArgumentException("Worker address must have a nonempty host");
        }
        String workerIpPort = host.getLogicalIpPort();
        String site = host.getSite();
        String group = host.getGroup();
        while (true) {
            WorkerStatus workerStatus =
                    endpointRegistry.currentOrDiscover(
                            roleType,
                            workerIpPort,
                            () -> WorkerStatus.createDiscovered(roleType, host));

            if (!workerStatus.isAvailableForScheduling()) {
                return workerStatus;
            }
            WorkerStatus.TopologySnapshot topology = workerStatus.topologySnapshot();
            boolean sameTransport =
                    topology.grpcPort() == host.getGrpcPort()
                            && topology.workerStatusPort() == host.getWorkerStatusPort()
                            && topology.multiEngineNum() == host.getMultiEngineNum();
            if (sameTransport
                    && Objects.equals(topology.group(), group)
                    && Objects.equals(topology.site(), site)
                    && Objects.equals(topology.deploymentName(), host.getDeploymentName())) {
                return workerStatus;
            }
            EndpointRegistry.Retirement retirement;
            if (!workerStatus.lock.tryLock()) {
                return sameTransport && Objects.equals(topology.group(), group)
                        ? workerStatus
                        : null;
            }
            try {
                if (!endpointRegistry.isCurrentStatus(roleType, workerIpPort, workerStatus)) {
                    continue;
                }
                if (!workerStatus.isActiveGeneration()) {
                    return workerStatus;
                }

                RoleType currentRole = workerStatus.getRole();
                String currentGroup = workerStatus.getGroup();
                boolean roleChanged = currentRole != roleType;
                boolean groupChanged = !Objects.equals(currentGroup, group);
                if (!roleChanged && !groupChanged && sameTransport) {
                    // Site changes do not change scheduling ownership. Publish
                    // the discovery labels atomically on the same generation.
                    workerStatus.updateDiscoveryLabels(site, group, host.getDeploymentName());
                    return workerStatus;
                }

                // Group/role ownership is a generation boundary. Keep the old
                // status identity published as RETIRING until the endpoint's
                // real retirement completion runs the exact finalizer. Cache
                // cleanup is generation-scoped and must finish before replacement.
                RoleType generationRole = currentRole == null ? roleType : currentRole;
                retirement =
                        endpointRegistry.beginRetirement(
                                generationRole, workerIpPort, workerStatus);
            } finally {
                workerStatus.lock.unlock();
            }

            retirement.completeAsync(cacheAwareService, logger);
            logger.info(
                    "[replace] retiring worker topology generation, model={}, role={}, ipPort={}, generation={}, newGroup={}",
                    modelName,
                    roleType,
                    workerIpPort,
                    workerStatus.getGenerationId(),
                    group);
            // A later discovery pass can publish the replacement only after
            // real endpoint retirement removes this RETIRING holder.
            return workerStatus;
        }
    }

    @Scheduled(
            fixedRateString =
                    "#{@configService.loadBalanceConfig().workerRegistry.health.cleanupIntervalMs}")
    public void retireUnavailableWorkers() {
        List<EndpointRegistry.Retirement> retirements = new ArrayList<>();
        Throwable failure = null;
        try {
            // Close every unavailable routing gate before scheduling cleanup.
            for (RoleType role : RoleType.values()) {
                for (var item : endpointRegistry.statusSnapshot(role).entrySet()) {
                    WorkerStatus status = item.getValue();
                    var retirement =
                            endpointRegistry.beginRequestedRetirement(role, item.getKey(), status);
                    if (retirement != null) {
                        retirements.add(retirement);
                    }
                }
            }
        } catch (Throwable detachFailure) {
            failure = detachFailure;
        }
        for (EndpointRegistry.Retirement retirement : retirements) {
            failure =
                    Failures.run(
                            failure, () -> retirement.completeAsync(cacheAwareService, logger));
            WorkerStatus status = retirement.status();
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    logger.warn(
                                            "Retiring unavailable worker: {}, role: {}, generation={}, trace={}",
                                            status.getIpPort(),
                                            status.getRole(),
                                            status.getGenerationId(),
                                            status.trace().snapshot().compact()));
        }
        Failures.rethrow(failure, "Worker retirement scan failed");
    }
}

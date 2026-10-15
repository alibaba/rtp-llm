package org.flexlb.sync.synchronizer;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.EngineSyncRunner;
import org.flexlb.sync.status.WorkerDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;

/**
 * Master engine status synchronizer
 */
@Component
public final class MasterEngineSynchronizer {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String modelName;
    private final List<RoleType> requiredRoles;
    private final WorkerAddressService workerAddressService;
    private final WorkerDirectory workerDirectory;
    private final EngineHealthReporter engineHealthReporter;
    private final FlexlbConfig flexlbConfig;
    private final EngineGrpcService engineGrpcService;
    private final CacheAwareService cacheAwareService;
    private final DynamicCacheIntervalService cacheIntervalService;
    private final long syncRequestTimeoutMs;
    private final AtomicLong syncCount = new AtomicLong();
    private final Long syncEngineStatusInterval;
    private final long statusStaleAfterUs;
    private final ScheduledThreadPoolExecutor scheduler;
    private final ThreadPoolExecutor statusCheckExecutor;
    private final ThreadPoolExecutor cacheStatusExecutor;
    private final ThreadPoolExecutor engineSyncExecutor;
    // Retain only operations that currently occupy an executor thread or a caller thread.
    private final Map<EndpointTrace.Poll, RoleType> activeSyncs = new ConcurrentHashMap<>();

    public MasterEngineSynchronizer(WorkerAddressService workerAddressService,
                                    EngineHealthReporter engineHealthReporter,
                                    WorkerDirectory workerDirectory,
                                    EngineGrpcService engineGrpcService,
                                    ModelMetaConfig modelMetaConfig,
                                    CacheAwareService cacheAwareService,
                                    DynamicCacheIntervalService cacheIntervalService,
                                    ConfigService configService) {

        this.workerAddressService = workerAddressService;
        this.engineHealthReporter = engineHealthReporter;
        this.workerDirectory = workerDirectory;
        this.flexlbConfig = configService.loadBalanceConfig();
        this.engineGrpcService = engineGrpcService;
        this.cacheAwareService = cacheAwareService;
        this.cacheIntervalService = cacheIntervalService;
        this.modelName = modelMetaConfig.modelName();
        this.requiredRoles = modelMetaConfig.requiredRoles();

        this.syncEngineStatusInterval = flexlbConfig.getWorkerRegistry().getHealth()
                .getStatusPollIntervalMs();
        this.syncRequestTimeoutMs = flexlbConfig.getWorkerRegistry().getHealth()
                .getStatusRpcTimeoutMs();
        this.statusStaleAfterUs = flexlbConfig.getWorkerRegistry().getHealth()
                .getStatusStaleAfterMs() * 1000L;
        int engineThreads = flexlbConfig.getInternalRuntime()
                .getEngineSyncExecutorThreads();
        engineSyncExecutor = executor(engineThreads, "engine-sync-executor");
        int statusThreads = flexlbConfig.getInternalRuntime()
                .getStatusCheckExecutorThreads();
        statusCheckExecutor = executor(statusThreads, "status-checker-executor");
        // Cache overload must not run cache work on a health/discovery caller.
        int cacheThreads = flexlbConfig.getInternalRuntime().getCacheStatusExecutorThreads();
        cacheStatusExecutor = new ThreadPoolExecutor(
                cacheThreads, cacheThreads, 60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(15_000),
                new NamedThreadFactory("cache-status-executor"),
                new ThreadPoolExecutor.AbortPolicy());
        this.scheduler = new ScheduledThreadPoolExecutor(5, new NamedThreadFactory("sync-status-scheduler"),
                new ThreadPoolExecutor.AbortPolicy());
        this.scheduler.scheduleAtFixedRate(
                this::syncEngineStatus,
                0,
                syncEngineStatusInterval,
                TimeUnit.MILLISECONDS);
        this.scheduler.scheduleAtFixedRate(
                this::reportExecutorMetrics, 2, 2, TimeUnit.SECONDS);
    }

    private static ThreadPoolExecutor executor(int threads, String name) {
        return new ThreadPoolExecutor(
                threads, threads, 60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(15_000),
                new NamedThreadFactory(name),
                new ThreadPoolExecutor.CallerRunsPolicy());
    }

    private void reportExecutorMetrics() {
        try {
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    "engineSyncExecutor", engineSyncExecutor);
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    "statusCheckExecutor", statusCheckExecutor);
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    "cacheStatusExecutor", cacheStatusExecutor);
        } catch (Throwable failure) {
            logger.warn("Failed to report worker sync executor metrics", failure);
        }
        reportSlowTraces();
    }

    /** Read active operations on the existing reporting tick; no per-cycle history or queue tracking. */
    void reportSlowTraces() {
        try {
            if (!logger.isWarnEnabled()) {return;}
            long slowAfterUs = Math.max(syncEngineStatusInterval * 1000L, statusStaleAfterUs / 2L);
            String enginePool = null;
            String statusPool = null;
            for (var active : activeSyncs.entrySet()) {
                var snapshot = active.getKey().snapshot();
                if (snapshot.thread() == null || Stage.RUNNER_ENDED.name().equals(snapshot.stage())
                        || (snapshot.runUs() < slowAfterUs && snapshot.queueUs() < slowAfterUs)) {continue;}
                if (enginePool == null) {
                    enginePool = poolSnapshot(engineSyncExecutor);
                    statusPool = poolSnapshot(statusCheckExecutor);
                }
                logger.warn("event=worker_sync_trace model={} role={} tasksPerCycle={} engine={} status={} {}",
                        modelName, active.getValue(), requiredRoles.size(), enginePool, statusPool, snapshot.compact());
            }
        } catch (RuntimeException | LinkageError failure) {
            logger.warn("Failed to report worker sync traces", failure);
        }
    }

    private static String poolSnapshot(ThreadPoolExecutor executor) {
        return "{active:" + executor.getActiveCount() + ",size:" + executor.getPoolSize()
                + ",queued:" + executor.getQueue().size() + ",completed:" + executor.getCompletedTaskCount() + "}";
    }

    public void syncEngineStatus() {
        long syncCycleId = syncCount.incrementAndGet();
        logger.debug("sync engine status start, times:{}, modelName:{}",
                syncCycleId, modelName);
        try {
            for (RoleType roleType : requiredRoles) {
                EngineSyncRunner runner = new EngineSyncRunner(
                        modelName, workerDirectory,
                        workerAddressService, statusCheckExecutor, cacheStatusExecutor, engineHealthReporter,
                        engineGrpcService, roleType, cacheAwareService,
                        cacheIntervalService,
                        syncRequestTimeoutMs, syncCycleId, syncEngineStatusInterval,
                        flexlbConfig.getWorkerRegistry().getCacheStatus()
                                .isFullSnapshotDebugMode(),
                        statusStaleAfterUs,
                        flexlbConfig.getWorkerRegistry().getHealth().getStatusCommitGraceMs() * 1000L
                );
                engineSyncExecutor.submit(() -> {
                    activeSyncs.put(runner.trace(), roleType);
                    try {
                        runner.run();
                    } finally {
                        activeSyncs.remove(runner.trace());
                    }
                });
            }
        } catch (Exception e) {
            logger.error("sync engine prefill status error", e);
        }
    }

    public boolean isReady() {
        return requiredRoles.stream()
                .allMatch(role -> workerDirectory.routingCapacity(role) > 0);
    }

    @PreDestroy
    public void destroy() {
        scheduler.shutdown();
        engineSyncExecutor.shutdown();
        statusCheckExecutor.shutdown();
        cacheStatusExecutor.shutdown();
        activeSyncs.clear();
    }

}

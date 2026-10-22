package org.flexlb.sync.synchronizer;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.EngineSyncRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.EnumMap;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

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
                    ENGINE_BALANCING_THREAD_POOL_INFO, "engineSyncExecutor", engineSyncExecutor);
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO, "statusCheckExecutor", statusCheckExecutor);
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    "statusApplicationExecutor",
                    statusApplicationExecutor);
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO, "cacheStatusExecutor", cacheStatusExecutor);
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
                                    new EngineSyncRunner(
                                                    modelName,
                                                    endpointRegistry,
                                                    workerAddressService,
                                                    statusCheckExecutor,
                                                    engineHealthReporter,
                                                    workerStatusRpcClient,
                                                    roleType,
                                                    cacheAwareService,
                                                    cacheIntervalService,
                                                    syncRequestTimeoutMs,
                                                    syncCycleId,
                                                    syncEngineStatusInterval,
                                                    flexlbConfig
                                                            .getWorkerRegistry()
                                                            .getCacheStatus()
                                                            .isFullSnapshotDebugMode(),
                                                    cacheStatusExecutor,
                                                    statusApplicationExecutor)
                                            .run();
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
}

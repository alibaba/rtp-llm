package org.flexlb.sync.synchronizer;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.EngineSyncRunner;
import org.flexlb.sync.status.WorkerDirectory;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

/**
 * Master engine status synchronizer
 */
@Component
public class MasterEngineSynchronizer extends AbstractEngineStatusSynchronizer {

    private final String modelName;
    private final List<RoleType> requiredRoles;
    private final EngineGrpcService engineGrpcService;
    private final CacheAwareService cacheAwareService;
    private final DynamicCacheIntervalService cacheIntervalService;
    private final EndpointEventSink endpointEventSink;
    private final EndpointRegistry endpointRegistry;
    private final long syncRequestTimeoutMs;
    private final LongAdder syncCount = new LongAdder();
    private final Long syncEngineStatusInterval;
    private final long statusStaleAfterUs;

    public MasterEngineSynchronizer(WorkerAddressService workerAddressService,
                                    EngineHealthReporter engineHealthReporter,
                                    WorkerDirectory workerDirectory,
                                    EngineGrpcService engineGrpcService,
                                    ModelMetaConfig modelMetaConfig,
                                    CacheAwareService cacheAwareService,
                                    DynamicCacheIntervalService cacheIntervalService,
                                    EndpointEventSink endpointEventSink,
                                    EndpointRegistry endpointRegistry,
                                    ConfigService configService) {

        super(workerAddressService, engineHealthReporter, workerDirectory,
                modelMetaConfig, configService);

        this.engineGrpcService = engineGrpcService;
        this.cacheAwareService = cacheAwareService;
        this.cacheIntervalService = cacheIntervalService;
        this.endpointEventSink = java.util.Objects.requireNonNull(
                endpointEventSink, "endpointEventSink");
        this.endpointRegistry = endpointRegistry;
        this.modelName = modelMetaConfig.modelName();
        this.requiredRoles = modelMetaConfig.requiredRoles();

        this.syncEngineStatusInterval = flexlbConfig.getWorkerRegistry().getHealth()
                .getStatusPollIntervalMs();
        this.syncRequestTimeoutMs = flexlbConfig.getWorkerRegistry().getHealth()
                .getStatusRpcTimeoutMs();
        this.statusStaleAfterUs = flexlbConfig.getWorkerRegistry().getHealth()
                .getStatusStaleAfterMs() * 1000L;
        this.scheduler = new ScheduledThreadPoolExecutor(5, new NamedThreadFactory("sync-status-scheduler"),
                new ThreadPoolExecutor.AbortPolicy());
        this.scheduler.scheduleAtFixedRate(
                this::syncEngineStatus,
                0,
                syncEngineStatusInterval,
                TimeUnit.MILLISECONDS);
    }

    public void syncEngineStatus() {
        syncCount.increment();
        logger.debug("sync engine status start, times:{}, modelName:{}",
                syncCount.longValue(), modelName);
        try {
            for (RoleType roleType : requiredRoles) {
                engineSyncExecutor.submit(new EngineSyncRunner(
                        modelName, workerDirectory.statusMap(roleType),
                        workerAddressService, statusCheckExecutor, engineHealthReporter,
                        engineGrpcService, roleType, cacheAwareService,
                        cacheIntervalService,
                        syncRequestTimeoutMs, syncCount, syncEngineStatusInterval,
                        flexlbConfig.getWorkerRegistry().getCacheStatus()
                                .isFullSnapshotDebugMode(),
                        endpointEventSink, endpointRegistry,
                        statusStaleAfterUs
                ));
            }
        } catch (Exception e) {
            logger.error("sync engine prefill status error", e);
        }
    }

    public boolean isReady() {
        return requiredRoles.stream()
                .allMatch(role -> endpointRegistry.getEndpointCount(role) > 0);
    }

}

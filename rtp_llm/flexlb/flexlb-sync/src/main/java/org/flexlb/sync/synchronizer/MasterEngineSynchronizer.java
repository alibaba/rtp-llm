package org.flexlb.sync.synchronizer;

import com.fasterxml.jackson.core.type.TypeReference;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.EngineSyncRunner;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.EnvUtils;
import org.flexlb.util.IdUtils;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.flexlb.util.RateLimitedWarn;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Consumer;

/** Master engine-status synchronizer. */
@Component
public class MasterEngineSynchronizer extends AbstractEngineStatusSynchronizer {

    private final List<String> modelNames = new ArrayList<>();
    private final Map<String, Long> lastDiscoverySuccessUs = new ConcurrentHashMap<>();
    private final SingleFlightGate syncGate = new SingleFlightGate();
    private final Map<String, RateLimitedWarn> slowRoundWarns = new ConcurrentHashMap<>();
    private final Map<String, RateLimitedWarn> discoveryGapWarns = new ConcurrentHashMap<>();
    private final EngineGrpcService engineGrpcService;
    private final CacheAwareService localKvCacheAwareManager;
    private final FlexlbBatchScheduler batchScheduler;
    private final EndpointRegistry endpointRegistry;
    private final long syncRequestTimeoutMs;
    private final LongAdder syncCount = new LongAdder();
    private final long syncEngineStatusInterval;
    private volatile int completedSyncCount;

    @Autowired
    public MasterEngineSynchronizer(
            WorkerAddressService workerAddressService,
            EngineHealthReporter engineHealthReporter,
            EngineWorkerStatus engineWorkerStatus,
            EngineGrpcService engineGrpcService,
            ModelMetaConfig modelMetaConfig,
            CacheAwareService localKvCacheAwareManager,
            @Autowired(required = false) FlexlbBatchScheduler batchScheduler,
            EndpointRegistry endpointRegistry,
            ConfigService configService) {
        this(
                workerAddressService,
                engineHealthReporter,
                engineWorkerStatus,
                engineGrpcService,
                modelMetaConfig,
                localKvCacheAwareManager,
                batchScheduler,
                endpointRegistry,
                configService,
                System.getenv("MODEL_SERVICE_CONFIG"),
                MasterEngineSynchronizer::startPeriodicSync);
    }

    MasterEngineSynchronizer(
            WorkerAddressService workerAddressService,
            EngineHealthReporter engineHealthReporter,
            EngineWorkerStatus engineWorkerStatus,
            EngineGrpcService engineGrpcService,
            ModelMetaConfig modelMetaConfig,
            CacheAwareService localKvCacheAwareManager,
            FlexlbBatchScheduler batchScheduler,
            EndpointRegistry endpointRegistry,
            ConfigService configService,
            String modelConfig,
            Consumer<MasterEngineSynchronizer> schedulerStarter) {
        super(
                workerAddressService,
                engineHealthReporter,
                engineWorkerStatus,
                modelMetaConfig,
                configService);
        this.engineGrpcService = engineGrpcService;
        this.localKvCacheAwareManager = localKvCacheAwareManager;
        this.batchScheduler = batchScheduler;
        this.endpointRegistry = endpointRegistry;
        this.syncEngineStatusInterval =
                EnvUtils.readPositiveLong("SYNC_STATUS_INTERVAL", 20L);
        this.syncRequestTimeoutMs =
                EnvUtils.readPositiveLong("SYNC_REQUEST_TIMEOUT_MS", 5000L);

        if (StringUtils.isEmpty(modelConfig)) {
            Logger.warn("master load balancer env MODEL_SERVICE_CONFIG is empty");
            throw new IllegalStateException(
                    "master load balancer env MODEL_SERVICE_CONFIG is empty");
        }
        ServiceRoute serviceRoute = JsonUtils.toObject(
                modelConfig, new TypeReference<>() { });
        ModelMetaConfig.putServiceRoute(serviceRoute.getServiceId(), serviceRoute);
        modelNames.add(IdUtils.getModelNameByServiceId(serviceRoute.getServiceId()));

        flexlbConfig.validateEngineTypeConfig(serviceRoute.getAllRoleTypes());
        Logger.info("engine type: {}", flexlbConfig.getEngineType());
        schedulerStarter.accept(this);
    }

    /** Compatibility seam retained for focused tests that do not exercise endpoint calibration. */
    MasterEngineSynchronizer(
            WorkerAddressService workerAddressService,
            EngineHealthReporter engineHealthReporter,
            EngineWorkerStatus engineWorkerStatus,
            EngineGrpcService engineGrpcService,
            ModelMetaConfig modelMetaConfig,
            CacheAwareService localKvCacheAwareManager,
            ConfigService configService,
            String modelConfig,
            Consumer<MasterEngineSynchronizer> schedulerStarter) {
        this(
                workerAddressService,
                engineHealthReporter,
                engineWorkerStatus,
                engineGrpcService,
                modelMetaConfig,
                localKvCacheAwareManager,
                null,
                null,
                configService,
                modelConfig,
                schedulerStarter);
    }

    private void startPeriodicSync() {
        this.scheduler = new ScheduledThreadPoolExecutor(
                5,
                new NamedThreadFactory("sync-status-scheduler"),
                new ThreadPoolExecutor.AbortPolicy());
        this.scheduler.scheduleAtFixedRate(
                this::syncEngineStatus,
                0,
                syncEngineStatusInterval,
                TimeUnit.MILLISECONDS);
    }

    @Override
    public void syncEngineStatus() {
        syncCount.increment();
        logger.debug("sync engine status start, times:{}, modelNames:{}",
                syncCount.longValue(), modelNames);
        try {
            for (String modelName : modelNames) {
                ModelWorkerStatus modelWorkerStatus =
                        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
                String serviceId = IdUtils.getServiceIdByModelName(modelName);
                if (serviceId.isEmpty()) {
                    logger.error("serviceId not found for model:{}", modelName);
                    continue;
                }
                ServiceRoute serviceRoute = modelMetaConfig.getServiceRoute(serviceId);
                if (serviceRoute == null) {
                    logger.error("serviceRoute not found for serviceId:{}", serviceId);
                    continue;
                }

                for (RoleType roleType : serviceRoute.getAllRoleTypes()) {
                    List<Endpoint> roleEndpoints = serviceRoute.getRoleEndpoints(roleType);
                    if (roleEndpoints == null) {
                        logger.error("roleEndpoints is null, roleType:{}", roleType);
                        continue;
                    }
                    try {
                        submitRound(
                                modelName,
                                roleType,
                                modelWorkerStatus.getRoleStatusMap(roleType));
                    } catch (Throwable error) {
                        logger.error(
                                "submit sync round failed, model={}, role={}",
                                modelName, roleType, error);
                    }
                }
            }
            completedSyncCount++;
        } catch (Throwable error) {
            // An Error escaping scheduleAtFixedRate silently suppresses every future tick.
            logger.error("sync engine status error", error);
        }
    }

    void submitRound(
            String modelName,
            RoleType roleType,
            Map<String, WorkerStatus> roleStatusMap) {
        String key = modelName + "/" + roleType;
        boolean submitted = syncGate.submit(key, engineSyncExecutor, () ->
                new EngineSyncRunner(
                        modelName,
                        roleStatusMap,
                        workerAddressService,
                        statusCheckExecutor,
                        engineHealthReporter,
                        engineGrpcService,
                        roleType,
                        localKvCacheAwareManager,
                        syncRequestTimeoutMs,
                        syncCount,
                        syncEngineStatusInterval,
                        batchScheduler,
                        endpointRegistry,
                        flexlbConfig.getEngineType(),
                        flexlbConfig.getDiscoveryFailureGraceMs(),
                        lastDiscoverySuccessUs,
                        discoveryGapWarns.computeIfAbsent(
                                key, ignored -> new RateLimitedWarn(1, TimeUnit.SECONDS)))
                        .run());
        if (!submitted) {
            slowRoundWarns.computeIfAbsent(
                            key, ignored -> new RateLimitedWarn(1, TimeUnit.SECONDS))
                    .warn("sync round still in flight, skipping tick: key={}", key);
        }
    }

    public boolean isReady() {
        return completedSyncCount > 0;
    }
}

package org.flexlb.sync.synchronizer;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.EngineSyncRunner;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.IdUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

/**
 * Master engine status synchronizer
 */
@Component
public class MasterEngineSynchronizer extends AbstractEngineStatusSynchronizer {

    private static final long DEFAULT_SYNC_REQUEST_TIMEOUT_MS = 200L;

    private final List<String> modelNames = new ArrayList<>();
    private final EngineGrpcService engineGrpcService;
    private final CacheAwareService cacheAwareService;
    private final long syncRequestTimeoutMs;
    private final LongAdder syncCount = new LongAdder();
    private final Long syncEngineStatusInterval;

    @Autowired
    public MasterEngineSynchronizer(WorkerAddressService workerAddressService,
                                    EngineHealthReporter engineHealthReporter,
                                    EngineWorkerStatus engineWorkerStatus,
                                    EngineGrpcService engineGrpcService,
                                    ModelMetaConfig modelMetaConfig,
                                    CacheAwareService cacheAwareService,
                                    ConfigService configService) {
        this(workerAddressService, engineHealthReporter, engineWorkerStatus, engineGrpcService, modelMetaConfig,
                cacheAwareService, configService, true);
    }

    MasterEngineSynchronizer(WorkerAddressService workerAddressService,
                             EngineHealthReporter engineHealthReporter,
                             EngineWorkerStatus engineWorkerStatus,
                             EngineGrpcService engineGrpcService,
                             ModelMetaConfig modelMetaConfig,
                             CacheAwareService cacheAwareService,
                             ConfigService configService,
                             boolean startStatusSync) {

        super(workerAddressService, engineHealthReporter, engineWorkerStatus, modelMetaConfig, configService);

        this.engineGrpcService = engineGrpcService;
        this.cacheAwareService = cacheAwareService;

        this.syncEngineStatusInterval = System.getenv("SYNC_STATUS_INTERVAL") != null
                ? Long.parseLong(System.getenv("SYNC_STATUS_INTERVAL"))
                : 20;
        this.syncRequestTimeoutMs = System.getenv("SYNC_REQUEST_TIMEOUT_MS") != null
                ? Long.parseLong(System.getenv("SYNC_REQUEST_TIMEOUT_MS"))
                : DEFAULT_SYNC_REQUEST_TIMEOUT_MS;
        modelMetaConfig.getServiceRoutes().stream()
                .map(ServiceRoute::getServiceId)
                .map(IdUtils::getModelNameByServiceId)
                .forEach(modelNames::add);
        this.scheduler = new ScheduledThreadPoolExecutor(5, new NamedThreadFactory("sync-status-scheduler"),
                new ThreadPoolExecutor.AbortPolicy());
        if (startStatusSync) {
            this.scheduler.scheduleAtFixedRate(this::syncEngineStatus, 0, syncEngineStatusInterval, TimeUnit.MILLISECONDS);
        }
    }

    public void syncEngineStatus() {
        syncCount.increment();
        logger.debug("Sync engine status start, times={}, modelNames={}",
                syncCount.longValue(), modelNames);
        try {
            for (String modelName : modelNames) {
                ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
                String serviceId = IdUtils.getServiceIdByModelName(modelName);
                if (serviceId.isEmpty()) {
                    logger.error("serviceId not found, serviceId:{}", serviceId);
                }
                ServiceRoute serviceRoute = modelMetaConfig.getServiceRoute(serviceId);
                if (serviceRoute == null) {
                    logger.error("serviceRoute not found");
                    continue;
                }
                List<RoleType> roleTypes = serviceRoute.getAllRoleTypes();

                for (RoleType roleType : roleTypes) {
                    List<Endpoint> roleEndpoints = serviceRoute.getRoleEndpoints(roleType);
                    if (roleEndpoints != null) {
                        try {
                            engineSyncExecutor.submit(new EngineSyncRunner(
                                    modelName, modelWorkerStatus.getRoleStatusMap(roleType),
                                    workerAddressService, statusCheckExecutor, engineHealthReporter,
                                    engineGrpcService, roleType, cacheAwareService,
                                    syncRequestTimeoutMs, syncCount, syncEngineStatusInterval,
                                    serviceRoute.isKvcmEnabled()
                            ));
                        } catch (RejectedExecutionException exception) {
                            logger.debug("Engine status sync task rejected, modelName={}, roleType={}",
                                    modelName, roleType);
                        }
                    } else {
                        logger.error("roleEndpoints is null, by roleType : {}", roleType);
                    }
                }
            }
        } catch (Exception e) {
            logger.error("sync engine prefill status error", e);
        }
    }
}

package org.flexlb.sync.synchronizer;

import com.fasterxml.jackson.core.type.TypeReference;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.cache.service.CacheAwareService;
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
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
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

    private final List<String> modelNames = new ArrayList<>();
    private final EngineGrpcService engineGrpcService;
    private final CacheAwareService localKvCacheAwareManager;
    private final FlexlbBatchScheduler batchScheduler;
    private final EndpointRegistry endpointRegistry;
    private final long syncRequestTimeoutMs;
    private final LongAdder syncCount = new LongAdder();
    private final Long syncEngineStatusInterval;
    private volatile int completedSyncCount = 0;

    public MasterEngineSynchronizer(WorkerAddressService workerAddressService,
                                    EngineHealthReporter engineHealthReporter,
                                    EngineWorkerStatus engineWorkerStatus,
                                    EngineGrpcService engineGrpcService,
                                    ModelMetaConfig modelMetaConfig,
                                    CacheAwareService localKvCacheAwareManager,
                                    @org.springframework.beans.factory.annotation.Autowired(required = false)
                                    FlexlbBatchScheduler batchScheduler,
                                    EndpointRegistry endpointRegistry,
                                    ConfigService configService) {

        super(workerAddressService, engineHealthReporter, engineWorkerStatus, modelMetaConfig, configService);

        this.engineGrpcService = engineGrpcService;
        this.localKvCacheAwareManager = localKvCacheAwareManager;
        this.batchScheduler = batchScheduler;
        this.endpointRegistry = endpointRegistry;

        this.syncEngineStatusInterval = System.getenv("SYNC_STATUS_INTERVAL") != null
                ? Long.parseLong(System.getenv("SYNC_STATUS_INTERVAL"))
                : 20;
        this.syncRequestTimeoutMs = System.getenv("SYNC_REQUEST_TIMEOUT_MS") != null
                ? Long.parseLong(System.getenv("SYNC_REQUEST_TIMEOUT_MS"))
                : 5000;  // 5s default for gRPC calls, must not fallback to sync interval (20ms) which is too short
        this.scheduler = new ScheduledThreadPoolExecutor(5, new NamedThreadFactory("sync-status-scheduler"),
                new ThreadPoolExecutor.AbortPolicy());
        this.scheduler.scheduleAtFixedRate(this::syncEngineStatus, 0, syncEngineStatusInterval, TimeUnit.MILLISECONDS);

        // Get environment variable
        String modelConfig = System.getenv("MODEL_SERVICE_CONFIG");
        if (StringUtils.isEmpty(modelConfig)) {
            Logger.warn("prefill load balancer env:MODEL_CONFIG is empty");
            throw new RuntimeException("master load balancer env:MODEL_CONFIG is empty");
        }
        ServiceRoute serviceRoute = JsonUtils.toObject(modelConfig, new TypeReference<>() {
        });
        ModelMetaConfig.putServiceRoute(serviceRoute.getServiceId(), serviceRoute);
        modelNames.add(IdUtils.getModelNameByServiceId(serviceRoute.getServiceId()));
    }

    public void syncEngineStatus() {
        syncCount.increment();
        logger.debug("====================sync engine prefill status start, times:{} =========================", syncCount.longValue());
        logger.debug("modelNames:{}", modelNames);
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
                        engineSyncExecutor.submit(new EngineSyncRunner(
                                modelName, modelWorkerStatus.getRoleStatusMap(roleType),
                                workerAddressService, statusCheckExecutor, engineHealthReporter,
                                engineGrpcService, roleType, localKvCacheAwareManager,
                                syncRequestTimeoutMs, syncCount, syncEngineStatusInterval,
                                batchScheduler, endpointRegistry
                        ));
                    } else {
                        logger.error("roleEndpoints is null, by roleType : {}", roleType);
                    }
                }
            }
            completedSyncCount++;
        } catch (Exception e) {
            logger.error("sync engine prefill status error", e);
        }
    }

    public boolean isReady() {
        return completedSyncCount > 0;
    }

}

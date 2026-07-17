package org.flexlb.sync.synchronizer;

import com.fasterxml.jackson.core.type.TypeReference;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.apache.commons.lang3.StringUtils;
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

/**
 * Master engine status synchronizer
 */
@Component
public class MasterEngineSynchronizer extends AbstractEngineStatusSynchronizer {

    private final List<String> modelNames = new ArrayList<>();
    /** Last successful discovery time per {@code model/role}, shared across this bean's per-round runners. */
    private final Map<String, Long> lastDiscoverySuccessUs = new ConcurrentHashMap<>();
    /** One in-flight sync round per {@code model/role} — see {@link #submitRound}. */
    private final SingleFlightGate syncGate = new SingleFlightGate();
    /**
     * Ticks are 20ms apart; a slow-discovery deployment would otherwise flood the log. Keyed per
     * {@code model/role} so one key's suppression window cannot swallow another key's signal in a
     * multi-role deployment.
     */
    private final Map<String, RateLimitedWarn> slowRoundWarns = new ConcurrentHashMap<>();
    /**
     * Discovery-gap warn per {@code model/role}, handed to the per-round runners — runners are
     * recreated every round, so a runner-owned instance would never suppress anything.
     */
    private final Map<String, RateLimitedWarn> discoveryGapWarns = new ConcurrentHashMap<>();
    private final EngineGrpcService engineGrpcService;
    private final CacheAwareService localKvCacheAwareManager;
    private final long syncRequestTimeoutMs;
    private final LongAdder syncCount = new LongAdder();
    private final Long syncEngineStatusInterval;

    @Autowired
    public MasterEngineSynchronizer(WorkerAddressService workerAddressService,
                                    EngineHealthReporter engineHealthReporter,
                                    EngineWorkerStatus engineWorkerStatus,
                                    EngineGrpcService engineGrpcService,
                                    ModelMetaConfig modelMetaConfig,
                                    CacheAwareService localKvCacheAwareManager,
                                    ConfigService configService) {

        this(workerAddressService, engineHealthReporter, engineWorkerStatus, engineGrpcService,
                modelMetaConfig, localKvCacheAwareManager, configService,
                System.getenv("MODEL_SERVICE_CONFIG"), MasterEngineSynchronizer::startPeriodicSync);
    }

    /**
     * Wiring seam: production reads {@code MODEL_SERVICE_CONFIG} from the environment and starts
     * the real scheduler; tests inject the config string and the start hook, making "a failed
     * validation must not start the scheduler" observable without a live scheduler thread.
     */
    MasterEngineSynchronizer(WorkerAddressService workerAddressService,
                             EngineHealthReporter engineHealthReporter,
                             EngineWorkerStatus engineWorkerStatus,
                             EngineGrpcService engineGrpcService,
                             ModelMetaConfig modelMetaConfig,
                             CacheAwareService localKvCacheAwareManager,
                             ConfigService configService,
                             String modelConfig,
                             Consumer<MasterEngineSynchronizer> schedulerStarter) {

        super(workerAddressService, engineHealthReporter, engineWorkerStatus, modelMetaConfig, configService);

        this.engineGrpcService = engineGrpcService;
        this.localKvCacheAwareManager = localKvCacheAwareManager;

        this.syncEngineStatusInterval = System.getenv("SYNC_STATUS_INTERVAL") != null
                ? Long.parseLong(System.getenv("SYNC_STATUS_INTERVAL"))
                : 20;
        this.syncRequestTimeoutMs = System.getenv("SYNC_REQUEST_TIMEOUT_MS") != null
                ? Long.parseLong(System.getenv("SYNC_REQUEST_TIMEOUT_MS"))
                : syncEngineStatusInterval;

        if (StringUtils.isEmpty(modelConfig)) {
            Logger.warn("prefill load balancer env:MODEL_CONFIG is empty");
            throw new RuntimeException("master load balancer env:MODEL_CONFIG is empty");
        }
        ServiceRoute serviceRoute = JsonUtils.toObject(modelConfig, new TypeReference<>() {
        });
        ModelMetaConfig.putServiceRoute(serviceRoute.getServiceId(), serviceRoute);
        modelNames.add(IdUtils.getModelNameByServiceId(serviceRoute.getServiceId()));

        flexlbConfig.validateEngineTypeConfig(serviceRoute.getAllRoleTypes());
        Logger.warn("engine type: {}", flexlbConfig.getEngineType());

        // Start the periodic sync only after config parse + validation succeed, so a failed
        // construction never leaves a live scheduler thread running against partial state.
        schedulerStarter.accept(this);
    }

    private void startPeriodicSync() {
        this.scheduler = new ScheduledThreadPoolExecutor(5, new NamedThreadFactory("sync-status-scheduler"),
                new ThreadPoolExecutor.AbortPolicy());
        this.scheduler.scheduleAtFixedRate(this::syncEngineStatus, 0, syncEngineStatusInterval, TimeUnit.MILLISECONDS);
    }

    public void syncEngineStatus() {
        syncCount.increment();
        logger.info("====================sync engine prefill status start, times:{} =========================", syncCount.longValue());
        logger.info("modelNames:{}", modelNames);
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
                    if (roleEndpoints == null) {
                        logger.error("roleEndpoints is null, by roleType : {}", roleType);
                        continue;
                    }
                    try {
                        submitRound(modelName, roleType, modelWorkerStatus.getRoleStatusMap(roleType));
                    } catch (Throwable e) {
                        // Per-role rounds are independent: a rejected submit (saturated executor)
                        // for one role must not starve the remaining roles of this tick.
                        logger.error("submit sync round failed, model={}, role={}", modelName, roleType, e);
                    }
                }
            }
        } catch (Throwable e) {
            // Throwable, not Exception: an Error escaping a scheduleAtFixedRate task cancels
            // all future executions silently — the sync loop must survive anything.
            logger.error("sync engine prefill status error", e);
        }
    }

    /**
     * Submit one sync round for {@code model/role}, but only if the previous round for that key has
     * finished.
     *
     * <p>The scheduler ticks every {@code syncEngineStatusInterval} (default 20ms) and this method
     * only hands work to {@link #engineSyncExecutor} before returning, so without this gate the
     * rounds themselves overlap: a discovery lookup is bounded at 500ms, which is 25 ticks. Rounds
     * are not commutative — each applies its own snapshot as the truth (for EMBEDDING it marks
     * workers dead/alive purely from discovery presence), so a slow round finishing after a fast
     * one silently reverts the newer membership. Skipping the tick keeps exactly one round in
     * flight per key, which is what {@code scheduleWithFixedDelay} would give us if the work were
     * done inline.
     */
    void submitRound(String modelName, RoleType roleType, Map<String, WorkerStatus> roleStatusMap) {
        String key = modelName + "/" + roleType;
        boolean submitted = syncGate.submit(key, engineSyncExecutor, () -> new EngineSyncRunner(
                modelName, roleStatusMap,
                workerAddressService, statusCheckExecutor, engineHealthReporter,
                engineGrpcService, roleType, localKvCacheAwareManager,
                syncRequestTimeoutMs, syncCount, syncEngineStatusInterval,
                flexlbConfig.getEngineType(), flexlbConfig.getDiscoveryFailureGraceMs(),
                lastDiscoverySuccessUs,
                discoveryGapWarns.computeIfAbsent(key, k -> new RateLimitedWarn(1, TimeUnit.SECONDS))
        ).run());
        if (!submitted) {
            // Discovery is slower than the tick — worth surfacing, but at 20ms ticks it must not
            // flood the log.
            slowRoundWarns.computeIfAbsent(key, k -> new RateLimitedWarn(1, TimeUnit.SECONDS))
                    .warn("sync round still in flight, skipping tick: key={}", key);
        }
    }
}

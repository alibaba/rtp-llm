package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerGenerationFence;
import org.flexlb.sync.status.WorkerGenerationManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

public class EngineSyncRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String modelName;

    private final ConcurrentMap<String /*ipPort*/, WorkerStatus> workerStatusMap;

    private final WorkerAddressService workerAddressService;

    private final ExecutorService statusCheckExecutor;

    private final EngineHealthReporter engineHealthReporter;

    private final EngineGrpcService engineGrpcService;

    private final RoleType roleType;

    private final CacheAwareService localKvCacheAwareManager;

    private final long syncRequestTimeoutMs;

    private final LongAdder syncCount;

    private final Long syncEngineStatusInterval;

    private final FlexlbBatchScheduler batchScheduler;

    private final EndpointRegistry endpointRegistry;
    private final WorkerGenerationManager generationManager;
    private final WorkerGenerationFence generationFence;

    public EngineSyncRunner(String modelName,
                            ConcurrentMap<String, WorkerStatus> workerStatusMap,
                            WorkerAddressService workerAddressService,
                            ExecutorService statusCheckExecutor,
                            EngineHealthReporter engineHealthReporter,
                            EngineGrpcService engineGrpcService,
                            RoleType roleType,
                            CacheAwareService localKvCacheAwareManager,
                            long syncRequestTimeoutMs,
                            LongAdder syncCount,
                            Long syncEngineStatusInterval,
                            FlexlbBatchScheduler batchScheduler,
                            EndpointRegistry endpointRegistry,
                            WorkerGenerationManager generationManager,
                            WorkerGenerationFence generationFence) {

        this.modelName = modelName;
        this.workerAddressService = workerAddressService;
        this.workerStatusMap = workerStatusMap;
        this.statusCheckExecutor = statusCheckExecutor;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.roleType = roleType;
        this.localKvCacheAwareManager = localKvCacheAwareManager;
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
        this.syncCount = syncCount;
        this.syncEngineStatusInterval = syncEngineStatusInterval;
        this.batchScheduler = batchScheduler;
        this.endpointRegistry = endpointRegistry;
        this.generationManager = generationManager;
        this.generationFence = generationFence;
    }

    @Override
    public void run() {
        try {
            List<WorkerHost> latestEngineWorkerList = workerAddressService.getEngineWorkerList(modelName, roleType);
            engineHealthReporter.reportServiceDiscoveryResult(modelName, latestEngineWorkerList.size(), roleType.toString());
            ConcurrentMap<String/*ip*/, WorkerStatus> cachedWorkerStatuses = workerStatusMap;
            // Log if latest worker count differs from cached worker count
            if (cachedWorkerStatuses.size() != latestEngineWorkerList.size()) {
                logger.info("[update] engine ip changes, model={}, role={}, before={}, after={}",
                        modelName, roleType, cachedWorkerStatuses.size(), latestEngineWorkerList.size());
            }

            // Remove if not in latest engine list
            Set<String> latestValidIpPorts = latestEngineWorkerList.stream()
                    .map(WorkerHost::getIpPort)
                    .collect(Collectors.toSet());
            for (Map.Entry<String, WorkerStatus> entry: cachedWorkerStatuses.entrySet()) {
                WorkerStatus workerStatus = entry.getValue();
                String ipPort = entry.getKey();
                if (!latestValidIpPorts.contains(ipPort)) {
                    long lastTime = workerStatus.getStatusLastUpdateTime().get();
                    long actualIntervalUs = workerStatus.getStatusUpdateIntervalUs().get();
                    // Use max(3 * actual sync interval, 1s) as removal threshold to tolerate transient service discovery flaps
                    long removalThresholdUs = Math.max(3 * actualIntervalUs, 1_000_000L);
                    if (System.nanoTime() / 1000 - lastTime > removalThresholdUs) {
                        boolean removed = generationManager.retireIf(
                                cachedWorkerStatuses, roleType, ipPort, workerStatus,
                                current -> System.nanoTime() / 1000
                                        - current.getStatusLastUpdateTime().get()
                                        > Math.max(3 * current.getStatusUpdateIntervalUs().get(), 1_000_000L));
                        if (removed) {
                            logger.info("[remove] engine ip changes, model={}, role={}, ipPort={}",
                                    modelName, roleType, ipPort);
                        }
                    }
                }
            }
            if (latestEngineWorkerList.isEmpty()) {
                return;
            }

            for (WorkerHost host : latestEngineWorkerList) {
                String workerIpPort = host.getIpPort();
                String site = host.getSite();

                WorkerStatus workerStatus = getOrCreateWorkerStatus(cachedWorkerStatuses, workerIpPort);

                if (workerStatus.getStatusCheckInProgress().compareAndSet(false, true)) {
                    try {
                        GrpcWorkerStatusRunner grpcWorkerStatusRunner
                                = new GrpcWorkerStatusRunner(modelName, workerIpPort, site, roleType, host.getGroup(),
                                workerStatus, cachedWorkerStatuses, engineHealthReporter, engineGrpcService,
                                syncRequestTimeoutMs, batchScheduler, endpointRegistry,
                                generationManager, generationFence, statusCheckExecutor);
                        statusCheckExecutor.submit(grpcWorkerStatusRunner);
                    } catch (RejectedExecutionException e) {
                        workerStatus.getStatusCheckInProgress().set(false);
                    }
                }

                if (workerStatus.getCacheCheckInProgress().compareAndSet(false, true)) {
                    try {
                        GrpcCacheStatusCheckRunner grpcCacheStatusCheckRunner
                                = new GrpcCacheStatusCheckRunner(modelName, workerIpPort, roleType,
                                workerStatus, engineHealthReporter, engineGrpcService, localKvCacheAwareManager,
                                cachedWorkerStatuses, generationFence,
                                syncRequestTimeoutMs, syncCount, syncEngineStatusInterval, statusCheckExecutor);
                        statusCheckExecutor.submit(grpcCacheStatusCheckRunner);
                    } catch (RejectedExecutionException e) {
                        workerStatus.getCacheCheckInProgress().set(false);
                    }
                }
            }

        } catch (Exception e) {
            logger.error("sync engine workers status exception, modelName:{}, error:{}", modelName, e.getMessage(), e);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, null);
        } finally {
            int size = workerStatusMap.size();

            if (size >= 2) {
                double sumStepLatency = 0.0;
                double sumRunningLoad = 0.0;
                for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
                    WorkerStatus workerStatus = entry.getValue();
                    sumStepLatency += workerStatus.getStepLatencyMs();
                    sumRunningLoad += workerStatus.getReportedSchedulingLoad();
                }
                double meanStepLatency = sumStepLatency / size;
                double meanRunningLoad = sumRunningLoad / size;

                // Calculate variance (sample variance using Bessel correction)
                double sumStepLatencyOfSquaredDiffs = 0.0;
                double sumRunningLoadOfSquaredDiffs = 0.0;
                for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
                    WorkerStatus workerStatus = entry.getValue();
                    double diff = workerStatus.getStepLatencyMs() - meanStepLatency;
                    double diff2 = workerStatus.getReportedSchedulingLoad() - meanRunningLoad;
                    sumStepLatencyOfSquaredDiffs += diff * diff;
                    sumRunningLoadOfSquaredDiffs += diff2 * diff2;
                }
                double variance = sumStepLatencyOfSquaredDiffs / (size - 1); // Sample variance
                double variance2 = sumRunningLoadOfSquaredDiffs / (size - 1);

                engineHealthReporter.reportLatencyMetric(modelName, this.roleType.toString(), variance, variance2);
            }
        }
    }

    private WorkerStatus getOrCreateWorkerStatus(
            ConcurrentMap<String, WorkerStatus> workerStatuses, String workerIpPort) {
        return generationManager.getOrCreate(workerStatuses, roleType, workerIpPort);
    }
}

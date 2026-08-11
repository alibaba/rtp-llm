package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointRetireCause;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.CollectionUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

public class EngineSyncRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String modelName;

    private final Map<String /*ipPort*/, WorkerStatus> workerStatusMap;

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

    public EngineSyncRunner(String modelName,
                            Map<String, WorkerStatus> workerStatusMap,
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
                            EndpointRegistry endpointRegistry) {

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
    }

    @Override
    public void run() {
        logger.info("EngineSyncRunner start for model: {}, role: {}", modelName, roleType.toString());
        try {
            long startTimeInUs = System.nanoTime() / 1000;
            List<WorkerHost> latestEngineWorkerList = workerAddressService.getEngineWorkerList(modelName, roleType);
            logger.info("workerAddressService getEngineWorkerList, model: {}, role: {}, size: {}", modelName, roleType, latestEngineWorkerList.size());
            engineHealthReporter.reportServiceDiscoveryResult(modelName, latestEngineWorkerList.size(), roleType.toString());
            if (CollectionUtils.isEmpty(latestEngineWorkerList)) {
                logger.error("get engine worker list is empty, cost={}μs, model={}", System.nanoTime() / 1000 - startTimeInUs, modelName);
            }
            Map<String/*ip*/, WorkerStatus> cachedWorkerStatuses = workerStatusMap;
            // Log if latest worker count differs from cached worker count
            if (cachedWorkerStatuses.size() != latestEngineWorkerList.size()) {
                logger.info("[update] engine ip changes, model={}, role={}, before={}, after={}",
                        modelName, roleType, cachedWorkerStatuses.size(), latestEngineWorkerList.size());
            }

            // Retire workers that have disappeared from service discovery. Discovery
            // freshness is intentionally separate from the successful status heartbeat.
            Set<String> latestValidIpPorts = latestEngineWorkerList.stream()
                    .map(WorkerHost::getIpPort)
                    .collect(Collectors.toSet());
            logger.info("Current cached worker size: {}, latest worker list size: {}", cachedWorkerStatuses.size(), latestEngineWorkerList.size());
            for (Map.Entry<String, WorkerStatus> entry: cachedWorkerStatuses.entrySet()) {
                WorkerStatus workerStatus = entry.getValue();
                String ipPort = entry.getKey();
                if (!latestValidIpPorts.contains(ipPort)) {
                    long lastTime = workerStatus.getDiscoveryLastSeenTime().get();
                    long actualIntervalUs = workerStatus.getDiscoveryUpdateIntervalUs().get();
                    // Use max(3 * actual sync interval, 1s) as removal threshold to tolerate transient service discovery flaps
                    long removalThresholdUs = Math.max(3 * actualIntervalUs, 1_000_000L);
                    if (lastTime > 0L && System.nanoTime() / 1000 - lastTime > removalThresholdUs) {
                        retireWorkerGeneration(
                                cachedWorkerStatuses, ipPort, workerStatus,
                                lastTime, "discovery-missing");
                    }
                }
            }
            if (latestEngineWorkerList.isEmpty()) {
                logger.warn("latestEngineWorkerList is empty, role: {}", roleType);
                return;
            } else {
                logger.info("latestEngineWorkerList for role: {}, workers:{}", roleType, latestEngineWorkerList.size());
            }

            logger.info("Submitting status check tasks for {} workers", latestEngineWorkerList.size());
            for (WorkerHost host : latestEngineWorkerList) {
                String workerIpPort = host.getIpPort();
                String site = host.getSite();

                WorkerStatus workerStatus = getOrCreateWorkerStatus(cachedWorkerStatuses, workerIpPort);
                workerStatus.recordDiscoverySeen(System.nanoTime() / 1000);

                if (!workerStatus.isProbeable()) {
                    logger.debug("Skip retired worker generation: {}, state={}",
                            workerIpPort, workerStatus.getLifecycleState());
                    continue;
                }

                if (workerStatus.getStatusCheckInProgress().compareAndSet(false, true)) {
                    try {
                        logger.debug("Submitting GrpcWorkerStatusRunner for worker: {}, site: {}", workerIpPort, site);
                        GrpcWorkerStatusRunner grpcWorkerStatusRunner
                                = new GrpcWorkerStatusRunner(modelName, workerIpPort, site, roleType, host.getGroup(),
                                workerStatus, cachedWorkerStatuses, engineHealthReporter, engineGrpcService,
                                syncRequestTimeoutMs, batchScheduler, endpointRegistry, statusCheckExecutor);
                        statusCheckExecutor.submit(grpcWorkerStatusRunner);
                    } catch (RejectedExecutionException e) {
                        workerStatus.getStatusCheckInProgress().set(false);
                        logger.warn("Status check rejected for worker: {}, reset flag for retry", workerIpPort);
                    }
                } else {
                    logger.info("Skip status check for worker: {}, previous request in progress", workerIpPort);
                }

                // Cache data is generation-scoped scheduling input. Do not populate it
                // before the status channel has validated and published this generation.
                if (!workerStatus.isReady()) {
                    logger.debug("Skip cache check for non-ready worker generation: {}, state={}",
                            workerIpPort, workerStatus.getLifecycleState());
                } else if (workerStatus.getCacheCheckInProgress().compareAndSet(false, true)) {
                    try {
                        logger.debug("Submitting GrpcCacheStatusCheckRunner for worker: {}, site: {}", workerIpPort, site);
                        GrpcCacheStatusCheckRunner grpcCacheStatusCheckRunner
                                = new GrpcCacheStatusCheckRunner(modelName, workerIpPort, site, roleType,
                                workerStatus, cachedWorkerStatuses,
                                engineHealthReporter, engineGrpcService, localKvCacheAwareManager,
                                syncRequestTimeoutMs, syncCount, syncEngineStatusInterval, statusCheckExecutor);
                        statusCheckExecutor.submit(grpcCacheStatusCheckRunner);
                    } catch (RejectedExecutionException e) {
                        workerStatus.getCacheCheckInProgress().set(false);
                        logger.warn("Cache check rejected for worker: {}, reset flag for retry", workerIpPort);
                    }
                } else {
                    logger.info("Skip cache check for worker: {}, previous request in progress", workerIpPort);
                }
            }
            logger.info("Finished submitting status check tasks for model: {}, role: {}, worker count: {}", modelName,
                    roleType, latestEngineWorkerList.size());

        } catch (Exception e) {
            logger.error("sync engine workers status exception, modelName:{}, error:{}", modelName, e.getMessage(), e);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, null);
        } finally {
            logger.debug("Entering finally block for model: {}", modelName);
            int size = workerStatusMap.size();
            logger.debug("Worker status map size: {}", size);

            if (size >= 2) {
                double sumStepLatency = 0.0;
                double sumRunningLoad = 0.0;
                for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
                    WorkerStatus workerStatus = entry.getValue();
                    sumStepLatency += workerStatus.getStepLatencyMs();
                    WorkerEndpoint ep = endpointRegistry != null
                            ? endpointRegistry.get(roleType, entry.getKey()) : null;
                    sumRunningLoad += ep != null ? ep.getLoadMetric() : 0;
                }
                double meanStepLatency = sumStepLatency / size;
                double meanRunningLoad = sumRunningLoad / size;

                // Calculate variance (sample variance using Bessel correction)
                double sumStepLatencyOfSquaredDiffs = 0.0;
                double sumRunningLoadOfSquaredDiffs = 0.0;
                for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
                    WorkerStatus workerStatus = entry.getValue();
                    double diff = workerStatus.getStepLatencyMs() - meanStepLatency;
                    WorkerEndpoint ep = endpointRegistry != null
                            ? endpointRegistry.get(roleType, entry.getKey()) : null;
                    double diff2 = (ep != null ? ep.getLoadMetric() : 0) - meanRunningLoad;
                    sumStepLatencyOfSquaredDiffs += diff * diff;
                    sumRunningLoadOfSquaredDiffs += diff2 * diff2;
                }
                double variance = sumStepLatencyOfSquaredDiffs / (size - 1); // Sample variance
                double variance2 = sumRunningLoadOfSquaredDiffs / (size - 1);

                engineHealthReporter.reportLatencyMetric(modelName, this.roleType.toString(), variance, variance2);
                logger.info("EngineSyncRunner finished for model: {}, role: {}", modelName, roleType);
            } else {
                logger.debug("Less than 2 workers, skipping variance calculation for model: {}", modelName);
            }
        }
    }

    private WorkerStatus getOrCreateWorkerStatus(Map<String, WorkerStatus> workerStatuses, String workerIpPort) {
        WorkerStatus workerStatus = workerStatuses.computeIfAbsent(workerIpPort, ignored -> {
            WorkerStatus created = new WorkerStatus();
            String[] split = workerIpPort.split(":");
            created.setIp(split[0]);
            created.setPort(Integer.parseInt(split[1]));
            created.setGrpcPort(CommonUtils.toGrpcPort(created.getPort()));
            created.setRole(roleType);
            logger.info("Created new WorkerStatus for worker: {}", workerIpPort);
            return created;
        });
        // Publish topology metadata known from discovery before a status probe
        // callback can observe this persistent PROBING object.
        if (workerStatus.getRole() == null) {
            workerStatus.setRole(roleType);
        }
        return workerStatus;
    }

    /**
     * Fence and retire exactly the generation that disappeared through the
     * registry's identity-conditional retirement barrier.
     */
    private void retireWorkerGeneration(Map<String, WorkerStatus> statuses,
                                        String ipPort,
                                        WorkerStatus expected,
                                        long observedDiscoveryTimeUs,
                                        String cause) {
        boolean retirementStarted = false;
        expected.lock.lock();
        try {
            if (statuses.get(ipPort) != expected || !expected.isProbeable()) {
                return;
            }
            long currentDiscoveryTimeUs = expected.getDiscoveryLastSeenTime().get();
            long currentIntervalUs = expected.getDiscoveryUpdateIntervalUs().get();
            long removalThresholdUs = Math.max(3 * currentIntervalUs, 1_000_000L);
            long nowUs = System.nanoTime() / 1000;
            if (currentDiscoveryTimeUs != observedDiscoveryTimeUs
                    || currentDiscoveryTimeUs <= 0L
                    || nowUs - currentDiscoveryTimeUs <= removalThresholdUs
                    || !expected.tryBeginRetirement()) {
                return;
            }
            retirementStarted = true;
        } finally {
            expected.lock.unlock();
        }

        boolean endpointRemoved = false;
        boolean statusRemoved = false;
        try {
            endpointRemoved = endpointRegistry != null
                    && endpointRegistry.retire(roleType, ipPort, expected,
                    EndpointRetireCause.DISCOVERY_REMOVED);
        } finally {
            statusRemoved = statuses.remove(ipPort, expected);
            if (retirementStarted) {
                expected.markClosed();
            }
        }
        logger.info("[retire] engine generation, model={}, role={}, ipPort={}, cause={}, statusRemoved={}, endpointRemoved={}",
                modelName, roleType, ipPort, cause, statusRemoved, endpointRemoved);
    }
}

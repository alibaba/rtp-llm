package org.flexlb.sync.runner;

import org.flexlb.balance.dp.InflightBatchRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.CollectionUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
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

    private final InflightBatchRegistry inflightRegistry;

    private final long syncRequestTimeoutMs;

    private final LongAdder syncCount;

    private final Long syncEngineStatusInterval;

    public EngineSyncRunner(String modelName,
                            Map<String, WorkerStatus> workerStatusMap,
                            WorkerAddressService workerAddressService,
                            ExecutorService statusCheckExecutor,
                            EngineHealthReporter engineHealthReporter,
                            EngineGrpcService engineGrpcService,
                            RoleType roleType,
                            CacheAwareService localKvCacheAwareManager,
                            InflightBatchRegistry inflightRegistry,
                            long syncRequestTimeoutMs,
                            LongAdder syncCount,
                            Long syncEngineStatusInterval) {

        this.modelName = modelName;
        this.workerAddressService = workerAddressService;
        this.workerStatusMap = workerStatusMap;
        this.statusCheckExecutor = statusCheckExecutor;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.roleType = roleType;
        this.localKvCacheAwareManager = localKvCacheAwareManager;
        this.inflightRegistry = inflightRegistry;
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
        this.syncCount = syncCount;
        this.syncEngineStatusInterval = syncEngineStatusInterval;
    }

    @Override
    public void run() {
        logger.info("EngineSyncRunner start for model: {}, role: {}", modelName, roleType.toString());
        try {
            long startTimeInUs = System.nanoTime() / 1000;
            WorkerAddressService.EngineWorkerList engineWorkerList = workerAddressService.getEngineWorkers(modelName, roleType);
            if (engineWorkerList == null) {
                logger.warn("workerAddressService getEngineWorkers returned null, model: {}, role: {}", modelName, roleType);
                engineWorkerList = new WorkerAddressService.EngineWorkerList(List.of(), Set.of());
            }
            List<WorkerHost> discoveredEngineWorkerList = engineWorkerList.getWorkerHosts();
            if (discoveredEngineWorkerList == null) {
                discoveredEngineWorkerList = List.of();
            }
            long nowUs = System.nanoTime() / 1000;
            markUnavailableGroupWorkers(workerStatusMap, engineWorkerList.getUnavailableGroups(), nowUs);
            Set<String> discoveryFailedGroups = engineWorkerList.getDiscoveryFailedGroups();
            List<WorkerHost> latestEngineWorkerList = mergeDiscoveryFailedCachedWorkers(
                    discoveredEngineWorkerList, workerStatusMap, discoveryFailedGroups);
            logger.info("workerAddressService getEngineWorkerList, model: {}, role: {}, discoveredSize: {}, effectiveSize: {}",
                    modelName, roleType, discoveredEngineWorkerList.size(), latestEngineWorkerList.size());
            engineHealthReporter.reportServiceDiscoveryResult(modelName, discoveredEngineWorkerList.size(), roleType.toString());
            if (CollectionUtils.isEmpty(latestEngineWorkerList)) {
                logger.error("get engine worker list is empty, cost={}μs, model={}", System.nanoTime() / 1000 - startTimeInUs, modelName);
                return;
            }
            Map<String/*ip*/, WorkerStatus> cachedWorkerStatuses = workerStatusMap;
            // Log if latest worker count differs from cached worker count
            if (cachedWorkerStatuses.size() != latestEngineWorkerList.size()) {
                logger.info("[update] engine ip changes, model={}, role={}, before={}, after={}",
                        modelName, roleType, cachedWorkerStatuses.size(), latestEngineWorkerList.size());
            }

            // Remove if not in latest engine list
            Set<String> latestValidIpPorts = latestEngineWorkerList.stream()
                    .map(WorkerHost::getIpPort)
                    .collect(Collectors.toSet());
            logger.info("Current cached worker size: {}, latest worker list size: {}", cachedWorkerStatuses.size(), latestEngineWorkerList.size());
            for (Map.Entry<String, WorkerStatus> entry: cachedWorkerStatuses.entrySet()) {
                WorkerStatus workerStatus = entry.getValue();
                String ipPort = entry.getKey();
                if (!latestValidIpPorts.contains(ipPort)) {
                    if (isDiscoveryFailedGroup(workerStatus, discoveryFailedGroups)) {
                        logger.warn("[keep] engine discovery failed, keep cached worker, model={}, role={}, group={}, ipPort={}",
                                modelName, roleType, normalizeGroupName(workerStatus), ipPort);
                        continue;
                    }
                    if (shouldRemoveWorker(workerStatus, nowUs)) {
                        cachedWorkerStatuses.remove(ipPort);
                        logger.info("[remove] engine ip changes, model={}, role={}, ipPort={}", modelName, roleType, ipPort);
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
                workerStatus.setSite(site);
                workerStatus.setGroup(host.getGroup());

                if (workerStatus.getStatusCheckInProgress().compareAndSet(false, true)) {
                    logger.debug("Submitting GrpcWorkerStatusRunner for worker: {}, site: {}", workerIpPort, site);
                    GrpcWorkerStatusRunner grpcWorkerStatusRunner
                            = new GrpcWorkerStatusRunner(modelName, workerIpPort, site, roleType, host.getGroup(),
                            workerStatus, engineHealthReporter, engineGrpcService,
                            inflightRegistry, syncRequestTimeoutMs);
                    statusCheckExecutor.submit(grpcWorkerStatusRunner);
                } else {
                    logger.info("Skip status check for worker: {}, previous request in progress", workerIpPort);
                }

                if (workerStatus.getCacheCheckInProgress().compareAndSet(false, true)) {
                    logger.debug("Submitting GrpcCacheStatusCheckRunner for worker: {}, site: {}", workerIpPort, site);
                    GrpcCacheStatusCheckRunner grpcCacheStatusCheckRunner
                            = new GrpcCacheStatusCheckRunner(modelName, workerIpPort, site, roleType,
                            workerStatus, engineHealthReporter, engineGrpcService, localKvCacheAwareManager,
                            syncRequestTimeoutMs, syncCount, syncEngineStatusInterval);
                    statusCheckExecutor.submit(grpcCacheStatusCheckRunner);
                } else {
                    logger.info("Skip cache check for worker: {}, previous request in progress", workerIpPort);
                }
            }
            logger.info("Finished submitting status check tasks for model: {}, role: {}, worker count: {}", modelName,
                    roleType, latestEngineWorkerList.size());

        } catch (Exception e) {
            logger.error("sync engine workers status exception, modelName:{}, error:{}", modelName, e.getMessage(), e);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, null, null);
        } finally {
            logger.debug("Entering finally block for model: {}", modelName);
            int size = workerStatusMap.size();
            logger.debug("Worker status map size: {}", size);

            if (size >= 2) {
                double sumStepLatency = 0.0;
                double sumRunningQueryTime = 0.0;
                for (WorkerStatus workerStatus : workerStatusMap.values()) {
                    sumStepLatency += workerStatus.getStepLatencyMs();
                    sumRunningQueryTime += workerStatus.getRunningQueueTime().get();
                }
                double meanStepLatency = sumStepLatency / size;
                double meanRunningQueryLen = sumRunningQueryTime / size;

                // Calculate variance (sample variance using Bessel correction)
                double sumStepLatencyOfSquaredDiffs = 0.0;
                double sumRunningQueryLenOfSquaredDiffs = 0.0;
                for (WorkerStatus workerStatus : workerStatusMap.values()) {
                    double diff = workerStatus.getStepLatencyMs() - meanStepLatency;
                    double diff2 = workerStatus.getRunningQueueTime().get() - meanRunningQueryLen;
                    sumStepLatencyOfSquaredDiffs += diff * diff;
                    sumRunningQueryLenOfSquaredDiffs += diff2 * diff2;
                }
                double variance = sumStepLatencyOfSquaredDiffs / (size - 1); // Sample variance
                double variance2 = sumRunningQueryLenOfSquaredDiffs / (size - 1);

                engineHealthReporter.reportLatencyMetric(modelName, this.roleType.toString(), variance, variance2);
                logger.info("EngineSyncRunner finished for model: {}, role: {}", modelName, roleType);
            } else {
                logger.debug("Less than 2 workers, skipping variance calculation for model: {}", modelName);
            }
        }
    }

    private List<WorkerHost> mergeDiscoveryFailedCachedWorkers(List<WorkerHost> discoveredWorkerList,
                                                               Map<String, WorkerStatus> cachedWorkerStatuses,
                                                               Set<String> discoveryFailedGroups) {
        if (CollectionUtils.isEmpty(cachedWorkerStatuses) || CollectionUtils.isEmpty(discoveryFailedGroups)) {
            return discoveredWorkerList;
        }

        List<WorkerHost> mergedWorkerList = new ArrayList<>(
                discoveredWorkerList == null ? List.of() : discoveredWorkerList);
        Set<String> mergedIpPorts = mergedWorkerList.stream()
                .map(WorkerHost::getIpPort)
                .collect(Collectors.toSet());
        for (Map.Entry<String, WorkerStatus> entry : cachedWorkerStatuses.entrySet()) {
            WorkerStatus workerStatus = entry.getValue();
            if (!isDiscoveryFailedGroup(workerStatus, discoveryFailedGroups)) {
                continue;
            }
            WorkerHost cachedHost = cachedWorkerHost(entry.getKey(), workerStatus);
            if (cachedHost == null || !mergedIpPorts.add(cachedHost.getIpPort())) {
                continue;
            }
            mergedWorkerList.add(cachedHost);
            logger.warn("[keep] engine discovery failed, refresh cached worker, model={}, role={}, group={}, ipPort={}",
                    modelName, roleType, cachedHost.getGroup(), cachedHost.getIpPort());
        }
        return mergedWorkerList;
    }

    private WorkerHost cachedWorkerHost(String ipPort, WorkerStatus workerStatus) {
        if (workerStatus == null) {
            return null;
        }
        String ip = workerStatus.getIp();
        int port = workerStatus.getPort();
        if ((ip == null || ip.isEmpty() || port <= 0) && ipPort != null) {
            String[] split = ipPort.split(":");
            if (split.length == 2) {
                ip = split[0];
                try {
                    port = Integer.parseInt(split[1]);
                } catch (NumberFormatException e) {
                    logger.warn("invalid cached worker ipPort={}, model={}, role={}", ipPort, modelName, roleType);
                    return null;
                }
            }
        }
        if (ip == null || ip.isEmpty() || port <= 0) {
            logger.warn("skip cached worker with invalid address, model={}, role={}, ipPort={}, worker={}",
                    modelName, roleType, ipPort, workerStatus);
            return null;
        }
        return new WorkerHost(
                ip,
                port,
                port + 1,
                port + 5,
                workerStatus.getSite(),
                normalizeGroupName(workerStatus));
    }

    private WorkerStatus getOrCreateWorkerStatus(Map<String, WorkerStatus> workerStatuses, String workerIpPort) {
        WorkerStatus workerStatus = workerStatuses.get(workerIpPort);
        if (workerStatus == null) {
            workerStatus = new WorkerStatus();
            String[] split = workerIpPort.split(":");
            workerStatus.setIp(split[0]);
            workerStatus.setPort(Integer.parseInt(split[1]));
            workerStatuses.put(workerIpPort, workerStatus);
            logger.info("Created new WorkerStatus for worker: {}", workerIpPort);
        }
        return workerStatus;
    }

    private void markUnavailableGroupWorkers(Map<String, WorkerStatus> cachedWorkerStatuses, Set<String> unavailableGroups,
                                             long nowUs) {
        if (CollectionUtils.isEmpty(cachedWorkerStatuses) || CollectionUtils.isEmpty(unavailableGroups)) {
            return;
        }
        List<String> ipPortsToRemove = new ArrayList<>();
        for (Map.Entry<String, WorkerStatus> entry : cachedWorkerStatuses.entrySet()) {
            WorkerStatus workerStatus = entry.getValue();
            if (workerStatus == null) {
                continue;
            }
            String group = workerStatus.getGroup() == null ? "" : workerStatus.getGroup();
            if (unavailableGroups.contains(group)) {
                workerStatus.setAlive(false);
                workerStatus.getResourceAvailable().set(false);
                if (shouldRemoveWorker(workerStatus, nowUs)) {
                    ipPortsToRemove.add(entry.getKey());
                } else {
                    logger.warn("[mark unavailable] engine group unavailable, model={}, role={}, group={}, ipPort={}",
                            modelName, roleType, group, entry.getKey());
                }
            }
        }
        for (String ipPort : ipPortsToRemove) {
            WorkerStatus workerStatus = cachedWorkerStatuses.remove(ipPort);
            if (workerStatus != null) {
                String group = workerStatus.getGroup() == null ? "" : workerStatus.getGroup();
                logger.warn("[remove] engine group unavailable, model={}, role={}, group={}, ipPort={}",
                        modelName, roleType, group, ipPort);
            }
        }
    }

    private boolean shouldRemoveWorker(WorkerStatus workerStatus, long nowUs) {
        if (workerStatus == null) {
            return false;
        }
        long lastTime = workerStatus.getStatusLastUpdateTime().get();
        long removalBaseTimeUs = lastTime;
        if (removalBaseTimeUs <= 0) {
            removalBaseTimeUs = workerStatus.getStatusFirstSeenTimeUs().get();
            if (removalBaseTimeUs <= 0) {
                return true;
            }
        }
        long actualIntervalUs = workerStatus.getStatusUpdateIntervalUs().get();
        // Use max(3 * actual sync interval, 1s) as removal threshold to tolerate transient service discovery flaps.
        // Workers that never completed a status sync fall back to first-seen time,
        // so a bad host cannot stay in the cache forever after discovery removes it.
        long removalThresholdUs = Math.max(3 * Math.max(actualIntervalUs, 0L), 1_000_000L);
        return nowUs - removalBaseTimeUs > removalThresholdUs;
    }

    private boolean isDiscoveryFailedGroup(WorkerStatus workerStatus, Set<String> discoveryFailedGroups) {
        if (workerStatus == null || CollectionUtils.isEmpty(discoveryFailedGroups)) {
            return false;
        }
        return discoveryFailedGroups.contains(normalizeGroupName(workerStatus));
    }

    private String normalizeGroupName(WorkerStatus workerStatus) {
        return workerStatus.getGroup() == null ? "" : workerStatus.getGroup();
    }
}

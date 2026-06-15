package org.flexlb.sync.runner;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.EngineType;
import org.flexlb.exception.ServiceDiscoveryException;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.CollectionUtils;

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

    private final long syncRequestTimeoutMs;

    private final LongAdder syncCount;

    private final Long syncEngineStatusInterval;

    private final EngineType engineType;

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
                            EngineType engineType) {

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
        this.engineType = engineType;
    }

    @Override
    public void run() {
        logger.info("EngineSyncRunner start for model: {}, role: {}", modelName, roleType.toString());
        try {
            long startTimeInUs = System.nanoTime() / 1000;
            List<WorkerHost> latestEngineWorkerList = workerAddressService.getEngineWorkerList(modelName, roleType);
            logger.info("workerAddressService getEngineWorkerList, model: {}, role: {}, size: {}", modelName, roleType, latestEngineWorkerList.size());
            engineHealthReporter.reportServiceDiscoveryResult(modelName, latestEngineWorkerList.size(), roleType.toString());
            Set<String> latestValidIpPorts = latestEngineWorkerList.stream()
                    .map(WorkerHost::getIpPort)
                    .collect(Collectors.toSet());
            // Discovery presence is the only liveness signal for embedding engines (no gRPC
            // probe), so a worker missing from the list stops being routable immediately. This
            // runs before the empty-list short-circuit below: an empty list is exactly the case
            // where every previously-alive embedding worker has to be marked dead. The contract
            // holds because getEngineWorkerList throws on discovery failure, so an empty list
            // here always means a genuinely empty fleet.
            if (engineType == EngineType.EMBEDDING) {
                markDeadFromDiscovery(latestValidIpPorts);
            }
            if (CollectionUtils.isEmpty(latestEngineWorkerList)) {
                logger.error("get engine worker list is empty, cost={}μs, model={}", System.nanoTime() / 1000 - startTimeInUs, modelName);
                return;
            }
            if (workerStatusMap.size() != latestEngineWorkerList.size()) {
                logger.info("[update] engine ip changes, model={}, role={}, before={}, after={}",
                        modelName, roleType, workerStatusMap.size(), latestEngineWorkerList.size());
            }
            removeStaleWorkers(latestValidIpPorts);

            logger.info("Submitting status check tasks for {} workers", latestEngineWorkerList.size());
            for (WorkerHost host : latestEngineWorkerList) {
                submitStatusChecks(host);
            }
            logger.info("Finished submitting status check tasks for model: {}, role: {}, worker count: {}", modelName,
                    roleType, latestEngineWorkerList.size());

        } catch (ServiceDiscoveryException e) {
            // Already reported by WorkerAddressService; skip the round and keep the
            // previous worker state until discovery recovers.
            logger.error("service discovery failed, keeping previous worker state, model={}, role={}, error:{}",
                    modelName, roleType, e.getMessage());
        } catch (Exception e) {
            logger.error("sync engine workers status exception, modelName:{}, error:{}", modelName, e.getMessage(), e);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, null, null);
        } finally {
            reportLatencyVariance();
        }
    }

    /**
     * Mark workers that vanished from the latest discovery list as not-routable. For embedding
     * engines discovery presence is the only liveness signal, so this is how a dropped worker
     * stops receiving traffic; physical removal is handled separately by {@link #removeStaleWorkers}.
     */
    private void markDeadFromDiscovery(Set<String> latestValidIpPorts) {
        for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
            if (!latestValidIpPorts.contains(entry.getKey()) && entry.getValue().isAlive()) {
                entry.getValue().setAlive(false);
                logger.info("[dead] embedding worker dropped by discovery, model={}, role={}, ipPort={}",
                        modelName, roleType, entry.getKey());
            }
        }
    }

    /**
     * Drop workers no longer in the discovery list once they have been gone past the removal
     * threshold (max(3 * actual sync interval, 1s)), tolerating transient discovery flaps.
     */
    private void removeStaleWorkers(Set<String> latestValidIpPorts) {
        for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
            String ipPort = entry.getKey();
            if (latestValidIpPorts.contains(ipPort)) {
                continue;
            }
            WorkerStatus workerStatus = entry.getValue();
            long lastTime = workerStatus.getStatusLastUpdateTime().get();
            long actualIntervalUs = workerStatus.getStatusUpdateIntervalUs().get();
            long removalThresholdUs = Math.max(3 * actualIntervalUs, 1_000_000L);
            if (System.nanoTime() / 1000 - lastTime > removalThresholdUs) {
                workerStatusMap.remove(ipPort);
                logger.info("[remove] engine ip changes, model={}, role={}, ipPort={}", modelName, roleType, ipPort);
            }
        }
    }

    private void submitStatusChecks(WorkerHost host) {
        String workerIpPort = host.getIpPort();
        String site = host.getSite();
        WorkerStatus workerStatus = getOrCreateWorkerStatus(workerStatusMap, workerIpPort);

        if (engineType == EngineType.EMBEDDING) {
            markAliveFromDiscovery(workerStatus, host);
            return;
        }

        if (workerStatus.getStatusCheckInProgress().compareAndSet(false, true)) {
            logger.debug("Submitting GrpcWorkerStatusRunner for worker: {}, site: {}", workerIpPort, site);
            GrpcWorkerStatusRunner grpcWorkerStatusRunner
                    = new GrpcWorkerStatusRunner(modelName, workerIpPort, site, roleType, host.getGroup(),
                    workerStatus, engineHealthReporter, engineGrpcService,
                    syncRequestTimeoutMs);
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

    private void reportLatencyVariance() {
        int size = workerStatusMap.size();
        if (size < 2 || engineType == EngineType.EMBEDDING) {
            logger.debug("Less than 2 workers, skipping variance calculation for model: {}", modelName);
            return;
        }
        double sumStepLatency = 0.0;
        double sumRunningQueryTime = 0.0;
        for (WorkerStatus workerStatus : workerStatusMap.values()) {
            sumStepLatency += workerStatus.getStepLatencyMs();
            sumRunningQueryTime += workerStatus.getRunningQueueTime().get();
        }
        double meanStepLatency = sumStepLatency / size;
        double meanRunningQueryLen = sumRunningQueryTime / size;

        // Sample variance (Bessel correction)
        double sumStepLatencyOfSquaredDiffs = 0.0;
        double sumRunningQueryLenOfSquaredDiffs = 0.0;
        for (WorkerStatus workerStatus : workerStatusMap.values()) {
            double diff = workerStatus.getStepLatencyMs() - meanStepLatency;
            double diff2 = workerStatus.getRunningQueueTime().get() - meanRunningQueryLen;
            sumStepLatencyOfSquaredDiffs += diff * diff;
            sumRunningQueryLenOfSquaredDiffs += diff2 * diff2;
        }
        double variance = sumStepLatencyOfSquaredDiffs / (size - 1);
        double variance2 = sumRunningQueryLenOfSquaredDiffs / (size - 1);

        engineHealthReporter.reportLatencyMetric(modelName, this.roleType.toString(), variance, variance2);
        logger.info("EngineSyncRunner finished for model: {}, role: {}", modelName, roleType);
    }

    /**
     * EMBEDDING engines expose no {@code GetWorkerStatus}, so the host appearing in the
     * service-discovery list is the liveness signal: register it alive without probing.
     * Engine-level liveness degrades to the discovery service's own health check; no load
     * metrics are collected (callers are limited to load-unaware strategies, enforced at
     * boot). Refreshing statusLastUpdateTime keeps the stale-worker removal above working
     * unchanged when discovery drops a host.
     */
    private void markAliveFromDiscovery(WorkerStatus workerStatus, WorkerHost host) {
        workerStatus.setSite(host.getSite());
        workerStatus.setGroup(host.getGroup());
        workerStatus.setRole(roleType.getCode());
        workerStatus.setAlive(true);
        long nowUs = System.nanoTime() / 1000;
        long prevUpdateTime = workerStatus.getStatusLastUpdateTime().get();
        if (prevUpdateTime > 0) {
            workerStatus.getStatusUpdateIntervalUs().set(nowUs - prevUpdateTime);
        }
        workerStatus.getStatusLastUpdateTime().set(nowUs);
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
}

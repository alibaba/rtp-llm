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
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

public class EngineSyncRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    /**
     * Last successful discovery time per {@code model/role}. The runner is {@code new}-ed every
     * round, so this map is owned by the long-lived {@link org.flexlb.sync.synchronizer.MasterEngineSynchronizer}
     * and passed in — the same channel it already uses for {@link #workerStatusMap}, keeping the
     * grace clock out of process-global static state.
     */
    private final Map<String, Long> lastDiscoverySuccessUs;

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

    /**
     * Upper bound on how long a discovery outage may keep already-known workers routable. Past
     * this window known workers are allowed to age out normally. Comes from
     * {@code FlexlbConfig#discoveryFailureGraceMs} (env override
     * {@code DISCOVERY_FAILURE_GRACE_MS}); default 5 minutes.
     */
    private final long discoveryFailureGraceUs;

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
                            EngineType engineType,
                            long discoveryFailureGraceMs,
                            Map<String, Long> lastDiscoverySuccessUs) {

        this.discoveryFailureGraceUs = discoveryFailureGraceMs * 1000L;
        this.lastDiscoverySuccessUs = lastDiscoverySuccessUs;
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

            // "No hosts" and "the lookup did not work" are the same observation from here: a
            // discovery client that swallows a failed lookup reports it as an empty list, exactly
            // like a fleet that scaled to zero. Wiping a fleet is not recoverable within a sync
            // round, so an empty result never overwrites non-empty known state — it rides the same
            // grace window as an outright discovery failure, past which the workers age out
            // normally. A fleet that is already empty has nothing to protect and falls through.
            if (CollectionUtils.isEmpty(latestEngineWorkerList) && !workerStatusMap.isEmpty()) {
                rideOutDiscoveryGap("empty worker list while " + workerStatusMap.size() + " workers are known");
                return;
            }
            lastDiscoverySuccessUs.put(discoveryKey(), System.nanoTime() / 1000);

            Set<String> latestValidIpPorts = latestEngineWorkerList.stream()
                    .map(WorkerHost::getIpPort)
                    .collect(Collectors.toSet());
            // Discovery presence is the only liveness signal for embedding engines (no gRPC
            // probe), so a worker missing from the list stops being routable immediately.
            if (engineType == EngineType.EMBEDDING) {
                markDeadFromDiscovery(latestValidIpPorts);
            }
            removeStaleWorkers(latestValidIpPorts);
            if (CollectionUtils.isEmpty(latestEngineWorkerList)) {
                logger.error("get engine worker list is empty, cost={}μs, model={}", System.nanoTime() / 1000 - startTimeInUs, modelName);
                return;
            }
            if (workerStatusMap.size() != latestEngineWorkerList.size()) {
                logger.info("[update] engine ip changes, model={}, role={}, before={}, after={}",
                        modelName, roleType, workerStatusMap.size(), latestEngineWorkerList.size());
            }

            logger.info("Submitting status check tasks for {} workers", latestEngineWorkerList.size());
            for (WorkerHost host : latestEngineWorkerList) {
                try {
                    submitStatusChecks(host);
                } catch (Exception e) {
                    logger.error("skip worker with submit failure, model={}, role={}, ipPort={}, error:{}",
                            modelName, roleType, host.getIpPort(), e.getMessage());
                }
            }
            logger.info("Finished submitting status check tasks for model: {}, role: {}, worker count: {}", modelName,
                    roleType, latestEngineWorkerList.size());

        } catch (ServiceDiscoveryException e) {
            rideOutDiscoveryGap(e.getMessage());
        } catch (Exception e) {
            logger.error("sync engine workers status exception, modelName:{}, error:{}", modelName, e.getMessage(), e);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, null, null);
        } finally {
            reportLatencyVariance();
        }
    }

    /**
     * Discovery could not tell us who is alive — it either threw or handed back an empty list for a
     * fleet we know is not empty. Refresh the staleness clock of the known workers so they ride out
     * a transient outage, bounded by {@code discoveryFailureGraceUs}; past the window they age out
     * normally.
     */
    private void rideOutDiscoveryGap(String reason) {
        long nowUs = System.nanoTime() / 1000;
        Long lastSuccessUs = lastDiscoverySuccessUs.get(discoveryKey());
        boolean withinGrace = lastSuccessUs != null && nowUs - lastSuccessUs <= discoveryFailureGraceUs;
        if (withinGrace) {
            for (WorkerStatus workerStatus : workerStatusMap.values()) {
                workerStatus.getStatusLastUpdateTime().set(nowUs);
            }
            logger.error("service discovery unusable, keeping previous worker state within grace, model={}, role={}, reason:{}",
                    modelName, roleType, reason);
        } else {
            logger.error("service discovery unusable beyond grace ({}ms), letting workers age out, model={}, role={}, reason:{}",
                    discoveryFailureGraceUs / 1000, modelName, roleType, reason);
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
            submitOrReset(grpcWorkerStatusRunner, workerStatus.getStatusCheckInProgress(), workerIpPort, "status");
        } else {
            logger.info("Skip status check for worker: {}, previous request in progress", workerIpPort);
        }

        if (workerStatus.getCacheCheckInProgress().compareAndSet(false, true)) {
            logger.debug("Submitting GrpcCacheStatusCheckRunner for worker: {}, site: {}", workerIpPort, site);
            GrpcCacheStatusCheckRunner grpcCacheStatusCheckRunner
                    = new GrpcCacheStatusCheckRunner(modelName, workerIpPort, site, roleType,
                    workerStatus, engineHealthReporter, engineGrpcService, localKvCacheAwareManager,
                    syncRequestTimeoutMs, syncCount, syncEngineStatusInterval);
            submitOrReset(grpcCacheStatusCheckRunner, workerStatus.getCacheCheckInProgress(), workerIpPort, "cache");
        } else {
            logger.info("Skip cache check for worker: {}, previous request in progress", workerIpPort);
        }
    }

    /**
     * Submits a probe and, if the bounded executor rejects it (queue full or shutting down),
     * resets the in-progress flag the runner would have cleared on completion. Without this the
     * flag stays set forever and the worker is never probed again.
     */
    private void submitOrReset(Runnable runner, AtomicBoolean inProgress, String workerIpPort, String kind) {
        try {
            statusCheckExecutor.submit(runner);
        } catch (RejectedExecutionException e) {
            inProgress.set(false);
            logger.warn("status executor rejected {} check for worker: {}, skipping this round", kind, workerIpPort);
        }
    }

    private void reportLatencyVariance() {
        int size = workerStatusMap.size();
        if (engineType == EngineType.EMBEDDING) {
            // Embedding workers are never probed, so latency fields stay 0 — variance is noise.
            return;
        }
        if (size < 2) {
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
        return workerStatuses.computeIfAbsent(workerIpPort, key -> {
            WorkerStatus workerStatus = new WorkerStatus();
            String[] split = key.split(":");
            workerStatus.setIp(split[0]);
            workerStatus.setPort(Integer.parseInt(split[1]));
            logger.info("Created new WorkerStatus for worker: {}", key);
            return workerStatus;
        });
    }

    private String discoveryKey() {
        return modelName + "/" + roleType;
    }
}

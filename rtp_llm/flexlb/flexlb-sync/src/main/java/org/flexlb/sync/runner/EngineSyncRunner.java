package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
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
import org.flexlb.util.CommonUtils;
import org.flexlb.util.RateLimitedWarn;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.CollectionUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

public class EngineSyncRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String modelName;
    private final Map<String, WorkerStatus> workerStatusMap;
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
    private final EngineType engineType;
    private final long discoveryFailureGraceUs;
    private final Map<String, Long> lastDiscoverySuccessUs;
    private final RateLimitedWarn discoveryGapWarn;

    public EngineSyncRunner(
            String modelName,
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
            EndpointRegistry endpointRegistry,
            EngineType engineType,
            long discoveryFailureGraceMs,
            Map<String, Long> lastDiscoverySuccessUs,
            RateLimitedWarn discoveryGapWarn) {
        this.modelName = modelName;
        this.workerStatusMap = workerStatusMap;
        this.workerAddressService = workerAddressService;
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
        this.engineType = engineType;
        long effectiveGraceMs = discoveryFailureGraceMs > 0
                ? discoveryFailureGraceMs
                : 300_000L;
        this.discoveryFailureGraceUs =
                TimeUnit.MILLISECONDS.toMicros(effectiveGraceMs);
        this.lastDiscoverySuccessUs = lastDiscoverySuccessUs;
        this.discoveryGapWarn = discoveryGapWarn;
    }

    @Override
    public void run() {
        logger.debug("EngineSyncRunner start for model: {}, role: {}", modelName, roleType);
        try {
            long startTimeUs = System.nanoTime() / 1000;
            List<WorkerHost> latestWorkers =
                    workerAddressService.getEngineWorkerList(modelName, roleType);
            logger.debug(
                    "workerAddressService result, model: {}, role: {}, size: {}",
                    modelName, roleType, latestWorkers.size());
            engineHealthReporter.reportServiceDiscoveryResult(
                    modelName, latestWorkers.size(), roleType.toString());

            if (CollectionUtils.isEmpty(latestWorkers) && !workerStatusMap.isEmpty()) {
                rideOutDiscoveryGap(
                        "empty worker list while " + workerStatusMap.size()
                                + " workers are known");
                return;
            }
            if (!CollectionUtils.isEmpty(latestWorkers)) {
                lastDiscoverySuccessUs.put(discoveryKey(), System.nanoTime() / 1000);
            }

            Set<String> latestIpPorts = latestWorkers.stream()
                    .map(WorkerHost::getIpPort)
                    .collect(Collectors.toSet());
            if (engineType == EngineType.EMBEDDING) {
                markDeadFromDiscovery(latestIpPorts);
            }
            removeStaleWorkers(latestIpPorts);

            if (CollectionUtils.isEmpty(latestWorkers)) {
                logger.debug(
                        "empty worker list, cost={}us, model={}, role={}",
                        System.nanoTime() / 1000 - startTimeUs,
                        modelName,
                        roleType);
                return;
            }
            if (workerStatusMap.size() != latestWorkers.size()) {
                logger.info(
                        "[update] engine ip changes, model={}, role={}, cached={}, discovered={}",
                        modelName, roleType, workerStatusMap.size(), latestWorkers.size());
            }

            for (WorkerHost host : latestWorkers) {
                try {
                    submitStatusChecks(host);
                } catch (RuntimeException error) {
                    logger.error(
                            "skip worker with submit failure, model={}, role={}, ipPort={}, error:{}",
                            modelName, roleType, host.getIpPort(), error.getMessage(), error);
                }
            }
        } catch (ServiceDiscoveryException error) {
            rideOutDiscoveryGap(error.getMessage());
        } catch (Exception error) {
            logger.error(
                    "sync engine workers status exception, modelName:{}, error:{}",
                    modelName, error.getMessage(), error);
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.UNKNOWN_ERROR, roleType);
        } finally {
            reportLatencyVariance();
        }
    }

    private void rideOutDiscoveryGap(String reason) {
        long nowUs = System.nanoTime() / 1000;
        Long lastSuccessUs = lastDiscoverySuccessUs.get(discoveryKey());
        boolean withinGrace = lastSuccessUs != null
                && nowUs - lastSuccessUs <= discoveryFailureGraceUs;
        if (!withinGrace) {
            discoveryGapWarn.warn(
                    "service discovery unusable beyond grace ({}ms), letting workers age out, "
                            + "model={}, role={}, reason:{}",
                    TimeUnit.MICROSECONDS.toMillis(discoveryFailureGraceUs),
                    modelName,
                    roleType,
                    reason);
            return;
        }

        if (engineType == EngineType.EMBEDDING) {
            for (WorkerStatus status : workerStatusMap.values()) {
                if (status.isAlive()) {
                    status.getStatusLastUpdateTime().set(nowUs);
                }
            }
        } else {
            for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
                WorkerStatus status = entry.getValue();
                try {
                    submitProbes(
                            entry.getKey(), status.getSite(), status.getGroup(), status);
                } catch (RuntimeException error) {
                    logger.error(
                            "probe submit failed during discovery gap, model={}, role={}, "
                                    + "ipPort={}, error:{}",
                            modelName, roleType, entry.getKey(), error.getMessage(), error);
                }
            }
        }
        discoveryGapWarn.warn(
                "service discovery unusable, keeping previous worker state within grace, "
                        + "model={}, role={}, reason:{}",
                modelName,
                roleType,
                reason);
    }

    private void markDeadFromDiscovery(Set<String> latestIpPorts) {
        for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
            WorkerStatus status = entry.getValue();
            if (!latestIpPorts.contains(entry.getKey()) && status.isAlive()) {
                status.setAlive(false);
                logger.info(
                        "[dead] embedding worker dropped by discovery, model={}, role={}, ipPort={}",
                        modelName, roleType, entry.getKey());
            }
        }
    }

    private void removeStaleWorkers(Set<String> latestIpPorts) {
        long nowUs = System.nanoTime() / 1000;
        for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
            String ipPort = entry.getKey();
            if (latestIpPorts.contains(ipPort)) {
                continue;
            }
            WorkerStatus status = entry.getValue();
            long removalThresholdUs = Math.max(
                    3 * status.getStatusUpdateIntervalUs().get(), 1_000_000L);
            if (nowUs - status.getStatusLastUpdateTime().get() <= removalThresholdUs) {
                continue;
            }
            status.setAlive(false);
            boolean statusRemoved = workerStatusMap.remove(ipPort, status);
            boolean endpointRemoved = endpointRegistry != null
                    && endpointRegistry.remove(roleType, ipPort, status);
            logger.info(
                    "[remove] engine ip changes, model={}, role={}, ipPort={}, "
                            + "statusRemoved={}, endpointRemoved={}",
                    modelName, roleType, ipPort, statusRemoved, endpointRemoved);
        }
    }

    private void submitStatusChecks(WorkerHost host) {
        String ipPort = host.getIpPort();
        WorkerStatus status = getOrCreateWorkerStatus(ipPort);
        status.setSite(host.getSite());
        status.setGroup(host.getGroup());

        if (engineType == EngineType.EMBEDDING) {
            markAliveFromDiscovery(status);
            ensureEndpoint(ipPort, status);
            return;
        }

        if (!status.isAlive()) {
            status.setAlive(true);
        }
        ensureEndpoint(ipPort, status);
        submitProbes(ipPort, host.getSite(), host.getGroup(), status);
    }

    private void submitProbes(
            String ipPort, String site, String group, WorkerStatus status) {
        if (status.getStatusCheckInProgress().compareAndSet(false, true)) {
            GrpcWorkerStatusRunner statusRunner = new GrpcWorkerStatusRunner(
                    modelName,
                    ipPort,
                    site,
                    roleType,
                    group,
                    status,
                    workerStatusMap,
                    engineHealthReporter,
                    engineGrpcService,
                    syncRequestTimeoutMs,
                    batchScheduler,
                    endpointRegistry,
                    statusCheckExecutor);
            submitOrReset(
                    statusRunner, status.getStatusCheckInProgress(), ipPort, "status");
        }

        if (roleType != RoleType.VIT
                && status.getCacheCheckInProgress().compareAndSet(false, true)) {
            GrpcCacheStatusCheckRunner cacheRunner = new GrpcCacheStatusCheckRunner(
                    modelName,
                    ipPort,
                    site,
                    roleType,
                    status,
                    engineHealthReporter,
                    engineGrpcService,
                    localKvCacheAwareManager,
                    syncRequestTimeoutMs,
                    syncCount,
                    syncEngineStatusInterval,
                    statusCheckExecutor);
            submitOrReset(
                    cacheRunner, status.getCacheCheckInProgress(), ipPort, "cache");
        }
    }

    private void submitOrReset(
            Runnable runner,
            AtomicBoolean inProgress,
            String ipPort,
            String kind) {
        try {
            statusCheckExecutor.submit(runner);
        } catch (RejectedExecutionException error) {
            inProgress.set(false);
            logger.warn(
                    "status executor rejected {} check for worker: {}; retrying next round",
                    kind, ipPort);
        } catch (RuntimeException error) {
            inProgress.set(false);
            throw error;
        }
    }

    private void reportLatencyVariance() {
        if (engineType == EngineType.EMBEDDING) {
            return;
        }

        List<double[]> samples = new ArrayList<>();
        for (Map.Entry<String, WorkerStatus> entry : workerStatusMap.entrySet()) {
            WorkerEndpoint endpoint = endpointRegistry == null
                    ? null
                    : endpointRegistry.get(roleType, entry.getKey());
            samples.add(new double[]{
                    entry.getValue().getStepLatencyMs(),
                    endpoint == null ? 0 : endpoint.getLoadMetric()
            });
        }
        int size = samples.size();
        if (size < 2) {
            return;
        }

        double stepSum = 0;
        double loadSum = 0;
        for (double[] sample : samples) {
            stepSum += sample[0];
            loadSum += sample[1];
        }
        double stepMean = stepSum / size;
        double loadMean = loadSum / size;

        double stepSquaredDiffs = 0;
        double loadSquaredDiffs = 0;
        for (double[] sample : samples) {
            double stepDiff = sample[0] - stepMean;
            double loadDiff = sample[1] - loadMean;
            stepSquaredDiffs += stepDiff * stepDiff;
            loadSquaredDiffs += loadDiff * loadDiff;
        }
        engineHealthReporter.reportLatencyMetric(
                modelName,
                roleType.toString(),
                stepSquaredDiffs / (size - 1),
                loadSquaredDiffs / (size - 1));
    }

    private void markAliveFromDiscovery(WorkerStatus status) {
        status.setRole(roleType);
        status.setAlive(true);
        long nowUs = System.nanoTime() / 1000;
        long previousUs = status.getStatusLastUpdateTime().get();
        if (previousUs > 0) {
            status.getStatusUpdateIntervalUs().set(nowUs - previousUs);
        }
        status.getStatusLastUpdateTime().set(nowUs);
    }

    private WorkerStatus getOrCreateWorkerStatus(String ipPort) {
        WorkerStatus status = workerStatusMap.computeIfAbsent(ipPort, key -> {
            WorkerStatus created = new WorkerStatus();
            String[] address = key.split(":");
            created.setIp(address[0]);
            created.setPort(Integer.parseInt(address[1]));
            created.setRole(roleType);
            created.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
            logger.info("Created new WorkerStatus for worker: {}", key);
            return created;
        });
        if (status.getRole() == null) {
            status.setRole(roleType);
        }
        return status;
    }

    private void ensureEndpoint(String ipPort, WorkerStatus status) {
        if (endpointRegistry == null) {
            return;
        }
        status.setGrpcPort(CommonUtils.toGrpcPort(status.getPort()));
        if ((roleType == RoleType.PREFILL || roleType == RoleType.PDFUSION)
                && status.getDpSize() > 1) {
            throw new UnsupportedOperationException(String.format(
                    "%s DP group endpoint not yet supported: model=%s, ipPort=%s, dp_size=%d",
                    roleType, modelName, ipPort, status.getDpSize()));
        }
        endpointRegistry.ensureEndpoint(roleType, ipPort, status);
    }

    private String discoveryKey() {
        return modelName + "/" + roleType;
    }
}

package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.CollectionUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.stream.Collectors;

public class EngineSyncRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String modelName;

    private final WorkerDirectory workerDirectory;

    private final WorkerAddressService workerAddressService;

    private final ExecutorService statusCheckExecutor;
    private final ExecutorService cacheStatusExecutor;
    private final ExecutorService statusApplicationExecutor;

    private final EngineHealthReporter engineHealthReporter;

    private final EngineGrpcService engineGrpcService;

    private final RoleType roleType;

    private final CacheAwareService cacheAwareService;

    private final DynamicCacheIntervalService cacheIntervalService;

    private final long syncRequestTimeoutMs;

    private final long syncCycleId;

    private final Long syncEngineStatusInterval;

    private final boolean cacheFullSnapshotDebugMode;

    public EngineSyncRunner(String modelName,
                            WorkerDirectory workerDirectory,
                            WorkerAddressService workerAddressService,
                            ExecutorService statusCheckExecutor,
                            ExecutorService cacheStatusExecutor,
                            ExecutorService statusApplicationExecutor,
                            EngineHealthReporter engineHealthReporter,
                            EngineGrpcService engineGrpcService,
                            RoleType roleType,
                            CacheAwareService cacheAwareService,
                            DynamicCacheIntervalService cacheIntervalService,
                            long syncRequestTimeoutMs,
                            long syncCycleId,
                            Long syncEngineStatusInterval,
                            boolean cacheFullSnapshotDebugMode) {

        this.modelName = modelName;
        this.workerAddressService = workerAddressService;
        this.workerDirectory = Objects.requireNonNull(
                workerDirectory, "workerDirectory");
        this.statusCheckExecutor = statusCheckExecutor;
        this.statusApplicationExecutor = Objects.requireNonNull(statusApplicationExecutor, "statusApplicationExecutor");
        this.cacheStatusExecutor = Objects.requireNonNull(cacheStatusExecutor, "cacheStatusExecutor");
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.roleType = roleType;
        this.cacheAwareService = Objects.requireNonNull(
                cacheAwareService, "cacheAwareService");
        this.cacheIntervalService = Objects.requireNonNull(
                cacheIntervalService, "cacheIntervalService");
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
        this.syncCycleId = syncCycleId;
        this.syncEngineStatusInterval = syncEngineStatusInterval;
        this.cacheFullSnapshotDebugMode = cacheFullSnapshotDebugMode;
    }

    @Override
    public void run() {
        logger.debug("EngineSyncRunner start for model: {}, role: {}", modelName, roleType.toString());
        try {
            long startTimeInUs = System.nanoTime() / 1000;
            Map<String, WorkerStatus> cachedWorkerStatuses = workerDirectory.statusSnapshot(roleType);
            List<WorkerHost> latestEngineWorkerList;
            boolean discoverySucceeded = true;
            try {
                latestEngineWorkerList = workerAddressService.getEngineWorkerList(modelName, roleType);
            } catch (RuntimeException failure) {
                discoverySucceeded = false;
                latestEngineWorkerList = cachedWorkerStatuses.values().stream().map(status -> {
                    WorkerStatus.TopologySnapshot topology = status.topologySnapshot();
                    return new WorkerHost(topology.ip(), topology.port(), topology.grpcPort(),
                            topology.port() + 5, topology.site(), topology.group());
                }).toList();
                logger.warn("Service discovery failed; polling cached workers, model={}, role={}, workers={}, error={}",
                        modelName, roleType, latestEngineWorkerList.size(), failure.getMessage());
            }
            logger.debug("workerAddressService getEngineWorkerList, model: {}, role: {}, size: {}", modelName, roleType, latestEngineWorkerList.size());
            if (discoverySucceeded) {
                engineHealthReporter.reportServiceDiscoveryResult(modelName, latestEngineWorkerList.size(), roleType.toString());
            }
            if (CollectionUtils.isEmpty(latestEngineWorkerList)) {
                logger.debug("get engine worker list is empty, cost={}μs, model={}", System.nanoTime() / 1000 - startTimeInUs, modelName);
            }
            // Log if latest worker count differs from cached worker count
            if (discoverySucceeded && cachedWorkerStatuses.size() != latestEngineWorkerList.size()) {
                logger.info("[update] engine ip changes, model={}, role={}, before={}, after={}",
                        modelName, roleType, cachedWorkerStatuses.size(), latestEngineWorkerList.size());
            }

            // Remove if not in latest engine list
            Set<String> latestValidIpPorts = latestEngineWorkerList.stream()
                    .map(WorkerHost::getIpPort)
                    .collect(Collectors.toSet());
            logger.debug("Current cached worker size: {}, latest worker list size: {}", cachedWorkerStatuses.size(), latestEngineWorkerList.size());
            for (Map.Entry<String, WorkerStatus> entry: cachedWorkerStatuses.entrySet()) {
                WorkerStatus workerStatus = entry.getValue();
                String ipPort = entry.getKey();
                if (discoverySucceeded && !latestValidIpPorts.contains(ipPort)) {
                    retireMissingGeneration(
                            ipPort, workerStatus);
                }
            }
            if (latestEngineWorkerList.isEmpty()) {
                logger.debug("latestEngineWorkerList is empty, role: {}", roleType);
                return;
            } else {
                logger.debug("latestEngineWorkerList for role: {}, workers:{}", roleType, latestEngineWorkerList.size());
            }

            logger.debug("Submitting status check tasks for {} workers", latestEngineWorkerList.size());
            // Check cadence before allocating a cache poll or submitting a task.
            boolean pollCacheThisCycle = !roleType.requiresCacheKeys()
                    || syncCycleId % Math.max(cacheIntervalService.getCurrentIntervalMs()
                            / syncEngineStatusInterval, 1L) == 0L;
            for (WorkerHost host : latestEngineWorkerList) {
                String workerIpPort = host.getIpPort();
                String site = host.getSite();

                WorkerStatus workerStatus = discoverySucceeded
                        ? getOrCreateWorkerStatus(workerIpPort, site, host.getGroup())
                        : cachedWorkerStatuses.get(workerIpPort);

                if (workerStatus == null) { continue; }
                if (!workerStatus.isActiveGeneration()) {
                    workerStatus.trace().record(Stage.DISCOVERY_POLL_SKIPPED);
                    logger.debug(
                            "Skip retiring WorkerStatus generation {} for {}",
                            workerStatus.getGenerationId(), workerIpPort);
                    continue;
                }

                WorkerStatus.PollLease statusPollLease =
                        workerStatus.tryBeginStatusPoll(syncCycleId);
                if (statusPollLease != null) {
                    boolean handedOff = false;
                    try {
                        logger.debug("Submitting GrpcWorkerStatusRunner for worker: {}, site: {}", workerIpPort, site);
                        GrpcWorkerStatusRunner grpcWorkerStatusRunner
                                = new GrpcWorkerStatusRunner(modelName, workerIpPort, site, roleType,
                                workerStatus, statusPollLease, workerDirectory,
                                engineHealthReporter, engineGrpcService,
                                syncRequestTimeoutMs,
                                cacheAwareService, statusApplicationExecutor);
                        statusPollLease.pollTrace().record(Stage.RUNNER_SUBMITTED);
                        statusCheckExecutor.submit(grpcWorkerStatusRunner);
                        handedOff = true;
                    } catch (RejectedExecutionException e) {
                        statusPollLease.pollTrace().record(Stage.RUNNER_REJECTED, e);
                        logger.debug("Status check rejected for worker: {}, reset flag for retry", workerIpPort);
                    } finally {
                        if (!handedOff) {
                            statusPollLease.close();
                        }
                    }
                } else {
                    logger.debug("Skip status check for worker: {}, previous request in progress", workerIpPort);
                }

                if (!pollCacheThisCycle) { continue; }
                WorkerStatus.PollLease cachePollLease =
                        workerStatus.tryBeginCachePoll(syncCycleId);
                if (cachePollLease != null) {
                    boolean handedOff = false;
                    try {
                        logger.debug("Submitting GrpcCacheStatusCheckRunner for worker: {}, site: {}", workerIpPort, site);
                        GrpcCacheStatusCheckRunner grpcCacheStatusCheckRunner
                                = new GrpcCacheStatusCheckRunner(modelName, workerIpPort, site, roleType,
                                workerStatus, cachePollLease, workerDirectory,
                                engineHealthReporter, engineGrpcService,
                                cacheAwareService, cacheIntervalService, syncRequestTimeoutMs,
                                cacheFullSnapshotDebugMode, cacheStatusExecutor);
                        cachePollLease.pollTrace().record(Stage.RUNNER_SUBMITTED);
                        cacheStatusExecutor.submit(grpcCacheStatusCheckRunner);
                        handedOff = true;
                    } catch (RejectedExecutionException e) {
                        cachePollLease.pollTrace().record(Stage.RUNNER_REJECTED, e);
                        logger.debug("Cache check rejected for worker: {}, reset flag for retry", workerIpPort);
                    } finally {
                        if (!handedOff) {
                            cachePollLease.close();
                        }
                    }
                } else {
                    logger.debug("Skip cache check for worker: {}, previous request in progress", workerIpPort);
                }
            }
            logger.debug("Finished submitting status check tasks for model: {}, role: {}, worker count: {}", modelName,
                    roleType, latestEngineWorkerList.size());

        } catch (Exception e) {
            logger.error("sync engine workers status exception, modelName:{}, error:{}", modelName, e.getMessage(), e);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, null);
        } finally {
            logger.debug("Entering finally block for model: {}", modelName);
            Map<String, WorkerStatus> currentStatuses =
                    workerDirectory.statusSnapshot(roleType);
            logger.debug("Worker status map size: {}", currentStatuses.size());

            Map<String, WorkerStatus.EngineObservation> statusSnapshots =
                    new HashMap<>();
            Map<String, Long> observedRunningLoads = new HashMap<>();
            double sumStepLatency = 0.0;
            double sumRunningLoad = 0.0;
            for (Map.Entry<String, WorkerStatus> entry
                    : currentStatuses.entrySet()) {
                String workerIpPort = entry.getKey();
                WorkerStatus workerStatus = entry.getValue();
                if (!workerStatus.isActiveGeneration()) {
                    continue;
                }
                WorkerStatus.EngineObservation statusSnapshot =
                        workerStatus.committedEngineObservation();
                if (!workerDirectory.isCurrentStatus(
                        roleType, workerIpPort, workerStatus)) {
                    continue;
                }
                statusSnapshots.put(workerIpPort, statusSnapshot);
                sumStepLatency += statusSnapshot.stepLatencyMs();

                WorkerEndpoint endpoint = workerDirectory.exactEndpoint(
                        roleType, workerIpPort, workerStatus);
                OptionalLong load = endpoint == null
                        ? OptionalLong.empty() : endpoint.getLoadMetric();
                if (load.isPresent()) {
                    long value = load.getAsLong();
                    observedRunningLoads.put(workerIpPort, value);
                    sumRunningLoad += value;
                }
            }

            int observedStatusCount = statusSnapshots.size();
            if (observedStatusCount >= 2) {
                double meanStepLatency = sumStepLatency / observedStatusCount;
                double meanRunningLoad = observedRunningLoads.isEmpty()
                        ? 0.0 : sumRunningLoad / observedRunningLoads.size();

                // Calculate variance (sample variance using Bessel correction)
                double sumStepLatencyOfSquaredDiffs = 0.0;
                double sumRunningLoadOfSquaredDiffs = 0.0;
                for (Map.Entry<String, WorkerStatus.EngineObservation> entry
                        : statusSnapshots.entrySet()) {
                    double diff = entry.getValue().stepLatencyMs()
                            - meanStepLatency;
                    sumStepLatencyOfSquaredDiffs += diff * diff;
                    Long runningLoad = observedRunningLoads.get(entry.getKey());
                    if (runningLoad != null) {
                        double diff2 = runningLoad - meanRunningLoad;
                        sumRunningLoadOfSquaredDiffs += diff2 * diff2;
                    }
                }
                double variance = sumStepLatencyOfSquaredDiffs
                        / (observedStatusCount - 1); // Sample variance
                engineHealthReporter.reportStepLatencyVariance(
                        modelName, this.roleType.toString(), variance);
                if (observedRunningLoads.size() >= 2) {
                    double runningLoadVariance = sumRunningLoadOfSquaredDiffs
                            / (observedRunningLoads.size() - 1);
                    engineHealthReporter.reportRunningLoadVariance(
                            modelName,
                            this.roleType.toString(),
                            runningLoadVariance);
                }
                logger.debug("EngineSyncRunner finished for model: {}, role: {}", modelName, roleType);
            } else {
                logger.debug("Less than 2 workers, skipping variance calculation for model: {}", modelName);
            }
        }
    }

    private WorkerStatus getOrCreateWorkerStatus(
            String workerIpPort,
            String site,
            String group) {
        while (true) {
            WorkerStatus workerStatus = workerDirectory.currentOrDiscover(
                    roleType, workerIpPort,
                    () -> createWorkerStatus(workerIpPort, site, group));

            // Unchanged discovery must not wait for local status reduction.
            if (workerStatus.getRole() == roleType && Objects.equals(workerStatus.getGroup(), group)
                    && Objects.equals(workerStatus.getSite(), site)) { return workerStatus; }
            EndpointRegistry.DetachedGeneration endpointToRetire = null;
            RoleType generationRole = null;
            boolean retirementStarted = false;
            if (!workerStatus.lock.tryLock()) { return null; }
            try {
                if (!workerDirectory.isCurrentStatus(
                        roleType, workerIpPort, workerStatus)) {
                    continue;
                }
                if (!workerStatus.isActiveGeneration()) {
                    return workerStatus;
                }

                RoleType currentRole = workerStatus.getRole();
                String currentGroup = workerStatus.getGroup();
                boolean roleChanged = currentRole != roleType;
                boolean groupChanged = !Objects.equals(currentGroup, group);
                if (!roleChanged && !groupChanged) {
                    // Site changes do not change scheduling ownership. Publish
                    // the discovery labels atomically on the same generation.
                    workerStatus.updateDiscoveryLabels(site, group);
                    return workerStatus;
                }

                // Group/role ownership is a generation boundary. Keep the old
                // status identity published as RETIRING until the endpoint's
                // real retirement completion runs the exact finalizer. Cache
                // cleanup finishes before a replacement can publish.
                generationRole = currentRole == null ? roleType : currentRole;
                endpointToRetire = workerDirectory.beginRetirement(
                        generationRole, workerIpPort, workerStatus, "DISCOVERY_TOPOLOGY_CHANGED");
                retirementStarted = true;
            } finally {
                workerStatus.lock.unlock();
            }

            if (!retirementStarted) {
                return workerStatus;
            }
            workerDirectory.completeRetirementAsync(
                    generationRole, workerIpPort, workerStatus,
                    endpointToRetire, cacheAwareService, logger);
            logger.info(
                    "[replace] retiring worker topology generation, model={}, role={}, ipPort={}, generation={}, newGroup={}",
                    modelName,
                    roleType,
                    workerIpPort,
                    workerStatus.getGenerationId(),
                    group);
            // A later discovery pass can publish the replacement only after
            // real endpoint retirement removes this RETIRING holder.
            return workerStatus;
        }
    }

    private WorkerStatus createWorkerStatus(
            String workerIpPort,
            String site,
            String group) {
        int separator = workerIpPort.lastIndexOf(':');
        if (separator <= 0 || separator == workerIpPort.length() - 1) {
            throw new IllegalArgumentException(
                    "Invalid worker address: " + workerIpPort);
        }
        String ip = workerIpPort.substring(0, separator);
        int port = Integer.parseInt(workerIpPort.substring(separator + 1));
        WorkerStatus discovered = WorkerStatus.createDiscovered(
                roleType,
                group,
                ip,
                port,
                CommonUtils.toGrpcPort(port),
                site);
        logger.info("Created WorkerStatus generation {} for worker: {}",
                discovered.getGenerationId(), workerIpPort);
        return discovered;
    }

    private void retireMissingGeneration(
            String workerIpPort,
            WorkerStatus workerStatus) {
        EndpointRegistry.DetachedGeneration endpointToRetire = null;
        RoleType generationRole = null;
        boolean retirementStarted = false;
        if (!workerDirectory.isCurrentStatus(roleType, workerIpPort, workerStatus)) { return; }
        workerStatus.markRemovedFromDiscovery();
        if (!workerStatus.lock.tryLock()) { return; }
        try {
            if (!workerDirectory.isCurrentStatus(
                    roleType, workerIpPort, workerStatus)) {
                return;
            }
            if (!workerStatus.isActiveGeneration()) {
                return;
            }
            generationRole = workerStatus.getRole();
            endpointToRetire = workerDirectory.beginRetirement(
                    generationRole, workerIpPort, workerStatus, workerStatus.retirementReason().name());
            retirementStarted = true;
        } finally {
            workerStatus.lock.unlock();
        }

        if (retirementStarted) {
            workerDirectory.completeRetirementAsync(
                    generationRole, workerIpPort, workerStatus,
                    endpointToRetire, cacheAwareService, logger);
            logger.info(
                    "[remove] retiring missing worker, model={}, role={}, ipPort={}, generation={}",
                    modelName,
                    roleType,
                    workerIpPort,
                    workerStatus.getGenerationId());
        }
    }

}

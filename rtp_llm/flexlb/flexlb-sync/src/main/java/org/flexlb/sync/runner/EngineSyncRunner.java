package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.util.CommonUtils;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.lifecycle.WorkerGenerationRetirement;
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

    private final CacheAwareService cacheAwareService;

    private final DynamicCacheIntervalService cacheIntervalService;

    private final long syncRequestTimeoutMs;

    private final LongAdder syncCount;

    private final Long syncEngineStatusInterval;

    private final boolean cacheFullSnapshotDebugMode;

    private final EndpointEventSink endpointEventSink;

    private final EndpointRegistry endpointRegistry;

    private final long statusStaleAfterUs;

    public EngineSyncRunner(String modelName,
                            Map<String, WorkerStatus> workerStatusMap,
                            WorkerAddressService workerAddressService,
                            ExecutorService statusCheckExecutor,
                            EngineHealthReporter engineHealthReporter,
                            EngineGrpcService engineGrpcService,
                            RoleType roleType,
                            CacheAwareService cacheAwareService,
                            DynamicCacheIntervalService cacheIntervalService,
                            long syncRequestTimeoutMs,
                            LongAdder syncCount,
                            Long syncEngineStatusInterval,
                            boolean cacheFullSnapshotDebugMode,
                            EndpointEventSink endpointEventSink,
                            EndpointRegistry endpointRegistry) {

        this(modelName, workerStatusMap, workerAddressService,
                statusCheckExecutor, engineHealthReporter, engineGrpcService,
                roleType, cacheAwareService, cacheIntervalService,
                syncRequestTimeoutMs, syncCount, syncEngineStatusInterval,
                cacheFullSnapshotDebugMode, endpointEventSink,
                endpointRegistry, 10_000_000L);
    }

    public EngineSyncRunner(String modelName,
                            Map<String, WorkerStatus> workerStatusMap,
                            WorkerAddressService workerAddressService,
                            ExecutorService statusCheckExecutor,
                            EngineHealthReporter engineHealthReporter,
                            EngineGrpcService engineGrpcService,
                            RoleType roleType,
                            CacheAwareService cacheAwareService,
                            DynamicCacheIntervalService cacheIntervalService,
                            long syncRequestTimeoutMs,
                            LongAdder syncCount,
                            Long syncEngineStatusInterval,
                            boolean cacheFullSnapshotDebugMode,
                            EndpointEventSink endpointEventSink,
                            EndpointRegistry endpointRegistry,
                            long statusStaleAfterUs) {

        this.modelName = modelName;
        this.workerAddressService = workerAddressService;
        this.workerStatusMap = workerStatusMap;
        this.statusCheckExecutor = statusCheckExecutor;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.roleType = roleType;
        this.cacheAwareService = Objects.requireNonNull(
                cacheAwareService, "cacheAwareService");
        this.cacheIntervalService = Objects.requireNonNull(
                cacheIntervalService, "cacheIntervalService");
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
        this.syncCount = syncCount;
        this.syncEngineStatusInterval = syncEngineStatusInterval;
        this.cacheFullSnapshotDebugMode = cacheFullSnapshotDebugMode;
        this.endpointEventSink = Objects.requireNonNull(
                endpointEventSink, "endpointEventSink");
        this.endpointRegistry = endpointRegistry;
        if (statusStaleAfterUs <= 0L) {
            throw new IllegalArgumentException(
                    "statusStaleAfterUs must be positive");
        }
        this.statusStaleAfterUs = statusStaleAfterUs;
    }

    @Override
    public void run() {
        logger.debug("EngineSyncRunner start for model: {}, role: {}", modelName, roleType.toString());
        try {
            long startTimeInUs = System.nanoTime() / 1000;
            List<WorkerHost> latestEngineWorkerList = workerAddressService.getEngineWorkerList(modelName, roleType);
            logger.debug("workerAddressService getEngineWorkerList, model: {}, role: {}, size: {}", modelName, roleType, latestEngineWorkerList.size());
            engineHealthReporter.reportServiceDiscoveryResult(modelName, latestEngineWorkerList.size(), roleType.toString());
            if (CollectionUtils.isEmpty(latestEngineWorkerList)) {
                logger.debug("get engine worker list is empty, cost={}μs, model={}", System.nanoTime() / 1000 - startTimeInUs, modelName);
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
            logger.debug("Current cached worker size: {}, latest worker list size: {}", cachedWorkerStatuses.size(), latestEngineWorkerList.size());
            for (Map.Entry<String, WorkerStatus> entry: cachedWorkerStatuses.entrySet()) {
                WorkerStatus workerStatus = entry.getValue();
                String ipPort = entry.getKey();
                if (!latestValidIpPorts.contains(ipPort)) {
                    retireMissingGenerationIfExpired(
                            cachedWorkerStatuses, ipPort, workerStatus);
                }
            }
            if (latestEngineWorkerList.isEmpty()) {
                logger.debug("latestEngineWorkerList is empty, role: {}", roleType);
                return;
            } else {
                logger.debug("latestEngineWorkerList for role: {}, workers:{}", roleType, latestEngineWorkerList.size());
            }

            logger.debug("Submitting status check tasks for {} workers", latestEngineWorkerList.size());
            for (WorkerHost host : latestEngineWorkerList) {
                String workerIpPort = host.getIpPort();
                String site = host.getSite();

                WorkerStatus workerStatus = getOrCreateWorkerStatus(
                        cachedWorkerStatuses, workerIpPort, site, host.getGroup());

                if (!workerStatus.isActiveGeneration()) {
                    logger.debug(
                            "Skip retiring WorkerStatus generation {} for {}",
                            workerStatus.getGenerationId(), workerIpPort);
                    continue;
                }

                WorkerStatus.PollLease statusPollLease =
                        workerStatus.tryBeginStatusPoll();
                if (statusPollLease != null) {
                    boolean handedOff = false;
                    try {
                        logger.debug("Submitting GrpcWorkerStatusRunner for worker: {}, site: {}", workerIpPort, site);
                        GrpcWorkerStatusRunner grpcWorkerStatusRunner
                                = new GrpcWorkerStatusRunner(modelName, workerIpPort, site, roleType, host.getGroup(),
                                workerStatus, statusPollLease, cachedWorkerStatuses,
                                engineHealthReporter, engineGrpcService,
                                syncRequestTimeoutMs, endpointEventSink, endpointRegistry,
                                cacheAwareService, statusCheckExecutor);
                        statusCheckExecutor.submit(grpcWorkerStatusRunner);
                        handedOff = true;
                    } catch (RejectedExecutionException e) {
                        logger.debug("Status check rejected for worker: {}, reset flag for retry", workerIpPort);
                    } finally {
                        if (!handedOff) {
                            statusPollLease.close();
                        }
                    }
                } else {
                    logger.debug("Skip status check for worker: {}, previous request in progress", workerIpPort);
                }

                WorkerStatus.PollLease cachePollLease =
                        workerStatus.tryBeginCachePoll();
                if (cachePollLease != null) {
                    boolean handedOff = false;
                    try {
                        logger.debug("Submitting GrpcCacheStatusCheckRunner for worker: {}, site: {}", workerIpPort, site);
                        GrpcCacheStatusCheckRunner grpcCacheStatusCheckRunner
                                = new GrpcCacheStatusCheckRunner(modelName, workerIpPort, site, roleType,
                                workerStatus, cachePollLease, cachedWorkerStatuses,
                                engineHealthReporter, engineGrpcService,
                                cacheAwareService, cacheIntervalService,
                                syncRequestTimeoutMs, syncCount, syncEngineStatusInterval,
                                cacheFullSnapshotDebugMode, statusCheckExecutor);
                        statusCheckExecutor.submit(grpcCacheStatusCheckRunner);
                        handedOff = true;
                    } catch (RejectedExecutionException e) {
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
            logger.debug("Worker status map size: {}", workerStatusMap.size());

            Map<String, WorkerStatus.EngineObservation> statusSnapshots =
                    new HashMap<>();
            Map<String, Long> observedRunningLoads = new HashMap<>();
            double sumStepLatency = 0.0;
            double sumRunningLoad = 0.0;
            for (Map.Entry<String, WorkerStatus> entry
                    : workerStatusMap.entrySet()) {
                String workerIpPort = entry.getKey();
                WorkerStatus workerStatus = entry.getValue();
                if (!workerStatus.isActiveGeneration()) {
                    continue;
                }
                WorkerStatus.EngineObservation statusSnapshot =
                        workerStatus.committedEngineObservation();
                if (workerStatusMap.get(workerIpPort) != workerStatus) {
                    continue;
                }
                statusSnapshots.put(workerIpPort, statusSnapshot);
                sumStepLatency += statusSnapshot.stepLatencyMs();

                WorkerEndpoint endpoint = endpointRegistry == null
                        ? null
                        : endpointRegistry.get(
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
            Map<String, WorkerStatus> workerStatuses,
            String workerIpPort,
            String site,
            String group) {
        while (true) {
            WorkerStatus workerStatus = workerStatuses.computeIfAbsent(
                    workerIpPort,
                    ignored -> createWorkerStatus(
                            workerIpPort, site, group));

            EndpointRegistry.DetachedGeneration endpointToRetire = null;
            RoleType generationRole = null;
            boolean retirementStarted = false;
            workerStatus.lock.lock();
            try {
                if (workerStatuses.get(workerIpPort) != workerStatus) {
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
                // cleanup is generation-scoped and cannot block replacement.
                generationRole = currentRole == null ? roleType : currentRole;
                endpointToRetire = WorkerGenerationRetirement.begin(
                        workerStatus, endpointRegistry, generationRole,
                        workerIpPort);
                retirementStarted = true;
            } finally {
                workerStatus.lock.unlock();
            }

            if (!retirementStarted) {
                return workerStatus;
            }
            WorkerGenerationRetirement.complete(
                    workerStatus, workerStatuses, cacheAwareService,
                    workerIpPort, endpointToRetire, logger);
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

    private void retireMissingGenerationIfExpired(
            Map<String, WorkerStatus> workerStatuses,
            String workerIpPort,
            WorkerStatus workerStatus) {
        EndpointRegistry.DetachedGeneration endpointToRetire = null;
        RoleType generationRole = null;
        boolean retirementStarted = false;
        workerStatus.lock.lock();
        try {
            if (workerStatuses.get(workerIpPort) != workerStatus) {
                return;
            }
            if (!workerStatus.isActiveGeneration()) {
                return;
            }
            WorkerStatus.PollHealth health = workerStatus.pollHealth();
            if (System.nanoTime() / 1000
                    - health.lastSuccessfulPollUs()
                    <= statusStaleAfterUs) {
                return;
            }

            generationRole = workerStatus.getRole();
            endpointToRetire = WorkerGenerationRetirement.begin(
                    workerStatus, endpointRegistry, generationRole,
                    workerIpPort);
            retirementStarted = true;
        } finally {
            workerStatus.lock.unlock();
        }

        if (retirementStarted) {
            WorkerGenerationRetirement.complete(
                    workerStatus, workerStatuses, cacheAwareService,
                    workerIpPort, endpointToRetire, logger);
            logger.info(
                    "[remove] retiring missing worker, model={}, role={}, ipPort={}, generation={}",
                    modelName,
                    roleType,
                    workerIpPort,
                    workerStatus.getGenerationId());
        }
    }

}

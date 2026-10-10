package org.flexlb.sync.runner;

import com.google.common.math.StatsAccumulator;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Failures;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.google.common.base.Preconditions.checkArgument;

public class EngineSyncRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String modelName;

    private final EndpointRegistry endpointRegistry;

    private final WorkerAddressService workerAddressService;

    private final ExecutorService statusCheckExecutor;

    private final EngineHealthReporter engineHealthReporter;

    private final WorkerStatusRpcClient workerStatusRpcClient;

    private final RoleType roleType;

    private final CacheAwareService cacheAwareService;

    private final DynamicCacheIntervalService cacheIntervalService;

    private final long syncRequestTimeoutMs;

    private final LongAdder syncCount;

    private final Long syncEngineStatusInterval;

    private final boolean cacheFullSnapshotDebugMode;

    private final long statusStaleAfterUs;

    public EngineSyncRunner(String modelName,
                            EndpointRegistry endpointRegistry,
                            WorkerAddressService workerAddressService,
                            ExecutorService statusCheckExecutor,
                            EngineHealthReporter engineHealthReporter,
                            WorkerStatusRpcClient workerStatusRpcClient,
                            RoleType roleType,
                            CacheAwareService cacheAwareService,
                            DynamicCacheIntervalService cacheIntervalService,
                            long syncRequestTimeoutMs,
                            LongAdder syncCount,
                            Long syncEngineStatusInterval,
                            boolean cacheFullSnapshotDebugMode,
                            long statusStaleAfterUs) {

        this.modelName = modelName;
        this.workerAddressService = workerAddressService;
        this.endpointRegistry = Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        this.statusCheckExecutor = statusCheckExecutor;
        this.engineHealthReporter = engineHealthReporter;
        this.workerStatusRpcClient = workerStatusRpcClient;
        this.roleType = roleType;
        this.cacheAwareService = Objects.requireNonNull(
                cacheAwareService, "cacheAwareService");
        this.cacheIntervalService = Objects.requireNonNull(
                cacheIntervalService, "cacheIntervalService");
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
        this.syncCount = syncCount;
        this.syncEngineStatusInterval = syncEngineStatusInterval;
        this.cacheFullSnapshotDebugMode = cacheFullSnapshotDebugMode;
        checkArgument(statusStaleAfterUs > 0L, "statusStaleAfterUs must be positive");
        this.statusStaleAfterUs = statusStaleAfterUs;
    }

    @Override
    public void run() {
        logger.debug("EngineSyncRunner start for model: {}, role: {}", modelName, roleType.toString());
        try {
            long startTimeInUs = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
            List<WorkerHost> latestEngineWorkerList = workerAddressService.getEngineWorkerList(modelName, roleType);
            logger.debug("workerAddressService getEngineWorkerList, model: {}, role: {}, size: {}", modelName, roleType, latestEngineWorkerList.size());
            reportTelemetry(() -> engineHealthReporter.reportServiceDiscoveryResult(
                    modelName, latestEngineWorkerList.size(), roleType.toString()));
            if (latestEngineWorkerList.isEmpty()) {
                logger.debug("get engine worker list is empty, cost={}μs, model={}", TimeUnit.NANOSECONDS.toMicros(System.nanoTime()) - startTimeInUs, modelName);
            }
            Map<String, WorkerStatus> cachedWorkerStatuses =
                    endpointRegistry.statusSnapshot(roleType);
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
                    var retirement = endpointRegistry.beginRetirementIfStale(roleType, ipPort, workerStatus, statusStaleAfterUs);
                    if (retirement != null) {
                        retirement.complete(cacheAwareService, logger);
                        logger.info("[remove] retiring missing worker, model={}, role={}, ipPort={}, generation={}",
                                modelName, roleType, ipPort, workerStatus.getGenerationId());
                    }
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
                        workerIpPort, site, host.getGroup());

                if (!workerStatus.isActiveGeneration()) {
                    logger.debug(
                            "Skip retiring WorkerStatus generation {} for {}",
                            workerStatus.getGenerationId(), workerIpPort);
                    continue;
                }

                submitPoll(lease ->
                        new GrpcWorkerStatusRunner(modelName, workerStatus, lease, endpointRegistry,
                                engineHealthReporter, workerStatusRpcClient, syncRequestTimeoutMs,
                                cacheAwareService, statusCheckExecutor), workerStatus.tryBeginStatusPoll());
                submitPoll(lease ->
                        new GrpcCacheStatusCheckRunner(modelName, workerStatus, lease, endpointRegistry,
                                engineHealthReporter, workerStatusRpcClient, cacheAwareService,
                                cacheIntervalService, syncRequestTimeoutMs, syncCount,
                                syncEngineStatusInterval, cacheFullSnapshotDebugMode, statusCheckExecutor), workerStatus.tryBeginCachePoll());
            }
            logger.debug("Finished submitting status check tasks for model: {}, role: {}, worker count: {}", modelName,
                    roleType, latestEngineWorkerList.size());

        } catch (Exception e) {
            logger.error("sync engine workers status exception, modelName:{}, error:{}", modelName, e.getMessage(), e);
            reportTelemetry(() -> engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, null));
        } finally {
            reportTelemetry(this::reportLoadVariance);
        }
    }

    private void reportTelemetry(Runnable action) {
        Throwable failure = Failures.run(null, action);
        if (failure != null) { logger.warn("Worker sync telemetry failed for model: {}, role: {}", modelName, roleType, failure); }
    }

    /** Build the factory before acquiring a lease; failed submissions release it here. */
    private void submitPoll(Function<WorkerStatus.PollLease, Runnable> createRunner,
                            WorkerStatus.PollLease lease) {
        if (lease == null) {
            return;
        }
        boolean handedOff = false;
        try {
            statusCheckExecutor.submit(createRunner.apply(lease));
            handedOff = true;
        } catch (RejectedExecutionException rejected) {
            logger.debug("Worker poll rejected for model: {}, role: {}", modelName, roleType);
        } finally {
            if (!handedOff) {
                lease.close();
            }
        }
    }

    private void reportLoadVariance() {
        StatsAccumulator stepLatency = new StatsAccumulator();
        StatsAccumulator runningLoad = new StatsAccumulator();
        for (var entry : endpointRegistry.statusSnapshot(roleType).entrySet()) {
            WorkerStatus worker = entry.getValue();
            if (!worker.isActiveGeneration()) {
                continue;
            }
            WorkerStatus.EngineObservation observation = worker.committedEngineObservation();
            if (!endpointRegistry.isCurrentStatus(roleType, entry.getKey(), worker)) {
                continue;
            }
            stepLatency.add(observation.stepLatencyMs());
            WorkerEndpoint endpoint = endpointRegistry.get(roleType, entry.getKey(), worker);
            if (endpoint != null) {
                endpoint.getLoadMetric().ifPresent(runningLoad::add);
            }
        }
        if (stepLatency.count() >= 2) {
            engineHealthReporter.reportStepLatencyVariance(modelName, roleType.name(), stepLatency.sampleVariance());
        }
        if (runningLoad.count() >= 2) {
            engineHealthReporter.reportRunningLoadVariance(modelName, roleType.name(), runningLoad.sampleVariance());
        }
    }

    private WorkerStatus getOrCreateWorkerStatus(
            String workerIpPort,
            String site,
            String group) {
        while (true) {
            WorkerStatus workerStatus = endpointRegistry.currentOrDiscover(
                    roleType, workerIpPort,
                    () -> createWorkerStatus(workerIpPort, site, group));

            EndpointRegistry.Retirement retirement;
            workerStatus.lock.lock();
            try {
                if (!endpointRegistry.isCurrentStatus(
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
                // cleanup is generation-scoped and cannot block replacement.
                RoleType generationRole = currentRole == null ? roleType : currentRole;
                retirement = endpointRegistry.beginRetirement(
                        generationRole, workerIpPort, workerStatus);
            } finally {
                workerStatus.lock.unlock();
            }

            retirement.complete(cacheAwareService, logger);
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
        checkArgument(separator > 0 && separator != workerIpPort.length() - 1,
                "Invalid worker address: %s", workerIpPort);
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

}

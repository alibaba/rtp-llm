package org.flexlb.sync.runner;

import com.google.common.math.StatsAccumulator;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.match.localsync.DynamicCacheIntervalService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Failures;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.Function;
import java.util.stream.Collectors;

public class EngineSyncRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String modelName;

    private final EndpointRegistry endpointRegistry;

    private final WorkerAddressService workerAddressService;

    private final ExecutorService statusCheckExecutor;
    private final ExecutorService cacheStatusExecutor;
    private final ExecutorService statusApplicationExecutor;

    private final EngineHealthReporter engineHealthReporter;

    private final WorkerStatusRpcClient workerStatusRpcClient;

    private final RoleType roleType;

    private final CacheAwareService cacheAwareService;

    private final DynamicCacheIntervalService cacheIntervalService;

    private final long syncRequestTimeoutMs;

    private final long syncCycleId;

    private final Long syncEngineStatusInterval;

    private final boolean cacheFullSnapshotDebugMode;
    private final boolean kvcmEnabled;

    public EngineSyncRunner(
            String modelName,
            EndpointRegistry endpointRegistry,
            WorkerAddressService workerAddressService,
            ExecutorService statusCheckExecutor,
            EngineHealthReporter engineHealthReporter,
            WorkerStatusRpcClient workerStatusRpcClient,
            RoleType roleType,
            CacheAwareService cacheAwareService,
            DynamicCacheIntervalService cacheIntervalService,
            long syncRequestTimeoutMs,
            long syncCycleId,
            Long syncEngineStatusInterval,
            boolean kvcmEnabled,
            boolean cacheFullSnapshotDebugMode,
            ExecutorService cacheStatusExecutor,
            ExecutorService statusApplicationExecutor) {

        this.modelName = modelName;
        this.workerAddressService = workerAddressService;
        this.endpointRegistry = Objects.requireNonNull(endpointRegistry, "endpointRegistry");
        this.statusCheckExecutor = statusCheckExecutor;
        this.engineHealthReporter = engineHealthReporter;
        this.workerStatusRpcClient = workerStatusRpcClient;
        this.roleType = roleType;
        this.cacheAwareService = Objects.requireNonNull(cacheAwareService, "cacheAwareService");
        this.cacheIntervalService =
                Objects.requireNonNull(cacheIntervalService, "cacheIntervalService");
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
        this.syncCycleId = syncCycleId;
        this.syncEngineStatusInterval = syncEngineStatusInterval;
        this.cacheFullSnapshotDebugMode = cacheFullSnapshotDebugMode;
        this.kvcmEnabled = kvcmEnabled;
        this.cacheStatusExecutor =
                Objects.requireNonNull(cacheStatusExecutor, "cacheStatusExecutor");
        this.statusApplicationExecutor =
                Objects.requireNonNull(statusApplicationExecutor, "statusApplicationExecutor");
    }

    @Override
    public void run() {
        logger.debug(
                "EngineSyncRunner start for model: {}, role: {}", modelName, roleType.toString());
        try {
            Map<String, WorkerStatus> cachedWorkerStatuses =
                    endpointRegistry.statusSnapshot(roleType);
            List<WorkerHost> discovered;
            try {
                discovered =
                        Objects.requireNonNull(
                                workerAddressService.getEngineWorkerList(modelName, roleType),
                                "discovery result");
            } catch (RuntimeException discoveryFailure) {
                logger.warn(
                        "Worker discovery failed; polling cached generations, model={}, role={}",
                        modelName,
                        roleType,
                        discoveryFailure);
                for (WorkerStatus worker : cachedWorkerStatuses.values()) {
                    pollWorker(worker);
                }
                return;
            }
            List<WorkerHost> latestEngineWorkerList = discovered;
            Set<String> latestValidIpPorts =
                    latestEngineWorkerList.stream()
                            .map(WorkerHost::getLogicalIpPort)
                            .collect(Collectors.toSet());
            for (var entry : cachedWorkerStatuses.entrySet()) {
                if (!latestValidIpPorts.contains(entry.getKey())) {
                    WorkerStatus status = entry.getValue();
                    status.markRemovedFromDiscovery();
                    var retirement =
                            endpointRegistry.beginRequestedRetirement(
                                    roleType, entry.getKey(), status);
                    if (retirement != null) {
                        retirement.completeAsync(cacheAwareService, logger);
                    }
                }
            }

            logger.debug(
                    "Submitting status check tasks for {} workers", latestEngineWorkerList.size());
            for (WorkerHost host : latestEngineWorkerList) {
                try {
                    if (host.getIp() == null || host.getIp().isBlank()) {
                        logger.warn(
                                "Ignoring discovered worker without an IP, model={}, role={}",
                                modelName,
                                roleType);
                        continue;
                    }
                    WorkerStatus workerStatus =
                            getOrCreateWorkerStatus(
                                    host.getLogicalIpPort(),
                                    host.getSite(),
                                    host.getGroup(),
                                    host.getDeploymentName(),
                                    host.getIp(),
                                    host.getHttpPort(),
                                    host.getGrpcPort(),
                                    host.getEngineIndex(),
                                    host.getMultiEngineNum(),
                                    host.getWorkerStatusPort());
                    if (workerStatus != null) {
                        pollWorker(workerStatus);
                    }
                } catch (RuntimeException failure) {
                    logger.error(
                            "Worker sync failed, model={}, role={}, address={}",
                            modelName,
                            roleType,
                            host.getIpPort(),
                            failure);
                }
            }
            logger.debug(
                    "Finished submitting status check tasks for model: {}, role: {}, worker count: {}",
                    modelName,
                    roleType,
                    latestEngineWorkerList.size());

        } catch (Exception e) {
            logger.error(
                    "sync engine workers status exception, modelName:{}, error:{}",
                    modelName,
                    e.getMessage(),
                    e);
            reportTelemetry(
                    () ->
                            engineHealthReporter.reportStatusCheckerFail(
                                    BalanceStatusEnum.UNKNOWN_ERROR, null));
        } finally {
            reportTelemetry(this::reportLoadVariance);
        }
    }

    private void pollWorker(WorkerStatus workerStatus) {
        if (!workerStatus.isAvailableForScheduling()) {
            return;
        }
        submitPoll(
                statusCheckExecutor,
                lease ->
                        new GrpcWorkerStatusRunner(
                                modelName,
                                workerStatus,
                                lease,
                                endpointRegistry,
                                engineHealthReporter,
                                workerStatusRpcClient,
                                syncRequestTimeoutMs,
                                cacheAwareService,
                                statusApplicationExecutor,
                                workerStatus.getWorkerStatusPort()),
                workerStatus.tryBeginStatusPoll(syncCycleId));
        if (kvcmEnabled) {
            return;
        }
        long intervalCycles =
                Math.max(
                        1L, cacheIntervalService.getCurrentIntervalMs() / syncEngineStatusInterval);
        WorkerStatus.PollLease cacheLease = workerStatus.tryBeginCachePoll(syncCycleId);
        if (cacheLease == null) {
            return;
        }
        if (roleType.requiresCacheKeys()
                && !workerStatus.shouldPollCache(syncCycleId, intervalCycles)) {
            cacheLease.close();
            return;
        }
        submitPoll(
                cacheStatusExecutor,
                lease ->
                        new GrpcCacheStatusCheckRunner(
                                modelName,
                                workerStatus,
                                lease,
                                endpointRegistry,
                                engineHealthReporter,
                                workerStatusRpcClient,
                                cacheAwareService,
                                cacheIntervalService,
                                syncRequestTimeoutMs,
                                cacheFullSnapshotDebugMode,
                                cacheStatusExecutor),
                cacheLease);
    }

    private void reportTelemetry(Runnable action) {
        Throwable failure = Failures.run(null, action);
        if (failure != null) {
            logger.warn(
                    "Worker sync telemetry failed for model: {}, role: {}",
                    modelName,
                    roleType,
                    failure);
        }
    }

    /**
     * Build the factory before acquiring a lease; failed submissions release it here.
     */
    private void submitPoll(
            ExecutorService executor,
            Function<WorkerStatus.PollLease, Runnable> createRunner,
            WorkerStatus.PollLease lease) {
        if (lease == null) {
            return;
        }
        boolean handedOff = false;
        try {
            lease.trace().record(org.flexlb.dao.master.EndpointTrace.Stage.RUNNER_SUBMITTED);
            executor.submit(createRunner.apply(lease));
            handedOff = true;
        } catch (RejectedExecutionException rejected) {
            lease.trace()
                    .record(org.flexlb.dao.master.EndpointTrace.Stage.RUNNER_REJECTED, rejected);
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
                endpoint.tryGetLoadMetric().ifPresent(runningLoad::add);
            }
        }
        if (stepLatency.count() >= 2) {
            engineHealthReporter.reportStepLatencyVariance(
                    roleType.name(), stepLatency.sampleVariance());
        }
        if (runningLoad.count() >= 2) {
            engineHealthReporter.reportRunningLoadVariance(
                    roleType.name(), runningLoad.sampleVariance());
        }
    }

    private WorkerStatus getOrCreateWorkerStatus(
            String workerIpPort,
            String site,
            String group,
            String deploymentName,
            String ip,
            int port,
            int grpcPort,
            int engineIndex,
            int multiEngineNum,
            int workerStatusPort) {
        while (true) {
            WorkerStatus workerStatus =
                    endpointRegistry.currentOrDiscover(
                            roleType,
                            workerIpPort,
                            () ->
                                    createWorkerStatus(
                                            workerIpPort,
                                            site,
                                            group,
                                            deploymentName,
                                            ip,
                                            port,
                                            grpcPort,
                                            engineIndex,
                                            multiEngineNum,
                                            workerStatusPort));

            if (!workerStatus.isAvailableForScheduling()) {
                return workerStatus;
            }
            WorkerStatus.TopologySnapshot topology = workerStatus.topologySnapshot();
            boolean addressChanged =
                    topology.grpcPort() != grpcPort
                            || topology.port() != port
                            || !Objects.equals(topology.ip(), ip)
                            || topology.workerStatusPort() != workerStatusPort
                            || topology.engineIndex() != engineIndex
                            || topology.multiEngineNum() != multiEngineNum;
            if (!addressChanged
                    && Objects.equals(topology.group(), group)
                    && Objects.equals(topology.site(), site)
                    && Objects.equals(topology.deploymentName(), deploymentName)) {
                return workerStatus;
            }
            EndpointRegistry.Retirement retirement;
            if (!workerStatus.lock.tryLock()) {
                return !addressChanged && Objects.equals(topology.group(), group)
                        ? workerStatus
                        : null;
            }
            try {
                if (!endpointRegistry.isCurrentStatus(roleType, workerIpPort, workerStatus)) {
                    continue;
                }
                if (!workerStatus.isActiveGeneration()) {
                    return workerStatus;
                }

                RoleType currentRole = workerStatus.getRole();
                String currentGroup = workerStatus.getGroup();
                boolean roleChanged = currentRole != roleType;
                boolean groupChanged = !Objects.equals(currentGroup, group);
                if (!roleChanged && !groupChanged && !addressChanged) {
                    // Site changes do not change scheduling ownership. Publish
                    // the discovery labels atomically on the same generation.
                    workerStatus.updateDiscoveryLabels(site, group, deploymentName);
                    return workerStatus;
                }

                // Group/role ownership is a generation boundary. Keep the old
                // status identity published as RETIRING until the endpoint's
                // real retirement completion runs the exact finalizer. Cache
                // cleanup is generation-scoped and must finish before replacement.
                RoleType generationRole = currentRole == null ? roleType : currentRole;
                retirement =
                        endpointRegistry.beginRetirement(
                                generationRole, workerIpPort, workerStatus);
            } finally {
                workerStatus.lock.unlock();
            }

            retirement.completeAsync(cacheAwareService, logger);
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
            String group,
            String deploymentName,
            String ip,
            int port,
            int grpcPort,
            int engineIndex,
            int multiEngineNum,
            int workerStatusPort) {
        WorkerStatus discovered =
                WorkerStatus.createDiscovered(
                        roleType,
                        group,
                        ip,
                        port,
                        grpcPort,
                        site,
                        deploymentName,
                        engineIndex,
                        multiEngineNum,
                        workerStatusPort);
        logger.info(
                "Created WorkerStatus generation {} for worker: {}",
                discovered.getGenerationId(),
                workerIpPort);
        return discovered;
    }
}

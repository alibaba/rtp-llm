package org.flexlb.sync.runner;

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
                            Long syncEngineStatusInterval) {

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
                return;
            }
            Map<String/*ip*/, WorkerStatus> cachedWorkerStatuses = workerStatusMap;
            // 如果最新的机器数和缓存中的机器数不一致，则打印日志
            if (cachedWorkerStatuses.size() != latestEngineWorkerList.size()) {
                logger.info("[update] engine ip changes, model={}, role={}, before={}, after={}",
                        modelName, roleType, cachedWorkerStatuses.size(), latestEngineWorkerList.size());
            }

            // 如果不在最新的引擎列表中，则移除
            Set<String> latestValidIpPorts = latestEngineWorkerList.stream()
                    .map(WorkerHost::getIpPort)
                    .collect(Collectors.toSet());
            logger.info("Current cached worker size: {}, latest worker list size: {}", cachedWorkerStatuses.size(), latestEngineWorkerList.size());
            for (Map.Entry<String, WorkerStatus> entry: cachedWorkerStatuses.entrySet()) {
                WorkerStatus workerStatus = entry.getValue();
                String ipPort = entry.getKey();
                if (!latestValidIpPorts.contains(ipPort)) {
                    long lastTime = workerStatus.getStatusLastUpdateTime().get();
                    if (System.nanoTime() / 1000 - lastTime > 1000 * 1000) { // 如果上次更新时间超过1s，则移除
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

                //  计算方差（样本方差，使用 Bessel 校正）
                double sumStepLatencyOfSquaredDiffs = 0.0;
                double sumRunningQueryLenOfSquaredDiffs = 0.0;
                for (WorkerStatus workerStatus : workerStatusMap.values()) {
                    double diff = workerStatus.getStepLatencyMs() - meanStepLatency;
                    double diff2 = workerStatus.getRunningQueueTime().get() - meanRunningQueryLen;
                    sumStepLatencyOfSquaredDiffs += diff * diff;
                    sumRunningQueryLenOfSquaredDiffs += diff2 * diff2;
                }
                double variance = sumStepLatencyOfSquaredDiffs / (size - 1); // 样本方差
                double variance2 = sumRunningQueryLenOfSquaredDiffs / (size - 1);

                engineHealthReporter.reportLatencyMetric(modelName, this.roleType.toString(), variance, variance2);
                logger.info("EngineSyncRunner finished for model: {}, role: {}", modelName, roleType);
            } else {
                logger.debug("Less than 2 workers, skipping variance calculation for model: {}", modelName);
            }
        }
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

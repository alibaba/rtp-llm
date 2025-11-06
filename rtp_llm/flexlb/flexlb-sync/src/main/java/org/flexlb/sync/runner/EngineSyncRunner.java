package org.flexlb.sync.runner;

import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.WorkerHost;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.HttpNettyClientHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.CollectionUtils;

public class EngineSyncRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String modelName;

    private final ConcurrentHashMap<String /*ipPort*/, WorkerStatus> workerStatusMap;

    private final WorkerAddressService workerAddressService;

    private final ExecutorService statusCheckExecutor;

    private final EngineHealthReporter engineHealthReporter;

    private final HttpNettyClientHandler syncNettyClient;

    private final EngineGrpcService engineGrpcService;

    private final RoleType roleType;

    private final CacheAwareService localKvCacheAwareManager;

    private final long syncRequestTimeoutMs;

    private final LongAdder syncCount;

    private final Long syncEngineStatusInterval;

    public EngineSyncRunner(String modelName,
                            ConcurrentHashMap<String, WorkerStatus> workerStatusMap,
                            WorkerAddressService workerAddressService,
                            ExecutorService statusCheckExecutor,
                            EngineHealthReporter engineHealthReporter,
                            HttpNettyClientHandler syncNettyClient,
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
        this.syncNettyClient = syncNettyClient;
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
            long startTimeInMs = System.currentTimeMillis();
            List<WorkerHost> latestEngineWorkerList = workerAddressService.getEngineWorkerList(modelName, roleType);
            logger.info("workerAddressService getEngineWorkerList, model: {}, role: {}, size: {}", modelName, roleType, latestEngineWorkerList.size());
            engineHealthReporter.reportVipServerResult(modelName, latestEngineWorkerList.size(), roleType.toString());
            if (CollectionUtils.isEmpty(latestEngineWorkerList)) {
                logger.error("get engine worker list is empty, cost={}ms, model={}", (System.currentTimeMillis() - startTimeInMs), modelName);
                return;
            }
            ConcurrentHashMap<String/*ip*/, WorkerStatus> cachedWorkerStatuses = workerStatusMap;
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
            for (String ipPort : cachedWorkerStatuses.keySet()) {
                WorkerStatus workerStatus = cachedWorkerStatuses.get(ipPort);
                logger.info("worker status ip: {} , alive: {}", workerStatus.getIp(), workerStatus.isAlive());
                if (!latestValidIpPorts.contains(ipPort)) {
                    long last_time = workerStatus.getLastUpdateTime().get();
                    if (System.currentTimeMillis() - last_time > 1000) {
                        cachedWorkerStatuses.remove(ipPort);
                        logger.info("[remove] engine ip changes, model={}, role={}, ipPort={}", modelName, roleType, ipPort);
                    }
                } else {
                    workerStatus.getLastScheduleTime().set(System.currentTimeMillis());
                }
            }
            if (latestEngineWorkerList.isEmpty()) {
                logger.warn("latestEngineWorkerList is empty, role: {}", roleType);
                return;
            } else {
                logger.info("latestEngineWorkerList for role: {}, workers:\n{}", roleType,
                        latestEngineWorkerList.stream().map(WorkerHost::getIpPort).collect(Collectors.joining("\n")));
            }

            logger.info("Submitting status check tasks for {} workers", latestEngineWorkerList.size());
            for (WorkerHost host : latestEngineWorkerList) {
                String workerIpPort = host.getIpPort();
                String site = host.getSite();

                // Choose between gRPC and HTTP based on service availability
                if (engineGrpcService.isEngineStatusEnabled()) {
                    logger.debug("Submitting GrpcWorkerStatusRunner for worker: {}, site: {}", workerIpPort, site);
                    GrpcWorkerStatusRunner grpcWorkerStatusRunner
                            = new GrpcWorkerStatusRunner(modelName, workerIpPort, site, host.getGroup(),
                        cachedWorkerStatuses, engineHealthReporter, engineGrpcService, syncRequestTimeoutMs, roleType);
                    statusCheckExecutor.submit(grpcWorkerStatusRunner);
                } else {
                    logger.debug("Submitting HttpStatusCheckRunner (HTTP) for worker: {}, site: {}, model: {}", workerIpPort, site, modelName);
                    HttpStatusCheckRunner httpStatusCheckRunner
                            = new HttpStatusCheckRunner(modelName, workerIpPort, site, host.getGroup(), cachedWorkerStatuses,
                            engineHealthReporter, syncNettyClient, localKvCacheAwareManager);
                    statusCheckExecutor.submit(httpStatusCheckRunner);
                }

                // Submit separate cache status check if enabled
                if (engineGrpcService.isCacheStatusEnabled()) {
                    logger.debug("Submitting GrpcCacheStatusCheckRunner for worker: {}, site: {}", workerIpPort, site);
                    GrpcCacheStatusCheckRunner grpcCacheStatusCheckRunner
                            = new GrpcCacheStatusCheckRunner(modelName, workerIpPort, site, roleType,
                            cachedWorkerStatuses, engineHealthReporter, engineGrpcService, localKvCacheAwareManager,
                            syncRequestTimeoutMs, syncCount, syncEngineStatusInterval);
                    statusCheckExecutor.submit(grpcCacheStatusCheckRunner);
                }
            }
            logger.info("Finished submitting status check tasks for model: {}, role: {}, worker count: {}", modelName,
                    roleType, latestEngineWorkerList.size());

        } catch (Exception e) {
            logger.error("sync engine workers status exception, modelName:{}, error:{}", modelName, e.getMessage(), e);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR);
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
}

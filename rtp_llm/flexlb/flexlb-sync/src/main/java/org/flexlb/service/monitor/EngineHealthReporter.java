package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import io.netty.util.concurrent.EventExecutor;
import io.netty.util.concurrent.SingleThreadEventExecutor;
import lombok.Data;
import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.FlexStatisticsType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.sync.synchronizer.AbstractEngineStatusSynchronizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.netty.resources.LoopResources;

import javax.annotation.PostConstruct;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadPoolExecutor;

import static org.flexlb.constant.MetricConstant.CACHE_BLOCK_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_VISITOR_RT;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_ALL_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_FAIL_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SCHEDULE_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_DECODE_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_LOCAL_TASK_MAP_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT;
import static org.flexlb.constant.MetricConstant.ENGINE_PREFILL_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_QUEUE_TIME;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_AVAILABLE_CONCURRENCY;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_VISITOR_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_VISITOR_SUCCESS_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_INFO_STEP_LATENCY_VAR;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.PREFILL_BALANCE_SELECT_FAIL_QPS;
import static org.flexlb.constant.MetricConstant.PREFILL_BALANCE_SELECT_QPS;
import static org.flexlb.constant.MetricConstant.PREFILL_BALANCE_TOKENIZE_COST;
import static org.flexlb.constant.MetricConstant.PREFILL_MASTER_EVENT;
import static org.flexlb.constant.MetricConstant.PREFILL_MASTER_NODE;

/**
 * Engine health reporter for monitoring engine status and metrics
 */
@Data
@Component
public class EngineHealthReporter {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final FlexMonitor monitor;

    private final CacheMetricsReporter cacheMetricsReporter;

    private final EngineGrpcClient engineGrpcClient;

    private final Map<String, EventLoopGroup> eventLoopGroupMap;

    @Autowired
    public EngineHealthReporter(FlexMonitor monitor,
                                CacheMetricsReporter cacheMetricsReporter,
                                EngineGrpcClient engineGrpcClient,
                                LoopResources serverLoopResources) {
        this.monitor = monitor;
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.engineGrpcClient = engineGrpcClient;

        this.eventLoopGroupMap = Map.of(
                "serverWorker", serverLoopResources.onServer(true),
                "serverSelector", serverLoopResources.onServerSelect(true),
                "gRpcEventLoopGroup", engineGrpcClient.getEventLoopGroup()
        );
    }

    @PostConstruct
    public void init() {

        this.monitor.register(ENGINE_STATUS_CHECK_SUCCESS_PERIOD, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_AVAILABLE_CONCURRENCY, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_VISITOR_RT, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_VISITOR_SUCCESS_QPS, FlexMetricType.QPS);
        this.monitor.register(ENGINE_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_PREFILL_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_DECODE_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_CHECK_FAIL, FlexMetricType.QPS);
        this.monitor.register(ENGINE_BALANCING_THREAD_POOL_INFO, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO, FlexMetricType.GAUGE);

        this.monitor.register(ENGINE_BALANCING_MASTER_ALL_QPS, FlexMetricType.QPS);
        this.monitor.register(ENGINE_BALANCING_MASTER_FAIL_QPS, FlexMetricType.QPS);
        this.monitor.register(ENGINE_BALANCING_MASTER_SCHEDULE_RT, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_BALANCING_MASTER_SELECT_DETAIL, FlexMetricType.QPS);

        this.monitor.register(ENGINE_RUNNING_QUEUE_TIME, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_LOCAL_TASK_MAP_SIZE, FlexMetricType.GAUGE);

        this.monitor.register(PREFILL_BALANCE_SELECT_QPS, FlexMetricType.QPS);
        this.monitor.register(PREFILL_BALANCE_SELECT_FAIL_QPS, FlexMetricType.QPS);
        this.monitor.register(PREFILL_BALANCE_TOKENIZE_COST, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY | FlexStatisticsType.SUM);

        this.monitor.register(PREFILL_MASTER_NODE, FlexMetricType.GAUGE, FlexStatisticsType.SUM);
        this.monitor.register(PREFILL_MASTER_EVENT, FlexMetricType.GAUGE, FlexStatisticsType.SUM);

        this.monitor.register(ENGINE_WORKER_INFO_STEP_LATENCY_VAR, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_RT, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, FlexMetricType.QPS);
        this.monitor.register(CACHE_STATUS_CHECK_SUCCESS_PERIOD, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_FAIL, FlexMetricType.QPS);
        this.monitor.register(CACHE_BLOCK_SIZE, FlexMetricType.GAUGE);
    }

    public void reportLatencyMetric(String modelName, String role, double result, double result2) {
        FlexMetricTags metricTags = FlexMetricTags.of("model", modelName, "role", role);
        monitor.report(ENGINE_WORKER_INFO_STEP_LATENCY_VAR, metricTags, result);
        monitor.report(ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR, metricTags, result2);
        logger.debug("Latency metric - model: {}, role: {}, stepLatency: {}, queryLen: {}", modelName, role, result, result2);
    }

    @Scheduled(fixedRate = 2000)
    private void reportEngineMetric() {
        for (Map.Entry<String, ModelWorkerStatus> entry : EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.entrySet()) {
            String modelName = entry.getKey();
            FlexMetricTags tags = FlexMetricTags.of("model", modelName);
            ModelWorkerStatus modelWorkerStatus = entry.getValue();
            monitor.report(ENGINE_WORKER_NUMBER, tags, modelWorkerStatus.getWorkerTotalCount());
            monitor.report(ENGINE_PREFILL_WORKER_NUMBER, tags, modelWorkerStatus.getPrefillStatusMap().size());
            monitor.report(ENGINE_DECODE_WORKER_NUMBER, tags, modelWorkerStatus.getDecodeStatusMap().size());
        }

        if (AbstractEngineStatusSynchronizer.engineSyncExecutor != null && AbstractEngineStatusSynchronizer.statusCheckExecutor != null) {
            reportThreadPoolInfo(ENGINE_BALANCING_THREAD_POOL_INFO, "engineSyncExecutor",
                    (ThreadPoolExecutor) AbstractEngineStatusSynchronizer.engineSyncExecutor);
            reportThreadPoolInfo(ENGINE_BALANCING_THREAD_POOL_INFO, "statusCheckExecutor",
                    (ThreadPoolExecutor) AbstractEngineStatusSynchronizer.statusCheckExecutor);
            reportThreadPoolInfo(ENGINE_BALANCING_THREAD_POOL_INFO, "serviceDiscoveryExecutor",
                    (ThreadPoolExecutor) WorkerAddressService.serviceDiscoveryExecutor);
        }
        reportThreadPoolInfo(ENGINE_BALANCING_THREAD_POOL_INFO, "gRpcExecutor", (ThreadPoolExecutor) engineGrpcClient.getExecutor());

        eventLoopGroupMap.forEach(this::reportEventLoopGroup);
    }

    public void reportServiceDiscoveryResult(String modelName, int result, String role) {
        FlexMetricTags metricTags = FlexMetricTags.of("model", modelName, "role", role);
        monitor.report(ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT, metricTags, result);
    }

    public void reportStatusCheckRemoteInfo(String modelName, String engineIp, String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp,
                "role", role);
        monitor.report(ENGINE_STATUS_VISITOR_RT, metricTags, System.currentTimeMillis() - startTime);
        monitor.report(ENGINE_STATUS_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportCacheStatusCheckRemoteInfo(String modelName, String engineIp, String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp,
                "role", role);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_RT, metricTags, System.currentTimeMillis() - startTime);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportStatusCheckerFail(String modelName, BalanceStatusEnum errorEnum, String ip) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "code", String.valueOf(errorEnum.getCode()),
                "engineIp", ip == null ? "" : ip
        );
        monitor.report(ENGINE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportCacheStatusCheckerFail(String modelName, String engineIp, BalanceStatusEnum errorEnum) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp,
                "code", String.valueOf(errorEnum.getCode()));
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportStatusCheckerSuccess(String modelName, WorkerStatus workerStatus) {

        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "code", "0",
                "engineIp", workerStatus.getIp(),
                "role", workerStatus.getRole());

        Long availableConcurrency = workerStatus.getAvailableConcurrency();
        if (availableConcurrency != null) {
            monitor.report(ENGINE_STATUS_AVAILABLE_CONCURRENCY, metricTags, availableConcurrency);
        }
        long lastUpdateTime = workerStatus.getStatusLastUpdateTime().get();
        if (lastUpdateTime > 0) {
            monitor.report(ENGINE_STATUS_CHECK_SUCCESS_PERIOD, metricTags, System.currentTimeMillis() - lastUpdateTime);
        }
        monitor.report(ENGINE_RUNNING_QUEUE_TIME, metricTags, workerStatus.getRunningQueueTime().get());

        // 汇报本地任务缓存的大小
        int localTaskMapSize = workerStatus.getLocalTaskMap() != null ? workerStatus.getLocalTaskMap().size() : 0;
        monitor.report(ENGINE_LOCAL_TASK_MAP_SIZE, metricTags, localTaskMapSize);
    }

    public void reportCacheStatusCheckerSuccess(String modelName, WorkerStatus workerStatus) {
        long cacheLastUpdateTime = workerStatus.getCacheLastUpdateTime().get();
        if (cacheLastUpdateTime > 0) {
            FlexMetricTags metricTags = FlexMetricTags.of(
                    "model", modelName,
                    "code", "0",
                    "engineIp", workerStatus.getIp(),
                    "role", workerStatus.getRole());
            monitor.report(CACHE_STATUS_CHECK_SUCCESS_PERIOD, metricTags, System.currentTimeMillis() - cacheLastUpdateTime);
        }
        if (workerStatus.getCacheStatus() != null) {
            long blockSize = workerStatus.getCacheStatus().getBlockSize();
            FlexMetricTags blockSizeTags = FlexMetricTags.of(
                    "model", modelName,
                    "engineIp", workerStatus.getIp(),
                    "role", workerStatus.getRole());
            monitor.report(CACHE_BLOCK_SIZE, blockSizeTags, blockSize);
        }
    }

    public void reportBalancingService(BalanceContext ctx) {
        if (ctx == null) {
            return;
        }

        if (!ctx.isSuccess()) {
            FlexMetricTags metricTags = FlexMetricTags.of(
                    "model", ctx.getMasterRequest().getModel(),
                    "code", String.valueOf(ctx.getMasterResponse().getCode()));
            monitor.report(ENGINE_BALANCING_MASTER_FAIL_QPS, metricTags, 1.0);
            monitor.report(ENGINE_BALANCING_MASTER_ALL_QPS, metricTags, 1.0);
        } else {
            FlexMetricTags metricTags = FlexMetricTags.of("model", ctx.getMasterRequest().getModel());
            monitor.report(ENGINE_BALANCING_MASTER_ALL_QPS, metricTags, 1.0);
            monitor.report(ENGINE_BALANCING_MASTER_SCHEDULE_RT, metricTags, System.currentTimeMillis() - ctx.getStartTime());
        }

        // 汇报服务器状态选择结果（根据 roleType 和 ip 区分）
        if (ctx.getMasterResponse() != null && CollectionUtils.isNotEmpty(ctx.getMasterResponse().getServerStatus())) {
            String modelName = ctx.getMasterRequest().getModel();
            boolean isSuccess = ctx.getMasterResponse().isSuccess();
            int code = ctx.getMasterResponse().getCode();

            for (ServerStatus serverStatus : ctx.getMasterResponse().getServerStatus()) {
                if (serverStatus.getRole() != null && serverStatus.getServerIp() != null) {
                    // 汇报具体服务器选择QPS
                    FlexMetricTags serverSelectionTags = FlexMetricTags.of(
                            "model", modelName,
                            "role", serverStatus.getRole().name(),
                            "engineIp", serverStatus.getServerIp(),
                            "success", String.valueOf(isSuccess),
                            "code", String.valueOf(code)
                    );
                    monitor.report(ENGINE_BALANCING_MASTER_SELECT_DETAIL, serverSelectionTags, 1.0);
                }
            }
        }
    }

    public void reportPrefillBalanceSelectMetric(String modelName,
                                                 boolean success,
                                                 String errorCode,
                                                 long totalCost,
                                                 long tokenizeCost,
                                                 long calcPrefixCost,
                                                 long calcTtftCost) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName);
        monitor.report(PREFILL_BALANCE_SELECT_QPS, metricTags, 1.0);
        if (!success) {
            FlexMetricTags failMetricTags = FlexMetricTags.of(
                    "model", modelName,
                    "code", errorCode);
            monitor.report(PREFILL_BALANCE_SELECT_FAIL_QPS, failMetricTags, 1.0);
        }

        if (totalCost > 0) {
            monitor.report(PREFILL_BALANCE_TOKENIZE_COST,
                    FlexMetricTags.of(
                            "model", modelName,
                            "stage", "total"),
                    totalCost);
        }
        if (tokenizeCost > 0) {
            monitor.report(PREFILL_BALANCE_TOKENIZE_COST,
                    FlexMetricTags.of(
                            "model", modelName,
                            "stage", "tokenize"),
                    tokenizeCost);
        }
        if (calcPrefixCost > 0) {
            monitor.report(PREFILL_BALANCE_TOKENIZE_COST,
                    FlexMetricTags.of(
                            "model", modelName,
                            "stage", "calcPrefix"),
                    calcPrefixCost);
        }
        if (calcTtftCost > 0) {
            monitor.report(PREFILL_BALANCE_TOKENIZE_COST,
                    FlexMetricTags.of(
                            "model", modelName,
                            "stage", "calcTtft"),
                    calcTtftCost);
        }

    }

    public void reportPrefillBalanceMasterNode(String master) {
        monitor.report(PREFILL_MASTER_NODE, FlexMetricTags.of("masterNode", master), 1.0);
    }

    public void reportPrefillBalanceMasterEvent(String event) {
        monitor.report(PREFILL_MASTER_EVENT, FlexMetricTags.of("event", event), 1.0);
    }

    public void reportThreadPoolInfo(String metricName, String name, ThreadPoolExecutor engineSyncExecutor) {
        if (engineSyncExecutor == null) {
            return;
        }

        Map<String, String> metricMap = new HashMap<>();
        metricMap.put("threadPool", name);

        metricMap.put("type", "executingTaskThreadSize");
        monitor.report(metricName, FlexMetricTags.of(metricMap), engineSyncExecutor.getActiveCount());
        metricMap.put("type", "queueSize");
        monitor.report(metricName, FlexMetricTags.of(metricMap), engineSyncExecutor.getQueue().size());
        metricMap.put("type", "corePoolSize");
        monitor.report(metricName, FlexMetricTags.of(metricMap), engineSyncExecutor.getCorePoolSize());
        metricMap.put("type", "currentThreadSizeInPool");
        monitor.report(metricName, FlexMetricTags.of(metricMap), engineSyncExecutor.getPoolSize());
    }

    private void reportEventLoopGroup(String eventLoopGroupName, EventLoopGroup eventLoopGroup) {
        int totalActiveExecutorCount = 0;
        int totalPendingTask = 0;
        for (EventExecutor executor : eventLoopGroup) {
            boolean isShutdown = executor.isShutdown();
            boolean isTerminated = executor.isTerminated();
            boolean isShuttingDown = executor.isShuttingDown();
            // 记录 active 的 worker 数量
            if (!isShutdown && !isTerminated && !isShuttingDown) {
                totalActiveExecutorCount++;
            }
            if (executor instanceof SingleThreadEventExecutor singleThreadEventExecutor) {
                int pendingTasks = singleThreadEventExecutor.pendingTasks();
                totalPendingTask += pendingTasks;
            }
        }
        Map<String, String> metricMap = new HashMap<>();
        metricMap.put("name", eventLoopGroupName);
        metricMap.put("type", "active-executor-count");
        monitor.report(org.flexlb.constant.MetricConstant.ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO, FlexMetricTags.of(metricMap), totalActiveExecutorCount);
        metricMap.put("type", "pending-task-total-count");
        monitor.report(org.flexlb.constant.MetricConstant.ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO, FlexMetricTags.of(metricMap), totalPendingTask);
    }

    public void reportCacheHitMetrics(String modelName, RoleType roleType, String engineIp, long hitTokens, double hitRatio) {
        cacheMetricsReporter.reportCacheHitMetrics(modelName, roleType, engineIp, hitTokens, hitRatio);
    }
}

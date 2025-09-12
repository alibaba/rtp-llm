package org.flexlb.service.monitor;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

import com.taobao.kmonitor.ImmutableMetricTags;
import com.taobao.kmonitor.KMonitor;
import com.taobao.kmonitor.MetricType;
import com.taobao.kmonitor.PriorityType;
import com.taobao.kmonitor.StatisticsType;
import io.netty.channel.EventLoopGroup;
import io.netty.util.concurrent.EventExecutor;
import io.netty.util.concurrent.SingleThreadEventExecutor;
import lombok.Data;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.sync.status.EngineMetric;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.sync.synchronizer.AbstractEngineStatusSynchronizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import static org.flexlb.constant.MetricConstant.CACHE_BLOCK_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_VISITOR_RT;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_ALL_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_FAIL_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SCHEDULE_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_DECODE_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_NUMBER_VIP_RESULT;
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

    private final ScheduledExecutorService scheduler;

    private final KMonitor monitor;

    private final EngineWorkerStatus engineWorkerStatus;

    private final CacheMetricsReporter cacheMetricsReporter;

    private final EngineGrpcClient engineGrpcClient;

    private Set<String/*modelName*/> proxyEngineSet = ConcurrentHashMap.newKeySet();

    private long syncMetricPeriodInMs;

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    @Autowired
    public EngineHealthReporter(
            @Qualifier("engineStatusSyncScheduler") ScheduledExecutorService scheduler,
            KMonitor monitor,
            CacheMetricsReporter cacheMetricsReporter,
            EngineWorkerStatus engineWorkerStatus,
            EngineGrpcClient engineGrpcClient) {

        this.scheduler = scheduler;
        this.monitor = monitor;
        this.syncMetricPeriodInMs = 1000;
        this.engineWorkerStatus = engineWorkerStatus;
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.engineGrpcClient = engineGrpcClient;
    }

    @PostConstruct
    public void init() {

        this.monitor.register(ENGINE_STATUS_CHECK_SUCCESS_PERIOD, MetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_AVAILABLE_CONCURRENCY, MetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_VISITOR_RT, MetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_VISITOR_SUCCESS_QPS, MetricType.QPS);
        this.monitor.register(ENGINE_WORKER_NUMBER, MetricType.GAUGE);
        this.monitor.register(ENGINE_PREFILL_WORKER_NUMBER, MetricType.GAUGE);
        this.monitor.register(ENGINE_DECODE_WORKER_NUMBER, MetricType.GAUGE);
        this.monitor.register(ENGINE_NUMBER_VIP_RESULT, MetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_CHECK_FAIL, MetricType.QPS);
        this.monitor.register(ENGINE_BALANCING_THREAD_POOL_INFO, MetricType.GAUGE, PriorityType.PRECISE);
        this.monitor.register(ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO, MetricType.GAUGE, PriorityType.PRECISE);

        this.monitor.register(ENGINE_BALANCING_MASTER_ALL_QPS, MetricType.QPS);
        this.monitor.register(ENGINE_BALANCING_MASTER_FAIL_QPS, MetricType.QPS);
        this.monitor.register(ENGINE_BALANCING_MASTER_SCHEDULE_RT, MetricType.GAUGE);

        this.monitor.register(ENGINE_RUNNING_QUEUE_TIME, MetricType.GAUGE);

        this.monitor.register(PREFILL_BALANCE_SELECT_QPS, MetricType.QPS);
        this.monitor.register(PREFILL_BALANCE_SELECT_FAIL_QPS, MetricType.QPS);
        this.monitor.register(PREFILL_BALANCE_TOKENIZE_COST, MetricType.GAUGE, StatisticsType.SUMMARY | StatisticsType.SUM);

        this.monitor.register(PREFILL_MASTER_NODE, MetricType.GAUGE, StatisticsType.SUM);
        this.monitor.register(PREFILL_MASTER_EVENT, MetricType.GAUGE, StatisticsType.SUM);

        this.monitor.register(ENGINE_WORKER_INFO_STEP_LATENCY_VAR, MetricType.GAUGE, StatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR, MetricType.GAUGE, StatisticsType.SUMMARY);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_RT, MetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, MetricType.QPS);
        this.monitor.register(CACHE_STATUS_CHECK_SUCCESS_PERIOD, MetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_FAIL, MetricType.QPS);
        this.monitor.register(CACHE_BLOCK_SIZE, MetricType.GAUGE);
        this.scheduler.scheduleAtFixedRate(this::reportEngineMetric, 0, syncMetricPeriodInMs, TimeUnit.MILLISECONDS);
    }

    public void reportLatencyMetric(String modelName, String role, double result, double result2) {
        ImmutableMetricTags metricTags = new ImmutableMetricTags("model", modelName, "role", role);
        monitor.report(ENGINE_WORKER_INFO_STEP_LATENCY_VAR, metricTags, result);
        monitor.report(ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR, metricTags, result2);
        logger.debug("Latency metric - model: {}, role: {}, stepLatency: {}, queryLen: {}", modelName, role, result, result2);
    }

    private void reportEngineMetric() {
        for (Map.Entry<String, ModelWorkerStatus> entry :
                engineWorkerStatus.getModelRoleWorkerStatusMap().entrySet()) {
            String modelName = entry.getKey();
            ImmutableMetricTags tags = new ImmutableMetricTags("model", modelName);
            EngineMetric engineMetric = entry.getValue().getEngineMetric();
            monitor.report(ENGINE_WORKER_NUMBER, tags, engineMetric.getTotal());
            monitor.report(ENGINE_PREFILL_WORKER_NUMBER, tags, engineMetric.getPrefill());
            monitor.report(ENGINE_DECODE_WORKER_NUMBER, tags, engineMetric.getDecode());
            logger.debug("Engine metric - model: {}, total: {}, prefill: {}, decode: {}",
                    modelName, engineMetric.getTotal(), engineMetric.getPrefill(), engineMetric.getDecode());
        }

        if (AbstractEngineStatusSynchronizer.engineSyncExecutor != null
                && AbstractEngineStatusSynchronizer.statusCheckExecutor != null
                && WorkerAddressService.vipServerExecutor != null) {
            reportThreadPoolInfo(ENGINE_BALANCING_THREAD_POOL_INFO, "engineSyncExecutor",
                    (ThreadPoolExecutor) AbstractEngineStatusSynchronizer.engineSyncExecutor);
            reportThreadPoolInfo(ENGINE_BALANCING_THREAD_POOL_INFO, "statusCheckExecutor",
                    (ThreadPoolExecutor) AbstractEngineStatusSynchronizer.statusCheckExecutor);
            reportThreadPoolInfo(ENGINE_BALANCING_THREAD_POOL_INFO, "vipServerExecutor",
                    (ThreadPoolExecutor) WorkerAddressService.vipServerExecutor);
        }

        reportThreadPoolInfo(ENGINE_BALANCING_THREAD_POOL_INFO, "gRpcExecutor", (ThreadPoolExecutor) engineGrpcClient.getExecutor());
        reportEventLoopGroup(ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO, "gRpcEventLoopGroup", engineGrpcClient.getEventLoopGroup());
    }

    public void reportVipServerResult(String modelName, int result, String role) {
        ImmutableMetricTags metricTags = new ImmutableMetricTags("model", modelName, "role", role);
        monitor.report(ENGINE_NUMBER_VIP_RESULT, metricTags, result);
    }

    @PreDestroy
    public void destroy() {
        scheduler.shutdown();
    }

    public void reportStatusCheckRemoteInfo(String modelName, String engineIp, String role, Long startTime) {
        ImmutableMetricTags metricTags = new ImmutableMetricTags(
                "model", modelName,
                "engineIp", engineIp,
                "role", role);
        monitor.report(ENGINE_STATUS_VISITOR_RT, metricTags, System.currentTimeMillis() - startTime);
        monitor.report(ENGINE_STATUS_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportCacheStatusCheckRemoteInfo(String modelName, String engineIp, String role, Long startTime) {
        ImmutableMetricTags metricTags = new ImmutableMetricTags(
                "model", modelName,
                "engineIp", engineIp,
                "role", role);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_RT, metricTags, System.currentTimeMillis() - startTime);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportStatusCheckerFail(String modelName, BalanceStatusEnum errorEnum) {
        ImmutableMetricTags metricTags = new ImmutableMetricTags("model", modelName, "code", String.valueOf(errorEnum.getCode()));
        monitor.report(ENGINE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportCacheStatusCheckerFail(String modelName, String engineIp, BalanceStatusEnum errorEnum) {
        ImmutableMetricTags metricTags = new ImmutableMetricTags(
                "model", modelName,
                "engineIp", engineIp,
                "code", String.valueOf(errorEnum.getCode()));
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportStatusCheckerSuccess(String modelName, WorkerStatus workerStatus) {

        ImmutableMetricTags metricTags = new ImmutableMetricTags(
            "model", modelName,
            "code", "0",
            "engineIp", workerStatus.getIp(),
            "role", workerStatus.getRole());

        Long availableConcurrency = workerStatus.getAvailableConcurrency();
        if (availableConcurrency != null) {
            monitor.report(ENGINE_STATUS_AVAILABLE_CONCURRENCY, metricTags, availableConcurrency);
        }
        long lastUpdateTime = workerStatus.getLastUpdateTime().get();
        if (lastUpdateTime > 0) {
            monitor.report(ENGINE_STATUS_CHECK_SUCCESS_PERIOD, metricTags, System.currentTimeMillis() - lastUpdateTime);
        }
        monitor.report(ENGINE_RUNNING_QUEUE_TIME, metricTags, workerStatus.getRunningQueueTime().get());
    }

    public void reportCacheStatusCheckerSuccess(String modelName, WorkerStatus workerStatus) {
        long cacheLastUpdateTime = workerStatus.getCacheLastUpdateTime().get();
        if (cacheLastUpdateTime > 0) {
            ImmutableMetricTags metricTags = new ImmutableMetricTags(
                    "model", modelName,
                    "code", "0",
                    "engineIp", workerStatus.getIp(),
                    "role", workerStatus.getRole());
            monitor.report(CACHE_STATUS_CHECK_SUCCESS_PERIOD, metricTags, System.currentTimeMillis() - cacheLastUpdateTime);
        }
        if (workerStatus.getCacheStatus() != null) {
            long blockSize = workerStatus.getCacheStatus().getBlockSize();
            ImmutableMetricTags blockSizeTags = new ImmutableMetricTags(
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
            ImmutableMetricTags metricTags = new ImmutableMetricTags(
                    "model", ctx.getMasterRequest().getModel(),
                    "code", String.valueOf(ctx.getMasterResponse().getCode()));
            monitor.report(ENGINE_BALANCING_MASTER_FAIL_QPS, metricTags, 1.0);
            monitor.report(ENGINE_BALANCING_MASTER_ALL_QPS, metricTags, 1.0);
        } else {
            ImmutableMetricTags metricTags = new ImmutableMetricTags("model", ctx.getMasterRequest().getModel());
            monitor.report(ENGINE_BALANCING_MASTER_ALL_QPS, metricTags, 1.0);
            monitor.report(ENGINE_BALANCING_MASTER_SCHEDULE_RT, metricTags, System.currentTimeMillis() - ctx.getStartTime());
        }
    }

    public void reportPrefillBalanceSelectMetric(String modelName,
                                                 boolean success,
                                                 String errorCode,
                                                 long totalCost,
                                                 long tokenizeCost,
                                                 long calcPrefixCost,
                                                 long calcTtftCost) {
        ImmutableMetricTags metricTags = new ImmutableMetricTags(
                "model", modelName);
        monitor.report(PREFILL_BALANCE_SELECT_QPS, metricTags, 1.0);
        if (!success) {
            ImmutableMetricTags failMetricTags = new ImmutableMetricTags(
                    "model", modelName,
                    "code", errorCode);
            monitor.report(PREFILL_BALANCE_SELECT_FAIL_QPS, failMetricTags, 1.0);
        }

        if (totalCost > 0) {
            monitor.report(PREFILL_BALANCE_TOKENIZE_COST,
                    new ImmutableMetricTags(
                            "model", modelName,
                            "stage", "total"),
                    totalCost);
        }
        if (tokenizeCost > 0) {
            monitor.report(PREFILL_BALANCE_TOKENIZE_COST,
                    new ImmutableMetricTags(
                            "model", modelName,
                            "stage", "tokenize"),
                    tokenizeCost);
        }
        if (calcPrefixCost > 0) {
            monitor.report(PREFILL_BALANCE_TOKENIZE_COST,
                    new ImmutableMetricTags(
                            "model", modelName,
                            "stage", "calcPrefix"),
                    calcPrefixCost);
        }
        if (calcTtftCost > 0) {
            monitor.report(PREFILL_BALANCE_TOKENIZE_COST,
                    new ImmutableMetricTags(
                            "model", modelName,
                            "stage", "calcTtft"),
                    calcTtftCost);
        }

    }

    public void reportPrefillBalanceMasterNode(String master) {
        monitor.report(PREFILL_MASTER_NODE, new ImmutableMetricTags("masterNode", master), 1.0);
    }

    public void reportPrefillBalanceMasterEvent(String event) {
        monitor.report(PREFILL_MASTER_EVENT, new ImmutableMetricTags("event", event), 1.0);
    }

    public void reportThreadPoolInfo(String metricName, String name, ThreadPoolExecutor engineSyncExecutor) {
        if (engineSyncExecutor == null) {
            return;
        }

        Map<String, String> metricMap = new HashMap<>();
        metricMap.put("threadPool", name);

        metricMap.put("type", "executingTaskThreadSize");
        monitor.report(metricName, new ImmutableMetricTags(metricMap), engineSyncExecutor.getActiveCount());
        metricMap.put("type", "queueSize");
        monitor.report(metricName, new ImmutableMetricTags(metricMap), engineSyncExecutor.getQueue().size());
        metricMap.put("type", "corePoolSize");
        monitor.report(metricName, new ImmutableMetricTags(metricMap), engineSyncExecutor.getCorePoolSize());
        metricMap.put("type", "currentThreadSizeInPool");
        monitor.report(metricName, new ImmutableMetricTags(metricMap), engineSyncExecutor.getPoolSize());
    }

    private void reportEventLoopGroup(String metricName, String eventLoopGroupName, EventLoopGroup eventLoopGroup) {
        int totalActiveExecutorCount = 0;
        int totalPendingTask = 0;
        Map<String, Integer> pendingTasksMap = new HashMap<>();
        for (EventExecutor executor : eventLoopGroup) {
            boolean isShutdown = executor.isShutdown();
            boolean isTerminated = executor.isTerminated();
            boolean isShuttingDown = executor.isShuttingDown();
            // 记录 active 的 worker 数量
            if (!isShutdown && !isTerminated && !isShuttingDown) {
                totalActiveExecutorCount++;
            }
            if (executor instanceof SingleThreadEventExecutor) {
                SingleThreadEventExecutor singleThreadEventExecutor = (SingleThreadEventExecutor) executor;
                int pendingTasks = singleThreadEventExecutor.pendingTasks();
                totalPendingTask += pendingTasks;
                pendingTasksMap.put(singleThreadEventExecutor.threadProperties().name(), pendingTasks);
            }
        }
        Map<String, String> metricMap = new HashMap<>();
        metricMap.put("name", eventLoopGroupName);
        metricMap.put("type", "active-executor-count");
        monitor.report(metricName, new ImmutableMetricTags(metricMap), totalActiveExecutorCount);
        metricMap.put("type", "pending-task-total-count");
        monitor.report(metricName, new ImmutableMetricTags(metricMap), totalPendingTask);
        for (Map.Entry<String, Integer> pendingTaskEntry : pendingTasksMap.entrySet()) {
            metricMap.put("type", pendingTaskEntry.getKey() + "-pending-task-count");
            monitor.report(metricName, new ImmutableMetricTags(metricMap), pendingTaskEntry.getValue());
        }
    }

    public void reportCacheHitMetrics(String modelName, RoleType roleType, String engineIp, long hitTokens, double hitRatio) {
        cacheMetricsReporter.reportCacheHitMetrics(modelName, roleType, engineIp, hitTokens, hitRatio);
    }
}

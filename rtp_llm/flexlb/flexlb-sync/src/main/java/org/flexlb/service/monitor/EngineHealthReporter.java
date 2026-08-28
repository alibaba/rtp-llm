package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import io.netty.util.concurrent.EventExecutor;
import io.netty.util.concurrent.SingleThreadEventExecutor;
import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.FlexStatisticsType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.sync.status.WorkerDirectory;
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

import static org.flexlb.constant.MetricConstant.CACHE_AVAILABLE_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_BLOCK_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_KEY_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_VISITOR_RT;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS;
import static org.flexlb.constant.MetricConstant.CACHE_TOTAL_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_ALL_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_ALL_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_DECODE_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_FINISHED_TASK_LIST_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT;
import static org.flexlb.constant.MetricConstant.ENGINE_PREFILL_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_QUEUE_TIME;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_TASK_INFO_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_AVAILABLE_CONCURRENCY;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_VISITOR_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_VISITOR_SUCCESS_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_INFO_STEP_LATENCY_VAR;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.FORWARD_TO_MASTER_RESULT;
import static org.flexlb.constant.MetricConstant.GRPC_SERVER_PROCESS_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_ESTIMATED_TTFT_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_EXECUTION_TIME_MS;
import static org.flexlb.constant.MetricConstant.REQUEST_NETWORK_DELAY_MS;
import static org.flexlb.constant.MetricConstant.ZK_MASTER_EVENT;
import static org.flexlb.constant.MetricConstant.ZK_MASTER_NODE;

/**
 * Engine health reporter for monitoring engine status and metrics
 */
@Component
public class EngineHealthReporter {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final FlexMonitor monitor;

    private final CacheMetricsReporter cacheMetricsReporter;

    private final EngineGrpcClient engineGrpcClient;

    private final WorkerDirectory workerDirectory;

    private final Map<String, EventLoopGroup> eventLoopGroupMap;

    @Autowired
    public EngineHealthReporter(FlexMonitor monitor,
                                CacheMetricsReporter cacheMetricsReporter,
                                EngineGrpcClient engineGrpcClient,
                                LoopResources serverLoopResources,
                                WorkerDirectory workerDirectory) {
        this.monitor = monitor;
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.engineGrpcClient = engineGrpcClient;
        this.workerDirectory = workerDirectory;
        this.eventLoopGroupMap = Map.of(
                "serverWorker", serverLoopResources.onServer(true),
                "serverSelector", serverLoopResources.onServerSelect(true),
                "gRpcEventLoopGroup", engineGrpcClient.getEventLoopGroup()
        );
    }

    @PostConstruct
    public void init() {

        this.monitor.register(ENGINE_STATUS_CHECK_SUCCESS_PERIOD, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_AVAILABLE_CONCURRENCY, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_STATUS_VISITOR_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_STATUS_VISITOR_SUCCESS_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_PREFILL_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_DECODE_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_CHECK_FAIL, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_BALANCING_THREAD_POOL_INFO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_FINISHED_TASK_LIST_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_RUNNING_TASK_INFO_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_KEY_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        this.monitor.register(ENGINE_BALANCING_MASTER_ALL_QPS, FlexMetricType.QPS);
        this.monitor.register(ENGINE_BALANCING_MASTER_ALL_RT, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_BALANCING_MASTER_SELECT_DETAIL, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        this.monitor.register(ENGINE_RUNNING_QUEUE_TIME, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(PREFILL_SELECTED_ESTIMATED_TTFT_MS,
                FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        this.monitor.register(PREFILL_SELECTED_EXECUTION_TIME_MS,
                FlexMetricType.TIMER, FlexPriorityType.PRECISE);

        this.monitor.register(ZK_MASTER_NODE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ZK_MASTER_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        this.monitor.register(ENGINE_WORKER_INFO_STEP_LATENCY_VAR, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_RT, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, FlexMetricType.QPS);
        this.monitor.register(CACHE_STATUS_CHECK_SUCCESS_PERIOD, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_FAIL, FlexMetricType.QPS);
        this.monitor.register(CACHE_BLOCK_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_USED_KV_CACHE_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_AVAILABLE_KV_CACHE_TOKENS, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_TOTAL_KV_CACHE_TOKENS, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_USED_KV_CACHE_RATIO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(REQUEST_NETWORK_DELAY_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(GRPC_SERVER_PROCESS_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(FORWARD_TO_MASTER_RESULT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    public void reportStepLatencyVariance(
            String modelName, String role, double variance) {
        FlexMetricTags metricTags = FlexMetricTags.of("model", modelName, "role", role);
        monitor.report(ENGINE_WORKER_INFO_STEP_LATENCY_VAR, metricTags, variance);
        logger.debug("Step-latency variance - model: {}, role: {}, value: {}",
                modelName, role, variance);
    }

    public void reportRunningLoadVariance(
            String modelName, String role, double variance) {
        FlexMetricTags metricTags = FlexMetricTags.of("model", modelName, "role", role);
        monitor.report(ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR,
                metricTags, variance);
        logger.debug("Running-load variance - model: {}, role: {}, value: {}",
                modelName, role, variance);
    }

    @Scheduled(fixedRate = 2000)
    private void reportEngineMetric() {
        String modelName = "engine_service";
        FlexMetricTags tags = FlexMetricTags.of("model", modelName);
        monitor.report(ENGINE_WORKER_NUMBER, tags,
                workerDirectory.discoveredCount());
        monitor.report(ENGINE_PREFILL_WORKER_NUMBER, tags,
                workerDirectory.discoveredCount(RoleType.PREFILL));
        monitor.report(ENGINE_DECODE_WORKER_NUMBER, tags,
                workerDirectory.discoveredCount(RoleType.DECODE));

        if (AbstractEngineStatusSynchronizer.engineSyncExecutor != null
                && AbstractEngineStatusSynchronizer.statusCheckExecutor != null
                && WorkerAddressService.serviceDiscoveryExecutor != null) {
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

    public void reportStatusCheckRemoteInfo(String modelName, String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "role", role);
        monitor.report(ENGINE_STATUS_VISITOR_RT, metricTags, (double) System.nanoTime() / 1000 - startTime);
        monitor.report(ENGINE_STATUS_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportCacheStatusCheckRemoteInfo(String modelName, String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "role", role);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_RT, metricTags, (double) System.nanoTime() / 1000 - startTime);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportStatusCheckerFail(String modelName, BalanceStatusEnum errorEnum, RoleType role) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "code", String.valueOf(errorEnum.getCode()),
                "role", role == null ? "" : role.getCode()
        );
        monitor.report(ENGINE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportCacheStatusCheckerFail(String modelName, BalanceStatusEnum errorEnum, RoleType role) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "code", String.valueOf(errorEnum.getCode()),
                "role", role == null ? "" : role.getCode());
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportStatusCheckerSuccess(String modelName,
                                           WorkerStatus workerStatus,
                                           WorkerEndpoint ep,
                                           int runningTaskInfoSize,
                                           int finishedTaskListSize) {

        WorkerStatus.TopologySnapshot topology = workerStatus.topologySnapshot();
        WorkerStatus.EngineObservation status =
                workerStatus.committedEngineObservation();
        WorkerStatus.PollHealth pollHealth = workerStatus.pollHealth();

        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", topology.ip(),
                "role", status.role().name());

        Long availableConcurrency = status.availableConcurrency();
        if (availableConcurrency != null) {
            monitor.report(ENGINE_STATUS_AVAILABLE_CONCURRENCY, metricTags, availableConcurrency);
        }
        long pollIntervalUs = pollHealth.successfulPollIntervalUs();
        if (pollIntervalUs > 0) {
            monitor.report(ENGINE_STATUS_CHECK_SUCCESS_PERIOD,
                    metricTags, (double) pollIntervalUs);
        }
        if (ep != null) {
            ep.getLoadMetric().ifPresent(
                    value -> monitor.report(
                            ENGINE_RUNNING_QUEUE_TIME, metricTags, value));
        }

        monitor.report(ENGINE_FINISHED_TASK_LIST_SIZE, metricTags, finishedTaskListSize);
        monitor.report(ENGINE_RUNNING_TASK_INFO_SIZE, metricTags, runningTaskInfoSize);
    }

    public void reportCacheStatusCheckerSuccess(
            String modelName,
            WorkerStatus workerStatus,
            long successfulPollIntervalUs) {
        WorkerStatus.TopologySnapshot topology = workerStatus.topologySnapshot();
        WorkerStatus.EngineObservation status =
                workerStatus.committedEngineObservation();
        CacheStatus cacheStatus = workerStatus.getCacheStatus();
        if (successfulPollIntervalUs > 0L) {
            FlexMetricTags metricTags = FlexMetricTags.of(
                    "model", modelName,
                    "engineIp", topology.ip(),
                    "role", status.role().name());
            monitor.report(
                    CACHE_STATUS_CHECK_SUCCESS_PERIOD,
                    metricTags,
                    (double) successfulPollIntervalUs);
        }
        if (cacheStatus != null) {
            long blockSize = cacheStatus.getBlockSize();
            long cacheKeySize = cacheStatus.getCacheKeySize();
            FlexMetricTags roleMetricTags = FlexMetricTags.of(
                    "model", modelName,
                    "role", status.role().name());
            FlexMetricTags engineMetricTags = FlexMetricTags.of(
                    "model", modelName,
                    "engineIp", topology.ip(),
                    "role", status.role().name());
            monitor.report(CACHE_BLOCK_SIZE, roleMetricTags, blockSize);
            monitor.report(CACHE_KEY_SIZE, engineMetricTags, cacheKeySize);
        }

        long totalKvCacheTokens = status.totalKvCacheTokens();
        long availableKvCacheTokens = status.availableKvCacheTokens();
        long usedKvCacheTokens = totalKvCacheTokens - availableKvCacheTokens;

        FlexMetricTags kvCacheMetricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", topology.ip(),
                "role", status.role().name());

        monitor.report(CACHE_USED_KV_CACHE_TOKENS, kvCacheMetricTags, usedKvCacheTokens);
        monitor.report(CACHE_AVAILABLE_KV_CACHE_TOKENS, kvCacheMetricTags, availableKvCacheTokens);
        monitor.report(CACHE_TOTAL_KV_CACHE_TOKENS,
                FlexMetricTags.of("model", modelName, "role", status.role().name()),
                totalKvCacheTokens);
        if (totalKvCacheTokens > 0) {
            double usedRatio = (usedKvCacheTokens * 1.0 / totalKvCacheTokens) * 100;
            monitor.report(CACHE_USED_KV_CACHE_RATIO, kvCacheMetricTags, usedRatio);
        }
    }

    public void reportBalancingService(BalanceContext ctx) {
        if (ctx == null || ctx.getResponse() == null) {
            return;
        }

        FlexMetricTags metricTags = FlexMetricTags.of(
                "code", String.valueOf(ctx.getResponse().getCode()));
        monitor.report(ENGINE_BALANCING_MASTER_ALL_QPS, metricTags, 1.0);
        monitor.report(ENGINE_BALANCING_MASTER_ALL_RT, metricTags, System.currentTimeMillis() - ctx.getStartTime());

        // Report server selection results aggregated by role and outcome.
        if (ctx.getResponse() != null && CollectionUtils.isNotEmpty(ctx.getResponse().getServerStatus())) {
            boolean isSuccess = ctx.getResponse().isSuccess();
            int code = ctx.getResponse().getCode();

            for (ServerStatus serverStatus : ctx.getResponse().getServerStatus()) {
                if (serverStatus.getRole() != null) {
                    FlexMetricTags serverSelectionTags = FlexMetricTags.of(
                            "role", serverStatus.getRole().name(),
                            "success", String.valueOf(isSuccess),
                            "code", String.valueOf(code)
                    );
                    monitor.report(ENGINE_BALANCING_MASTER_SELECT_DETAIL, serverSelectionTags, 1.0);
                }
            }
        }
    }

    public void reportMasterNode(String master) {
        monitor.report(ZK_MASTER_NODE, FlexMetricTags.of("masterNode", master), 1.0);
    }

    public void reportPrefillBalanceMasterEvent(ZkMasterEvent event) {
        monitor.report(ZK_MASTER_EVENT, FlexMetricTags.of("event", event.name()), 1.0);
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
            // Record active worker count
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

    public void reportCacheHitMetrics(RoleType roleType, long hitTokens, double hitRatio) {
        cacheMetricsReporter.reportCacheHitMetrics(roleType, hitTokens, hitRatio);
    }

    /**
     * Report request-level estimates captured when a Prefill worker is selected.
     * These metrics are common to batch and non-batch delivery; the delivery mode
     * tag keeps their distributions independently queryable. One sample pair is
     * emitted per successful selection attempt, so retries and reroutes emit again.
     */
    public void reportPrefillSelectedEstimates(RoleType roleType,
                                               String engineIp,
                                               String deliveryMode,
                                               long estimatedTtftMs,
                                               long executionTimeMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", roleType.name(),
                "delivery_mode", deliveryMode);
        monitor.report(PREFILL_SELECTED_ESTIMATED_TTFT_MS, tags, estimatedTtftMs);
        monitor.report(PREFILL_SELECTED_EXECUTION_TIME_MS, tags, executionTimeMs);
    }

    /**
     * Delegate routing selected cache match metrics to {@link CacheMetricsReporter}.
     */
    public void reportRoutingSelectedCacheMatchMetrics(RoleType roleType,
                                                       long hitTokens,
                                                       long totalTokens) {
        cacheMetricsReporter.reportRoutingSelectedCacheMatchMetrics(roleType, hitTokens, totalTokens);
    }

    public void reportRoutingCandidateMaxCacheMatchMetrics(RoleType roleType,
                                                           long hitTokens) {
        cacheMetricsReporter.reportRoutingCandidateMaxCacheMatchMetrics(roleType, hitTokens);
    }

    public void reportCacheAffinityDecision(RoleType roleType,
                                            String engineIp,
                                            String decision) {
        cacheMetricsReporter.reportCacheAffinityDecision(roleType, engineIp, decision);
    }

    public void reportArriveDelayTime(BalanceContext ctx) {
        if (ctx.getRequest().getRequestTimeMs() == 0) {
            return;
        }
        long grpcEntryTime = ctx.getGrpcEntryTime();
        if (grpcEntryTime > 0) {
            long networkDelayMs = grpcEntryTime - ctx.getRequest().getRequestTimeMs();
            long grpcProcessMs = ctx.getStartTime() - grpcEntryTime;
            monitor.report(REQUEST_NETWORK_DELAY_MS, FlexMetricTags.of(), networkDelayMs);
            monitor.report(GRPC_SERVER_PROCESS_MS, FlexMetricTags.of(), grpcProcessMs);
        } else {
            // Fallback: if grpcEntryTime not set, report total delay as network delay
            long arrivalDelayMs = ctx.getStartTime() - ctx.getRequest().getRequestTimeMs();
            monitor.report(REQUEST_NETWORK_DELAY_MS, FlexMetricTags.of(), arrivalDelayMs);
        }
    }

    public void reportForwardToMasterResult(String type, String code) {
        monitor.report(FORWARD_TO_MASTER_RESULT, FlexMetricTags.of("type", type, "code", code), 1.0);
    }
}

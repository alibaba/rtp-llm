package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import io.netty.util.concurrent.EventExecutor;
import io.netty.util.concurrent.SingleThreadEventExecutor;
import lombok.Data;
import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
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

import static org.flexlb.constant.MetricConstant.CACHE_AVAILABLE_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_BLOCK_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_ACTUAL_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_DELTA_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_KVCM_P2P_TOTAL_MATCH_DELTA_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_PREDICTED_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_PREDICTED_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_KEY_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_LOCAL_STANDBY_BLOCK_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_VISITOR_RT;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS;
import static org.flexlb.constant.MetricConstant.CACHE_TOTAL_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_ALL_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SCHEDULE_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_DECODE_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_FINISHED_TASK_LIST_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_IN_TRANSIT_TASK_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_LOCAL_TASK_MAP_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT;
import static org.flexlb.constant.MetricConstant.ENGINE_PREFILL_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_QUEUE_TIME;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_TASK_INFO_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_AVAILABLE_CONCURRENCY;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL_TOTAL;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_VISITOR_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_VISITOR_SUCCESS_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_WAITING_TASK_INFO_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_INFO_STEP_LATENCY_VAR;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_MASTER_DECISION_TO_WAITING_CONFIRM_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_WAITING_TO_RUNNING_MS;
import static org.flexlb.constant.MetricConstant.FORWARD_TO_MASTER_RESULT;
import static org.flexlb.constant.MetricConstant.REQUEST_ARRIVAL_DELAY_MS;
import static org.flexlb.constant.MetricConstant.REQUEST_BODY_BYTES;
import static org.flexlb.constant.MetricConstant.REQUEST_INPUT_IDS_COUNT;
import static org.flexlb.constant.MetricConstant.ZK_MASTER_EVENT;
import static org.flexlb.constant.MetricConstant.ZK_MASTER_NODE;

/**
 * Engine health reporter for monitoring engine status and metrics
 */
@Data
@Component
public class EngineHealthReporter {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final FlexMonitor monitor;

    private final CacheMetricsReporter cacheMetricsReporter;

    private final CacheMatchConfiguration cacheMatchConfiguration;

    private final EngineGrpcClient engineGrpcClient;

    private final Map<String, EventLoopGroup> eventLoopGroupMap;

    @Autowired
    public EngineHealthReporter(FlexMonitor monitor,
                                CacheMetricsReporter cacheMetricsReporter,
                                CacheMatchConfiguration cacheMatchConfiguration,
                                EngineGrpcClient engineGrpcClient,
                                LoopResources serverLoopResources) {
        this.monitor = monitor;
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.cacheMatchConfiguration = cacheMatchConfiguration;
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
        this.monitor.register(ENGINE_STATUS_AVAILABLE_CONCURRENCY, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_STATUS_VISITOR_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_STATUS_VISITOR_SUCCESS_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_PREFILL_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_DECODE_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_CHECK_FAIL, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_STATUS_CHECK_FAIL_TOTAL, FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_STATUS_CHECK_FAIL_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_BALANCING_THREAD_POOL_INFO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_FINISHED_TASK_LIST_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_RUNNING_TASK_INFO_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_WAITING_TASK_INFO_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_KEY_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        this.monitor.register(ENGINE_BALANCING_MASTER_ALL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_BALANCING_MASTER_SCHEDULE_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_BALANCING_MASTER_SELECT_DETAIL, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        this.monitor.register(ENGINE_RUNNING_QUEUE_TIME, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_LOCAL_TASK_MAP_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_IN_TRANSIT_TASK_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        this.monitor.register(ZK_MASTER_NODE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ZK_MASTER_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        this.monitor.register(ENGINE_WORKER_INFO_STEP_LATENCY_VAR, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_MASTER_DECISION_TO_WAITING_CONFIRM_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_WAITING_TO_RUNNING_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_RT, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_STATUS_CHECK_SUCCESS_PERIOD, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_FAIL, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_BLOCK_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_LOCAL_STANDBY_BLOCK_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_HIT_COMPARISON_PREDICTED_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_ACTUAL_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_DELTA_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_KVCM_P2P_TOTAL_MATCH_DELTA_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_PREDICTED_RATIO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_ACTUAL_RATIO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_RATIO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_USED_KV_CACHE_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_AVAILABLE_KV_CACHE_TOKENS, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_TOTAL_KV_CACHE_TOKENS, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_USED_KV_CACHE_RATIO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(REQUEST_ARRIVAL_DELAY_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(REQUEST_INPUT_IDS_COUNT, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(REQUEST_BODY_BYTES, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(FORWARD_TO_MASTER_RESULT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    public void reportLatencyMetric(String modelName, String role, double result, double result2) {
        FlexMetricTags metricTags = FlexMetricTags.of("model", modelName, "role", role);
        monitor.report(ENGINE_WORKER_INFO_STEP_LATENCY_VAR, metricTags, result);
        monitor.report(ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR, metricTags, result2);
        logger.debug("Latency metric - model: {}, role: {}, stepLatency: {}, queryLen: {}", modelName, role, result, result2);
    }

    public void reportFlexlbObservedMasterDecisionToWaitingConfirmationLatency(String modelName,
                                                                               String engineIp,
                                                                               String role,
                                                                               String group,
                                                                               long latencyMs) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp,
                "role", role,
                "group", group);
        monitor.report(ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_MASTER_DECISION_TO_WAITING_CONFIRM_MS,
                metricTags, latencyMs);
    }

    public void reportFlexlbObservedWaitingToRunningLatency(String modelName,
                                                            String engineIp,
                                                            String role,
                                                            String group,
                                                            long latencyMs) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp,
                "role", role,
                "group", group);
        monitor.report(ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_WAITING_TO_RUNNING_MS, metricTags, latencyMs);
    }

    public void reportEngineObservedWaitingToRunningLatency(String modelName,
                                                            String engineIp,
                                                            String role,
                                                            String group,
                                                            long latencyMs) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp,
                "role", role,
                "group", group);
        monitor.report(ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS, metricTags, latencyMs);
    }

    public void reportEngineObservedReceivedToWaitingLatency(String modelName,
                                                             String engineIp,
                                                             String role,
                                                             String group,
                                                             long latencyMs) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp,
                "role", role,
                "group", group);
        monitor.report(ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS, metricTags, latencyMs);
    }

    @Scheduled(fixedRate = 2000)
    private void reportEngineMetric() {
        ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        if (modelWorkerStatus != null) {
            String modelName = "engine_service";
            FlexMetricTags tags = FlexMetricTags.of("model", modelName);
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
        // startTime and the reported value are both in microseconds.
        monitor.report(ENGINE_STATUS_VISITOR_RT, metricTags, (double) System.nanoTime() / 1000 - startTime);
        monitor.report(ENGINE_STATUS_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportCacheStatusCheckRemoteInfo(String modelName, String engineIp, String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp,
                "role", role);
        // startTime and the reported value are both in microseconds.
        monitor.report(CACHE_STATUS_CHECK_VISITOR_RT, metricTags, (double) System.nanoTime() / 1000 - startTime);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportStatusCheckerFail(String modelName, BalanceStatusEnum errorEnum, String ip, RoleType role) {
        FlexMetricTags metricTags = statusCheckFailureTags(modelName, errorEnum, ip, role);
        monitor.report(ENGINE_STATUS_CHECK_FAIL, metricTags, 1.0);
        monitor.report(ENGINE_STATUS_CHECK_FAIL_TOTAL, metricTags, 1.0);
    }

    /**
     * Reports only WorkerStatus RPC attempts that failed before a successful
     * response was available. Keeping it separate from visitor RT prevents
     * timeout latency from being hidden by successful probes.
     */
    public void reportStatusCheckFailureLatency(String modelName,
                                                BalanceStatusEnum errorEnum,
                                                String ip,
                                                RoleType role,
                                                long latencyUs) {
        FlexMetricTags metricTags = statusCheckFailureTags(modelName, errorEnum, ip, role);
        monitor.report(ENGINE_STATUS_CHECK_FAIL_RT, metricTags, latencyUs);
    }

    private FlexMetricTags statusCheckFailureTags(String modelName,
                                                   BalanceStatusEnum errorEnum,
                                                   String ip,
                                                   RoleType role) {
        return FlexMetricTags.of(
                "model", modelName,
                "code", String.valueOf(errorEnum.getCode()),
                "engineIp", ip == null ? "" : ip,
                "role", role == null ? "" : role.getCode());
    }

    public void reportCacheStatusCheckerFail(String modelName, String engineIp, BalanceStatusEnum errorEnum) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp,
                "code", String.valueOf(errorEnum.getCode()));
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportStatusCheckerSuccess(String modelName,
                                           WorkerStatus workerStatus,
                                           int waitingTaskInfoSize,
                                           int runningTaskInfoSize,
                                           int finishedTaskListSize) {

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
            monitor.report(ENGINE_STATUS_CHECK_SUCCESS_PERIOD, metricTags, (double) System.nanoTime() / 1000 - lastUpdateTime);
        }
        monitor.report(ENGINE_RUNNING_QUEUE_TIME, metricTags, workerStatus.getRunningQueueTime().get());

        // Report local task cache size
        int localTaskMapSize = workerStatus.getLocalTaskMap() != null ? workerStatus.getLocalTaskMap().size() : 0;
        monitor.report(ENGINE_LOCAL_TASK_MAP_SIZE, metricTags, localTaskMapSize);
        monitor.report(ENGINE_IN_TRANSIT_TASK_SIZE, metricTags, workerStatus.getInTransitTaskCount());

        reportCacheCapacityMetrics(modelName, workerStatus);

        metricTags = FlexMetricTags.of(
                "engineIp", workerStatus.getIp(),
                "role", workerStatus.getRole());

        monitor.report(ENGINE_FINISHED_TASK_LIST_SIZE, metricTags, finishedTaskListSize);
        monitor.report(ENGINE_WAITING_TASK_INFO_SIZE, metricTags, waitingTaskInfoSize);
        monitor.report(ENGINE_RUNNING_TASK_INFO_SIZE, metricTags, runningTaskInfoSize);
    }

    public void reportCacheStatusCheckerSuccess(String modelName, WorkerStatus workerStatus) {
        long cacheLastUpdateTime = workerStatus.getCacheLastUpdateTime().get();
        if (cacheLastUpdateTime > 0) {
            FlexMetricTags metricTags = FlexMetricTags.of(
                    "model", modelName,
                    "code", "0",
                    "engineIp", workerStatus.getIp(),
                    "role", workerStatus.getRole());
            monitor.report(CACHE_STATUS_CHECK_SUCCESS_PERIOD, metricTags, (double) System.nanoTime() / 1000 - cacheLastUpdateTime);
        }
        CacheStatus cacheStatus = workerStatus.getCacheStatus();
        if (cacheStatus != null) {
            // Cache key details are available only from the legacy GetCacheStatus response.
            FlexMetricTags metricTags = FlexMetricTags.of(
                    "model", modelName,
                    "engineIp", workerStatus.getIp(),
                    "role", workerStatus.getRole());
            monitor.report(CACHE_KEY_SIZE, metricTags, cacheStatus.getCacheKeySize());
        }
    }

    /**
     * Reports the shared capacity snapshot populated by either GetWorkerStatus or GetCacheStatus.
     */
    private void reportCacheCapacityMetrics(String modelName, WorkerStatus workerStatus) {
        CacheStatus cacheStatus = workerStatus.getCacheStatus();
        if (cacheStatus == null) {
            return;
        }

        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", workerStatus.getIp(),
                "role", workerStatus.getRole());
        if (cacheStatus.getBlockSize() > 0) {
            monitor.report(CACHE_BLOCK_SIZE, metricTags, cacheStatus.getBlockSize());
        }
        reportLocalStandbyBlockSize(metricTags, cacheStatus.getBlockSize());
        long usedKvCacheTokens = workerStatus.getUsedKvCacheTokens().get();
        long availableKvCacheTokens = workerStatus.getAvailableKvCacheTokens().get();
        long totalKvCacheTokens = usedKvCacheTokens + availableKvCacheTokens;

        monitor.report(CACHE_USED_KV_CACHE_TOKENS, metricTags, usedKvCacheTokens);
        monitor.report(CACHE_AVAILABLE_KV_CACHE_TOKENS, metricTags, availableKvCacheTokens);
        monitor.report(CACHE_TOTAL_KV_CACHE_TOKENS, metricTags, totalKvCacheTokens);
        if (totalKvCacheTokens > 0) {
            double usedRatio = (usedKvCacheTokens * 1.0 / totalKvCacheTokens) * 100;
            monitor.report(CACHE_USED_KV_CACHE_RATIO, metricTags, usedRatio);
        }
    }

    private void reportLocalStandbyBlockSize(FlexMetricTags metricTags, long engineBlockSize) {
        if (!cacheMatchConfiguration.isLocalStandbyEnabled()) {
            return;
        }
        LocalStandbyConfig localStandbyConfig = cacheMatchConfiguration.getLocalStandbyConfig();
        if (localStandbyConfig == null) {
            return;
        }
        long configuredBlockSize = localStandbyConfig.getBlockSize();
        long effectiveBlockSize = configuredBlockSize > 0 ? configuredBlockSize : engineBlockSize;
        if (effectiveBlockSize > 0) {
            monitor.report(CACHE_LOCAL_STANDBY_BLOCK_SIZE, metricTags, effectiveBlockSize);
        }
    }

    public void reportBalancingService(BalanceContext ctx) {
        if (ctx == null || ctx.getResponse() == null) {
            return;
        }

        FlexMetricTags metricTags = FlexMetricTags.of(
                "code", String.valueOf(ctx.getResponse().getCode()));
        monitor.report(ENGINE_BALANCING_MASTER_ALL_QPS, metricTags, 1.0);
        monitor.report(ENGINE_BALANCING_MASTER_SCHEDULE_RT, metricTags, System.currentTimeMillis() - ctx.getStartTime());

        // Report server status selection results (distinguished by roleType and ip)
        if (ctx.getResponse() != null && CollectionUtils.isNotEmpty(ctx.getResponse().getServerStatus())) {
            boolean isSuccess = ctx.getResponse().isSuccess();
            int code = ctx.getResponse().getCode();

            for (ServerStatus serverStatus : ctx.getResponse().getServerStatus()) {
                if (serverStatus.getRole() != null && serverStatus.getServerIp() != null) {
                    // Report specific server selection QPS
                    FlexMetricTags serverSelectionTags = FlexMetricTags.of(
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

    /**
     * Reports request payload dimensions captured at the HTTP boundary. The body size is taken
     * from Content-Length, so chunked requests without that header are intentionally omitted.
     */
    public void reportRequestPayload(BalanceContext ctx) {
        if (ctx == null) {
            return;
        }

        FlexMetricTags metricTags = FlexMetricTags.of("success", String.valueOf(ctx.isSuccess()));
        if (ctx.getInputIdsCount() != null) {
            monitor.report(REQUEST_INPUT_IDS_COUNT, metricTags, ctx.getInputIdsCount());
        }
        if (ctx.getRequestBodyBytes() != null) {
            monitor.report(REQUEST_BODY_BYTES, metricTags, ctx.getRequestBodyBytes());
        }
    }

    public void reportMasterNode(String master) {
        monitor.report(ZK_MASTER_NODE, FlexMetricTags.of("masterNode", master), 1.0);
    }

    public void reportPrefillBalanceMasterEvent(ZkMasterEvent event) {
        monitor.report(
                ZK_MASTER_EVENT,
                FlexMetricTags.of("event", event.name()),
                System.currentTimeMillis());
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

    public void reportCacheHitMetrics(RoleType roleType, String engineIp, long hitTokens, double hitRatio) {
        cacheMetricsReporter.reportCacheHitMetrics(roleType, engineIp, hitTokens, hitRatio);
    }

    public void reportKvcmSelectedMatch(RoleType roleType, String engineIp, long localMatchTokens,
                                        long p2pFetchTokens, long p2pTotalMatchTokens,
                                        boolean available) {
        if (!available) {
            return;
        }
        cacheMetricsReporter.reportKvcmSelectedMatch(
                roleType,
                engineIp,
                localMatchTokens,
                p2pFetchTokens,
                p2pTotalMatchTokens);
    }

    public void reportCacheHitComparisonMetrics(String modelName, CacheHitComparisonResult comparison) {
        if (comparison == null) {
            return;
        }
        CacheHitComparisonResult.HitComparison routing = comparison.routing();
        CacheHitComparisonResult.Actual actual = comparison.actual();
        CacheHitComparisonResult.KvcmDetails kvcmDetails = comparison.kvcmDetails();
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", comparison.worker(),
                "role", comparison.role(),
                "group", comparison.group(),
                "taskState", comparison.state(),
                "cacheMatchSource", comparison.source() == null ? "" : comparison.source());
        monitor.report(CACHE_HIT_COMPARISON_PREDICTED_TOKENS, metricTags, routing.hit());
        monitor.report(CACHE_HIT_COMPARISON_ACTUAL_TOKENS, metricTags, actual.hit());
        monitor.report(CACHE_HIT_COMPARISON_DELTA_TOKENS, metricTags, routing.delta());
        if (kvcmDetails != null) {
            monitor.report(CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS, metricTags, kvcmDetails.local().delta());
            monitor.report(CACHE_HIT_COMPARISON_KVCM_P2P_TOTAL_MATCH_DELTA_TOKENS, metricTags, kvcmDetails.p2pTotal().delta());
        }
        long inputTokens = comparison.inputTokens();
        if (inputTokens > 0) {
            monitor.report(CACHE_HIT_COMPARISON_PREDICTED_RATIO, metricTags, routing.hit() / (double) inputTokens);
            monitor.report(CACHE_HIT_COMPARISON_ACTUAL_RATIO, metricTags, actual.hit() / (double) inputTokens);
        }
        CacheHitComparisonResult.HitComparison localStandby = comparison.localStandby();
        if (localStandby != null) {
            monitor.report(CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_TOKENS, metricTags, localStandby.hit());
            monitor.report(CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS, metricTags, localStandby.delta());
            if (inputTokens > 0) {
                monitor.report(CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_RATIO, metricTags, localStandby.hit() / (double) inputTokens);
            }
        }
    }

    public void reportArriveDelayTime(BalanceContext ctx) {
        if (ctx.getRequest().getRequestTimeMs() == 0) {
            return;
        }
        monitor.report(REQUEST_ARRIVAL_DELAY_MS, FlexMetricTags.of(), ctx.getRequestArrivalDelayMs());
    }

    public void reportForwardToMasterResult(String type, String code) {
        monitor.report(FORWARD_TO_MASTER_RESULT, FlexMetricTags.of("type", type, "code", code), 1.0);
    }
}

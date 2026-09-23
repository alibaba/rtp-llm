package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import io.netty.util.concurrent.EventExecutor;
import io.netty.util.concurrent.SingleThreadEventExecutor;
import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.LocalStandbyConfig;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.FlexStatisticsType;
import org.flexlb.sync.status.WorkerDirectory;
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
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS;
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
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL_TOTAL;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_VISITOR_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_VISITOR_SUCCESS_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_INFO_STEP_LATENCY_VAR;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_MASTER_DECISION_TO_WAITING_CONFIRM_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_WAITING_TO_RUNNING_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_HBM_LOCAL_MATCH_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_INPUT_QUEUE_WAIT_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MIN;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_PREFILL_STEP_COUNT;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_REMOTE_KV_ADDED_MATCH_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_REMOTE_KV_WAIT_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_RUNNING_TO_FIRST_TOKEN_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_SCHEDULER_TO_RUNNING_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STATUS_SCHEDULER_WAIT_MS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STEP_BUDGET_FILL_RATIO;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STEP_PREFILL_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STEP_PREFILL_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STEP_TOKEN_BUDGET;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_STEP_TOTAL_SCHEDULED_TOKENS;
import static org.flexlb.constant.MetricConstant.FORWARD_TO_MASTER_RESULT;
import static org.flexlb.constant.MetricConstant.GRPC_SERVER_PROCESS_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_ESTIMATED_TTFT_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_EXECUTION_TIME_MS;
import static org.flexlb.constant.MetricConstant.REQUEST_BODY_BYTES;
import static org.flexlb.constant.MetricConstant.REQUEST_INPUT_IDS_COUNT;
import static org.flexlb.constant.MetricConstant.REQUEST_MESSAGE_BYTES;
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

    private final CacheMatchConfiguration cacheMatchConfiguration;

    private final EngineGrpcClient engineGrpcClient;

    private final WorkerDirectory workerDirectory;

    private final Map<String, EventLoopGroup> eventLoopGroupMap;

    @Autowired
    public EngineHealthReporter(FlexMonitor monitor,
                                CacheMetricsReporter cacheMetricsReporter,
                                CacheMatchConfiguration cacheMatchConfiguration,
                                EngineGrpcClient engineGrpcClient,
                                LoopResources serverLoopResources,
                                WorkerDirectory workerDirectory) {
        this.monitor = monitor;
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.cacheMatchConfiguration = cacheMatchConfiguration;
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

        monitor.register(ENGINE_WORKER_STEP_TOTAL_SCHEDULED_TOKENS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        monitor.register(ENGINE_WORKER_STEP_PREFILL_REQUEST_COUNT,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        monitor.register(ENGINE_WORKER_STEP_PREFILL_TOKENS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        monitor.register(ENGINE_WORKER_STEP_TOKEN_BUDGET,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        monitor.register(ENGINE_WORKER_STEP_BUDGET_FILL_RATIO,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);

        this.monitor.register(ENGINE_STATUS_CHECK_SUCCESS_PERIOD, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_AVAILABLE_CONCURRENCY, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_STATUS_VISITOR_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_STATUS_VISITOR_SUCCESS_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_PREFILL_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_DECODE_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_STATUS_CHECK_FAIL, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_STATUS_CHECK_FAIL_TOTAL,
                FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_STATUS_CHECK_FAIL_RT,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
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
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(PREFILL_SELECTED_EXECUTION_TIME_MS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

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
        this.monitor.register(ENGINE_WORKER_STATUS_INPUT_QUEUE_WAIT_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_SCHEDULER_TO_RUNNING_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_SCHEDULER_WAIT_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_REMOTE_KV_WAIT_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_RUNNING_TO_FIRST_TOKEN_MS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_HBM_LOCAL_MATCH_TOKENS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_REMOTE_KV_ADDED_MATCH_TOKENS,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_PREFILL_STEP_COUNT,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MIN,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_RT, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, FlexMetricType.QPS);
        this.monitor.register(CACHE_STATUS_CHECK_SUCCESS_PERIOD, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_FAIL, FlexMetricType.QPS);
        this.monitor.register(CACHE_BLOCK_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_LOCAL_STANDBY_BLOCK_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_HIT_COMPARISON_PREDICTED_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_ACTUAL_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_PREDICTED_RATIO,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_ACTUAL_RATIO,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_RATIO,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_USED_KV_CACHE_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_AVAILABLE_KV_CACHE_TOKENS, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_TOTAL_KV_CACHE_TOKENS, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_USED_KV_CACHE_RATIO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(REQUEST_NETWORK_DELAY_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(GRPC_SERVER_PROCESS_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(REQUEST_INPUT_IDS_COUNT,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(REQUEST_MESSAGE_BYTES,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(REQUEST_BODY_BYTES,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
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

    public void reportStatusCheckRemoteInfo(
            String modelName, String engineIp, String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp == null ? "" : engineIp,
                "role", role);
        monitor.report(ENGINE_STATUS_VISITOR_RT, metricTags,
                (double) System.nanoTime() / 1000 - startTime);
        monitor.report(ENGINE_STATUS_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportCacheStatusCheckRemoteInfo(String modelName, String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "role", role);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_RT, metricTags, (double) System.nanoTime() / 1000 - startTime);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportCacheStatusCheckRemoteInfo(
            String modelName, String engineIp, String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp == null ? "" : engineIp,
                "role", role);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_RT, metricTags,
                (double) System.nanoTime() / 1000 - startTime);
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

    public void reportStatusCheckerFail(
            String modelName, BalanceStatusEnum errorEnum, String engineIp, RoleType role) {
        FlexMetricTags metricTags = statusCheckFailureTags(
                modelName, errorEnum, engineIp, role);
        monitor.report(ENGINE_STATUS_CHECK_FAIL, metricTags, 1.0);
        monitor.report(ENGINE_STATUS_CHECK_FAIL_TOTAL, metricTags, 1.0);
    }

    public void reportStatusCheckFailureLatency(
            String modelName, BalanceStatusEnum errorEnum,
            String engineIp, RoleType role, long latencyUs) {
        monitor.report(ENGINE_STATUS_CHECK_FAIL_RT,
                statusCheckFailureTags(modelName, errorEnum, engineIp, role), latencyUs);
    }

    private static FlexMetricTags statusCheckFailureTags(
            String modelName, BalanceStatusEnum errorEnum,
            String engineIp, RoleType role) {
        return FlexMetricTags.of(
                "model", modelName,
                "code", String.valueOf(errorEnum.getCode()),
                "engineIp", engineIp == null ? "" : engineIp,
                "role", role == null ? "" : role.getCode());
    }

    public void reportCacheStatusCheckerFail(String modelName, BalanceStatusEnum errorEnum, RoleType role) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "code", String.valueOf(errorEnum.getCode()),
                "role", role == null ? "" : role.getCode());
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportCacheStatusCheckerFail(String modelName,
                                             WorkerStatus workerStatus,
                                             BalanceStatusEnum errorEnum) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", workerStatus.getMetricIpPort(),
                "code", String.valueOf(errorEnum.getCode()),
                "role", workerStatus.getRole().getCode());
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportRequestPayload(BalanceContext context) {
        if (context == null) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                "success", String.valueOf(context.isSuccess()));
        if (context.getInputIdsCount() != null) {
            monitor.report(REQUEST_INPUT_IDS_COUNT, tags, context.getInputIdsCount());
        }
        if (context.getRequestMessageBytes() != null) {
            monitor.report(REQUEST_MESSAGE_BYTES, tags, context.getRequestMessageBytes());
        }
        if (context.getRequestBodyBytes() != null) {
            monitor.report(REQUEST_BODY_BYTES, tags, context.getRequestBodyBytes());
        }
    }

    public void reportFlexlbObservedMasterDecisionToWaitingConfirmationLatency(
            String modelName, String engineIp, String role, String group, long latencyMs) {
        monitor.report(ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_MASTER_DECISION_TO_WAITING_CONFIRM_MS,
                lifecycleTags(modelName, engineIp, role, group), latencyMs);
    }

    public void reportFlexlbObservedWaitingToRunningLatency(
            String modelName, String engineIp, String role, String group, long latencyMs) {
        monitor.report(ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_WAITING_TO_RUNNING_MS,
                lifecycleTags(modelName, engineIp, role, group), latencyMs);
    }

    public void reportEngineObservedWaitingToRunningLatency(
            String modelName, String engineIp, String role, String group, long latencyMs) {
        monitor.report(ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS,
                lifecycleTags(modelName, engineIp, role, group), latencyMs);
    }

    public void reportEngineObservedReceivedToWaitingLatency(
            String modelName, String engineIp, String role, String group, long latencyMs) {
        monitor.report(ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS,
                lifecycleTags(modelName, engineIp, role, group), latencyMs);
    }

    public void reportPrefillWorkerStatusTask(
            String modelName, String engineIp, String role, String group, TaskInfo task) {
        FlexMetricTags tags = lifecycleTags(modelName, engineIp, role, group);
        monitor.report(ENGINE_WORKER_STATUS_HBM_LOCAL_MATCH_TOKENS,
                tags, task.getHbmLocalMatchTokens());
        monitor.report(ENGINE_WORKER_STATUS_REMOTE_KV_ADDED_MATCH_TOKENS,
                tags, task.getRemoteKvAddedMatchTokens());
        monitor.report(ENGINE_WORKER_STATUS_PREFILL_STEP_COUNT,
                tags, task.getPrefillStepCount());
        monitor.report(ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MIN,
                tags, task.getPrefillNonfinalChunkTokensMin());
        monitor.report(ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX,
                tags, task.getPrefillNonfinalChunkTokensMax());
        reportDuration(ENGINE_WORKER_STATUS_INPUT_QUEUE_WAIT_MS, tags,
                task.getInputQueueDrainTimeMs(), task.getInputQueueEnqueueTimeMs());
        monitor.report(ENGINE_WORKER_STATUS_REMOTE_KV_WAIT_MS,
                tags, task.getRemoteKvWaitMs());
        long schedulerToRunningMs = reportDuration(
                ENGINE_WORKER_STATUS_SCHEDULER_TO_RUNNING_MS, tags,
                task.getRunningEnteredTimeMs(), task.getWaitingEnteredTimeMs());
        if (schedulerToRunningMs >= 0L) {
            monitor.report(ENGINE_WORKER_STATUS_SCHEDULER_WAIT_MS,
                    tags, Math.max(0L, schedulerToRunningMs - task.getRemoteKvWaitMs()));
        }
        reportDuration(ENGINE_WORKER_STATUS_RUNNING_TO_FIRST_TOKEN_MS, tags,
                task.getFirstTokenTimeMs(), task.getRunningEnteredTimeMs());
    }

    private static FlexMetricTags lifecycleTags(
            String modelName, String engineIp, String role, String group) {
        return FlexMetricTags.of(
                "model", modelName,
                "engineIp", engineIp,
                "role", role,
                "group", group);
    }

    private long reportDuration(
            String metric, FlexMetricTags tags, long endTimeMs, long startTimeMs) {
        if (endTimeMs <= 0L || startTimeMs <= 0L) {
            return -1L;
        }
        long durationMs = Math.max(0L, endTimeMs - startTimeMs);
        monitor.report(metric, tags, durationMs);
        return durationMs;
    }

    public void reportWorkerStepMetrics(
            String modelName,
            WorkerStatus worker,
            WorkerStatus.StepMetrics step) {
        FlexMetricTags tags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", worker.getMetricIpPort(),
                "role", worker.getRole().name(),
                "group", worker.topologySnapshot().group(),
                "phase", step.prefillRequestCount() > 0 ? "prefill" : "decode");
        monitor.report(ENGINE_WORKER_STEP_TOTAL_SCHEDULED_TOKENS,
                tags, step.totalScheduledTokens());
        monitor.report(ENGINE_WORKER_STEP_PREFILL_REQUEST_COUNT,
                tags, step.prefillRequestCount());
        monitor.report(ENGINE_WORKER_STEP_PREFILL_TOKENS,
                tags, step.prefillTokens());
        monitor.report(ENGINE_WORKER_STEP_TOKEN_BUDGET,
                tags, step.tokenBudget());
        monitor.report(ENGINE_WORKER_STEP_BUDGET_FILL_RATIO,
                tags, step.budgetFillRatio());
    }

    public void reportStatusCheckerSuccess(String modelName,
                                           WorkerStatus workerStatus,
                                           WorkerEndpoint ep,
                                           int runningTaskInfoSize,
                                           int finishedTaskListSize) {

        WorkerStatus.EngineObservation status =
                workerStatus.committedEngineObservation();
        WorkerStatus.PollHealth pollHealth = workerStatus.pollHealth();

        FlexMetricTags metricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", workerStatus.getMetricIpPort(),
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
        WorkerStatus.EngineObservation status =
                workerStatus.committedEngineObservation();
        CacheStatus cacheStatus = workerStatus.getCacheStatus();
        if (successfulPollIntervalUs > 0L) {
            FlexMetricTags metricTags = FlexMetricTags.of(
                    "model", modelName,
                    "engineIp", workerStatus.getMetricIpPort(),
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
                    "engineIp", workerStatus.getMetricIpPort(),
                    "role", status.role().name());
            monitor.report(CACHE_BLOCK_SIZE, roleMetricTags, blockSize);
            monitor.report(CACHE_KEY_SIZE, engineMetricTags, cacheKeySize);
            reportLocalStandbyBlockSize(engineMetricTags, blockSize);
        }

        long totalKvCacheTokens = status.totalKvCacheTokens();
        long availableKvCacheTokens = status.availableKvCacheTokens();
        long usedKvCacheTokens = totalKvCacheTokens - availableKvCacheTokens;

        FlexMetricTags kvCacheMetricTags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", workerStatus.getMetricIpPort(),
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

    private void reportLocalStandbyBlockSize(
            FlexMetricTags metricTags, long engineBlockSize) {
        if (!cacheMatchConfiguration.isLocalStandbyEnabled()) {
            return;
        }
        LocalStandbyConfig localStandbyConfig =
                cacheMatchConfiguration.getLocalStandbyConfig();
        if (localStandbyConfig == null) {
            return;
        }
        long configuredBlockSize = localStandbyConfig.getBlockSize();
        long effectiveBlockSize = configuredBlockSize > 0
                ? configuredBlockSize : engineBlockSize;
        if (effectiveBlockSize > 0) {
            monitor.report(CACHE_LOCAL_STANDBY_BLOCK_SIZE,
                    metricTags, effectiveBlockSize);
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
                            "reason", selectionReason(ctx, serverStatus.getRole()),
                            "engineIp", serverStatus.getMetricIpPort(),
                            "success", String.valueOf(isSuccess),
                            "code", String.valueOf(code)
                    );
                    monitor.report(ENGINE_BALANCING_MASTER_SELECT_DETAIL, serverSelectionTags, 1.0);
                }
            }
        }
    }

    private static String selectionReason(
            BalanceContext context, RoleType roleType) {
        String selectionReason = context.selectionReason(roleType);
        return selectionReason == null ? "UNKNOWN" : selectionReason;
    }

    public void reportMasterNode(String master) {
        monitor.report(ZK_MASTER_NODE, FlexMetricTags.of("masterNode", master), 1.0);
    }

    public void reportPrefillBalanceMasterEvent(ZkMasterEvent event) {
        monitor.report(ZK_MASTER_EVENT, FlexMetricTags.of("event", event.name()),
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

    public void reportCacheHitMetrics(
            RoleType roleType, String ipIndex, long hitTokens, double hitRatio) {
        cacheMetricsReporter.reportCacheHitMetrics(roleType, ipIndex, hitTokens, hitRatio);
    }

    /** Report request-level estimates captured when a Prefill worker is selected. */
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

    /** Delegate a cache-affinity routing decision to the cache metric reporter. */
    public void reportCacheAffinityDecision(RoleType roleType,
                                            String engineIp,
                                            String decision) {
        cacheMetricsReporter.reportCacheAffinityDecision(roleType, engineIp, decision);
    }

    public void reportKvcmSelectedMatch(RoleType roleType,
                                        String engineIp,
                                        long localMatchTokens,
                                        long globalMatchTokens,
                                        boolean available) {
        if (available) {
            cacheMetricsReporter.reportKvcmSelectedMatch(
                    roleType, engineIp, localMatchTokens, globalMatchTokens);
        }
    }

    public void reportCacheHitComparisonMetrics(
            String modelName, CacheHitComparisonResult comparison) {
        if (comparison == null) {
            return;
        }
        CacheHitComparisonResult.HitComparison routing = comparison.routing();
        CacheHitComparisonResult.Actual actual = comparison.actual();
        CacheHitComparisonResult.KvcmDetails kvcmDetails = comparison.kvcmDetails();
        FlexMetricTags tags = FlexMetricTags.of(
                "model", modelName,
                "engineIp", comparison.worker(),
                "role", comparison.role(),
                "group", comparison.group(),
                "taskState", comparison.state(),
                "cacheMatchSource", comparison.source() == null ? "" : comparison.source());
        monitor.report(CACHE_HIT_COMPARISON_PREDICTED_TOKENS, tags, routing.hit());
        monitor.report(CACHE_HIT_COMPARISON_ACTUAL_TOKENS, tags, actual.hit());
        monitor.report(CACHE_HIT_COMPARISON_DELTA_TOKENS, tags, routing.delta());
        if (kvcmDetails != null) {
            monitor.report(CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS,
                    tags, kvcmDetails.local().delta());
            monitor.report(CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS,
                    tags, kvcmDetails.global().delta());
        }
        long inputTokens = comparison.inputTokens();
        if (inputTokens > 0) {
            monitor.report(CACHE_HIT_COMPARISON_PREDICTED_RATIO,
                    tags, routing.hit() / (double) inputTokens);
            monitor.report(CACHE_HIT_COMPARISON_ACTUAL_RATIO,
                    tags, actual.hit() / (double) inputTokens);
        }
        CacheHitComparisonResult.HitComparison localStandby = comparison.localStandby();
        if (localStandby != null) {
            monitor.report(CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_TOKENS,
                    tags, localStandby.hit());
            monitor.report(CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS,
                    tags, localStandby.delta());
            if (inputTokens > 0) {
                monitor.report(CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_RATIO,
                        tags, localStandby.hit() / (double) inputTokens);
            }
        }
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

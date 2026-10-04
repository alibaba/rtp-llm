package org.flexlb.service.monitor;

import io.netty.channel.EventLoopGroup;
import io.netty.util.concurrent.EventExecutor;
import io.netty.util.concurrent.SingleThreadEventExecutor;
import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
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
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_INPUT_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_KVCM_PREDICTED_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_KEY_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_VISITOR_RT;
import static org.flexlb.constant.MetricConstant.CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS;
import static org.flexlb.constant.MetricConstant.CACHE_TOTAL_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_USED_KV_CACHE_TOKENS;
import static org.flexlb.constant.MetricConstant.ENCODER_PENDING_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.ENCODER_SELECTION_LOAD;
import static org.flexlb.constant.MetricConstant.ENCODER_UNCACHED_TOKEN_LOAD;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_ALL_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_ALL_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_DECODE_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_ENCODER_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_FINISHED_TASK_LIST_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_PREFILL_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_QUEUE_TIME;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_TASK_INFO_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_FAIL_TOTAL;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_CHECK_SUCCESS_PERIOD;
import static org.flexlb.constant.MetricConstant.ENGINE_STATUS_VISITOR_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR;
import static org.flexlb.constant.MetricConstant.ENGINE_WORKER_INFO_STEP_LATENCY_VAR;
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
import static org.flexlb.constant.MetricConstant.REQUEST_BLOCK_SIZE;
import static org.flexlb.constant.MetricConstant.REQUEST_BODY_BYTES;
import static org.flexlb.constant.MetricConstant.REQUEST_MESSAGE_BYTES;
import static org.flexlb.constant.MetricConstant.REQUEST_NETWORK_DELAY_MS;
import static org.flexlb.constant.MetricConstant.REQUEST_SEQ_LEN;
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
        this.monitor.register(ENGINE_STATUS_VISITOR_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_PREFILL_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_DECODE_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_ENCODER_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENCODER_PENDING_REQUEST_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENCODER_SELECTION_LOAD, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ENCODER_UNCACHED_TOKEN_LOAD, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
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
                FlexMetricType.GAUGE, FlexPriorityType.TRIVIAL);
        this.monitor.register(ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX,
                FlexMetricType.GAUGE, FlexPriorityType.TRIVIAL);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_RT, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, FlexMetricType.QPS);
        this.monitor.register(CACHE_STATUS_CHECK_SUCCESS_PERIOD, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_STATUS_CHECK_FAIL, FlexMetricType.QPS);
        this.monitor.register(CACHE_BLOCK_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_HIT_COMPARISON_ACTUAL_TOKENS,
                FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_KVCM_PREDICTED_TOKENS,
                FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_INPUT_TOKENS,
                FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_RATIO,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_HIT_COMPARISON_ACTUAL_RATIO,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_USED_KV_CACHE_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(CACHE_AVAILABLE_KV_CACHE_TOKENS, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_TOTAL_KV_CACHE_TOKENS, FlexMetricType.GAUGE);
        this.monitor.register(CACHE_USED_KV_CACHE_RATIO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(REQUEST_NETWORK_DELAY_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(GRPC_SERVER_PROCESS_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(REQUEST_BLOCK_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(REQUEST_SEQ_LEN,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(REQUEST_MESSAGE_BYTES,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(REQUEST_BODY_BYTES,
                FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(FORWARD_TO_MASTER_RESULT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    public void reportStepLatencyVariance(
            String role, double variance) {
        FlexMetricTags metricTags = FlexMetricTags.of("role", role);
        monitor.report(ENGINE_WORKER_INFO_STEP_LATENCY_VAR, metricTags, variance);
        logger.debug("Step-latency variance - role: {}, value: {}",
                role, variance);
    }

    public void reportRunningLoadVariance(
            String role, double variance) {
        FlexMetricTags metricTags = FlexMetricTags.of("role", role);
        monitor.report(ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR,
                metricTags, variance);
        logger.debug("Running-load variance - role: {}, value: {}",
                role, variance);
    }

    @Scheduled(fixedRate = 2000)
    private void reportEngineMetric() {
        FlexMetricTags tags = FlexMetricTags.of();
        monitor.report(ENGINE_PREFILL_WORKER_NUMBER, tags,
                workerDirectory.discoveredCount(RoleType.PREFILL));
        monitor.report(ENGINE_DECODE_WORKER_NUMBER, tags,
                workerDirectory.discoveredCount(RoleType.DECODE));
        monitor.report(ENGINE_ENCODER_WORKER_NUMBER, tags,
                workerDirectory.discoveredCount(RoleType.ENCODER));

        reportThreadPoolInfo(ENGINE_BALANCING_THREAD_POOL_INFO, "gRpcExecutor", (ThreadPoolExecutor) engineGrpcClient.getExecutor());

        eventLoopGroupMap.forEach(this::reportEventLoopGroup);
    }

    @Scheduled(fixedRate = 2000)
    private void reportWorkerBlockSizes() {
        reportWorkerBlockSize(RoleType.PREFILL);
        reportWorkerBlockSize(RoleType.DECODE);
    }

    private void reportWorkerBlockSize(RoleType role) {
        for (WorkerStatus worker : workerDirectory.getWorkerStatuses(role, null)) {
            long blockSize = worker.committedEngineObservation().blockSize();
            if (blockSize > 0) {
                monitor.report(CACHE_BLOCK_SIZE, FlexMetricTags.of("role", role.name()), blockSize);
                return;
            }
        }
    }

    public void reportStatusCheckRemoteInfo(String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "role", role);
        monitor.report(ENGINE_STATUS_VISITOR_RT, metricTags, (double) System.nanoTime() / 1000 - startTime);
    }

    public void reportStatusCheckRemoteInfo(
            String engineIp, String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "engineIp", engineIp == null ? "" : engineIp,
                "role", role);
        monitor.report(ENGINE_STATUS_VISITOR_RT, metricTags,
                (double) System.nanoTime() / 1000 - startTime);
    }

    public void reportCacheStatusCheckRemoteInfo(String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "role", role);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_RT, metricTags, (double) System.nanoTime() / 1000 - startTime);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportCacheStatusCheckRemoteInfo(
            String engineIp, String role, Long startTime) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "engineIp", engineIp == null ? "" : engineIp,
                "role", role);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_RT, metricTags,
                (double) System.nanoTime() / 1000 - startTime);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportStatusCheckerFail(BalanceStatusEnum errorEnum, RoleType role) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "code", String.valueOf(errorEnum.getCode()),
                "role", role == null ? "" : role.getCode()
        );
        monitor.report(ENGINE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportStatusCheckerFail(
            BalanceStatusEnum errorEnum, String engineIp, RoleType role) {
        FlexMetricTags metricTags = statusCheckFailureTags(
                errorEnum, engineIp, role);
        monitor.report(ENGINE_STATUS_CHECK_FAIL, metricTags, 1.0);
        monitor.report(ENGINE_STATUS_CHECK_FAIL_TOTAL, metricTags, 1.0);
    }

    public void reportStatusCheckFailureLatency(
            BalanceStatusEnum errorEnum,
            String engineIp, RoleType role, long latencyUs) {
        monitor.report(ENGINE_STATUS_CHECK_FAIL_RT,
                statusCheckFailureTags(errorEnum, engineIp, role), latencyUs);
    }

    private static FlexMetricTags statusCheckFailureTags(
            BalanceStatusEnum errorEnum,
            String engineIp, RoleType role) {
        return FlexMetricTags.of(
                "code", String.valueOf(errorEnum.getCode()),
                "engineIp", engineIp == null ? "" : engineIp,
                "role", role == null ? "" : role.getCode());
    }

    public void reportCacheStatusCheckerFail(BalanceStatusEnum errorEnum, RoleType role) {
        FlexMetricTags metricTags = FlexMetricTags.of(
                "code", String.valueOf(errorEnum.getCode()),
                "role", role == null ? "" : role.getCode());
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportCacheStatusCheckerFail(WorkerStatus workerStatus,
                                             BalanceStatusEnum errorEnum) {
        RoleType role = workerStatus.getRole();
        FlexMetricTags metricTags = FlexMetricTags.of(
                "engineIp", workerStatus.getMetricIpPort(),
                "code", String.valueOf(errorEnum.getCode()),
                "role", role == null ? "" : role.getCode());
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportRequestPayload(BalanceContext context) {
        if (context == null) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                "success", String.valueOf(context.isSuccess()));
        if (context.getRequest() != null) {
            monitor.report(REQUEST_SEQ_LEN, tags, context.getRequest().getSeqLen());
            monitor.report(REQUEST_BLOCK_SIZE, tags, context.getRequest().getCacheKeyBlockSize());
        }
        if (context.getRequestMessageBytes() != null) {
            monitor.report(REQUEST_MESSAGE_BYTES, tags, context.getRequestMessageBytes());
        }
        if (context.getRequestBodyBytes() != null) {
            monitor.report(REQUEST_BODY_BYTES, tags, context.getRequestBodyBytes());
        }
    }

    public void reportFlexlbObservedMasterDecisionToWaitingConfirmationLatency(
            String engineIp, String role, String group, long latencyMs) {
        monitor.report(ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_MASTER_DECISION_TO_WAITING_CONFIRM_MS,
                lifecycleTags(engineIp, role, group), latencyMs);
    }

    public void reportFlexlbObservedWaitingToRunningLatency(
            String engineIp, String role, String group, long latencyMs) {
        monitor.report(ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_WAITING_TO_RUNNING_MS,
                lifecycleTags(engineIp, role, group), latencyMs);
    }

    public void reportEngineObservedWaitingToRunningLatency(
            String engineIp, String role, String group, long latencyMs) {
        monitor.report(ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS,
                lifecycleTags(engineIp, role, group), latencyMs);
    }

    public void reportEngineObservedReceivedToWaitingLatency(
            String engineIp, String role, String group, long latencyMs) {
        monitor.report(ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS,
                lifecycleTags(engineIp, role, group), latencyMs);
    }

    public void reportPrefillWorkerStatusTask(
            String engineIp, String role, String group,
            WorkerStatus.TaskTelemetry task) {
        FlexMetricTags tags = lifecycleTags(engineIp, role, group);
        monitor.report(ENGINE_WORKER_STATUS_HBM_LOCAL_MATCH_TOKENS,
                tags, task.hbmLocalMatchTokens());
        monitor.report(ENGINE_WORKER_STATUS_REMOTE_KV_ADDED_MATCH_TOKENS,
                tags, task.remoteKvAddedMatchTokens());
        monitor.report(ENGINE_WORKER_STATUS_PREFILL_STEP_COUNT,
                tags, task.prefillStepCount());
        // Zero means the request has no nonfinal chunk sample.
        if (task.prefillNonfinalChunkTokensMin() > 0 && task.prefillNonfinalChunkTokensMax() > 0) {
            monitor.report(ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MIN,
                    tags, task.prefillNonfinalChunkTokensMin());
            monitor.report(ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX,
                    tags, task.prefillNonfinalChunkTokensMax());
        }
        reportDuration(ENGINE_WORKER_STATUS_INPUT_QUEUE_WAIT_MS, tags,
                task.inputQueueDrainTimeMs(), task.inputQueueEnqueueTimeMs());
        monitor.report(ENGINE_WORKER_STATUS_REMOTE_KV_WAIT_MS,
                tags, task.remoteKvWaitMs());
        long schedulerToRunningMs = reportDuration(
                ENGINE_WORKER_STATUS_SCHEDULER_TO_RUNNING_MS, tags,
                task.runningEnteredTimeMs(), task.waitingEnteredTimeMs());
        if (schedulerToRunningMs >= 0L) {
            monitor.report(ENGINE_WORKER_STATUS_SCHEDULER_WAIT_MS,
                    tags, Math.max(0L, schedulerToRunningMs - task.remoteKvWaitMs()));
        }
        reportDuration(ENGINE_WORKER_STATUS_RUNNING_TO_FIRST_TOKEN_MS, tags,
                task.firstTokenTimeMs(), task.runningEnteredTimeMs());
        reportDuration(ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS,
                tags, task.waitingEnteredTimeMs(), task.requestReceivedTimeMs());
        reportDuration(ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS,
                tags, task.runningEnteredTimeMs(), task.waitingEnteredTimeMs());
    }

    private static FlexMetricTags lifecycleTags(
            String engineIp, String role, String group) {
        return FlexMetricTags.of(
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
            WorkerStatus worker,
            WorkerStatus.StepMetrics step) {
        FlexMetricTags tags = FlexMetricTags.of(
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

    public void reportStatusCheckerSuccess(WorkerStatus workerStatus,
                                           WorkerEndpoint ep,
                                           int runningTaskInfoSize,
                                           int finishedTaskListSize) {

        WorkerStatus.EngineObservation status =
                workerStatus.committedEngineObservation();
        WorkerStatus.PollHealth pollHealth = workerStatus.pollHealth();

        FlexMetricTags metricTags = FlexMetricTags.of(
                "engineIp", workerStatus.getMetricIpPort(),
                "role", status.role().name());

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
        if (status.role() == RoleType.ENCODER) {
            int pendingRequests = ep == null ? 0 : ((EncoderEndpoint) ep).pendingEncoderRequestCount();
            monitor.report(ENCODER_PENDING_REQUEST_COUNT, metricTags, pendingRequests);
            monitor.report(ENCODER_SELECTION_LOAD, metricTags,
                    Math.max(0, status.runningQueryLen())
                            + Math.max(0, status.waitingQueryLen()) + pendingRequests);
            monitor.report(ENCODER_UNCACHED_TOKEN_LOAD, metricTags,
                    ep == null ? 0 : ((EncoderEndpoint) ep).inflightUncachedTokenEstimate());
        }
        reportKvCacheCapacity(metricTags, status);
    }

    public void reportCacheStatusCheckerSuccess(WorkerStatus workerStatus,
                                               long successfulPollIntervalUs) {
        WorkerStatus.EngineObservation status =
                workerStatus.committedEngineObservation();
        CacheStatus cacheStatus = workerStatus.getCacheStatus();
        if (successfulPollIntervalUs > 0L) {
            FlexMetricTags metricTags = FlexMetricTags.of(
                    "engineIp", workerStatus.getMetricIpPort(),
                    "role", status.role().name());
            monitor.report(
                    CACHE_STATUS_CHECK_SUCCESS_PERIOD,
                    metricTags,
                    (double) successfulPollIntervalUs);
        }
        if (cacheStatus != null) {
            long cacheKeySize = cacheStatus.getCacheKeySize();
            FlexMetricTags engineMetricTags = FlexMetricTags.of(
                    "engineIp", workerStatus.getMetricIpPort(),
                    "role", status.role().name());
            monitor.report(CACHE_KEY_SIZE, engineMetricTags, cacheKeySize);
        }

    }

    /**
     * WorkerStatus is the common capacity source for both regular engines and
     * KVCM deployments.  KVCM does not poll GetCacheStatus, so cache capacity
     * telemetry must be emitted from the WorkerStatus path rather than the
     * optional cache-status checker.
     */
    private void reportKvCacheCapacity(FlexMetricTags metricTags,
                                       WorkerStatus.EngineObservation status) {
        long totalKvCacheTokens = status.totalKvCacheTokens();
        if (totalKvCacheTokens <= 0) {
            return;
        }
        long availableKvCacheTokens = Math.max(0, status.availableKvCacheTokens());
        long usedKvCacheTokens = Math.max(0, totalKvCacheTokens - availableKvCacheTokens);
        monitor.report(CACHE_USED_KV_CACHE_TOKENS, metricTags, usedKvCacheTokens);
        monitor.report(CACHE_AVAILABLE_KV_CACHE_TOKENS, metricTags, availableKvCacheTokens);
        monitor.report(CACHE_TOTAL_KV_CACHE_TOKENS, metricTags, totalKvCacheTokens);
        monitor.report(CACHE_USED_KV_CACHE_RATIO, metricTags,
                (usedKvCacheTokens * 100.0) / totalKvCacheTokens);
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

    /**
     * Report selected-worker cache hits and the matching request's input tokens.
     *
     * @param roleType    selected role
     * @param ipIndex     selected worker address
     * @param hitTokens   cache-hit tokens
     * @param inputTokens request input tokens
     * @param hitRatio    hit fraction for this request
     */
    public void reportCacheHitMetrics(
            RoleType roleType, String ipIndex, long hitTokens, long inputTokens, double hitRatio) {
        cacheMetricsReporter.reportCacheHitMetrics(roleType, ipIndex, hitTokens, inputTokens, hitRatio);
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

    /**
     * Report KVCM matches only when KVCM supplied a result for the selected worker.
     *
     * @param roleType          selected role
     * @param engineIp          selected worker address
     * @param localMatchTokens  tokens matched in the worker's local cache
     * @param globalMatchTokens tokens matched across local and remote caches
     * @param inputTokens       request input tokens
     * @param available         whether KVCM supplied a match result
     */
    public void reportKvcmSelectedMatch(RoleType roleType,
                                         String engineIp,
                                         long localMatchTokens,
                                         long globalMatchTokens,
                                         long inputTokens,
                                         boolean available) {
        if (available) {
            cacheMetricsReporter.reportKvcmSelectedMatch(
                    roleType, engineIp, localMatchTokens, globalMatchTokens, inputTokens);
        }
    }

    public void reportCacheHitComparisonMetrics(CacheHitComparisonResult comparison) {
        if (comparison == null) {
            return;
        }
        CacheHitComparisonResult.CachePrediction kvcmPrediction = comparison.kvcmPrediction();
        CacheHitComparisonResult.CachePrediction localSyncPrediction = comparison.localSyncPrediction();
        CacheHitComparisonResult.CachePrediction localStandbyPrediction = comparison.localStandbyPrediction();
        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", comparison.worker(),
                "role", comparison.role(),
                "group", comparison.group(),
                "taskState", comparison.state(),
                "cacheMatchSource", comparison.source() == null ? "" : comparison.source());
        CacheHitComparisonResult.CachePrediction sourcePrediction = kvcmPrediction != null
                ? kvcmPrediction
                : localSyncPrediction != null ? localSyncPrediction : localStandbyPrediction;
        if (sourcePrediction != null) {
            monitor.report(CACHE_HIT_COMPARISON_DELTA_TOKENS, tags,
                    comparison.actualHitTokens() - sourcePrediction.predictedHitTokens());
        }
        if (kvcmPrediction != null && kvcmPrediction.localPredictionTokens() >= 0) {
            monitor.report(CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS, tags,
                    comparison.actualHitTokens() - kvcmPrediction.localPredictionTokens());
            monitor.report(CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS, tags,
                    comparison.actualHitTokens() - kvcmPrediction.globalPredictionTokens());
        }
        long inputTokens = comparison.inputTokens();
        if (inputTokens > 0) {
            monitor.report(CACHE_HIT_COMPARISON_INPUT_TOKENS, tags, inputTokens);
            monitor.report(CACHE_HIT_COMPARISON_ACTUAL_TOKENS, tags, comparison.actualHitTokens());
            if (kvcmPrediction != null) {
                monitor.report(CACHE_HIT_COMPARISON_KVCM_PREDICTED_TOKENS,
                        tags, kvcmPrediction.predictedHitTokens());
            }
            monitor.report(CACHE_HIT_COMPARISON_ACTUAL_RATIO,
                    tags, comparison.actualHitTokens() / (double) inputTokens);
        }
        if (localStandbyPrediction != null) {
            monitor.report(CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS, tags,
                    comparison.actualHitTokens() - localStandbyPrediction.predictedHitTokens());
            if (inputTokens > 0) {
                monitor.report(CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_RATIO, tags,
                        (comparison.actualHitTokens() - localStandbyPrediction.predictedHitTokens())
                                / (double) inputTokens);
            }
        }
    }

    /**
     * Delegate routing selected cache match metrics to {@link CacheMetricsReporter}.
     */
    public void reportRoutingSelectedCacheMatchMetrics(RoleType roleType, long hitTokens) {
        cacheMetricsReporter.reportRoutingSelectedCacheMatchMetrics(roleType, hitTokens);
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

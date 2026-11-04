package org.flexlb.service.monitor;

import org.flexlb.balance.endpoint.EndpointStats;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.constant.ZkMasterEvent;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.FlexStatisticsType;
import org.flexlb.metric.SafeFlexMonitor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Locale;
import java.util.Objects;
import java.util.OptionalLong;

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
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_ALL_QPS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_ALL_RT;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_WORKER_SELECT_DETAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;
import static org.flexlb.constant.MetricConstant.ENGINE_DECODE_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_ENCODER_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_FINISHED_TASK_LIST_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT;
import static org.flexlb.constant.MetricConstant.ENGINE_PREFILL_WORKER_NUMBER;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_QUEUE_TIME;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_TASK_INFO_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_SERVICE_DISCOVERY_RAW_HOST_COUNT;
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
import static org.flexlb.constant.MetricConstant.GRACEFUL_LIFECYCLE_EVENT;
import static org.flexlb.constant.MetricConstant.GRPC_SERVER_PROCESS_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_ESTIMATED_TTFT_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_SELECTED_EXECUTION_TIME_MS;
import static org.flexlb.constant.MetricConstant.REQUEST_BLOCK_SIZE;
import static org.flexlb.constant.MetricConstant.REQUEST_MESSAGE_BYTES;
import static org.flexlb.constant.MetricConstant.REQUEST_NETWORK_DELAY_MS;
import static org.flexlb.constant.MetricConstant.REQUEST_SEQ_LEN;
import static org.flexlb.constant.MetricConstant.ZK_MASTER_EVENT;
import static org.flexlb.constant.MetricConstant.ZK_MASTER_EVENT_TIME_MS;
import static org.flexlb.constant.MetricConstant.ZK_MASTER_NODE;

/**
 * Engine health reporter for monitoring engine status and metrics
 */
@Component
public class EngineHealthReporter {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final FlexMonitor monitor;

    private final EndpointStats endpointStats;

    @Autowired
    public EngineHealthReporter(FlexMonitor monitor, EndpointStats endpointStats) {
        this.monitor = SafeFlexMonitor.wrap(monitor);
        this.endpointStats = endpointStats;
        this.monitor.register(
                GRACEFUL_LIFECYCLE_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    @PostConstruct
    public void init() {
        this.monitor.register(ENGINE_ENCODER_WORKER_NUMBER, FlexMetricType.GAUGE);
        this.monitor.register(ENGINE_SERVICE_DISCOVERY_RAW_HOST_COUNT, FlexMetricType.GAUGE);
        this.monitor.register(
                ENGINE_STATUS_CHECK_FAIL_TOTAL, FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        this.monitor.register(
                ENGINE_STATUS_CHECK_FAIL_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(
                ENCODER_PENDING_REQUEST_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(
                ENCODER_SELECTION_LOAD, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(
                ENCODER_UNCACHED_TOKEN_LOAD, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(
                ENGINE_WORKER_STEP_TOTAL_SCHEDULED_TOKENS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        monitor.register(
                ENGINE_WORKER_STEP_PREFILL_REQUEST_COUNT,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        monitor.register(
                ENGINE_WORKER_STEP_PREFILL_TOKENS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        monitor.register(
                ENGINE_WORKER_STEP_TOKEN_BUDGET, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        monitor.register(
                ENGINE_WORKER_STEP_BUDGET_FILL_RATIO,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_INPUT_QUEUE_WAIT_MS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_SCHEDULER_TO_RUNNING_MS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_SCHEDULER_WAIT_MS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_REMOTE_KV_WAIT_MS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_RUNNING_TO_FIRST_TOKEN_MS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_HBM_LOCAL_MATCH_TOKENS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_REMOTE_KV_ADDED_MATCH_TOKENS,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_PREFILL_STEP_COUNT,
                FlexMetricType.GAUGE,
                FlexStatisticsType.SUMMARY);
        this.monitor.register(
                ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MIN,
                FlexMetricType.GAUGE,
                FlexPriorityType.TRIVIAL);
        this.monitor.register(
                ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX,
                FlexMetricType.GAUGE,
                FlexPriorityType.TRIVIAL);
        this.monitor.register(
                CACHE_HIT_COMPARISON_ACTUAL_TOKENS,
                FlexMetricType.COUNTER,
                FlexPriorityType.PRECISE);
        this.monitor.register(
                CACHE_HIT_COMPARISON_KVCM_PREDICTED_TOKENS,
                FlexMetricType.COUNTER,
                FlexPriorityType.PRECISE);
        this.monitor.register(
                CACHE_HIT_COMPARISON_INPUT_TOKENS,
                FlexMetricType.COUNTER,
                FlexPriorityType.PRECISE);
        this.monitor.register(
                CACHE_HIT_COMPARISON_DELTA_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(
                CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS,
                FlexMetricType.GAUGE,
                FlexPriorityType.PRECISE);
        this.monitor.register(
                CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS,
                FlexMetricType.GAUGE,
                FlexPriorityType.PRECISE);
        this.monitor.register(
                CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS,
                FlexMetricType.GAUGE,
                FlexPriorityType.PRECISE);
        this.monitor.register(
                CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_RATIO,
                FlexMetricType.GAUGE,
                FlexPriorityType.PRECISE);
        this.monitor.register(
                CACHE_HIT_COMPARISON_ACTUAL_RATIO, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(REQUEST_BLOCK_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(REQUEST_SEQ_LEN, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(
                REQUEST_MESSAGE_BYTES, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);

        for (String metric :
                new String[] {
                    ENGINE_STATUS_CHECK_SUCCESS_PERIOD,
                    ENGINE_WORKER_NUMBER,
                    ENGINE_PREFILL_WORKER_NUMBER,
                    ENGINE_DECODE_WORKER_NUMBER,
                    ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT,
                    CACHE_STATUS_CHECK_VISITOR_RT,
                    CACHE_STATUS_CHECK_SUCCESS_PERIOD,
                    CACHE_BLOCK_SIZE,
                    CACHE_AVAILABLE_KV_CACHE_TOKENS,
                    CACHE_TOTAL_KV_CACHE_TOKENS,
                }) {
            monitor.register(metric, FlexMetricType.GAUGE);
        }
        for (String metric :
                new String[] {
                    ENGINE_STATUS_AVAILABLE_CONCURRENCY,
                    ENGINE_STATUS_VISITOR_RT,
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    ENGINE_FINISHED_TASK_LIST_SIZE,
                    ENGINE_RUNNING_TASK_INFO_SIZE,
                    CACHE_KEY_SIZE,
                    ENGINE_RUNNING_QUEUE_TIME,
                    PREFILL_SELECTED_ESTIMATED_TTFT_MS,
                    PREFILL_SELECTED_EXECUTION_TIME_MS,
                    ZK_MASTER_NODE,
                    ZK_MASTER_EVENT,
                    ZK_MASTER_EVENT_TIME_MS,
                    CACHE_USED_KV_CACHE_TOKENS,
                    CACHE_USED_KV_CACHE_RATIO,
                    REQUEST_NETWORK_DELAY_MS,
                    GRPC_SERVER_PROCESS_MS,
                }) {
            monitor.register(metric, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        }
        for (String metric :
                new String[] {
                    ENGINE_STATUS_VISITOR_SUCCESS_QPS,
                    ENGINE_STATUS_CHECK_FAIL,
                    ENGINE_BALANCING_MASTER_SELECT_DETAIL,
                    ENGINE_BALANCING_MASTER_WORKER_SELECT_DETAIL,
                    FORWARD_TO_MASTER_RESULT,
                }) {
            monitor.register(metric, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        }
        for (String metric :
                new String[] {
                    ENGINE_BALANCING_MASTER_ALL_QPS,
                    CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS,
                    CACHE_STATUS_CHECK_FAIL,
                }) {
            monitor.register(metric, FlexMetricType.QPS);
        }
        monitor.register(
                ENGINE_BALANCING_MASTER_ALL_RT, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        for (String metric :
                new String[] {
                    ENGINE_WORKER_INFO_STEP_LATENCY_VAR, ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR,
                }) {
            monitor.register(metric, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        }
    }

    public void reportStepLatencyVariance(String modelName, String role, double variance) {
        FlexMetricTags metricTags = FlexMetricTags.of("model", modelName, "role", role);
        monitor.report(ENGINE_WORKER_INFO_STEP_LATENCY_VAR, metricTags, variance);
        logger.debug(
                "Step-latency variance - model: {}, role: {}, value: {}",
                modelName,
                role,
                variance);
    }

    public void reportRunningLoadVariance(String modelName, String role, double variance) {
        FlexMetricTags metricTags = FlexMetricTags.of("model", modelName, "role", role);
        monitor.report(ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR, metricTags, variance);
        logger.debug(
                "Running-load variance - model: {}, role: {}, value: {}",
                modelName,
                role,
                variance);
    }

    @Scheduled(fixedRate = 2000)
    private void reportEngineMetric() {
        String modelName = "engine_service";
        FlexMetricTags tags = FlexMetricTags.of("model", modelName);
        monitor.report(ENGINE_WORKER_NUMBER, tags, endpointStats.discoveredCount());
        monitor.report(
                ENGINE_PREFILL_WORKER_NUMBER,
                tags,
                endpointStats.counts(RoleType.PREFILL).discovered());
        monitor.report(
                ENGINE_DECODE_WORKER_NUMBER,
                tags,
                endpointStats.counts(RoleType.DECODE).discovered());
        monitor.report(
                ENGINE_ENCODER_WORKER_NUMBER,
                tags,
                endpointStats.counts(RoleType.ENCODER).discovered());
    }

    @Scheduled(fixedRate = 2000)
    private void reportWorkerBlockSizes() {
        for (RoleType role : RoleType.values()) {
            reportWorkerBlockSize(role);
        }
    }

    private void reportWorkerBlockSize(RoleType role) {
        long blockSize = endpointStats.blockSize(role);
        if (blockSize > 0) {
            monitor.report(CACHE_BLOCK_SIZE, FlexMetricTags.of("role", role.name()), blockSize);
        }
    }

    public void reportRawServiceDiscoveryHostCount(String model, RoleType role, int count) {
        monitor.report(
                ENGINE_SERVICE_DISCOVERY_RAW_HOST_COUNT,
                FlexMetricTags.of("model", model, "role", role.name()),
                count);
    }

    public void reportServiceDiscoveryResult(String modelName, int result, String role) {
        FlexMetricTags metricTags = FlexMetricTags.of("model", modelName, "role", role);
        monitor.report(ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT, metricTags, result);
    }

    public void reportStatusCheckRemoteInfo(String modelName, String role, Long startTime) {
        FlexMetricTags metricTags =
                FlexMetricTags.of(
                        "model", modelName,
                        "role", role);
        monitor.report(
                ENGINE_STATUS_VISITOR_RT,
                metricTags,
                (double) System.nanoTime() / 1000 - startTime);
        monitor.report(ENGINE_STATUS_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportCacheStatusCheckRemoteInfo(String modelName, String role, Long startTime) {
        FlexMetricTags metricTags =
                FlexMetricTags.of(
                        "model", modelName,
                        "role", role);
        monitor.report(
                CACHE_STATUS_CHECK_VISITOR_RT,
                metricTags,
                (double) System.nanoTime() / 1000 - startTime);
        monitor.report(CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS, metricTags, 1.0);
    }

    public void reportStatusCheckerFail(BalanceStatusEnum errorEnum, RoleType role) {
        reportStatusCheckerFail(errorEnum, "", role);
    }

    public void reportStatusCheckerFail(
            BalanceStatusEnum errorEnum, String engineIp, RoleType role) {
        FlexMetricTags metricTags = statusCheckFailureTags(errorEnum, engineIp, role);
        monitor.report(ENGINE_STATUS_CHECK_FAIL, metricTags, 1.0);
        monitor.report(ENGINE_STATUS_CHECK_FAIL_TOTAL, metricTags, 1.0);
    }

    public void reportStatusCheckFailureLatency(
            BalanceStatusEnum errorEnum, String engineIp, RoleType role, long latencyUs) {
        monitor.report(
                ENGINE_STATUS_CHECK_FAIL_RT,
                statusCheckFailureTags(errorEnum, engineIp, role),
                latencyUs);
    }

    private static FlexMetricTags statusCheckFailureTags(
            BalanceStatusEnum errorEnum, String engineIp, RoleType role) {
        return FlexMetricTags.of(
                "code", String.valueOf(errorEnum.getCode()),
                "engineIp", engineIp == null ? "" : engineIp,
                "role", role == null ? "" : role.getCode());
    }

    public void reportCacheStatusCheckerFail(BalanceStatusEnum errorEnum, RoleType role) {
        FlexMetricTags metricTags =
                FlexMetricTags.of(
                        "code",
                        String.valueOf(errorEnum.getCode()),
                        "role",
                        role == null ? "" : role.getCode());
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportCacheStatusCheckerFail(
            WorkerStatus workerStatus, BalanceStatusEnum errorEnum) {
        RoleType role = workerStatus.getRole();
        FlexMetricTags metricTags =
                FlexMetricTags.of(
                        "engineIp", workerStatus.getMetricIpPort(),
                        "code", String.valueOf(errorEnum.getCode()),
                        "role", role == null ? "" : role.getCode());
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportStatusCheckerSuccess(
            WorkerStatus workerStatus,
            OptionalLong load,
            int runningTaskInfoSize,
            int finishedTaskListSize) {

        WorkerStatus.EngineObservation status = workerStatus.committedEngineObservation();
        WorkerStatus.PollHealth pollHealth = workerStatus.pollHealth();

        FlexMetricTags metricTags =
                FlexMetricTags.of(
                        "engineIp", workerStatus.getMetricIpPort(),
                        "role", status.role().name());

        long pollIntervalUs = pollHealth.successfulPollIntervalUs();
        if (pollIntervalUs > 0) {
            reportWorkerMetric(ENGINE_STATUS_CHECK_SUCCESS_PERIOD, workerStatus, pollIntervalUs);
        }
        load.ifPresent(value -> reportWorkerMetric(ENGINE_RUNNING_QUEUE_TIME, workerStatus, value));
        if (status.availableConcurrency() != null) {
            reportWorkerMetric(
                    ENGINE_STATUS_AVAILABLE_CONCURRENCY,
                    workerStatus,
                    status.availableConcurrency());
        }
        reportWorkerMetric(ENGINE_FINISHED_TASK_LIST_SIZE, workerStatus, finishedTaskListSize);
        reportWorkerMetric(ENGINE_RUNNING_TASK_INFO_SIZE, workerStatus, runningTaskInfoSize);
        reportKvCacheCapacity(workerStatus, status);
    }

    public void reportCacheStatusCheckerSuccess(
            WorkerStatus workerStatus, long successfulPollIntervalUs) {
        CacheStatus cacheStatus = workerStatus.getCacheStatus();
        if (successfulPollIntervalUs > 0L) {
            reportWorkerMetric(
                    CACHE_STATUS_CHECK_SUCCESS_PERIOD, workerStatus, successfulPollIntervalUs);
        }
        if (cacheStatus != null) {
            long cacheKeySize = cacheStatus.getCacheKeySize();
            reportWorkerMetric(CACHE_KEY_SIZE, workerStatus, cacheKeySize);
        }
    }

    /**
     * WorkerStatus is the common capacity source for both regular engines and
     * KVCM deployments.  KVCM does not poll GetCacheStatus, so cache capacity
     * telemetry must be emitted from the WorkerStatus path rather than the
     * optional cache-status checker.
     */
    private void reportKvCacheCapacity(WorkerStatus worker, WorkerStatus.EngineObservation status) {
        long totalKvCacheTokens = status.totalKvCacheTokens();
        if (totalKvCacheTokens <= 0) {
            return;
        }
        long availableKvCacheTokens = Math.max(0, status.availableKvCacheTokens());
        long usedKvCacheTokens = Math.max(0, totalKvCacheTokens - availableKvCacheTokens);
        reportWorkerMetric(CACHE_USED_KV_CACHE_TOKENS, worker, usedKvCacheTokens);
        reportWorkerMetric(CACHE_AVAILABLE_KV_CACHE_TOKENS, worker, availableKvCacheTokens);
        reportWorkerMetric(CACHE_TOTAL_KV_CACHE_TOKENS, worker, totalKvCacheTokens);
        reportWorkerMetric(
                CACHE_USED_KV_CACHE_RATIO,
                worker,
                (usedKvCacheTokens * 100.0) / totalKvCacheTokens);
    }

    private void reportWorkerMetric(String metric, WorkerStatus worker, double value) {
        monitor.report(
                metric,
                FlexMetricTags.of(
                        "engineIp", valueOrEmpty(worker.getIp()), "role", worker.getRole().name()),
                value);
    }

    public void reportEncoderStatus(
            String engineAddress, long pending, long total, long uncachedTokens) {
        var tags = FlexMetricTags.of("engineIp", engineAddress, "role", "ENCODER");
        monitor.report(ENCODER_PENDING_REQUEST_COUNT, tags, pending);
        monitor.report(ENCODER_SELECTION_LOAD, tags, total);
        monitor.report(ENCODER_UNCACHED_TOKEN_LOAD, tags, uncachedTokens);
    }

    public void reportBalancingService(int code, long elapsedMs) {
        FlexMetricTags metricTags = FlexMetricTags.of("code", String.valueOf(code));
        monitor.report(ENGINE_BALANCING_MASTER_ALL_QPS, metricTags, 1.0);
        monitor.report(ENGINE_BALANCING_MASTER_ALL_RT, metricTags, elapsedMs);
    }

    public void reportSelectionResult(RoleType role, boolean success, int code) {
        FlexMetricTags tags =
                FlexMetricTags.of(
                        "role",
                        role.name(),
                        "success",
                        String.valueOf(success),
                        "code",
                        String.valueOf(code));
        monitor.report(ENGINE_BALANCING_MASTER_SELECT_DETAIL, tags, 1.0);
    }

    public void reportSelectedWorker(
            RoleType role, String reason, String address, boolean success, int code) {
        monitor.report(
                ENGINE_BALANCING_MASTER_WORKER_SELECT_DETAIL,
                FlexMetricTags.of(
                        "role",
                        role.name(),
                        "reason",
                        reason == null ? "UNKNOWN" : reason,
                        "engineIp",
                        address,
                        "success",
                        String.valueOf(success),
                        "code",
                        String.valueOf(code)),
                1);
    }

    public void reportMasterNode(String master) {
        monitor.report(ZK_MASTER_NODE, FlexMetricTags.of("masterNode", master), 1.0);
    }

    public void reportPrefillBalanceMasterEvent(ZkMasterEvent event) {
        monitor.report(ZK_MASTER_EVENT, FlexMetricTags.of("event", event.name()), 1.0);
        monitor.report(
                ZK_MASTER_EVENT_TIME_MS,
                FlexMetricTags.of("event", event.name()),
                System.currentTimeMillis());
    }

    public void reportThreadPoolInfo(
            String metricName, String name, int active, int queued, int core, int pool) {
        monitor.report(
                metricName,
                FlexMetricTags.of("threadPool", name, "type", "executingTaskThreadSize"),
                active);
        monitor.report(
                metricName, FlexMetricTags.of("threadPool", name, "type", "queueSize"), queued);
        monitor.report(
                metricName, FlexMetricTags.of("threadPool", name, "type", "corePoolSize"), core);
        monitor.report(
                metricName,
                FlexMetricTags.of("threadPool", name, "type", "currentThreadSizeInPool"),
                pool);
    }

    /** Report request-level estimates captured when a Prefill worker is selected. */
    public void reportPrefillSelectedEstimates(
            RoleType roleType,
            String engineIp,
            String deliveryMode,
            long estimatedTtftMs,
            long executionTimeMs) {
        FlexMetricTags tags =
                FlexMetricTags.ofEngine(
                        engineIp, "role", roleType.name(), "delivery_mode", deliveryMode);
        monitor.report(PREFILL_SELECTED_ESTIMATED_TTFT_MS, tags, estimatedTtftMs);
        monitor.report(PREFILL_SELECTED_EXECUTION_TIME_MS, tags, executionTimeMs);
    }

    public void reportArriveDelayTime(long requestTimeMs, long grpcEntryTime, long startTimeMs) {
        if (requestTimeMs == 0) {
            return;
        }
        if (grpcEntryTime > 0) {
            long networkDelayMs = grpcEntryTime - requestTimeMs;
            long grpcProcessMs = startTimeMs - grpcEntryTime;
            monitor.report(REQUEST_NETWORK_DELAY_MS, FlexMetricTags.of(), networkDelayMs);
            monitor.report(GRPC_SERVER_PROCESS_MS, FlexMetricTags.of(), grpcProcessMs);
        } else {
            // Fallback: if grpcEntryTime not set, report total delay as network delay
            long arrivalDelayMs = startTimeMs - requestTimeMs;
            monitor.report(REQUEST_NETWORK_DELAY_MS, FlexMetricTags.of(), arrivalDelayMs);
        }
    }

    public void reportForwardToMasterResult(String type, String code) {
        monitor.report(
                FORWARD_TO_MASTER_RESULT, FlexMetricTags.of("type", type, "code", code), 1.0);
    }

    public enum Event {
        HEALTH_CHECK_OFFLINE,
        ZK_NODE_OFFLINE,
        SHUTDOWN_COMPLETE,
        ZK_NODE_ONLINE,
        WARMER_COMPLETE,
        ONLINE_COMPLETE
    }

    public void reportDuration(Event event, long durationMs) {
        monitor.report(
                GRACEFUL_LIFECYCLE_EVENT,
                FlexMetricTags.of(
                        "type",
                        event.name().toLowerCase(Locale.ROOT),
                        "duration_ms",
                        String.valueOf(durationMs)),
                1);
    }

    public void reportProcessOk() {
        monitor.report(GRACEFUL_LIFECYCLE_EVENT, FlexMetricTags.of("type", "process_ok"), 1);
    }

    public void reportPrefillWorkerStatusTask(
            String engineIp, String role, String group, WorkerStatus.TaskTelemetry task) {
        FlexMetricTags tags = lifecycleTags(engineIp, role, group);
        monitor.report(
                ENGINE_WORKER_STATUS_HBM_LOCAL_MATCH_TOKENS, tags, task.hbmLocalMatchTokens());
        monitor.report(
                ENGINE_WORKER_STATUS_REMOTE_KV_ADDED_MATCH_TOKENS,
                tags,
                task.remoteKvAddedMatchTokens());
        monitor.report(ENGINE_WORKER_STATUS_PREFILL_STEP_COUNT, tags, task.prefillStepCount());
        // Zero means the request has no nonfinal chunk sample.
        if (task.prefillNonfinalChunkTokensMin() > 0 && task.prefillNonfinalChunkTokensMax() > 0) {
            monitor.report(
                    ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MIN,
                    tags,
                    task.prefillNonfinalChunkTokensMin());
            monitor.report(
                    ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX,
                    tags,
                    task.prefillNonfinalChunkTokensMax());
        }
        reportDuration(
                ENGINE_WORKER_STATUS_INPUT_QUEUE_WAIT_MS,
                tags,
                task.inputQueueDrainTimeMs(),
                task.inputQueueEnqueueTimeMs());
        monitor.report(ENGINE_WORKER_STATUS_REMOTE_KV_WAIT_MS, tags, task.remoteKvWaitMs());
        long schedulerToRunningMs =
                reportDuration(
                        ENGINE_WORKER_STATUS_SCHEDULER_TO_RUNNING_MS,
                        tags,
                        task.runningEnteredTimeMs(),
                        task.waitingEnteredTimeMs());
        if (schedulerToRunningMs >= 0L) {
            monitor.report(
                    ENGINE_WORKER_STATUS_SCHEDULER_WAIT_MS,
                    tags,
                    Math.max(0L, schedulerToRunningMs - task.remoteKvWaitMs()));
        }
        reportDuration(
                ENGINE_WORKER_STATUS_RUNNING_TO_FIRST_TOKEN_MS,
                tags,
                task.firstTokenTimeMs(),
                task.runningEnteredTimeMs());
        reportDuration(
                ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS,
                tags,
                task.waitingEnteredTimeMs(),
                task.requestReceivedTimeMs());
        reportDuration(
                ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS,
                tags,
                task.runningEnteredTimeMs(),
                task.waitingEnteredTimeMs());
    }

    private static FlexMetricTags lifecycleTags(String engineIp, String role, String group) {
        return FlexMetricTags.of(
                "engineIp", engineIp == null ? "" : engineIp,
                "role", role == null ? "" : role,
                "group", group == null ? "" : group);
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

    public void reportWorkerStepMetrics(WorkerStatus worker, WorkerStatus.StepMetrics step) {
        FlexMetricTags tags =
                FlexMetricTags.of(
                        "engineIp", worker.getMetricIpPort(),
                        "role", worker.getRole().name(),
                        "group", worker.topologySnapshot().group(),
                        "phase", step.prefillRequestCount() > 0 ? "prefill" : "decode");
        monitor.report(
                ENGINE_WORKER_STEP_TOTAL_SCHEDULED_TOKENS, tags, step.totalScheduledTokens());
        monitor.report(ENGINE_WORKER_STEP_PREFILL_REQUEST_COUNT, tags, step.prefillRequestCount());
        monitor.report(ENGINE_WORKER_STEP_PREFILL_TOKENS, tags, step.prefillTokens());
        monitor.report(ENGINE_WORKER_STEP_TOKEN_BUDGET, tags, step.tokenBudget());
        monitor.report(ENGINE_WORKER_STEP_BUDGET_FILL_RATIO, tags, step.budgetFillRatio());
    }

    public void reportCacheHitComparisonMetrics(CacheHitComparisonResult comparison) {
        reportCacheHitComparisonMetrics(null, comparison);
    }

    public void reportCacheHitComparisonMetrics(
            WorkerStatus worker, CacheHitComparisonResult comparison) {
        if (comparison == null) {
            return;
        }
        CacheHitComparisonResult.CachePrediction kvcmPrediction = comparison.kvcmPrediction();
        CacheHitComparisonResult.CachePrediction localSyncPrediction =
                comparison.localSyncPrediction();
        CacheHitComparisonResult.CachePrediction localStandbyPrediction =
                comparison.localStandbyPrediction();
        String metricIp =
                worker != null
                                && Objects.equals(
                                        worker.getWorkerIdentity(), comparison.workerIdentity())
                        ? worker.getMetricIpPort()
                        : comparison.worker();
        FlexMetricTags tags =
                FlexMetricTags.of(
                        "engineIp", valueOrEmpty(metricIp),
                        "role", valueOrEmpty(comparison.role()),
                        "group", valueOrEmpty(comparison.group()),
                        "taskState", valueOrEmpty(comparison.state()),
                        "cacheMatchSource", valueOrEmpty(comparison.source()));
        CacheHitComparisonResult.CachePrediction sourcePrediction =
                kvcmPrediction != null
                        ? kvcmPrediction
                        : localSyncPrediction != null
                                ? localSyncPrediction
                                : localStandbyPrediction;
        if (sourcePrediction != null) {
            monitor.report(
                    CACHE_HIT_COMPARISON_DELTA_TOKENS,
                    tags,
                    comparison.actualHitTokens() - sourcePrediction.predictedHitTokens());
        }
        if (kvcmPrediction != null && kvcmPrediction.localPredictionTokens() >= 0) {
            monitor.report(
                    CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS,
                    tags,
                    comparison.actualHitTokens() - kvcmPrediction.localPredictionTokens());
            monitor.report(
                    CACHE_HIT_COMPARISON_KVCM_GLOBAL_MATCH_DELTA_TOKENS,
                    tags,
                    comparison.actualHitTokens() - kvcmPrediction.globalPredictionTokens());
        }
        long inputTokens = comparison.inputTokens();
        if (inputTokens > 0) {
            monitor.report(CACHE_HIT_COMPARISON_INPUT_TOKENS, tags, inputTokens);
            monitor.report(CACHE_HIT_COMPARISON_ACTUAL_TOKENS, tags, comparison.actualHitTokens());
            if (kvcmPrediction != null) {
                monitor.report(
                        CACHE_HIT_COMPARISON_KVCM_PREDICTED_TOKENS,
                        tags,
                        kvcmPrediction.predictedHitTokens());
            }
            monitor.report(
                    CACHE_HIT_COMPARISON_ACTUAL_RATIO,
                    tags,
                    comparison.actualHitTokens() / (double) inputTokens);
        }
        if (localStandbyPrediction != null) {
            monitor.report(
                    CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS,
                    tags,
                    comparison.actualHitTokens() - localStandbyPrediction.predictedHitTokens());
            if (inputTokens > 0) {
                monitor.report(
                        CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_RATIO,
                        tags,
                        (comparison.actualHitTokens() - localStandbyPrediction.predictedHitTokens())
                                / (double) inputTokens);
            }
        }
    }

    private static String valueOrEmpty(String value) {
        return value == null ? "" : value;
    }

    public void reportRequestPayload(
            boolean success, long inputTokens, long blockSize, Long messageBytes) {
        var tags = FlexMetricTags.of("success", String.valueOf(success));
        if (inputTokens >= 0) monitor.report(REQUEST_SEQ_LEN, tags, inputTokens);
        if (blockSize >= 0) monitor.report(REQUEST_BLOCK_SIZE, tags, blockSize);
        if (messageBytes != null) monitor.report(REQUEST_MESSAGE_BYTES, tags, messageBytes);
    }
}

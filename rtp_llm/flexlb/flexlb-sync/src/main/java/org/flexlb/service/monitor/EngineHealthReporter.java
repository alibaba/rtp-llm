package org.flexlb.service.monitor;

import org.flexlb.balance.endpoint.EndpointStats;
import org.flexlb.constant.ZkMasterEvent;
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
import java.util.OptionalLong;

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
import static org.flexlb.constant.MetricConstant.GRACEFUL_LIFECYCLE_EVENT;
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

    public void reportStatusCheckerFail(
            String modelName, BalanceStatusEnum errorEnum, RoleType role) {
        FlexMetricTags metricTags =
                FlexMetricTags.of(
                        "model",
                        modelName,
                        "code",
                        String.valueOf(errorEnum.getCode()),
                        "role",
                        role == null ? "" : role.getCode());
        monitor.report(ENGINE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportCacheStatusCheckerFail(
            String modelName, BalanceStatusEnum errorEnum, RoleType role) {
        FlexMetricTags metricTags =
                FlexMetricTags.of(
                        "model",
                        modelName,
                        "code",
                        String.valueOf(errorEnum.getCode()),
                        "role",
                        role == null ? "" : role.getCode());
        monitor.report(CACHE_STATUS_CHECK_FAIL, metricTags, 1.0);
    }

    public void reportStatusCheckerSuccess(
            String modelName,
            WorkerStatus.TopologySnapshot topology,
            WorkerStatus.EngineObservation status,
            WorkerStatus.PollHealth pollHealth,
            OptionalLong load,
            int runningTaskInfoSize,
            int finishedTaskListSize) {

        FlexMetricTags metricTags =
                FlexMetricTags.of(
                        "model", modelName,
                        "engineIp", topology.ip(),
                        "role", status.role().name());

        Long availableConcurrency = status.availableConcurrency();
        if (availableConcurrency != null) {
            monitor.report(ENGINE_STATUS_AVAILABLE_CONCURRENCY, metricTags, availableConcurrency);
        }
        long pollIntervalUs = pollHealth.successfulPollIntervalUs();
        if (pollIntervalUs > 0) {
            monitor.report(ENGINE_STATUS_CHECK_SUCCESS_PERIOD, metricTags, (double) pollIntervalUs);
        }
        load.ifPresent(value -> monitor.report(ENGINE_RUNNING_QUEUE_TIME, metricTags, value));

        monitor.report(ENGINE_FINISHED_TASK_LIST_SIZE, metricTags, finishedTaskListSize);
        monitor.report(ENGINE_RUNNING_TASK_INFO_SIZE, metricTags, runningTaskInfoSize);
    }

    public void reportCacheStatusCheckerSuccess(
            String modelName,
            WorkerStatus.TopologySnapshot topology,
            WorkerStatus.EngineObservation status,
            long successfulPollIntervalUs,
            Long blockSize,
            Long cacheKeySize) {
        FlexMetricTags engineTags =
                FlexMetricTags.of(
                        "model", modelName,
                        "engineIp", topology.ip(),
                        "role", status.role().name());
        FlexMetricTags roleTags =
                FlexMetricTags.of("model", modelName, "role", status.role().name());
        if (successfulPollIntervalUs > 0L) {
            monitor.report(CACHE_STATUS_CHECK_SUCCESS_PERIOD, engineTags, successfulPollIntervalUs);
        }
        if (blockSize != null && cacheKeySize != null) {
            monitor.report(CACHE_BLOCK_SIZE, roleTags, blockSize);
            monitor.report(CACHE_KEY_SIZE, engineTags, cacheKeySize);
        }

        long totalKvCacheTokens = status.totalKvCacheTokens();
        long availableKvCacheTokens = status.availableKvCacheTokens();
        long usedKvCacheTokens = totalKvCacheTokens - availableKvCacheTokens;
        monitor.report(CACHE_USED_KV_CACHE_TOKENS, engineTags, usedKvCacheTokens);
        monitor.report(CACHE_AVAILABLE_KV_CACHE_TOKENS, engineTags, availableKvCacheTokens);
        monitor.report(CACHE_TOTAL_KV_CACHE_TOKENS, roleTags, totalKvCacheTokens);
        if (totalKvCacheTokens > 0) {
            double usedRatio = (usedKvCacheTokens * 1.0 / totalKvCacheTokens) * 100;
            monitor.report(CACHE_USED_KV_CACHE_RATIO, engineTags, usedRatio);
        }
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

    public void reportMasterNode(String master) {
        monitor.report(ZK_MASTER_NODE, FlexMetricTags.of("masterNode", master), 1.0);
    }

    public void reportPrefillBalanceMasterEvent(ZkMasterEvent event) {
        monitor.report(ZK_MASTER_EVENT, FlexMetricTags.of("event", event.name()), 1.0);
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
}

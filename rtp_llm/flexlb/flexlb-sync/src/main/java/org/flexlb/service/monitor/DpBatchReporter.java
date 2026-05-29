package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.HashMap;
import java.util.Map;

import static org.flexlb.constant.MetricConstant.ROUTING_SLO_VIOLATION_QPS;
import static org.flexlb.constant.MetricConstant.V1_DP_FAKE_PAD_COUNT_PER_BATCH;
import static org.flexlb.constant.MetricConstant.V1_DP_FAKE_PAD_SLOT_QPS;
import static org.flexlb.constant.MetricConstant.V1_DP_INFLIGHT_BATCH_COUNT;
import static org.flexlb.constant.MetricConstant.V1_DP_INFLIGHT_EVICTED_COUNT;
import static org.flexlb.constant.MetricConstant.V1_DP_INFLIGHT_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.V1_DP_RANK_HIT_QPS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_BATCH_ACTUAL_TOKENS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_BATCH_FLUSH_QPS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_BATCH_REQUESTS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_BATCH_TARGET_TOKENS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_FAILURE_QPS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_QUEUE_DEPTH;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_QUEUE_TOKENS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_QUEUE_WAIT_MS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_TICK_DURATION_US;

@Slf4j
@Component
public class DpBatchReporter {

    public enum FlushReason {
        BUCKET_FULL,
        PER_REQUEST_TIMEOUT,
        DEADLINE,
        WINDOW_TIMER,
        EDF_URGENT,
        BATCH_READY,
        SLO_DROPPED
    }

    public enum LoopOutcome {
        DISPATCH,
        FAIL,
        PARK
    }

    public enum FailureCause {
        SLO_DROPPED,
        PLANNER_ERROR,
        DISPATCH_ERROR
    }

    private final FlexMonitor monitor;
    private final FlexMetricTags emptyTags = FlexMetricTags.of();

    @Autowired
    public DpBatchReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        // Multi-DP path
        monitor.register(V1_DP_RANK_HIT_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_FAKE_PAD_SLOT_QPS, FlexMetricType.QPS);
        monitor.register(V1_DP_FAKE_PAD_COUNT_PER_BATCH, FlexMetricType.GAUGE);
        monitor.register(V1_DP_INFLIGHT_BATCH_COUNT, FlexMetricType.GAUGE);
        monitor.register(V1_DP_INFLIGHT_REQUEST_COUNT, FlexMetricType.GAUGE);
        monitor.register(V1_DP_INFLIGHT_EVICTED_COUNT, FlexMetricType.GAUGE);
        monitor.register(ROUTING_SLO_VIOLATION_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        // SLO Batcher
        monitor.register(V1_DP_SLO_QUEUE_DEPTH, FlexMetricType.GAUGE);
        monitor.register(V1_DP_SLO_QUEUE_TOKENS, FlexMetricType.GAUGE);
        monitor.register(V1_DP_SLO_QUEUE_WAIT_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_BATCH_FLUSH_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_BATCH_REQUESTS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_BATCH_TARGET_TOKENS, FlexMetricType.GAUGE);
        monitor.register(V1_DP_SLO_BATCH_ACTUAL_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_FAILURE_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_TICK_DURATION_US, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        log.info("DpBatchReporter initialized and registered with KMonitor");
    }

    // ================== SLO Batcher metrics ==================

    public void reportSloQueueSnapshot(String model, int queueDepth, long queueTokens) {
        FlexMetricTags tags = FlexMetricTags.of("model", model == null ? "default" : model);
        monitor.report(V1_DP_SLO_QUEUE_DEPTH, tags, queueDepth);
        monitor.report(V1_DP_SLO_QUEUE_TOKENS, tags, queueTokens);
    }

    public void reportSloQueueWait(String model, FlushReason reason, long waitMs) {
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("reason", reason.name());
        monitor.report(V1_DP_SLO_QUEUE_WAIT_MS, FlexMetricTags.of(m), waitMs);
    }

    public void reportSloBatchFlush(String model, FlushReason reason, int requestCount) {
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("reason", reason.name());
        FlexMetricTags tags = FlexMetricTags.of(m);
        monitor.report(V1_DP_SLO_BATCH_FLUSH_QPS, tags, 1.0);
        monitor.report(V1_DP_SLO_BATCH_REQUESTS, tags, requestCount);
        if (reason == FlushReason.DEADLINE || reason == FlushReason.EDF_URGENT
                || reason == FlushReason.SLO_DROPPED) {
            monitor.report(ROUTING_SLO_VIOLATION_QPS,
                    FlexMetricTags.of("source", "dp_batch_dispatch_past_deadline"), 1.0);
        }
    }

    public void reportSloBatchTokens(String model, FlushReason reason,
                                     long targetTokens, long actualTokens) {
        monitor.report(V1_DP_SLO_BATCH_TARGET_TOKENS,
                FlexMetricTags.of("model", model == null ? "default" : model), targetTokens);
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("reason", reason.name());
        monitor.report(V1_DP_SLO_BATCH_ACTUAL_TOKENS, FlexMetricTags.of(m), actualTokens);
    }

    public void reportSloFailure(String model, FailureCause cause) {
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("cause", cause.name());
        monitor.report(V1_DP_SLO_FAILURE_QPS, FlexMetricTags.of(m), 1.0);
    }

    public void reportSloTickDuration(String model, LoopOutcome outcome, long durationMicros) {
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("outcome", outcome.name());
        monitor.report(V1_DP_SLO_TICK_DURATION_US, FlexMetricTags.of(m), durationMicros);
    }

    // ================== Multi-DP path metrics ==================

    public void reportDpRankHit(int dpRank) {
        monitor.report(V1_DP_RANK_HIT_QPS, FlexMetricTags.of("rank", String.valueOf(dpRank)), 1.0);
    }

    public void reportFakePadSlots(int fakePadCount, int dpSize) {
        if (fakePadCount > 0) {
            monitor.report(V1_DP_FAKE_PAD_SLOT_QPS, emptyTags, fakePadCount);
        }
        monitor.report(V1_DP_FAKE_PAD_COUNT_PER_BATCH,
                FlexMetricTags.of("dpSize", String.valueOf(dpSize)), fakePadCount);
    }

    public void reportSloViolation(String source) {
        monitor.report(ROUTING_SLO_VIOLATION_QPS,
                FlexMetricTags.of("source", source == null ? "unknown" : source), 1.0);
    }

    public void reportInflightStats(int batchCount, int requestCount, long evictedCount) {
        monitor.report(V1_DP_INFLIGHT_BATCH_COUNT, emptyTags, batchCount);
        monitor.report(V1_DP_INFLIGHT_REQUEST_COUNT, emptyTags, requestCount);
        monitor.report(V1_DP_INFLIGHT_EVICTED_COUNT, emptyTags, evictedCount);
    }
}

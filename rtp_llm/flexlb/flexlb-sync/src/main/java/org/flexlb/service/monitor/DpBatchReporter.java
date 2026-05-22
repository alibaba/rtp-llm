package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.ROUTING_SLO_VIOLATION_QPS;
import static org.flexlb.constant.MetricConstant.V1_DP_BATCHER_COUNT;
import static org.flexlb.constant.MetricConstant.V1_DP_BATCH_FLUSH_QPS;
import static org.flexlb.constant.MetricConstant.V1_DP_BATCH_SIZE;
import static org.flexlb.constant.MetricConstant.V1_DP_BATCH_WAIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.V1_DP_FAKE_PAD_COUNT_PER_BATCH;
import static org.flexlb.constant.MetricConstant.V1_DP_FAKE_PAD_SLOT_QPS;
import static org.flexlb.constant.MetricConstant.V1_DP_INFLIGHT_BATCH_COUNT;
import static org.flexlb.constant.MetricConstant.V1_DP_INFLIGHT_EVICTED_COUNT;
import static org.flexlb.constant.MetricConstant.V1_DP_INFLIGHT_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.V1_DP_QUEUE_DEPTH;
import static org.flexlb.constant.MetricConstant.V1_DP_RANK_HIT_QPS;

/**
 * V1 DP-batch metrics reporter.
 *
 * <p>Covers the gap left by {@link RoutingQueueReporter}: that reporter is
 * scoped to the legacy direct-routing path and produces aggregate routing
 * QPS/latency. The V1 path (DpBatchScheduler + GlobalPrefillBatcher) needs
 * its own observability for batch composition, fake-pad emissions, dp_rank
 * load distribution, and InflightBatchRegistry health.
 *
 * <p>Also owns the {@link #reportSloViolation} signal so SLO pressure
 * (batch dispatched past head deadline) can be alerted on independently of
 * the worker-reject failures already counted in {@code ROUTING_FAILURE_QPS}.
 */
@Slf4j
@Component
public class DpBatchReporter {

    public enum FlushReason {
        BUCKET_FULL,
        PER_REQUEST_TIMEOUT,
        DEADLINE,
        WINDOW_TIMER
    }

    private final FlexMonitor monitor;
    private final FlexMetricTags emptyTags = FlexMetricTags.of();

    @Autowired
    public DpBatchReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        monitor.register(V1_DP_BATCH_FLUSH_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_BATCH_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_RANK_HIT_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_FAKE_PAD_SLOT_QPS, FlexMetricType.QPS);
        monitor.register(V1_DP_FAKE_PAD_COUNT_PER_BATCH, FlexMetricType.GAUGE);
        monitor.register(V1_DP_INFLIGHT_BATCH_COUNT, FlexMetricType.GAUGE);
        monitor.register(V1_DP_INFLIGHT_REQUEST_COUNT, FlexMetricType.GAUGE);
        monitor.register(V1_DP_INFLIGHT_EVICTED_COUNT, FlexMetricType.GAUGE);
        monitor.register(V1_DP_BATCHER_COUNT, FlexMetricType.GAUGE);
        monitor.register(V1_DP_QUEUE_DEPTH, FlexMetricType.GAUGE);
        monitor.register(V1_DP_BATCH_WAIT_TIME_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(ROUTING_SLO_VIOLATION_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        log.info("DpBatchReporter initialized and registered with KMonitor");
    }

    /**
     * Report a batch flush event with its trigger reason and real-request size.
     * Caller must also separately report fake-pad slot count via
     * {@link #reportFakePadSlots} if applicable.
     */
    public void reportBatchFlush(FlushReason reason, int realRequestCount) {
        FlexMetricTags tags = FlexMetricTags.of("reason", reason.name());
        monitor.report(V1_DP_BATCH_FLUSH_QPS, tags, 1.0);
        monitor.report(V1_DP_BATCH_SIZE, tags, realRequestCount);
        if (reason == FlushReason.DEADLINE) {
            monitor.report(ROUTING_SLO_VIOLATION_QPS,
                    FlexMetricTags.of("source", "dp_batch_dispatch_past_deadline"), 1.0);
        }
    }

    /**
     * Report a real (non-fake) assignment landing on a specific dp_rank.
     * Call once per filled rank per batch (not per request) so a single batch
     * with 4 requests on rank=2 still emits one hit for rank=2.
     */
    public void reportDpRankHit(int dpRank) {
        monitor.report(V1_DP_RANK_HIT_QPS, FlexMetricTags.of("rank", String.valueOf(dpRank)), 1.0);
    }

    /**
     * Report fake-pad slot emission. {@code fakePadCount} may be 0 for a full
     * batch — still reported so the per-batch gauge reflects the "no fakes"
     * common case under healthy concurrency.
     */
    public void reportFakePadSlots(int fakePadCount, int dpSize) {
        if (fakePadCount > 0) {
            monitor.report(V1_DP_FAKE_PAD_SLOT_QPS, emptyTags, fakePadCount);
        }
        monitor.report(V1_DP_FAKE_PAD_COUNT_PER_BATCH,
                FlexMetricTags.of("dpSize", String.valueOf(dpSize)), fakePadCount);
    }

    /**
     * Report an SLO violation from a non-V1-batch source. Sources are tagged
     * so deployers can filter; existing V1 callers use
     * {@link #reportBatchFlush} which already emits this on DEADLINE.
     */
    public void reportSloViolation(String source) {
        monitor.report(ROUTING_SLO_VIOLATION_QPS,
                FlexMetricTags.of("source", source == null ? "unknown" : source), 1.0);
    }

    public void reportInflightStats(int batchCount, int requestCount, long evictedCount) {
        monitor.report(V1_DP_INFLIGHT_BATCH_COUNT, emptyTags, batchCount);
        monitor.report(V1_DP_INFLIGHT_REQUEST_COUNT, emptyTags, requestCount);
        monitor.report(V1_DP_INFLIGHT_EVICTED_COUNT, emptyTags, evictedCount);
    }

    /**
     * Report a single request's wait time in the batcher queue. Called once
     * per request at dispatch — KMonitor's PRECISE aggregator computes
     * p50/p95/p99 from the per-event stream.
     */
    public void reportRequestWaitTime(FlushReason reason, long waitMs) {
        monitor.report(V1_DP_BATCH_WAIT_TIME_MS,
                FlexMetricTags.of("reason", reason.name()), waitMs);
    }

    public void reportBatcherStats(int batcherCount, int totalQueueDepth) {
        monitor.report(V1_DP_BATCHER_COUNT, emptyTags, batcherCount);
        monitor.report(V1_DP_QUEUE_DEPTH, emptyTags, totalQueueDepth);
    }
}

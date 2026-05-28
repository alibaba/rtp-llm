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
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_BATCH_ACTUAL_TOKENS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_BATCH_DP_REQ_COUNT;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_BATCH_DP_TOKENS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_BATCH_TARGET_TOKENS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_FAILURE_QPS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_LOOPS_PER_DISPATCH;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_LOOP_DURATION_US;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_QUEUE_REQUESTS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_QUEUE_TOKENS;
import static org.flexlb.constant.MetricConstant.V1_DP_SLO_QUEUE_WAIT_MS;

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
        WINDOW_TIMER,
        EDF_URGENT,
        BATCH_READY,
        SLO_DROPPED
    }

    /** Outcome of a single SloBudgetBatcher main-loop iteration. */
    public enum LoopOutcome {
        DISPATCH,
        FAIL,
        PARK
    }

    /** Failure cause emitted via {@link #reportSloFailure}. */
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

        // SloBudgetBatcher 专属指标
        monitor.register(V1_DP_SLO_BATCH_DP_REQ_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_BATCH_DP_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_BATCH_TARGET_TOKENS, FlexMetricType.GAUGE);
        monitor.register(V1_DP_SLO_BATCH_ACTUAL_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_QUEUE_REQUESTS, FlexMetricType.GAUGE);
        monitor.register(V1_DP_SLO_QUEUE_TOKENS, FlexMetricType.GAUGE);
        monitor.register(V1_DP_SLO_QUEUE_WAIT_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_FAILURE_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_LOOP_DURATION_US, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(V1_DP_SLO_LOOPS_PER_DISPATCH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        log.info("DpBatchReporter initialized and registered with KMonitor");
    }

    // ================== SloBudgetBatcher metrics ==================

    /**
     * 构造一个含 model/role/group/endpoint 的复合 tag，dp_rank 可选。
     * 永远不直接用 raw IP — 至少带上 role + group，便于在多机多角色场景做切片。
     */
    private static FlexMetricTags machineTags(String model, String role, String group,
                                              String endpoint, Integer dpRank) {
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("role", role == null ? "PREFILL" : role);
        m.put("group", group == null ? "default" : group);
        m.put("endpoint", endpoint == null ? "unknown" : endpoint);
        if (dpRank != null) {
            m.put("dp_rank", String.valueOf(dpRank));
        }
        return FlexMetricTags.of(m);
    }

    /**
     * 算法结果上报 - 一次 dispatch 中分配给某个 DP rank 的请求条数和 token 数。
     *
     * @param model    模型名
     * @param role     worker 角色 (通常 PREFILL)
     * @param group    worker group
     * @param endpoint worker endpoint, 格式 ip:grpcPort (不直接 raw IP，组合形式)
     * @param dpRank   DP rank (dpSize=1 时恒为 0)
     * @param reqCount 该 rank 上的请求数
     * @param tokens   该 rank 上的 token 总数
     */
    public void reportSloBatchDpSlot(String model, String role, String group, String endpoint,
                                     int dpRank, int reqCount, long tokens) {
        FlexMetricTags tags = machineTags(model, role, group, endpoint, dpRank);
        monitor.report(V1_DP_SLO_BATCH_DP_REQ_COUNT, tags, reqCount);
        monitor.report(V1_DP_SLO_BATCH_DP_TOKENS, tags, tokens);
    }

    /**
     * 算法结果上报 - 一次 dispatch 的 batch token 数与配置的目标上限。
     * target 与 actual 对比即 batch 利用率，配合 reason 可知"为什么没打满"。
     */
    public void reportSloBatchTokens(String model, FlushReason reason,
                                     long targetTokens, long actualTokens) {
        monitor.report(V1_DP_SLO_BATCH_TARGET_TOKENS,
                FlexMetricTags.of("model", model == null ? "default" : model), targetTokens);
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("reason", reason.name());
        monitor.report(V1_DP_SLO_BATCH_ACTUAL_TOKENS, FlexMetricTags.of(m), actualTokens);
    }

    /**
     * 算法状态上报 - 每轮 loop 末尾上报队列深度和总 token 数。
     * 配对使用即可得到队列平均请求规模。
     */
    public void reportSloQueueSnapshot(String model, int queueRequests, long queueTokens) {
        FlexMetricTags tags = FlexMetricTags.of("model", model == null ? "default" : model);
        monitor.report(V1_DP_SLO_QUEUE_REQUESTS, tags, queueRequests);
        monitor.report(V1_DP_SLO_QUEUE_TOKENS, tags, queueTokens);
    }

    /** 算法状态上报 - 单条请求实际排队时间 (ms)，按 reason 切分。PRECISE -> p50/p95/p99。 */
    public void reportSloQueueWait(String model, FlushReason reason, long waitMs) {
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("reason", reason.name());
        monitor.report(V1_DP_SLO_QUEUE_WAIT_MS, FlexMetricTags.of(m), waitMs);
    }

    /** 算法状态上报 - 请求失败 QPS，按 cause 切分。 */
    public void reportSloFailure(String model, FailureCause cause) {
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("cause", cause.name());
        monitor.report(V1_DP_SLO_FAILURE_QPS, FlexMetricTags.of(m), 1.0);
    }

    /** 算法性能上报 - 单次 stepOnce 耗时 (微秒)，按 outcome 切分。 */
    public void reportSloLoopDuration(String model, LoopOutcome outcome, long durationMicros) {
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("outcome", outcome.name());
        monitor.report(V1_DP_SLO_LOOP_DURATION_US, FlexMetricTags.of(m), durationMicros);
    }

    /**
     * 算法性能上报 - 本次成功 dispatch 之前累计经历了多少次 loop (含本次)。
     * 值大表示 SLO 宽松、队列常 park 等伴侣；值=1 表示直接打包成功。
     */
    public void reportSloLoopsPerDispatch(String model, FlushReason reason, int loops) {
        Map<String, String> m = new HashMap<>();
        m.put("model", model == null ? "default" : model);
        m.put("reason", reason.name());
        monitor.report(V1_DP_SLO_LOOPS_PER_DISPATCH, FlexMetricTags.of(m), loops);
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
        if (reason == FlushReason.DEADLINE || reason == FlushReason.EDF_URGENT
                || reason == FlushReason.SLO_DROPPED) {
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

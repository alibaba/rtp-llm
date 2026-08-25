package org.flexlb.sync.shadow;

import java.util.Map;
import java.util.Objects;
import org.flexlb.constant.MetricConstant;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.state.CounterDriftReport;
import org.flexlb.state.LedgerMetrics;
import org.flexlb.state.LedgerMetricsSample;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 账本观测指标适配器（flexlb-sync 侧消费点）：低频拉取
 * {@link LedgerMetrics#sample(long)} 纯数据快照并上报 {@link FlexMonitor}。
 *
 * <p>指标出口设计（依赖最干净方案）：flexlb-state 不依赖任何 metric 库，
 * 只暴露只读采样接口；指标语义（名称/类型/发布通道）全部收敛在本适配器。
 * 采样 tick 由 StateShadowBridge 的维护调度线程驱动（5s），tick 内先
 * {@link LedgerMetrics#collectAges(long)}（相位年龄轮转抽样 + drift 对账
 * 刷新）再 sample + 全量上报。</p>
 *
 * <h2>超车三分口径</h2>
 * <ul>
 *   <li><b>正常通道胜</b>：fastpath.settles = 快路径 CAS 结算总数 − janitor
 *       兜底结算数（store 计数器对一切 settleRemove 胜者递增，含兜底通道——
 *       此处派生净额：引擎 finished / 本地 settle / cancel 双清 / F1 因果闭包
 *       的 CAS 胜者）；</li>
 *   <li><b>兜底通道胜</b>：janitor.vanished/ttl/hardcap.settles（janitor
 *       通道结算胜者）；</li>
 *   <li><b>超车败</b>：overtaken（相位推进 CAS 败者）+
 *       janitor.lost.to.fastpath（janitor 结算被快路径抢先——兜底败）。</li>
 * </ul>
 *
 * <p>drift 接线：对账偏差条目数上报 gauge（0 = 干净），非零时 WARN
 * （不静默修正语义保持——修正决策归人工/上层）。所有上报 catch-all，
 * 绝不外抛（观测层铁律：指标通道异常不影响任何主路径）。</p>
 */
public final class LedgerMetricsReporter {

    /** 采样 tick 间隔（低频：相位年龄轮转抽样 + 全部指标发布）。 */
    public static final long SAMPLE_INTERVAL_MS = 5_000L;

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final LedgerMetrics metrics;
    private final FlexMonitor monitor;
    /** 最近一次采样快照（诊断端点/测试观测钩子；未跑过 tick 时 null）。 */
    private volatile LedgerMetricsSample lastSample;

    public LedgerMetricsReporter(LedgerMetrics metrics, FlexMonitor monitor) {
        this.metrics = Objects.requireNonNull(metrics, "metrics");
        this.monitor = monitor; // null = 测试通道（上报全 no-op，采样照常）
    }

    /** 注册全部账本观测指标（装配时一次；monitor null 时 no-op）。 */
    public void registerMetrics() {
        if (monitor == null) {
            return;
        }
        monitor.register(MetricConstant.STATE_LEDGER_PREFILL_ACTIVE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_DECODE_ACTIVE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_PREFILL_TOMBSTONES, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_DECODE_TOMBSTONES, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_LATE_EVENTS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_LATE_CANCELS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_UNKNOWN_EVENTS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_FASTPATH_SETTLES, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_OVERTAKEN, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_DRIFT_ENTRIES, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_JANITOR_VANISHED_SETTLES, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_JANITOR_TTL_SETTLES, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_JANITOR_HARDCAP_SETTLES, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_JANITOR_LOST_TO_FASTPATH, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_JANITOR_FENCE_HOLDS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_JANITOR_ERRORS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_ENDPOINT_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_ENDPOINT_ACTIVE_P50, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_ENDPOINT_ACTIVE_P95, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_ENDPOINT_ACTIVE_MAX, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_PHASE_POPULATION, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_PHASE_AGE_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_REASON_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.STATE_LEDGER_ENDPOINT_ACTIVE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    /**
     * 采样 tick（StateShadowBridge 维护调度线程 5s 驱动）：相位年龄抽样 +
     * drift 对账刷新 + 全量指标上报 + drift 非零 WARN。catch-all 绝不外抛。
     */
    public void tick() {
        try {
            long nowMs = System.currentTimeMillis();
            metrics.collectAges(nowMs);
            LedgerMetricsSample sample = metrics.sample(nowMs);
            this.lastSample = sample;
            report(sample);
            warnDrift(sample);
        } catch (Throwable t) {
            logger.warn("[state-ledger-metrics] sampling tick failed (observation only): {}",
                    t.getMessage(), t);
        }
    }

    /** 最近一次采样快照（诊断/测试观测钩子；未跑过 tick 时 null）。 */
    public LedgerMetricsSample lastSample() {
        return lastSample;
    }

    // ---- 上报 ----

    private void report(LedgerMetricsSample s) {
        if (monitor == null) {
            return;
        }
        // 全局级：活跃/墓碑水位/迟到吸收/unknown/超车三分/drift
        reportValue(MetricConstant.STATE_LEDGER_PREFILL_ACTIVE, s.snapshot().prefill().inflight());
        reportValue(MetricConstant.STATE_LEDGER_DECODE_ACTIVE, s.snapshot().decode().activeTotal());
        reportValue(MetricConstant.STATE_LEDGER_PREFILL_TOMBSTONES, s.snapshot().prefillTombstones());
        reportValue(MetricConstant.STATE_LEDGER_DECODE_TOMBSTONES, s.snapshot().decodeTombstones());
        reportValue(MetricConstant.STATE_LEDGER_LATE_EVENTS, s.snapshot().lateEventsAbsorbed());
        reportValue(MetricConstant.STATE_LEDGER_LATE_CANCELS, s.snapshot().lateCancelsAbsorbed());
        reportValue(MetricConstant.STATE_LEDGER_UNKNOWN_EVENTS,
                s.snapshot().unknownRunningEvents() + s.snapshot().unknownFinishedEvents());
        reportValue(MetricConstant.STATE_LEDGER_FASTPATH_SETTLES,
                fastPathNetSettles(s));
        reportValue(MetricConstant.STATE_LEDGER_OVERTAKEN,
                s.prefillOvertaken() + s.decodeOvertaken());
        CounterDriftReport drift = s.drift();
        reportValue(MetricConstant.STATE_LEDGER_DRIFT_ENTRIES,
                drift == null ? 0L : drift.discrepancies().size());

        // janitor 通道计数（未挂载 janitor 时跳过）
        if (s.janitorStats() != null) {
            reportValue(MetricConstant.STATE_LEDGER_JANITOR_VANISHED_SETTLES, s.janitorStats().vanishedSettles());
            reportValue(MetricConstant.STATE_LEDGER_JANITOR_TTL_SETTLES, s.janitorStats().ttlSettles());
            reportValue(MetricConstant.STATE_LEDGER_JANITOR_HARDCAP_SETTLES, s.janitorStats().hardCapSettles());
            reportValue(MetricConstant.STATE_LEDGER_JANITOR_LOST_TO_FASTPATH, s.janitorStats().lostToFastPath());
            reportValue(MetricConstant.STATE_LEDGER_JANITOR_FENCE_HOLDS, s.janitorStats().fenceHoldSkips());
            reportValue(MetricConstant.STATE_LEDGER_JANITOR_ERRORS, s.janitorStats().errors());
        }

        // 端点池级聚合（分布分位——端点 × 相位矩阵刻意不打点）
        LedgerMetricsSample.EndpointDistributionSummary dist = s.endpointDistribution();
        if (dist != null) {
            reportValue(MetricConstant.STATE_LEDGER_ENDPOINT_COUNT, dist.endpointCount());
            reportValue(MetricConstant.STATE_LEDGER_ENDPOINT_ACTIVE_P50, dist.activeP50());
            reportValue(MetricConstant.STATE_LEDGER_ENDPOINT_ACTIVE_P95, dist.activeP95());
            reportValue(MetricConstant.STATE_LEDGER_ENDPOINT_ACTIVE_MAX, dist.activeMax());
        }

        // 相位人口直方图（全局级，tags: side/phase）
        for (LedgerMetricsSample.PhasePopulation p : s.phasePopulation()) {
            reportTagged(MetricConstant.STATE_LEDGER_PHASE_POPULATION,
                    FlexMetricTags.of("side", p.side(), "phase", p.phaseName()), p.count());
        }

        // 相位驻留年龄分位（累积采样分布，tags: side/phase/quantile）
        for (LedgerMetricsSample.PhaseAgeSummary a : s.prefillAges()) {
            reportAge(a);
        }
        for (LedgerMetricsSample.PhaseAgeSummary a : s.decodeAges()) {
            reportAge(a);
        }

        // 受控 reason 计数（全枚举覆盖，tags: kind/reason）
        for (Map.Entry<org.flexlb.state.SettleReason, Long> e : s.settleReasonCounts().entrySet()) {
            reportReason("settle", e.getKey().name(), e.getValue());
        }
        for (Map.Entry<org.flexlb.state.CleanupReason, Long> e : s.cleanupReasonCounts().entrySet()) {
            reportReason("cleanup", e.getKey().name(), e.getValue());
        }
        for (Map.Entry<org.flexlb.state.TransitionReason, Long> e : s.transitionReasonCounts().entrySet()) {
            reportReason("transition", e.getKey().name(), e.getValue());
        }
        for (Map.Entry<org.flexlb.state.TerminalReason, Long> e : s.terminalReasonCounts().entrySet()) {
            reportReason("terminal", e.getKey().name(), e.getValue());
        }

        // 端点级 per-EP 活跃（端点细级，tags: side/endpoint——series = 端点数，
        // 与 engine health check per-endpoint 指标同量级）
        for (LedgerMetricsSample.EndpointLedgerSummary ep : s.endpoints()) {
            String endpoint = String.valueOf(ep.endpointId());
            reportTagged(MetricConstant.STATE_LEDGER_ENDPOINT_ACTIVE,
                    FlexMetricTags.of("side", "P", "endpoint", endpoint), ep.prefillActive());
            reportTagged(MetricConstant.STATE_LEDGER_ENDPOINT_ACTIVE,
                    FlexMetricTags.of("side", "D", "endpoint", endpoint), ep.decodeActive());
        }
    }

    private void reportAge(LedgerMetricsSample.PhaseAgeSummary age) {
        reportTagged(MetricConstant.STATE_LEDGER_PHASE_AGE_MS,
                FlexMetricTags.of("side", age.side(), "phase", age.phaseName(), "quantile", "p50"),
                age.p50Ms());
        reportTagged(MetricConstant.STATE_LEDGER_PHASE_AGE_MS,
                FlexMetricTags.of("side", age.side(), "phase", age.phaseName(), "quantile", "p95"),
                age.p95Ms());
    }

    private void reportReason(String kind, String reason, long value) {
        reportTagged(MetricConstant.STATE_LEDGER_REASON_COUNT,
                FlexMetricTags.of("kind", kind, "reason", reason), value);
    }

    /** 正常通道胜净额：快路径 CAS 结算总数 − janitor 兜底结算数（store 计数含兜底胜者）。 */
    private static long fastPathNetSettles(LedgerMetricsSample s) {
        long total = s.prefillFastPathSettles() + s.decodeFastPathSettles();
        if (s.janitorStats() == null) {
            return total;
        }
        long janitorSettles = s.janitorStats().vanishedSettles()
                + s.janitorStats().ttlSettles()
                + s.janitorStats().hardCapSettles();
        return Math.max(total - janitorSettles, 0L);
    }

    /** drift 非零 WARN（不静默修正——每 tick 提醒；零漂移静默）。 */
    private void warnDrift(LedgerMetricsSample sample) {
        CounterDriftReport drift = sample.drift();
        if (drift == null || drift.clean()) {
            return;
        }
        logger.warn("[state-ledger-metrics] counter drift detected (no silent correction): {} entries: {}",
                drift.discrepancies().size(), drift.discrepancies());
    }

    private void reportValue(String name, long value) {
        try {
            monitor.report(name, value);
        } catch (Throwable ignored) {
            // 指标通道异常绝不影响观测链路
        }
    }

    private void reportTagged(String name, FlexMetricTags tags, long value) {
        try {
            monitor.report(name, tags, value);
        } catch (Throwable ignored) {
            // 指标通道异常绝不影响观测链路
        }
    }
}

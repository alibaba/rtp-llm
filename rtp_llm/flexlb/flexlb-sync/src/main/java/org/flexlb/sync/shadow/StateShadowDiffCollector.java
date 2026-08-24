package org.flexlb.sync.shadow;

import org.flexlb.constant.MetricConstant;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.state.TerminalReason;
import org.flexlb.state.TerminalState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * G1 影子对账：旧路径终态（InflightStore InflightItem CAS 终局）与影子账本终态
 * （StateLedger 墓碑 TerminalOutcome）按 requestId 比对，产出 diff 计数——
 * M7 全链路验收的 gate 数据源。
 *
 * <h2>对比语义</h2>
 * <ul>
 *   <li><b>terminal_state</b>：等价类不一致（COMPLETED↔COMPLETED、FAILED↔FAILED、
 *       CANCELLED↔CANCELLED、TIMED_OUT↔SLO_TIMEOUT 之外均计 diff；
 *       PREEMPTED 为新语义回边态，旧路径无对应——计入 diff 供归因观测）。</li>
 *   <li><b>terminal_reason</b>：影子 TerminalReason 不在旧终态等价 reason 集内
 *       （按旧终态分组；state 与 reason 两指标独立计数——state 不一致时 reason
 *       大概率同时不一致，各自计数供归因）。</li>
 *   <li><b>terminal_missing_on_new / _on_old</b>：单侧到达后对比窗口内对侧未达。</li>
 * </ul>
 *
 * <h2>滑动有界窗口</h2>
 * 两侧各持 bounded map（默认上限 {@value #DEFAULT_MAX_ENTRIES} 条），仅 diff 用：
 * 到达即查对侧——命中则比对并双清；未命中入窗等待。窗口过期（默认
 * {@value #DEFAULT_WINDOW_MS} ms）或容量超限时单侧条目淘汰并计 missing
 * （+超限丢弃计数）。淘汰为惰性触发（每秒至多一次过期扫描），无需后台线程。
 *
 * <p>线程安全：两侧 map 独立 ConcurrentHashMap，先到侧入窗、后到侧删除比对
 * （remove 返回值即竞争胜者，天然幂等——并发双到达只有一方能 remove 成功）。
 */
public final class StateShadowDiffCollector {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    static final long DEFAULT_WINDOW_MS = 10 * 60 * 1000L;
    static final int DEFAULT_MAX_ENTRIES = 65_536;

    /** 惰性过期扫描限频间隔（ms）：高频 put 下每次全表 removeIf 是 O(窗口容量) 热路径开销。 */
    static final long EVICT_SWEEP_MIN_INTERVAL_MS = 1_000L;
    /** 溢出 WARN 限频：满载时每条 put 一条 WARN 是日志风暴源。 */
    static final int OVERFLOW_WARN_EVERY = 10_000;

    /** 旧侧终态记录（requestId → 终态快照）。 */
    private final ConcurrentHashMap<Long, TerminalRecord> oldTerminals = new ConcurrentHashMap<>();
    /** 新侧（影子账本）终态记录。 */
    private final ConcurrentHashMap<Long, TerminalRecord> newTerminals = new ConcurrentHashMap<>();

    private final long windowMs;
    private final int maxEntries;
    private final FlexMonitor monitor;

    /** 惰性过期扫描限频水位（无锁 volatile；并发重复扫描幂等无害）。 */
    private volatile long lastEvictSweepMs;
    /** 溢出 WARN 限频计数（漏斗式：多打几条无害，风暴防护目的达成）。 */
    private final java.util.concurrent.atomic.AtomicLong overflowWarnCounter =
            new java.util.concurrent.atomic.AtomicLong();

    // ---- 计数（LongAdder：日志/诊断/指标双通道）----

    private final LongAdder eventCount = new LongAdder();
    private final LongAdder errorCount = new LongAdder();
    private final LongAdder diffTerminalState = new LongAdder();
    private final LongAdder diffTerminalReason = new LongAdder();
    private final LongAdder diffMissingOnNew = new LongAdder();
    private final LongAdder diffMissingOnOld = new LongAdder();
    private final LongAdder windowOverflowDropped = new LongAdder();
    private final LongAdder matchedCount = new LongAdder();

    public StateShadowDiffCollector(FlexMonitor monitor) {
        this(monitor, DEFAULT_WINDOW_MS, DEFAULT_MAX_ENTRIES);
    }

    public StateShadowDiffCollector(FlexMonitor monitor, long windowMs, int maxEntries) {
        if (windowMs <= 0 || maxEntries <= 0) {
            throw new IllegalArgumentException("windowMs > 0 且 maxEntries > 0，实际: " + windowMs + "/" + maxEntries);
        }
        this.monitor = monitor;
        this.windowMs = windowMs;
        this.maxEntries = maxEntries;
    }

    /** 注册全部影子指标（装配时一次）。 */
    public void registerMetrics() {
        if (monitor == null) {
            return;
        }
        monitor.register(MetricConstant.SHADOW_EVENT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.SHADOW_ERROR, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.SHADOW_DIFF_TERMINAL_STATE, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.SHADOW_DIFF_TERMINAL_REASON, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.SHADOW_DIFF_TERMINAL_MISSING_ON_NEW, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.SHADOW_DIFF_TERMINAL_MISSING_ON_OLD, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    /** shadow.event：一次引擎观察泵入（含空翻译跳过的报文计数）。 */
    void onEvent() {
        eventCount.increment();
        report(MetricConstant.SHADOW_EVENT, 1.0);
    }

    /** shadow.error：影子链路异常（catch-all 包裹后调用，绝不外抛）。 */
    public void onError(Throwable t) {
        errorCount.increment();
        report(MetricConstant.SHADOW_ERROR, 1.0);
        logger.warn("[state-shadow] shadow pipeline error (main path unaffected): {}", t.getMessage(), t);
    }

    /**
     * 旧路径终态（InflightItem 终局，state 为 InflightState 名称）。
     * 到达即查新侧窗口：命中比对双清，未命中入窗。
     */
    public void recordOldTerminal(long requestId, String oldStateName) {
        TerminalRecord record = new TerminalRecord(oldStateName, "OLD", System.currentTimeMillis());
        TerminalRecord counterpart = newTerminals.remove(requestId);
        if (counterpart != null) {
            compare(requestId, record, counterpart);
            return;
        }
        putBounded(oldTerminals, requestId, record);
    }

    /**
     * 新侧（影子账本）终态。到达即查旧侧窗口：命中比对双清，未命中入窗。
     */
    public void recordNewTerminal(long requestId, TerminalState newState, TerminalReason newReason) {
        TerminalRecord record = new TerminalRecord(newState.name(), newReason.name(), System.currentTimeMillis());
        TerminalRecord counterpart = oldTerminals.remove(requestId);
        if (counterpart != null) {
            compare(requestId, counterpart, record);
            return;
        }
        putBounded(newTerminals, requestId, record);
    }

    /** 窗口过期/容量淘汰（惰性触发；测试可显式调用）。返回本侧仍驻留条数。 */
    public int evictExpired(long nowMs) {
        evictSide(oldTerminals, nowMs, true);
        evictSide(newTerminals, nowMs, false);
        return oldTerminals.size() + newTerminals.size();
    }

    // ---- 比对核心 ----

    private void compare(long requestId, TerminalRecord oldRecord, TerminalRecord newRecord) {
        matchedCount.increment();
        String oldState = oldRecord.stateName();
        TerminalState newState;
        try {
            newState = TerminalState.valueOf(newRecord.stateName());
        } catch (IllegalArgumentException e) {
            newState = null;
        }
        boolean stateEquivalent = newState != null && equivalentOldName(newState).equals(oldState);
        if (!stateEquivalent) {
            diffTerminalState.increment();
            report(MetricConstant.SHADOW_DIFF_TERMINAL_STATE, 1.0);
            logger.info("[state-shadow] terminal-state diff: requestId={}, old={}, new={}, newReason={}",
                    requestId, oldState, newRecord.stateName(), newRecord.reasonName());
        }
        if (!reasonInEquivalentSet(oldState, newRecord.reasonName())) {
            diffTerminalReason.increment();
            report(MetricConstant.SHADOW_DIFF_TERMINAL_REASON, 1.0);
        }
    }

    /** 新终态 → 旧路径等价终态名（InflightState 值域）。PREEMPTED 无对应（回边态）。 */
    static String equivalentOldName(TerminalState newState) {
        return switch (newState) {
            case COMPLETED -> "COMPLETED";
            case FAILED -> "FAILED";
            case CANCELLED -> "CANCELLED";
            case SLO_TIMEOUT -> "TIMED_OUT";
            case PREEMPTED -> "PREEMPTED";
        };
    }

    /** 旧终态的等价 reason 集（宽松观测口径：同族 reason 不计 diff）。 */
    static boolean reasonInEquivalentSet(String oldState, String newReason) {
        if (newReason == null) {
            return false;
        }
        return switch (oldState) {
            case "COMPLETED" -> Set.of(TerminalReason.SUCCEEDED.name()).contains(newReason);
            case "FAILED" -> Set.of(TerminalReason.ENGINE_FAILED.name()).contains(newReason);
            case "CANCELLED" -> Set.of(TerminalReason.CANCELLED_ACK.name(), TerminalReason.CANCELLED_IMPLICIT.name(),
                    TerminalReason.CANCELLED_NEVER_ARRIVED.name()).contains(newReason);
            case "TIMED_OUT" -> Set.of(TerminalReason.TTL_EXPIRED.name(), TerminalReason.SLO_BUDGET_EXHAUSTED.name(),
                    TerminalReason.VANISHED.name()).contains(newReason);
            default -> false; // PREEMPTED 等新语义独有终态：一律计 reason diff 供归因
        };
    }

    private void putBounded(ConcurrentHashMap<Long, TerminalRecord> side, long requestId, TerminalRecord record) {
        // 惰性淘汰限频：过期扫描从每次 put 改为至多每秒一次——漂移只影响
        // missing 计数的结算时机（窗口 10 分钟≫秒级），不影响配对正确性；
        // 高频 put 下每次全表 removeIf 是 O(窗口容量) 热路径开销（真机轮
        // 窗口满载 13 万条时曾拖垮全部终态路径线程）。
        long nowMs = System.currentTimeMillis();
        if (nowMs - lastEvictSweepMs >= EVICT_SWEEP_MIN_INTERVAL_MS) {
            lastEvictSweepMs = nowMs;
            evictExpired(nowMs);
        }
        if (side.size() >= maxEntries) {
            windowOverflowDropped.increment();
            if (overflowWarnCounter.incrementAndGet() % OVERFLOW_WARN_EVERY == 1) {
                logger.warn("[state-shadow] diff window overflow ({}), dropping oldest entries on pending side "
                        + "(warn throttled every {}, total dropped={})",
                        maxEntries, OVERFLOW_WARN_EVERY, windowOverflowDropped.sum());
            }
            // 容量满：迭代器取首个淘汰（O(1)）——diff 窗口为观测辅助，近似随机
            // 淘汰与旧实现的“扫描最老”同可接受（旧实现 O(容量) 扫描在高频 put
            // 下是热路径瓶颈）。
            java.util.Iterator<Map.Entry<Long, TerminalRecord>> it = side.entrySet().iterator();
            if (it.hasNext()) {
                it.next();
                it.remove();
                countMissing(side == oldTerminals);
            }
        }
        side.put(requestId, record);
    }

    private void evictSide(ConcurrentHashMap<Long, TerminalRecord> side, long nowMs, boolean isOldSide) {
        side.entrySet().removeIf(e -> {
            if (nowMs - e.getValue().terminalAtMs() > windowMs) {
                countMissing(isOldSide);
                return true;
            }
            return false;
        });
    }

    /** 旧侧条目无对侧 → missing_on_new；新侧条目无对侧 → missing_on_old。 */
    private void countMissing(boolean oldSideRecord) {
        if (oldSideRecord) {
            diffMissingOnNew.increment();
            report(MetricConstant.SHADOW_DIFF_TERMINAL_MISSING_ON_NEW, 1.0);
        } else {
            diffMissingOnOld.increment();
            report(MetricConstant.SHADOW_DIFF_TERMINAL_MISSING_ON_OLD, 1.0);
        }
    }

    private void report(String metricName, double value) {
        if (monitor == null) {
            return;
        }
        try {
            monitor.report(metricName, value);
        } catch (Throwable ignored) {
            // 指标通道异常绝不影响影子链路
        }
    }

    // ---- 计数读口（诊断/测试）----

    public long eventCount() {
        return eventCount.sum();
    }

    public long errorCount() {
        return errorCount.sum();
    }

    public long diffTerminalState() {
        return diffTerminalState.sum();
    }

    public long diffTerminalReason() {
        return diffTerminalReason.sum();
    }

    public long diffMissingOnNew() {
        return diffMissingOnNew.sum();
    }

    public long diffMissingOnOld() {
        return diffMissingOnOld.sum();
    }

    public long matchedCount() {
        return matchedCount.sum();
    }

    public long windowOverflowDropped() {
        return windowOverflowDropped.sum();
    }

    public int pendingOld() {
        return oldTerminals.size();
    }

    public int pendingNew() {
        return newTerminals.size();
    }

    /**
     * 全量统计单行摘要（shutdown 诊断日志用：全部计数读口聚合）。
     * <p>指标通道未部署时（如压测环境无 pushgateway），本行即 G3/G4 验收
     * （missing/error/diff 归零 gate）的权威日志证据。</p>
     */
    public String summaryLine() {
        return "event=" + eventCount.sum()
                + " error=" + errorCount.sum()
                + " matched=" + matchedCount.sum()
                + " diffTerminalState=" + diffTerminalState.sum()
                + " diffTerminalReason=" + diffTerminalReason.sum()
                + " missingOnNew=" + diffMissingOnNew.sum()
                + " missingOnOld=" + diffMissingOnOld.sum()
                + " overflowDropped=" + windowOverflowDropped.sum()
                + " pendingOld=" + oldTerminals.size()
                + " pendingNew=" + newTerminals.size();
    }

    /** 单条终态快照（state 等价名 + reason 名 + 到达时刻）。 */
    private record TerminalRecord(String stateName, String reasonName, long terminalAtMs) {
    }
}

package org.flexlb.state.internal.prefill;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.InternalApi;
import org.flexlb.state.PrefillEndpointCounters;

/**
 * P 侧端点级派生计数簿（读取换权阶段调度读数的 O(1) 数据源）。
 *
 * <p>设计纪律与全局 {@link PrefillCounters} 一致：LongAdder 分带增量账，
 * mutator 全 package-private、仅 {@link PrefillSideStore} 在状态转换的
 * CAS 胜者临界区内调用；读侧 {@link #countersOf} 无锁 O(1)。</p>
 *
 * <p>口径：桶账只含<b>已派发绑定</b>条目（register 后排队/攒批窗口的
 * 未绑定条目不在任何桶——该窗口的调度读数由派发编排侧的 batcher 队列
 * 深度覆盖，与按需遍历聚合时代的语义一致）。条目在 dispatch 绑定后
 * 首次入桶（现态全账）；派发前重绑做桶间全账迁移；终局随移除归位。</p>
 */
@InternalApi
final class PrefillEndpointCountersBook {

    /** 端点桶：全部字段 LongAdder（读 sum 无锁；写竞争下分带）。 */
    private static final class Bucket {
        final LongAdder activeTotal = new LongAdder();
        final LongAdder engineOwned = new LongAdder();
        final LongAdder[] phaseCounts;

        Bucket() {
            PrefillPhase[] phases = PrefillPhase.values();
            phaseCounts = new LongAdder[phases.length];
            for (int i = 0; i < phases.length; i++) {
                phaseCounts[i] = new LongAdder();
            }
        }
    }

    private final ConcurrentHashMap<Integer, Bucket> buckets = new ConcurrentHashMap<>();

    // ---- mutator（仅 PrefillSideStore 在写者临界区内调用）----

    /**
     * 相位转换（CAS 胜者分支）：旧相位人口 -1、新相位 +1。未绑定条目
     * 不在任何桶（排队/攒批窗口）——no-op。
     */
    void onPhaseTransition(int endpointId, PrefillPhase from, PrefillPhase to) {
        Bucket b = buckets.get(endpointId);
        if (b == null) {
            return;
        }
        b.phaseCounts[from.ordinal()].decrement();
        b.phaseCounts[to.ordinal()].increment();
    }

    /** 引擎首见：engineOwned +1。 */
    void onEngineOwned(int endpointId) {
        Bucket b = buckets.get(endpointId);
        if (b != null) {
            b.engineOwned.increment();
        }
    }

    /** 条目移除归位（终局）：全账回退。 */
    void onRemoved(int endpointId, PrefillRequestState entry) {
        Bucket b = buckets.get(endpointId);
        if (b == null) {
            return;
        }
        b.activeTotal.decrement();
        b.phaseCounts[entry.phase().ordinal()].decrement();
        if (entry.engineOwned()) {
            b.engineOwned.decrement();
        }
    }

    /** 条目现态入桶（dispatch 绑定 / 重绑目标桶承接）。与 {@link #onRemoved} 严格对称。 */
    void onEntryAdded(int endpointId, PrefillRequestState entry) {
        Bucket b = buckets.computeIfAbsent(endpointId, k -> new Bucket());
        b.activeTotal.increment();
        b.phaseCounts[entry.phase().ordinal()].increment();
        if (entry.engineOwned()) {
            b.engineOwned.increment();
        }
    }

    /**
     * 派发前重绑迁移：条目现态全账从源端点桶划入目标端点桶（调用方保证
     * 持条目锁、条目状态稳定；两桶操作同一临界区内完成）。
     */
    void transferEntry(int fromEndpointId, int toEndpointId, PrefillRequestState entry) {
        onRemoved(fromEndpointId, entry);
        onEntryAdded(toEndpointId, entry);
    }

    // ---- 读（无锁 O(1)）----

    /** 端点级派生计数快照（无桶 = 无活跃条目 → 全零视图）。 */
    PrefillEndpointCounters countersOf(int endpointId) {
        Bucket b = buckets.get(endpointId);
        if (b == null) {
            return PrefillEndpointCounters.empty();
        }
        Long[] counts = new Long[b.phaseCounts.length];
        for (int i = 0; i < b.phaseCounts.length; i++) {
            counts[i] = b.phaseCounts[i].sum();
        }
        return new PrefillEndpointCounters(
                (int) b.activeTotal.sum(),
                (int) b.engineOwned.sum(),
                List.of(counts));
    }

    /**
     * 对账：桶增量账 vs 按活跃条目全量重算（仅已绑定条目——桶口径）。
     * 返回偏差描述列表（空 = 一致；不静默修正）。
     */
    List<String> driftAgainst(Map<Integer, List<PrefillRequestState>> boundEntriesByEndpoint) {
        List<String> drift = new java.util.ArrayList<>();
        java.util.Set<Integer> endpointIds = new java.util.HashSet<>(buckets.keySet());
        endpointIds.addAll(boundEntriesByEndpoint.keySet());
        for (int endpointId : endpointIds) {
            Bucket b = buckets.get(endpointId);
            List<PrefillRequestState> entries = boundEntriesByEndpoint.getOrDefault(endpointId, List.of());
            long recountActive = entries.size();
            long recountOwned = 0;
            long[] recountPhases = new long[PrefillPhase.values().length];
            for (PrefillRequestState e : entries) {
                recountPhases[e.phase().ordinal()]++;
                if (e.engineOwned()) {
                    recountOwned++;
                }
            }
            String prefix = "P ep" + endpointId;
            if (b == null) {
                if (recountActive != 0) {
                    drift.add(prefix + ": no bucket but recount activeTotal=" + recountActive);
                }
                continue;
            }
            if (b.activeTotal.sum() != recountActive) {
                drift.add(prefix + " activeTotal: counter=" + b.activeTotal.sum() + " recount=" + recountActive);
            }
            if (b.engineOwned.sum() != recountOwned) {
                drift.add(prefix + " engineOwned: counter=" + b.engineOwned.sum() + " recount=" + recountOwned);
            }
            for (PrefillPhase p : PrefillPhase.values()) {
                long counted = b.phaseCounts[p.ordinal()].sum();
                if (counted != recountPhases[p.ordinal()]) {
                    drift.add(prefix + " phase " + p + ": counter=" + counted
                            + " recount=" + recountPhases[p.ordinal()]);
                }
            }
        }
        return drift;
    }

    /** 清零（rebuild 用；单线程调用）。 */
    void reset() {
        buckets.clear();
    }
}

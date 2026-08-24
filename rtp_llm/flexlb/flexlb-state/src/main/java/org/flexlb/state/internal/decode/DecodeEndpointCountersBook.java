package org.flexlb.state.internal.decode;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.DecodeEndpointCounters;
import org.flexlb.state.InternalApi;

/**
 * D 侧端点级派生计数簿（读取换权阶段调度读数的 O(1) 数据源）。
 *
 * <p>设计纪律与全局 {@link DecodeCounters} 一致：LongAdder 分带增量账，
 * mutator 全 package-private、仅 {@link DecodeSideStore} 在状态转换的
 * CAS 胜者临界区内调用（单一写者位置）；读侧 {@link #countersOf}
 * 无锁 O(1)（LongAdder.sum 常数个字段聚合，零条目遍历）。</p>
 *
 * <p>与按需遍历聚合（每次读 O(端点活跃条目)，端点状态 tick 刷新缓存
 * 时反复全量扫描）的差异正是本类存在的理由：状态机单一事实源 +
 * 派生计数器增量维护——把 O(n) 读从调度热路径（端点校准 tick）上
 * 拿掉，写路径每事件多常数个原子加法（量级微秒级）换读路径零扫描。</p>
 *
 * <p>口径：桶账只含<b>已绑定</b>条目（D 侧 reserve 即绑定——全部活跃
 * 条目均在某桶内）。条目跨相位推进在原桶内"旧相位减、新相位加"；
 * 预占确认（首次进入引擎加载临界相位）在同一写者临界区内撤未确认
 * 三项（数量 / 期望 KV / prompt KV）并计确认数——瞬态负值不可能出现
 * （同一临界区内完成）。条目终局/释放随移除归位全账。</p>
 */
@InternalApi
final class DecodeEndpointCountersBook {

    /** 端点桶：全部字段 LongAdder（读 sum 无锁；写竞争下分带）。 */
    private static final class Bucket {
        final LongAdder activeTotal = new LongAdder();
        final LongAdder unconfirmedCount = new LongAdder();
        final LongAdder unconfirmedExpectedKv = new LongAdder();
        final LongAdder unconfirmedSeqKv = new LongAdder();
        final LongAdder engineOwned = new LongAdder();
        final LongAdder kvTokensReported = new LongAdder();
        final LongAdder confirmed = new LongAdder();
        final LongAdder[] phaseCounts;

        Bucket() {
            DecodePhase[] phases = DecodePhase.values();
            phaseCounts = new LongAdder[phases.length];
            for (int i = 0; i < phases.length; i++) {
                phaseCounts[i] = new LongAdder();
            }
        }
    }

    private final ConcurrentHashMap<Integer, Bucket> buckets = new ConcurrentHashMap<>();

    // ---- mutator（仅 DecodeSideStore 在写者临界区内调用）----

    /** 预约入账（reserve 即绑定）：RESERVED 人口 + 未确认三项 + 活跃总数。 */
    void onReserved(int endpointId, long seqLen, long expectedKv) {
        Bucket b = buckets.computeIfAbsent(endpointId, k -> new Bucket());
        b.activeTotal.increment();
        b.phaseCounts[DecodePhase.RESERVED.ordinal()].increment();
        b.unconfirmedCount.increment();
        b.unconfirmedExpectedKv.add(expectedKv);
        b.unconfirmedSeqKv.add(seqLen);
    }

    /**
     * 相位转换（CAS 胜者分支）：旧相位人口 -1、新相位 +1。未绑定条目
     * 不在任何桶（防御性 no-op——D 侧常态恒绑定）。
     */
    void onPhaseTransition(int endpointId, DecodePhase from, DecodePhase to) {
        Bucket b = buckets.get(endpointId);
        if (b == null) {
            return;
        }
        b.phaseCounts[from.ordinal()].decrement();
        b.phaseCounts[to.ordinal()].increment();
    }

    /**
     * 预占确认（首次进入引擎加载临界相位，同一胜者临界区内）：撤未确认
     * 三项（数量 / 被撤出的期望 KV / prompt KV）+ 确认数 +1。
     */
    void onReservationConfirmed(int endpointId, long withdrawnExpectedKv, long seqLen) {
        Bucket b = buckets.get(endpointId);
        if (b == null) {
            return;
        }
        b.unconfirmedCount.decrement();
        b.unconfirmedExpectedKv.add(-withdrawnExpectedKv);
        b.unconfirmedSeqKv.add(-seqLen);
        b.confirmed.increment();
    }

    /** 引擎首见：engineOwned +1。 */
    void onEngineOwned(int endpointId) {
        Bucket b = buckets.get(endpointId);
        if (b != null) {
            b.engineOwned.increment();
        }
    }

    /** 引擎事实 KV 增量（观察前后差值）。 */
    void onKvReportedDelta(int endpointId, long delta) {
        Bucket b = buckets.get(endpointId);
        if (b != null && delta != 0) {
            b.kvTokensReported.add(delta);
        }
    }

    /** rebuild 引擎收养入账（直接落到收养相位 + 引擎事实 KV + engineOwned）。 */
    void onAdopted(int endpointId, DecodePhase adoptedPhase, long kvTokens, boolean owned) {
        Bucket b = buckets.computeIfAbsent(endpointId, k -> new Bucket());
        b.activeTotal.increment();
        b.phaseCounts[adoptedPhase.ordinal()].increment();
        if (kvTokens > 0) {
            b.kvTokensReported.add(kvTokens);
        }
        if (owned) {
            b.engineOwned.increment();
        }
        if (adoptedPhase.ordinal() >= DecodePhase.D_LOADING.ordinal()) {
            b.confirmed.increment();
        } else {
            b.unconfirmedCount.increment();
        }
    }

    /** 条目移除归位（终局 / 主动释放）：全账回退（相位相关的未确认/确认按移除时相位判定）。 */
    void onRemoved(int endpointId, DecodeRequestState entry) {
        Bucket b = buckets.get(endpointId);
        if (b == null) {
            return;
        }
        b.activeTotal.decrement();
        b.phaseCounts[entry.phase().ordinal()].decrement();
        if (entry.phase().ordinal() < DecodePhase.D_LOADING.ordinal()) {
            // 未跨确认临界点即被移除：未确认三项随预占账回退
            b.unconfirmedCount.decrement();
            b.unconfirmedExpectedKv.add(-entry.reservedKv());
            b.unconfirmedSeqKv.add(-entry.seqLen());
        } else {
            b.confirmed.decrement();
        }
        b.kvTokensReported.add(-entry.kvTokensReported());
        if (entry.engineOwned()) {
            b.engineOwned.decrement();
        }
    }

    /**
     * 条目现态入桶（绑定后入账：dispatch 绑定路径 / 派发前重绑的目标桶承接）。
     * 与 {@link #onRemoved} 严格对称（同一套"现态全账"口径）。
     */
    void onEntryAdded(int endpointId, DecodeRequestState entry) {
        Bucket b = buckets.computeIfAbsent(endpointId, k -> new Bucket());
        b.activeTotal.increment();
        b.phaseCounts[entry.phase().ordinal()].increment();
        if (entry.phase().ordinal() < DecodePhase.D_LOADING.ordinal()) {
            b.unconfirmedCount.increment();
            b.unconfirmedExpectedKv.add(entry.reservedKv());
            b.unconfirmedSeqKv.add(entry.seqLen());
        } else {
            b.confirmed.increment();
        }
        b.kvTokensReported.add(entry.kvTokensReported());
        if (entry.engineOwned()) {
            b.engineOwned.increment();
        }
    }

    /**
     * 派发前重绑迁移：条目现态全账从源端点桶划入目标端点桶（调用方保证
     * 持条目锁、条目状态稳定；两桶操作同一临界区内完成）。
     */
    void transferEntry(int fromEndpointId, int toEndpointId, DecodeRequestState entry) {
        onRemoved(fromEndpointId, entry);
        onEntryAdded(toEndpointId, entry);
    }

    // ---- 读（无锁 O(1)）----

    /** 端点级派生计数快照（无桶 = 无活跃条目 → 全零视图）。 */
    DecodeEndpointCounters countersOf(int endpointId) {
        Bucket b = buckets.get(endpointId);
        if (b == null) {
            return DecodeEndpointCounters.empty();
        }
        Long[] counts = new Long[b.phaseCounts.length];
        for (int i = 0; i < b.phaseCounts.length; i++) {
            counts[i] = b.phaseCounts[i].sum();
        }
        return new DecodeEndpointCounters(
                (int) b.activeTotal.sum(),
                (int) b.unconfirmedCount.sum(),
                b.unconfirmedExpectedKv.sum(),
                b.unconfirmedSeqKv.sum(),
                (int) b.engineOwned.sum(),
                b.kvTokensReported.sum(),
                List.of(counts));
    }

    /**
     * 对账：桶增量账 vs 按活跃条目全量重算（仅已绑定条目——桶口径）。
     * 返回偏差描述列表（空 = 一致；不静默修正）。
     */
    List<String> driftAgainst(Map<Integer, List<DecodeRequestState>> boundEntriesByEndpoint) {
        List<String> drift = new java.util.ArrayList<>();
        java.util.Set<Integer> endpointIds = new java.util.HashSet<>(buckets.keySet());
        endpointIds.addAll(boundEntriesByEndpoint.keySet());
        for (int endpointId : endpointIds) {
            Bucket b = buckets.get(endpointId);
            List<DecodeRequestState> entries = boundEntriesByEndpoint.getOrDefault(endpointId, List.of());
            long recountActive = entries.size();
            long[] recountPhases = new long[DecodePhase.values().length];
            long recountUnconfirmedCount = 0;
            long recountUnconfirmedExpectedKv = 0;
            long recountUnconfirmedSeqKv = 0;
            long recountConfirmed = 0;
            long recountOwned = 0;
            long recountKv = 0;
            for (DecodeRequestState e : entries) {
                recountPhases[e.phase().ordinal()]++;
                if (e.phase().ordinal() < DecodePhase.D_LOADING.ordinal()) {
                    recountUnconfirmedCount++;
                    recountUnconfirmedExpectedKv += e.reservedKv();
                    recountUnconfirmedSeqKv += e.seqLen();
                } else {
                    recountConfirmed++;
                }
                if (e.engineOwned()) {
                    recountOwned++;
                }
                recountKv += e.kvTokensReported();
            }
            String prefix = "D ep" + endpointId;
            if (b == null) {
                if (recountActive != 0) {
                    drift.add(prefix + ": no bucket but recount activeTotal=" + recountActive);
                }
                continue;
            }
            if (b.activeTotal.sum() != recountActive) {
                drift.add(prefix + " activeTotal: counter=" + b.activeTotal.sum() + " recount=" + recountActive);
            }
            if (b.unconfirmedCount.sum() != recountUnconfirmedCount) {
                drift.add(prefix + " unconfirmedCount: counter=" + b.unconfirmedCount.sum()
                        + " recount=" + recountUnconfirmedCount);
            }
            if (b.unconfirmedExpectedKv.sum() != recountUnconfirmedExpectedKv) {
                drift.add(prefix + " unconfirmedExpectedKv: counter=" + b.unconfirmedExpectedKv.sum()
                        + " recount=" + recountUnconfirmedExpectedKv);
            }
            if (b.unconfirmedSeqKv.sum() != recountUnconfirmedSeqKv) {
                drift.add(prefix + " unconfirmedSeqKv: counter=" + b.unconfirmedSeqKv.sum()
                        + " recount=" + recountUnconfirmedSeqKv);
            }
            if (b.confirmed.sum() != recountConfirmed) {
                drift.add(prefix + " confirmed: counter=" + b.confirmed.sum() + " recount=" + recountConfirmed);
            }
            if (b.engineOwned.sum() != recountOwned) {
                drift.add(prefix + " engineOwned: counter=" + b.engineOwned.sum() + " recount=" + recountOwned);
            }
            if (b.kvTokensReported.sum() != recountKv) {
                drift.add(prefix + " kvTokensReported: counter=" + b.kvTokensReported.sum()
                        + " recount=" + recountKv);
            }
            for (DecodePhase p : DecodePhase.values()) {
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

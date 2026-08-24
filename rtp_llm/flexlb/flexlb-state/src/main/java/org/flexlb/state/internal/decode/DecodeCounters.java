package org.flexlb.state.internal.decode;

import java.util.List;
import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.DecodeCounterSnapshot;

/**
 * D 侧派生计数器（LongAdder 分带增量账）。
 *
 * <p>单写者强制：mutator 全 package-private，仅 {@link DecodeSideStore} 在
 * advance 的 CAS 胜者分支 / reserve / settleRemove 等固定位置调用——
 * 类型（package-private）+ 调用位置（Store 独占）双重约束。</p>
 *
 * <p>口径约定（auditAndDrift 可比对的前提）：全部合计为<b>活跃条目</b>口径——
 * 条目终局/释放移除时随 {@link #onRemoved} 归位。</p>
 */
final class DecodeCounters {

    private final LongAdder[] phaseCounts;
    private final LongAdder reservedKvTotal = new LongAdder();
    private final LongAdder expectedKvTotal = new LongAdder();
    private final LongAdder kvTokensReportedTotal = new LongAdder();
    private final LongAdder confirmed = new LongAdder();
    private final LongAdder engineOwned = new LongAdder();

    DecodeCounters() {
        DecodePhase[] phases = DecodePhase.values();
        phaseCounts = new LongAdder[phases.length];
        for (int i = 0; i < phases.length; i++) {
            phaseCounts[i] = new LongAdder();
        }
    }

    /** 预约入账（RESERVED 入口 +1、影子预占 + 期望 KV 双轨起点）。仅 Store.reserve 调用。 */
    void onReserved(long expectedKv) {
        phaseCounts[DecodePhase.RESERVED.ordinal()].increment();
        reservedKvTotal.add(expectedKv);
        expectedKvTotal.add(expectedKv);
    }

    /** 相位转换（CAS 胜者分支）：旧相位人口 -1、新相位人口 +1。仅 Store.advance 调用。 */
    void onPhaseTransition(DecodePhase from, DecodePhase to) {
        phaseCounts[from.ordinal()].decrement();
        phaseCounts[to.ordinal()].increment();
    }

    /** 引擎首见：engineOwned +1。仅 Store.noteEngineObserved 调用。 */
    void onEngineOwned() {
        engineOwned.increment();
    }

    /** 计费归属移交撤预占：reservedKvTotal 减去被撤出的量（调用方先读旧值再清零）。仅 Store.advance 调用。 */
    void onReservationWithdrawn(long withdrawnKv) {
        if (withdrawnKv != 0) {
            reservedKvTotal.add(-withdrawnKv);
        }
    }

    /** 引擎事实 KV 增量（noteEngineObserved 前后差值）。仅 Store.noteEngineObserved 调用。 */
    void onKvReportedDelta(long oldKv, long newKv) {
        kvTokensReportedTotal.add(newKv - oldKv);
    }

    /** 确认计数（首次进入 ≥ D_LOADING：引擎事实已接管预占）。仅 Store.advance 调用。 */
    void onConfirmed() {
        confirmed.increment();
    }

    /** rebuild 引擎收养入账（直接落到收养相位 + 引擎事实 KV + engineOwned）。仅 Store.adoptEngineOwned 调用。 */
    void onAdopted(DecodePhase adoptedPhase, long kvTokens, boolean owned) {
        phaseCounts[adoptedPhase.ordinal()].increment();
        if (kvTokens > 0) {
            kvTokensReportedTotal.add(kvTokens);
        }
        if (owned) {
            engineOwned.increment();
        }
        if (adoptedPhase.ordinal() >= DecodePhase.D_LOADING.ordinal()) {
            confirmed.increment();
        }
    }

    /** 条目移除归位（settle/release）。仅 Store 移除路径调用。 */
    void onRemoved(DecodeRequestState entry) {
        phaseCounts[entry.phase().ordinal()].decrement();
        reservedKvTotal.add(-entry.reservedKv());
        expectedKvTotal.add(-entry.reservedExpectedKv());
        kvTokensReportedTotal.add(-entry.kvTokensReported());
        if (entry.engineOwned()) {
            engineOwned.decrement();
        }
        if (entry.phase().ordinal() >= DecodePhase.D_LOADING.ordinal()) {
            confirmed.decrement();
        }
    }

    /** 全量重算快照（audit 与发布共用）。 */
    DecodeCounterSnapshot recompute(int activeTotal) {
        Long[] counts = new Long[phaseCounts.length];
        for (int i = 0; i < phaseCounts.length; i++) {
            counts[i] = phaseCounts[i].sum();
        }
        return new DecodeCounterSnapshot(
                java.util.List.of(counts),
                activeTotal,
                reservedKvTotal.sum(),
                expectedKvTotal.sum(),
                kvTokensReportedTotal.sum(),
                confirmed.sum());
    }

    /** 对账比对：全量重算值 vs 计数器账（返回偏差描述，空 = 一致）。 */
    List<String> driftAgainst(List<DecodeRequestState> allEntries) {
        List<String> drift = new java.util.ArrayList<>();
        long[] recount = new long[phaseCounts.length];
        long recountReserved = 0;
        long recountExpected = 0;
        long recountKv = 0;
        long recountConfirmed = 0;
        long recountOwned = 0;
        for (DecodeRequestState e : allEntries) {
            recount[e.phase().ordinal()]++;
            recountReserved += e.reservedKv();
            recountExpected += e.reservedExpectedKv();
            recountKv += e.kvTokensReported();
            if (e.phase().ordinal() >= DecodePhase.D_LOADING.ordinal()) {
                recountConfirmed++;
            }
            if (e.engineOwned()) {
                recountOwned++;
            }
        }
        for (DecodePhase p : DecodePhase.values()) {
            long counted = phaseCounts[p.ordinal()].sum();
            if (counted != recount[p.ordinal()]) {
                drift.add("D phase " + p + ": counter=" + counted + " recount=" + recount[p.ordinal()]);
            }
        }
        if (reservedKvTotal.sum() != recountReserved) {
            drift.add("D reservedKv: counter=" + reservedKvTotal.sum() + " recount=" + recountReserved);
        }
        if (expectedKvTotal.sum() != recountExpected) {
            drift.add("D expectedKv: counter=" + expectedKvTotal.sum() + " recount=" + recountExpected);
        }
        if (kvTokensReportedTotal.sum() != recountKv) {
            drift.add("D kvTokensReported: counter=" + kvTokensReportedTotal.sum() + " recount=" + recountKv);
        }
        if (confirmed.sum() != recountConfirmed) {
            drift.add("D confirmed: counter=" + confirmed.sum() + " recount=" + recountConfirmed);
        }
        if (engineOwned.sum() != recountOwned) {
            drift.add("D engineOwned: counter=" + engineOwned.sum() + " recount=" + recountOwned);
        }
        return drift;
    }

    /** 清零（rebuild 用；单线程调用）。 */
    void reset() {
        for (LongAdder a : phaseCounts) {
            a.reset();
        }
        reservedKvTotal.reset();
        expectedKvTotal.reset();
        kvTokensReportedTotal.reset();
        confirmed.reset();
        engineOwned.reset();
    }
}

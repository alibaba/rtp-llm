package org.flexlb.state.internal.prefill;

import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.PrefillCounterSnapshot;

/**
 * P 侧派生计数器（LongAdder 分带增量账）。
 *
 * <p>P3 单写者强制：mutator 全 package-private，仅 {@link PrefillSideStore} 在
 * transitionTo CAS 胜者分支 / register / settleRemove 等固定位置调用——
 * 类型（package-private）+ 调用位置（Store 独占）双重约束，条目与其他组件无法绕过。</p>
 */
final class PrefillCounters {

    private final LongAdder[] phaseCounts;
    private final LongAdder engineOwned = new LongAdder();

    PrefillCounters() {
        PrefillPhase[] phases = PrefillPhase.values();
        phaseCounts = new LongAdder[phases.length];
        for (int i = 0; i < phases.length; i++) {
            phaseCounts[i] = new LongAdder();
        }
    }

    /** 登记入账（INIT 人口 +1）。仅 Store.register 调用。 */
    void onRegistered() {
        phaseCounts[PrefillPhase.INIT.ordinal()].increment();
    }

    /** 相位转换（CAS 胜者分支）：旧相位人口 -1、新相位人口 +1。仅 Store.advance 调用。 */
    void onPhaseTransition(PrefillPhase from, PrefillPhase to) {
        phaseCounts[from.ordinal()].decrement();
        phaseCounts[to.ordinal()].increment();
    }

    /** 引擎首见（B 道）：engineOwned +1。仅 Store.noteEngineObserved 调用。 */
    void onEngineOwned() {
        engineOwned.increment();
    }

    /** 条目移除归位（settle/remove）。仅 Store 移除路径调用。 */
    void onRemoved(PrefillRequestState entry) {
        phaseCounts[entry.phase().ordinal()].decrement();
        if (entry.engineOwned()) {
            engineOwned.decrement();
        }
    }

    /** rebuild 引擎收养入账（直接落到收养相位 + engineOwned）。仅 Store.adoptEngineOwned 调用。 */
    void onAdopted(PrefillPhase adoptedPhase, boolean owned) {
        phaseCounts[adoptedPhase.ordinal()].increment();
        if (owned) {
            engineOwned.increment();
        }
    }

    /** 全量重算快照（audit 与发布共用）。 */
    PrefillCounterSnapshot recompute(int inflight) {
        Long[] counts = new Long[phaseCounts.length];
        for (int i = 0; i < phaseCounts.length; i++) {
            counts[i] = phaseCounts[i].sum();
        }
        return new PrefillCounterSnapshot(
                java.util.List.of(counts),
                inflight,
                engineOwned.sum(),
                counts[PrefillPhase.DISPATCHING.ordinal()]);
    }

    /** 对账比对：全量重算值 vs 计数器账（返回偏差描述，空 = 一致）。 */
    java.util.List<String> driftAgainst(java.util.List<PrefillRequestState> allEntries) {
        java.util.List<String> drift = new java.util.ArrayList<>();
        long[] recount = new long[phaseCounts.length];
        long recountOwned = 0;
        for (PrefillRequestState e : allEntries) {
            recount[e.phase().ordinal()]++;
            if (e.engineOwned()) {
                recountOwned++;
            }
        }
        for (PrefillPhase p : PrefillPhase.values()) {
            long counted = phaseCounts[p.ordinal()].sum();
            if (counted != recount[p.ordinal()]) {
                drift.add("P phase " + p + ": counter=" + counted + " recount=" + recount[p.ordinal()]);
            }
        }
        if (engineOwned.sum() != recountOwned) {
            drift.add("P engineOwned: counter=" + engineOwned.sum() + " recount=" + recountOwned);
        }
        return drift;
    }

    /** 清零（rebuild 用；单线程调用）。 */
    void reset() {
        for (LongAdder a : phaseCounts) {
            a.reset();
        }
        engineOwned.reset();
    }
}

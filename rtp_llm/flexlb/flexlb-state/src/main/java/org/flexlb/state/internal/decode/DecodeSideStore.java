package org.flexlb.state.internal.decode;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.DecodeCounterSnapshot;
import org.flexlb.state.GenerationTriple;
import org.flexlb.state.InternalApi;
import org.flexlb.state.ReserveResult;
import org.flexlb.state.TerminalOutcome;
import org.flexlb.state.internal.TombstoneStore;

/**
 * D 侧条目容器 + 计数挂点。
 *
 * <p>计数纪律（P3）：{@link DecodeCounters} 的 mutator 仅在本类固定位置调用
 * ——advance 的 CAS 胜者分支（含 C1 撤预占）/ reserve / settleRemove /
 * releaseRemove / adoptEngineOwned / noteEngineObserved。</p>
 */
@InternalApi
public final class DecodeSideStore {

    private final ConcurrentHashMap<Long, DecodeRequestState> entries = new ConcurrentHashMap<>();
    private final DecodeCounters counters = new DecodeCounters();
    private final TombstoneStore tombstones;
    private final int snapshotInterval;

    private volatile DecodeCounterSnapshot publishedSnapshot;
    private final AtomicInteger transitionTick = new AtomicInteger();
    private final LongAdder overtakenEvents = new LongAdder();

    public DecodeSideStore(TombstoneStore tombstones, int snapshotInterval) {
        this.tombstones = tombstones;
        this.snapshotInterval = snapshotInterval;
        this.publishedSnapshot = counters.recompute(0);
    }

    /**
     * 预约（RESERVED 起步）：登记 D① 影子预占双轨（reservedKv = reservedExpectedKv = expectedKv）
     * 并绑定世代三元组。判重——存活条目 → DUPLICATE_ALIVE；墓碑 → DUPLICATE_TOMBSTONE。
     */
    public ReserveResult reserve(long requestId, long seqLen, long expectedKv,
                                 GenerationTriple binding, long nowMs) {
        if (entries.containsKey(requestId)) {
            return ReserveResult.DUPLICATE_ALIVE;
        }
        if (tombstones.isTombstoned(requestId)) {
            return ReserveResult.DUPLICATE_TOMBSTONE;
        }
        DecodeRequestState fresh = new DecodeRequestState(requestId, seqLen, expectedKv, nowMs);
        fresh.setBindingOnce(binding);
        if (entries.putIfAbsent(requestId, fresh) != null) {
            return ReserveResult.DUPLICATE_ALIVE;
        }
        counters.onReserved(expectedKv);
        return ReserveResult.OK;
    }

    /** 条目查询（终局后已移除，返回 null）。 */
    public DecodeRequestState get(long requestId) {
        return entries.get(requestId);
    }

    /**
     * 相位推进（裁决 ACCEPT 后调用）：CAS 胜者分支内更新计数；
     * <b>C1 临界点</b>——target ≥ D_LOADING 且 from &lt; D_LOADING 的胜者分支
     * 撤 D① 影子预占（reservedKvTotal 减、条目清 0、confirmed +1），
     * 引擎事实 KV 由随后的 noteEngineObserved 接管。
     *
     * @return 是否本调用为 CAS 胜者
     */
    public boolean advance(DecodeRequestState entry, DecodePhase target, long version, long nowMs) {
        DecodePhase from;
        synchronized (entry) {
            from = entry.phase();
            if (!entry.transitionTo(target, version, nowMs)) {
                overtakenEvents.increment();
                return false;
            }
            counters.onPhaseTransition(from, target);
            if (target.ordinal() >= DecodePhase.D_LOADING.ordinal()
                    && from.ordinal() < DecodePhase.D_LOADING.ordinal()) {
                // C1：KV_ALLOCATED（或越级更高）起撤影子预占，引擎事实接管
                counters.onReservationWithdrawn(entry.reservedKv());
                entry.clearReservation();
                counters.onConfirmed();
            }
        }
        tickPublish();
        return true;
    }

    /**
     * B 道观察入账（裁决接受后调用）：引擎首见计数 + D② 引擎事实 KV 增量。
     */
    public void noteEngineObserved(DecodeRequestState entry, long round, long kvTokens, long version) {
        synchronized (entry) {
            boolean first = !entry.engineOwned();
            long oldKv = entry.kvTokensReported();
            entry.markEngineObserved(round, kvTokens, version);
            if (first) {
                counters.onEngineOwned();
            }
            long newKv = entry.kvTokensReported();
            if (newKv != oldKv) {
                counters.onKvReportedDelta(oldKv, newKv);
            }
        }
    }

    /**
     * 终局移除（CAS 单出口）：finishTransition 守卫 + 移除 + 计数归位（含预占/引擎事实 KV 归位）+ 墓碑吸收。
     *
     * @return 是否本调用完成终局（false = 已终局或不存在）
     */
    public boolean settleRemove(DecodeRequestState entry, TerminalOutcome outcome, long nowMs) {
        if (!entry.finishTransition(outcome.state(), nowMs)) {
            return false;
        }
        entries.remove(entry.requestId(), entry);
        synchronized (entry) {
            counters.onRemoved(entry);
        }
        tombstones.absorb(entry.requestId(), outcome, nowMs);
        tickPublish();
        return true;
    }

    /**
     * 释放预约（未终态主动放弃）：撤预占账并移除条目——<b>不进墓碑</b>
     * （释放不是终局，同 requestId 可重新 reserve）。
     */
    public boolean releaseRemove(DecodeRequestState entry, long nowMs) {
        synchronized (entry) {
            if (entry.isFinished()) {
                return false;
            }
            entries.remove(entry.requestId(), entry);
            counters.onRemoved(entry);
        }
        tickPublish();
        return true;
    }

    /**
     * rebuild 引擎收养（P2）：不认识 requestId 的 running 条目按 engineOwned=true
     * 直接入账（P1 重启重建）。收养条目无预占历史（reservedKv=0、expectedKv=0）。
     */
    public DecodeRequestState adoptEngineOwned(long requestId, int endpointId, long generation,
                                               long nowMs, DecodePhase adoptedPhase,
                                               long kvTokens, long version) {
        DecodeRequestState adopted = new DecodeRequestState(requestId, 0L, 0L, nowMs);
        adopted.setBindingOnce(new GenerationTriple(endpointId, generation, -1L));
        adopted.markEngineObserved(0L, kvTokens, version);
        // 条目相位实际推进到收养相位（trace 按格闭包补记）——保证 driftAgainst 全量重算与账一致。
        adopted.transitionTo(adoptedPhase, version, nowMs);
        DecodeRequestState prev = entries.putIfAbsent(requestId, adopted);
        if (prev != null) {
            return prev;
        }
        // 单次记账：收养相位人口 + 引擎事实 KV + engineOwned（+ confirmed 若 ≥ D_LOADING）。
        counters.onAdopted(adoptedPhase, adopted.kvTokensReported(), true);
        tickPublish();
        return adopted;
    }

    /** 墓碑判重委托。 */
    public boolean isTombstoned(long requestId) {
        return tombstones.isTombstoned(requestId);
    }

    /** 迟到事件吸收委托。 */
    public void absorbLateEvent() {
        tombstones.absorbLateEvent();
    }

    /** 迟到取消吸收委托。 */
    public void absorbLateCancel() {
        tombstones.absorbLateCancel();
    }

    /** 活跃条目快照视图（遍历用）。 */
    public List<DecodeRequestState> entriesSnapshot() {
        return List.copyOf(entries.values());
    }

    public long size() {
        return entries.size();
    }

    // ---- 快照发布 ----

    /** 已发布快照（零锁 volatile 读）。 */
    public DecodeCounterSnapshot snapshot() {
        return publishedSnapshot;
    }

    /** 强制重算并发布。 */
    public DecodeCounterSnapshot refreshSnapshot() {
        DecodeCounterSnapshot fresh = recompute();
        publishedSnapshot = fresh;
        return fresh;
    }

    /** 全量重算（audit 与 refresh 共用；尽力快照语义）。 */
    public DecodeCounterSnapshot recompute() {
        return counters.recompute(entries.size());
    }

    /** 对账：计数器增量账 vs 全量重算（不静默修正）。 */
    public List<String> auditDrift() {
        return counters.driftAgainst(entriesSnapshot());
    }

    /** reset（rebuild 用；单线程调用）。 */
    public void reset() {
        entries.clear();
        counters.reset();
        transitionTick.set(0);
        overtakenEvents.reset();
        publishedSnapshot = counters.recompute(0);
    }

    public long overtakenEvents() {
        return overtakenEvents.sum();
    }

    public TombstoneStore tombstones() {
        return tombstones;
    }

    private void tickPublish() {
        if (transitionTick.incrementAndGet() % snapshotInterval == 0) {
            refreshSnapshot();
        }
    }

    // 排除误用：Map 快照仅供调试
    Map<Long, DecodeRequestState> rawView() {
        return Map.copyOf(entries);
    }
}

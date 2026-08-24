package org.flexlb.state.internal.decode;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.DecodeCounterSnapshot;
import org.flexlb.state.DecodeEndpointCounters;
import org.flexlb.state.GenerationTriple;
import org.flexlb.state.InternalApi;
import org.flexlb.state.ReserveResult;
import org.flexlb.state.TerminalOutcome;
import org.flexlb.state.internal.TombstoneStore;

/**
 * D 侧条目容器 + 计数挂点。
 *
 * <p>计数纪律：{@link DecodeCounters} 与端点级 {@link DecodeEndpointCountersBook}
 * 的 mutator 仅在本类固定位置调用——advance 的 CAS 胜者分支（含计费
 * 归属移交撤预占）/ reserve / settleRemove / releaseRemove /
 * adoptEngineOwned / noteEngineObserved；全局账与端点账在同一临界区内
 * 同步更新（单一写者纪律，两套账天然一致）。</p>
 */
@InternalApi
public final class DecodeSideStore {

    private final ConcurrentHashMap<Long, DecodeRequestState> entries = new ConcurrentHashMap<>();
    /** byEndpoint 二级索引（清理层证据通道/TTL 扫描结构）：endpointId → 名下条目。 */
    private final ConcurrentHashMap<Integer, ConcurrentHashMap<Long, DecodeRequestState>> byEndpoint = new ConcurrentHashMap<>();
    private final DecodeCounters counters = new DecodeCounters();
    /** 端点级增量计数簿（调度读数 O(1) 数据源；与全局账同一写者位置同步更新）。 */
    private final DecodeEndpointCountersBook epCounters = new DecodeEndpointCountersBook();
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
     * 预约（RESERVED 起步）：登记影子预占双轨（reservedKv = reservedExpectedKv = expectedKv）
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
        indexEndpoint(fresh); // reserve 即绑定——入 byEndpoint 索引
        counters.onReserved(expectedKv);
        epCounters.onReserved(binding.endpointId(), seqLen, expectedKv);
        return ReserveResult.OK;
    }

    /** 条目查询（终局后已移除，返回 null）。 */
    public DecodeRequestState get(long requestId) {
        return entries.get(requestId);
    }

    /**
     * 相位推进（裁决 ACCEPT 后调用）：CAS 胜者分支内更新计数；
     * <b>计费归属临界点</b>——target ≥ D_LOADING 且 from &lt; D_LOADING 的胜者分支
     * 撤影子预占（reservedKvTotal 减、条目清 0、confirmed +1），
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
            Integer epId = boundEndpointId(entry);
            if (epId != null) {
                epCounters.onPhaseTransition(epId, from, target);
            }
            counters.onPhaseTransition(from, target);
            if (target.ordinal() >= DecodePhase.D_LOADING.ordinal()
                    && from.ordinal() < DecodePhase.D_LOADING.ordinal()) {
                // 计费归属移交：引擎加载临界相位起撤影子预占，引擎事实接管
                counters.onReservationWithdrawn(entry.reservedKv());
                if (epId != null) {
                    epCounters.onReservationConfirmed(epId, entry.reservedKv(), entry.seqLen());
                }
                entry.clearReservation();
                counters.onConfirmed();
            }
        }
        tickPublish();
        return true;
    }

    /**
     * 引擎上报观察入账（裁决接受后调用）：引擎首见计数 + 引擎事实 KV 增量。
     */
    public void noteEngineObserved(DecodeRequestState entry, long round, long kvTokens, long version) {
        synchronized (entry) {
            boolean first = !entry.engineOwned();
            long oldKv = entry.kvTokensReported();
            entry.markEngineObserved(round, kvTokens, version);
            Integer epId = boundEndpointId(entry);
            if (first) {
                counters.onEngineOwned();
                if (epId != null) {
                    epCounters.onEngineOwned(epId);
                }
            }
            long newKv = entry.kvTokensReported();
            if (newKv != oldKv) {
                counters.onKvReportedDelta(oldKv, newKv);
                if (epId != null) {
                    epCounters.onKvReportedDelta(epId, newKv - oldKv);
                }
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
        unindexEndpoint(entry);
        synchronized (entry) {
            counters.onRemoved(entry);
            Integer epId = boundEndpointId(entry);
            if (epId != null) {
                epCounters.onRemoved(epId, entry);
            }
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
            unindexEndpoint(entry);
            counters.onRemoved(entry);
            Integer epId = boundEndpointId(entry);
            if (epId != null) {
                epCounters.onRemoved(epId, entry);
            }
        }
        tickPublish();
        return true;
    }

    /**
     * rebuild 引擎收养：不认识 requestId 的 running 条目按 engineOwned=true
     * 直接入账（重启重建）。收养条目无预占历史（reservedKv=0、expectedKv=0）。
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
        indexEndpoint(adopted); // 收养即绑定观察端点——入 byEndpoint 索引
        // 单次记账：收养相位人口 + 引擎事实 KV + engineOwned（+ confirmed 若 ≥ 引擎加载临界相位）。
        counters.onAdopted(adoptedPhase, adopted.kvTokensReported(), true);
        epCounters.onAdopted(endpointId, adoptedPhase, adopted.kvTokensReported(), true);
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

    // ---- byEndpoint 二级索引（M4 janitor 扫描结构）----

    /**
     * 端点索引登记（幂等 put；重绑后旧桶残留由 {@link #entriesByEndpoint} 自愈清除）。
     * 调用点：reserve / adoptEngineOwned（store 内部）与 onDispatched 重绑后（StateLedger 门面）。
     * 判据是 {@code isBound()}（UNBOUND 三元组哨兵）——endpointId 本身可为负
     * （如 flexlb-sync 影子桥的 ipPort 哈希），不得用符号判定。
     */
    public void indexEndpoint(DecodeRequestState entry) {
        if (!entry.isBound()) {
            return;
        }
        GenerationTriple binding = entry.binding();
        byEndpoint.computeIfAbsent(binding.endpointId(), k -> new ConcurrentHashMap<>())
                .put(entry.requestId(), entry);
    }

    /** 端点索引移除（终局/释放时；UNBOUND 条目 no-op）。 */
    public void unindexEndpoint(DecodeRequestState entry) {
        if (!entry.isBound()) {
            return;
        }
        GenerationTriple binding = entry.binding();
        ConcurrentHashMap<Long, DecodeRequestState> bucket = byEndpoint.get(binding.endpointId());
        if (bucket != null) {
            bucket.remove(entry.requestId(), entry);
            if (bucket.isEmpty()) {
                byEndpoint.remove(binding.endpointId(), bucket);
            }
        }
    }

    /**
     * 该端点名下活跃条目视图（量级 = 每 endpoint 活跃条目数，非全账本扫描）。
     * 自愈：DISPATCHED 前重绑的旧桶残留顺手清除。
     */
    public List<DecodeRequestState> entriesByEndpoint(int endpointId) {
        ConcurrentHashMap<Long, DecodeRequestState> bucket = byEndpoint.get(endpointId);
        if (bucket == null) {
            return List.of();
        }
        List<DecodeRequestState> out = new ArrayList<>(bucket.size());
        for (DecodeRequestState e : bucket.values()) {
            if (e.isBound() && e.binding().endpointId() == endpointId) {
                out.add(e);
            } else {
                bucket.remove(e.requestId(), e); // 重绑后旧桶残留自愈
            }
        }
        return out;
    }

    /** 未绑定条目视图（D 侧 reserve/adopt 均即绑定——恒空，对称提供）。 */
    public List<DecodeRequestState> unboundEntries() {
        List<DecodeRequestState> out = new ArrayList<>();
        for (DecodeRequestState e : entries.values()) {
            if (!e.isBound()) {
                out.add(e);
            }
        }
        return out;
    }

    /** 已登记端点集合快照（janitor 轮转游标用）。 */
    public Set<Integer> trackedEndpointIds() {
        return Set.copyOf(byEndpoint.keySet());
    }

    /**
     * 绑定/重绑（dispatch 挂点）：首次绑定入桶（现态全账）；派发前重绑
     * 做桶间全账迁移（同一临界区内完成——旧桶减/新桶加原子）；
     * 引擎加载临界相位后不可变（拒绝重绑，保留原绑定）。索引随绑定同步维护。
     *
     * @return 绑定是否生效（false = 已不可变，保留原绑定）
     */
    public boolean bindEndpoint(DecodeRequestState entry, GenerationTriple binding) {
        Integer fromEndpointId;
        synchronized (entry) {
            GenerationTriple old = entry.binding();
            boolean wasBound = entry.isBound();
            if (!entry.setBindingOnce(binding)) {
                return false; // 已不可变（保留原绑定）：账不动
            }
            fromEndpointId = wasBound ? old.endpointId() : null;
            if (fromEndpointId != null && fromEndpointId != binding.endpointId()) {
                // 派发前重绑：全账迁移（本临界区内）
                epCounters.transferEntry(fromEndpointId, binding.endpointId(), entry);
            } else if (fromEndpointId == null) {
                // 首次绑定：现态入桶
                epCounters.onEntryAdded(binding.endpointId(), entry);
            }
        }
        if (fromEndpointId != null && fromEndpointId != binding.endpointId()) {
            unindexFromEndpoint(entry, fromEndpointId); // 旧桶索引清除（防残留）
        }
        indexEndpoint(entry);
        return true;
    }

    /** 旧桶索引清除（重绑专用：按指定旧端点移除，当前 binding 已指向新端点）。 */
    private void unindexFromEndpoint(DecodeRequestState entry, int oldEndpointId) {
        ConcurrentHashMap<Long, DecodeRequestState> bucket = byEndpoint.get(oldEndpointId);
        if (bucket != null) {
            bucket.remove(entry.requestId(), entry);
            if (bucket.isEmpty()) {
                byEndpoint.remove(oldEndpointId, bucket);
            }
        }
    }

    /** 已绑定条目的端点 id（未绑定返回 null——不入端点计数桶）。 */
    private static Integer boundEndpointId(DecodeRequestState entry) {
        return entry.isBound() ? entry.binding().endpointId() : null;
    }

    /**
     * 端点级派生计数（读取换权阶段调度读数数据源）：端点级增量计数簿的
     * O(1) 无锁快照（写路径与全局账同一临界区增量维护；不再按需遍历
     * 聚合——把 O(端点活跃条目) 扫描从调度热路径上拿掉）。未确认口径
     * （phase &lt; 引擎加载临界相位）对应旧双层账本预占层的期望 KV /
     * prompt KV 双轨计数。
     */
    public DecodeEndpointCounters endpointCounters(int endpointId) {
        return epCounters.countersOf(endpointId);
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
        List<String> drift = new ArrayList<>(counters.driftAgainst(entriesSnapshot()));
        drift.addAll(auditDriftPerEndpoint());
        return drift;
    }

    /** 对账（端点级）：桶增量账 vs 按已绑定活跃条目全量重算（不静默修正）。 */
    public List<String> auditDriftPerEndpoint() {
        Map<Integer, List<DecodeRequestState>> bound = new java.util.HashMap<>();
        for (DecodeRequestState e : entries.values()) {
            if (e.isBound()) {
                bound.computeIfAbsent(e.binding().endpointId(), k -> new ArrayList<>()).add(e);
            }
        }
        return epCounters.driftAgainst(bound);
    }

    /** reset（rebuild 用；单线程调用）。 */
    public void reset() {
        entries.clear();
        byEndpoint.clear();
        counters.reset();
        epCounters.reset();
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

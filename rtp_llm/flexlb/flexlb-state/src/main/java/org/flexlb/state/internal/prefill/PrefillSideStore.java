package org.flexlb.state.internal.prefill;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.GenerationTriple;
import org.flexlb.state.InternalApi;
import org.flexlb.state.PrefillCounterSnapshot;
import org.flexlb.state.RegisterResult;
import org.flexlb.state.SettleReason;
import org.flexlb.state.TerminalOutcome;
import org.flexlb.state.internal.TombstoneStore;

/**
 * P 侧条目容器 + 计数挂点。
 *
 * <p>计数纪律：{@link PrefillCounters} 的 mutator 仅在本类固定位置调用
 * ——advance 的 CAS 胜者分支 / register / settleRemove / adoptEngineOwned /
 * noteEngineObserved。条目与其他组件不可直达计数器。</p>
 */
@InternalApi
public final class PrefillSideStore {

    private final ConcurrentHashMap<Long, PrefillRequestState> entries = new ConcurrentHashMap<>();
    /** byEndpoint 二级索引（M4 janitor 证据通道/TTL 扫描结构）：endpointId → 名下条目。 */
    private final ConcurrentHashMap<Integer, ConcurrentHashMap<Long, PrefillRequestState>> byEndpoint = new ConcurrentHashMap<>();
    private final PrefillCounters counters = new PrefillCounters();
    private final TombstoneStore tombstones;
    private final int snapshotInterval;

    private volatile PrefillCounterSnapshot publishedSnapshot;
    private final AtomicInteger transitionTick = new AtomicInteger();
    private final LongAdder overtakenEvents = new LongAdder();

    public PrefillSideStore(TombstoneStore tombstones, int snapshotInterval) {
        this.tombstones = tombstones;
        this.snapshotInterval = snapshotInterval;
        this.publishedSnapshot = counters.recompute(0);
    }

    /**
     * 登记（INIT 起步）：判重——存活条目 → DUPLICATE_ALIVE；墓碑 → DUPLICATE_TOMBSTONE。
     */
    public RegisterResult register(long requestId, long batchId) {
        if (entries.containsKey(requestId)) {
            return RegisterResult.DUPLICATE_ALIVE;
        }
        if (tombstones.isTombstoned(requestId)) {
            return RegisterResult.DUPLICATE_TOMBSTONE;
        }
        PrefillRequestState fresh = new PrefillRequestState(requestId, batchId, System.currentTimeMillis());
        if (entries.putIfAbsent(requestId, fresh) != null) {
            return RegisterResult.DUPLICATE_ALIVE;
        }
        // register 时未绑定（UNBOUND）——不进 byEndpoint 索引，onDispatched 绑定后进
        counters.onRegistered();
        return RegisterResult.OK;
    }

    /** 条目查询（终局后已移除，返回 null）。 */
    public PrefillRequestState get(long requestId) {
        return entries.get(requestId);
    }

    /**
     * 相位推进（裁决 ACCEPT 后调用）：CAS 胜者分支内更新计数；
     * CAS 失败（并发超车）计 overtaken。
     *
     * @return 是否本调用为 CAS 胜者
     */
    public boolean advance(PrefillRequestState entry, PrefillPhase target, long version, long nowMs) {
        PrefillPhase from;
        synchronized (entry) {
            from = entry.phase();
            if (!entry.transitionTo(target, version, nowMs)) {
                overtakenEvents.increment();
                return false;
            }
            counters.onPhaseTransition(from, target);
            if (target == PrefillPhase.DISPATCHED) {
                entry.noteDispatched(nowMs);
            }
        }
        tickPublish();
        return true;
    }

    /**
     * 引擎上报观察入账（裁决接受后调用）：引擎首见计数。
     */
    public void noteEngineObserved(PrefillRequestState entry, long round, long kvTokens, long version) {
        synchronized (entry) {
            boolean first = !entry.engineOwned();
            entry.markEngineObserved(round, kvTokens, version);
            if (first) {
                counters.onEngineOwned();
            }
        }
    }

    /**
     * 终局移除（CAS 单出口）：finishTransition 守卫 + 移除 + 计数归位 + 墓碑吸收。
     *
     * @return 是否本调用完成终局（false = 已终局或不存在）
     */
    public boolean settleRemove(PrefillRequestState entry, TerminalOutcome outcome, long nowMs) {
        if (!entry.finishTransition(outcome.state(), nowMs)) {
            return false;
        }
        entries.remove(entry.requestId(), entry);
        unindexEndpoint(entry);
        synchronized (entry) {
            counters.onRemoved(entry);
        }
        tombstones.absorb(entry.requestId(), outcome, nowMs);
        tickPublish();
        return true;
    }

    /**
     * rebuild 引擎收养：不认识 requestId 的 running 条目按 batchId=-1、
     * engineOwned=true 直接入账（重启重建）。
     */
    public PrefillRequestState adoptEngineOwned(long requestId, int endpointId, long generation,
                                                long nowMs, PrefillPhase adoptedPhase,
                                                long kvTokens, long version) {
        PrefillRequestState adopted = new PrefillRequestState(requestId, PrefillRequestState.NO_BATCH, nowMs);
        adopted.setBindingOnce(new org.flexlb.state.GenerationTriple(endpointId, generation, -1L));
        adopted.markEngineObserved(0L, kvTokens, version);
        // 条目相位实际推进到收养相位（trace 按格闭包补记沿途）——保证 driftAgainst 全量重算与账一致。
        adopted.transitionTo(adoptedPhase, version, nowMs);
        PrefillRequestState prev = entries.putIfAbsent(requestId, adopted);
        if (prev != null) {
            return prev;
        }
        indexEndpoint(adopted); // 收养即绑定观察端点——入 byEndpoint 索引
        // 单次记账：收养相位人口 +1、engineOwned +1（构造时未走 register，无 INIT 账可减）。
        counters.onAdopted(adoptedPhase, true);
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
    public List<PrefillRequestState> entriesSnapshot() {
        return List.copyOf(entries.values());
    }

    // ---- byEndpoint 二级索引（M4 janitor 扫描结构）----

    /**
     * 端点索引登记（幂等 put；重绑后旧桶残留由 {@link #entriesByEndpoint} 自愈清除）。
     * 调用点：adoptEngineOwned（store 内部）与 onDispatched 绑定后（StateLedger 门面）。
     * 判据是 {@code isBound()}（UNBOUND 三元组哨兵）——endpointId 本身可为负
     * （如 flexlb-sync 影子桥的 ipPort 哈希），不得用符号判定。
     */
    public void indexEndpoint(PrefillRequestState entry) {
        if (!entry.isBound()) {
            return;
        }
        GenerationTriple binding = entry.binding();
        byEndpoint.computeIfAbsent(binding.endpointId(), k -> new ConcurrentHashMap<>())
                .put(entry.requestId(), entry);
    }

    /** 端点索引移除（终局/释放时；UNBOUND 条目 no-op）。 */
    public void unindexEndpoint(PrefillRequestState entry) {
        if (!entry.isBound()) {
            return;
        }
        GenerationTriple binding = entry.binding();
        ConcurrentHashMap<Long, PrefillRequestState> bucket = byEndpoint.get(binding.endpointId());
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
    public List<PrefillRequestState> entriesByEndpoint(int endpointId) {
        ConcurrentHashMap<Long, PrefillRequestState> bucket = byEndpoint.get(endpointId);
        if (bucket == null) {
            return List.of();
        }
        List<PrefillRequestState> out = new ArrayList<>(bucket.size());
        for (PrefillRequestState e : bucket.values()) {
            if (e.isBound() && e.binding().endpointId() == endpointId) {
                out.add(e);
            } else {
                bucket.remove(e.requestId(), e); // 重绑后旧桶残留自愈
            }
        }
        return out;
    }

    /** 未绑定条目视图（P 侧排队中：register 后未 dispatch；TTL/hard cap 由 janitor 每 tick 全扫）。 */
    public List<PrefillRequestState> unboundEntries() {
        List<PrefillRequestState> out = new ArrayList<>();
        for (PrefillRequestState e : entries.values()) {
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

    /** 按批次聚类（BatchShadowView 数据源）。 */
    public List<PrefillRequestState> batchMembers(long batchId) {
        List<PrefillRequestState> members = new ArrayList<>();
        for (PrefillRequestState e : entries.values()) {
            if (e.batchId() == batchId) {
                members.add(e);
            }
        }
        return members;
    }

    public long size() {
        return entries.size();
    }

    // ---- 快照发布 ----

    /** 已发布快照（零锁 volatile 读）。 */
    public PrefillCounterSnapshot snapshot() {
        return publishedSnapshot;
    }

    /** 强制重算并发布。 */
    public PrefillCounterSnapshot refreshSnapshot() {
        PrefillCounterSnapshot fresh = recompute();
        publishedSnapshot = fresh;
        return fresh;
    }

    /** 全量重算（audit 与 refresh 共用；尽力快照语义）。 */
    public PrefillCounterSnapshot recompute() {
        return counters.recompute(entries.size());
    }

    /** 对账：计数器增量账 vs 全量重算（不静默修正）。 */
    public List<String> auditDrift() {
        return counters.driftAgainst(entriesSnapshot());
    }

    /** reset（rebuild 用；单线程调用）。 */
    public void reset() {
        entries.clear();
        byEndpoint.clear();
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
    Map<Long, PrefillRequestState> rawView() {
        return Map.copyOf(entries);
    }
}

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
import org.flexlb.state.PrefillEndpointCounters;
import org.flexlb.state.RegisterResult;
import org.flexlb.state.SettleReason;
import org.flexlb.state.TerminalOutcome;
import org.flexlb.state.internal.TombstoneStore;

/**
 * P 侧条目容器 + 计数挂点。
 *
 * <p>计数纪律：{@link PrefillCounters} 与端点级 {@link PrefillEndpointCountersBook}
 * 的 mutator 仅在本类固定位置调用——advance 的 CAS 胜者分支 / register /
 * settleRemove / adoptEngineOwned / noteEngineObserved；全局账与端点账
 * 在同一临界区内同步更新（单一写者纪律）。条目与其他组件不可直达计数器。</p>
 */
@InternalApi
public final class PrefillSideStore {

    private final ConcurrentHashMap<Long, PrefillRequestState> entries = new ConcurrentHashMap<>();
    /** byEndpoint 二级索引（清理层证据通道/TTL 扫描结构）：endpointId → 名下条目。 */
    private final ConcurrentHashMap<Integer, ConcurrentHashMap<Long, PrefillRequestState>> byEndpoint = new ConcurrentHashMap<>();
    private final PrefillCounters counters = new PrefillCounters();
    /** 端点级增量计数簿（调度读数 O(1) 数据源；与全局账同一写者位置同步更新）。 */
    private final PrefillEndpointCountersBook epCounters = new PrefillEndpointCountersBook();
    private final TombstoneStore tombstones;
    private final int snapshotInterval;

    private volatile PrefillCounterSnapshot publishedSnapshot;
    private final AtomicInteger transitionTick = new AtomicInteger();
    private final LongAdder overtakenEvents = new LongAdder();
    /** 快路径 settle 胜者计数（引擎 finished / 本地 settle / cancel 双清——超车三分的正常通道胜）。 */
    private final LongAdder fastPathSettles = new LongAdder();

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
            Integer epId = boundEndpointId(entry);
            if (epId != null) {
                epCounters.onPhaseTransition(epId, from, target);
            }
            if (target == PrefillPhase.DISPATCHED) {
                entry.noteDispatched(nowMs);
            }
        }
        tickPublish();
        return true;
    }

    /**
     * 引擎上报观察入账（裁决接受后调用）：引擎首见计数。
     *
     * <p><b>在册守卫</b>：调用方（dispatchRunning）经 get 拿到条目引用与
     * 本方法记账之间存在窗口——并发终局移除（settleRemove）可能已先完成
     * 出账（出账时 engineOwned=false 不减首见账，因从未加过），随后本方法
     * 的首见 +1 打在已移除条目上<b>永久悬挂</b>（重启演练实测 engineOwned
     * counter 恒高 1）。守卫在条目锁内验证在册归属：移除已发生则整体 no-op
     * （迟到引擎观察，墓碑已吸收）；守卫通过则后续并发移除的出账（同一
     * 条目锁临界区）必读到本方法置位的现态而对称出账——任意交错恒配平。</p>
     */
    public void noteEngineObserved(PrefillRequestState entry, long round, long kvTokens, long version) {
        synchronized (entry) {
            if (entries.get(entry.requestId()) != entry) {
                return; // 已被并发移除：迟到引擎观察，不产生任何计数
            }
            boolean first = !entry.engineOwned();
            entry.markEngineObserved(round, kvTokens, version);
            if (first) {
                counters.onEngineOwned();
                Integer epId = boundEndpointId(entry);
                if (epId != null) {
                    epCounters.onEngineOwned(epId);
                }
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
        // 墓碑先落、条目后移：若先移除条目再吸收墓碑，两者之间存在观察窗口——
        // 并发的迟到 finished/settle 读到“条目不在且墓碑未落”会误记 unknown。
        // 先吸收墓碑可闭合该窗口（absorb 幂等，CAS 守卫保证唯一胜者）：
        // 条目仍在时 CAS 守卫拦截双重结算，条目移除后迟到事件被墓碑吸收。
        // 墓碑携带终局时刻的 trace 环快照（诊断端点故事线，保留期内可查）。
        tombstones.absorb(entry.requestId(), outcome, nowMs, entry.traceSnapshot());
        entries.remove(entry.requestId(), entry);
        unindexEndpoint(entry);
        synchronized (entry) {
            counters.onRemoved(entry);
            Integer epId = boundEndpointId(entry);
            if (epId != null) {
                epCounters.onRemoved(epId, entry);
            }
        }
        fastPathSettles.increment();
        tickPublish();
        return true;
    }

    /**
     * rebuild 引擎收养：不认识 requestId 的 running 条目按 batchId=-1、
     * engineOwned=true 直接入账（重启重建）。
     *
     * <p>记账口径：条目经 {@code putIfAbsent} 对外可见后与本方法的入账之间
     * 存在窗口——并发观察线程（事件泵后续 tick）可能已推进条目相位（advance
     * 的桶迁移账以收养相位为 from 先行执行）。因此入账必须以<b>收养时刻口径</b>
     * （adoptedPhase 参数 + 可见前预取的 predictedMs）记账，与并发迁移账
     * 任意交错下恒配平；不得在窗口后按条目现态入账（会造成收养相位 −1 /
     * 推进相位 +1 的端点簿永久漂移——与 D 侧 DecodeSideStore.adoptEngineOwned
     * 的记账口径对称）。</p>
     */
    public PrefillRequestState adoptEngineOwned(long requestId, int endpointId, long generation,
                                                long nowMs, PrefillPhase adoptedPhase,
                                                long kvTokens, long version) {
        PrefillRequestState adopted = new PrefillRequestState(requestId, PrefillRequestState.NO_BATCH, nowMs);
        adopted.setBindingOnce(new org.flexlb.state.GenerationTriple(endpointId, generation, -1L));
        adopted.markEngineObserved(0L, kvTokens, version);
        // 条目相位实际推进到收养相位（trace 按格闭包补记沿途）——保证 driftAgainst 全量重算与账一致。
        adopted.transitionTo(adoptedPhase, version, nowMs);
        // 入账口径预取（条目此刻不对外可见，无并发写；收养条目构造后 predictedBatchMs 亦不可变）。
        long adoptedPredictedMs = Math.max(adopted.predictedBatchMs(), 0L);
        PrefillRequestState prev = entries.putIfAbsent(requestId, adopted);
        if (prev != null) {
            return prev;
        }
        indexEndpoint(adopted); // 收养即绑定观察端点——入 byEndpoint 索引
        // 单次记账：收养相位人口 +1、engineOwned +1（构造时未走 register，无 INIT 账可减）。
        counters.onAdopted(adoptedPhase, true);
        epCounters.onAdopted(endpointId, adoptedPhase, true, adoptedPredictedMs);
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

    // ---- byEndpoint 二级索引（janitor 扫描结构）----

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

    /**
     * 绑定/重绑（dispatch 挂点）：首次绑定入桶（现态全账）；派发前重绑
     * 做桶间全账迁移（同一临界区内完成——旧桶减/新桶加原子）；
     * DISPATCHED 后不可变（拒绝重绑，保留原绑定）。索引随绑定同步维护。
     *
     * @return 绑定是否生效（false = 已不可变，保留原绑定）
     */
    public boolean bindEndpoint(PrefillRequestState entry, GenerationTriple binding) {
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
    private void unindexFromEndpoint(PrefillRequestState entry, int oldEndpointId) {
        ConcurrentHashMap<Long, PrefillRequestState> bucket = byEndpoint.get(oldEndpointId);
        if (bucket != null) {
            bucket.remove(entry.requestId(), entry);
            if (bucket.isEmpty()) {
                byEndpoint.remove(oldEndpointId, bucket);
            }
        }
    }

    /** 已绑定条目的端点 id（未绑定返回 null——排队/攒批窗口不入端点计数桶）。 */
    private static Integer boundEndpointId(PrefillRequestState entry) {
        return entry.isBound() ? entry.binding().endpointId() : null;
    }

    /**
     * 端点级派生计数（读取换权阶段调度读数数据源）：端点级增量计数簿的
     * O(1) 无锁快照（写路径与全局账同一临界区增量维护；不再按需遍历
     * 聚合——把 O(端点活跃条目) 扫描从调度热路径上拿掉）。条目在
     * dispatch 绑定后进端点计数桶——排队/攒批窗口由派发编排侧
     * （batcher 队列深度）单独覆盖，此处不含。
     */
    public PrefillEndpointCounters endpointCounters(int endpointId) {
        return epCounters.countersOf(endpointId);
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
        List<String> drift = new ArrayList<>(counters.driftAgainst(entriesSnapshot()));
        drift.addAll(auditDriftPerEndpoint());
        return drift;
    }

    /** 对账（端点级）：桶增量账 vs 按已绑定活跃条目全量重算（不静默修正）。 */
    public List<String> auditDriftPerEndpoint() {
        Map<Integer, List<PrefillRequestState>> bound = new java.util.HashMap<>();
        for (PrefillRequestState e : entries.values()) {
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
        fastPathSettles.reset();
        publishedSnapshot = counters.recompute(0);
    }

    public long overtakenEvents() {
        return overtakenEvents.sum();
    }

    /** 快路径 settle 胜者计数（观测层超车三分之正常通道胜）。 */
    public long fastPathSettles() {
        return fastPathSettles.sum();
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

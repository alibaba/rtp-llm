package org.flexlb.state;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.LongAdder;
import org.flexlb.state.internal.FenceRegistry;
import org.flexlb.state.internal.GenerationTracker;
import org.flexlb.state.internal.TombstoneStore;
import org.flexlb.state.internal.decode.DecodeLattice;
import org.flexlb.state.internal.decode.DecodePhase;
import org.flexlb.state.internal.decode.DecodeRequestState;
import org.flexlb.state.internal.decode.DecodeSideStore;
import org.flexlb.state.internal.prefill.PrefillLattice;
import org.flexlb.state.internal.prefill.PrefillPhase;
import org.flexlb.state.internal.prefill.PrefillRequestState;
import org.flexlb.state.internal.prefill.PrefillSideStore;
import org.flexlb.state.spi.EngineObservation;
import org.flexlb.state.spi.StateEndpointRef;
import org.flexlb.state.spi.StateRole;

/**
 * 状态账本门面：P/D 双侧状态核心的唯一组装点与<b>跨侧规则唯一入口</b>。
 *
 * <h2>事件单一入口</h2>
 * 引擎事件（running/finished 观察）不暴露在 {@link PrefillSide}/{@link DecodeSide} 上——
 * 一律经 {@link #observe(EngineObservation)} 分发，保证跨侧规则只有一个触发点。
 * 本地决策事件（register/queued/dispatching/dispatched/reserve/release/settle）
 * 走类型化侧门面 {@link #prefill()} / {@link #decode()}。
 *
 * <h2>跨侧规则（observe 链路独占，本类是唯一实现处）</h2>
 * <ol>
 *   <li><b>C7/L4 P 释放点 = D 确认点</b>：D 侧收到 KV_ALLOCATED（进 D_LOADING）的同 tick，
 *       若 P 侧同请求条目仍处 P_RECEIVED..P_WAITING_LOADED，闭包推进到 PREFILL_DONE
 *       （P 账释放无中间窗口）；P_RUNNING 不收缩（边算边传重叠窗口，见 {@link #crossSide()}）。</li>
 *   <li><b>C1 临界点（超卖窗口修正）</b>：D 侧 DISPATCHED 相位收到 RECEIVED 期间<b>双记</b>
 *       （D① 影子预占保持 + 引擎已见标记）；KV_ALLOCATED 起撤 D① 预占
 *       （reservedKv 出账）、D② 引擎事实 KV 接管。</li>
 *   <li><b>F1 因果闭包</b>：D 侧 finished(success) ⇒ 整请求完成 ⇒ 同 tick 收缩未终局的
 *       P 侧条目为 COMPLETED（SettleReason.CAUSAL_CLOSURE 通道）。</li>
 *   <li><b>cancel 双清</b>：任一侧 settle(CANCELLED) 成功的同 tick 清对侧同请求条目
 *       （两侧计数独立减，互不依赖）。</li>
 * </ol>
 *
 * <h2>世代屏障（S8）</h2>
 * observe 先做整报级世代校验（端点当前登记代 vs 报文代，不匹配整报拒绝），
 * 条目级再按 binding 世代裁决（REJECT_GENERATION）。换代经 {@link #newGeneration}。
 */
public final class StateLedger {

    private final StateLedgerConfig config;
    private final GenerationTracker generations;
    private final FenceRegistry fences;
    private final PrefillSideStore pStore;
    private final DecodeSideStore dStore;
    private final PrefillSide pFacade;
    private final DecodeSide dFacade;

    private final EnumMap<PhaseVerdict, LongAdder> verdictCounts = new EnumMap<>(PhaseVerdict.class);
    private final LongAdder unknownRunningEvents = new LongAdder();
    private final LongAdder unknownFinishedEvents = new LongAdder();

    public StateLedger() {
        this(StateLedgerConfig.defaults());
    }

    public StateLedger(StateLedgerConfig config) {
        this(config, System.currentTimeMillis());
    }

    /** 测试可注入 epoch（GenerationTracker 兜底基准）。 */
    StateLedger(StateLedgerConfig config, long epochMs) {
        this.config = Objects.requireNonNull(config, "config");
        this.generations = new GenerationTracker(epochMs);
        this.fences = new FenceRegistry(config.fenceTtlMs());
        TombstoneStore pTombstones = new TombstoneStore(config.tombstoneRetentionMs());
        TombstoneStore dTombstones = new TombstoneStore(config.tombstoneRetentionMs());
        this.pStore = new PrefillSideStore(pTombstones, config.snapshotIntervalTransitions());
        this.dStore = new DecodeSideStore(dTombstones, config.snapshotIntervalTransitions());
        this.pFacade = new PrefillSideImpl();
        this.dFacade = new DecodeSideImpl();
        for (PhaseVerdict v : PhaseVerdict.values()) {
            verdictCounts.put(v, new LongAdder());
        }
    }

    /** P 侧类型化门面。 */
    public PrefillSide prefill() {
        return pFacade;
    }

    /** D 侧类型化门面。 */
    public DecodeSide decode() {
        return dFacade;
    }

    /**
     * EP 换代登记（S8）：分配并登记该端点新代际
     * （= max(进程 epoch, 上一代 + 1)，单调且防 master 重启归零）。
     */
    public long newGeneration(StateEndpointRef endpoint) {
        Objects.requireNonNull(endpoint, "endpoint");
        return generations.nextGeneration(endpoint.endpointId());
    }

    /**
     * 引擎观察唯一入口：整报世代校验（S8）→ 逐条按 side 分发 → 裁决 →
     * 推进/丢弃/终局 + 跨侧规则（见类 javadoc）。
     */
    public void observe(EngineObservation observation) {
        observeInternal(observation, false);
    }

    /**
     * 跨侧推导视图（S9：只读推导，不参与裁决）。
     *
     * <p>KV_TRANSFERRING = P 条目 P_RUNNING ∧ D 条目已报（engineOwned）——
     * 边算边传的传输重叠窗口。</p>
     */
    public CrossSideView crossSide() {
        List<Long> transferring = new ArrayList<>();
        for (PrefillRequestState p : pStore.entriesSnapshot()) {
            if (p.phase() == PrefillPhase.P_RUNNING) {
                DecodeRequestState d = dStore.get(p.requestId());
                if (d != null && d.engineOwned()) {
                    transferring.add(p.requestId());
                }
            }
        }
        return new CrossSideView(transferring);
    }

    /**
     * P1 重启重建：清空两侧账后按序重放全量历史。
     *
     * <p>重放中不认识的 running 条目按引擎收养入账（P2：batchId=-1、engineOwned=true、
     * binding=观察端点世代）；历史 finished 对未开/已收条目跳过。世代登记经
     * observeGeneration merge max（防重建归零）。</p>
     */
    public void rebuild(List<EngineObservation> fullHistory) {
        Objects.requireNonNull(fullHistory, "fullHistory");
        pStore.reset();
        dStore.reset();
        pStore.tombstones().reset();
        dStore.tombstones().reset();
        verdictCounts.values().forEach(LongAdder::reset);
        unknownRunningEvents.reset();
        unknownFinishedEvents.reset();
        for (EngineObservation obs : fullHistory) {
            observeInternal(obs, true);
        }
        pStore.refreshSnapshot();
        dStore.refreshSnapshot();
    }

    /** 全局聚合快照（零锁读：聚合两侧已发布 volatile 快照 + 观测计数）。 */
    public LedgerSnapshot snapshot() {
        Map<PhaseVerdict, Long> counts = new EnumMap<>(PhaseVerdict.class);
        for (Map.Entry<PhaseVerdict, LongAdder> e : verdictCounts.entrySet()) {
            counts.put(e.getKey(), e.getValue().sum());
        }
        return new LedgerSnapshot(
                pStore.snapshot(),
                dStore.snapshot(),
                pStore.tombstones().size(),
                dStore.tombstones().size(),
                generations.crossGenerationRejects(),
                pStore.tombstones().lateEventCount() + dStore.tombstones().lateEventCount(),
                pStore.tombstones().lateCancelCount() + dStore.tombstones().lateCancelCount(),
                unknownRunningEvents.sum(),
                unknownFinishedEvents.sum(),
                counts);
    }

    /**
     * 10s 对账扫描（周期由上层调度）：全量重算 vs 增量计数器比对。
     * M2 只比对<b>不静默修正</b>（指标/告警 M6 接入）。
     */
    public CounterDriftReport auditAndDrift() {
        List<String> drift = new ArrayList<>();
        drift.addAll(pStore.auditDrift());
        drift.addAll(dStore.auditDrift());
        return new CounterDriftReport(drift);
    }

    /**
     * janitor 驱动入口（M4 前占位）：当前仅做过期清理的安全子集
     * （墓碑/fence TTL）；TTL 收尾、缺席判死、fence 驱逐路径 M4 接入。
     */
    public Runnable ledgerJanitor() {
        return () -> {
            long now = System.currentTimeMillis();
            pStore.tombstones().evictExpired(now);
            dStore.tombstones().evictExpired(now);
            fences.evictExpired(now);
        };
    }

    /** fence 仓库（同包测试/门面协作；R4 断言见 FenceRegistry.canEvict）。 */
    FenceRegistry fences() {
        return fences;
    }

    // ---- observe 内部分发 ----

    private void observeInternal(EngineObservation obs, boolean rebuildMode) {
        Objects.requireNonNull(obs, "obs");
        StateEndpointRef ep = obs.endpointRef();
        if (rebuildMode) {
            generations.observeGeneration(ep.endpointId(), ep.generation());
        }
        // 整报世代屏障（S8）：端点当前登记代 vs 报文代，不匹配整报拒绝
        if (!generations.isCurrent(new GenerationTriple((int) ep.endpointId(), ep.generation(), -1L))) {
            generations.recordCrossGenerationReject();
            return;
        }
        long nowMs = obs.statusMs();
        for (EngineObservation.RunningObservation r : obs.running()) {
            dispatchRunning(r, ep, obs.round(), nowMs, rebuildMode);
        }
        for (EngineObservation.FinishedObservation f : obs.finished()) {
            dispatchFinished(f, ep, nowMs, rebuildMode);
        }
    }

    private void dispatchRunning(EngineObservation.RunningObservation r, StateEndpointRef ep,
                                 long round, long nowMs, boolean rebuildMode) {
        long id = r.requestId();
        if (r.side() == StateRole.PREFILL) {
            PrefillRequestState e = pStore.get(id);
            if (e == null) {
                if (rebuildMode) {
                    // P2 引擎收养：batchId=-1、engineOwned=true
                    pStore.adoptEngineOwned(id, (int) ep.endpointId(), ep.generation(), nowMs,
                            PrefillPhase.fromEnginePhase(r.enginePhase()), r.kvTokens(), r.version());
                } else if (pStore.isTombstoned(id)) {
                    pStore.absorbLateEvent();
                } else {
                    unknownRunningEvents.increment();
                }
                return;
            }
            PrefillPhase eventPhase = PrefillPhase.fromEnginePhase(r.enginePhase());
            PhaseVerdict v = PrefillLattice.arbitrate(e.phase(), e.lastVersion(),
                    eventPhase, r.version(), false, bindingMatches(e.binding(), ep));
            countVerdict(v);
            if (v == PhaseVerdict.REJECT_GENERATION) {
                generations.recordCrossGenerationReject();
                return;
            }
            // 同相位新鲜观察（DROP_DUP/迟到）不推进相位，但版本不落后时仍更新 B 道观察账
            if (r.version() >= e.lastVersion()) {
                pStore.noteEngineObserved(e, round, r.kvTokens(), r.version());
            }
            if (v == PhaseVerdict.ACCEPT_ADVANCE) {
                pStore.advance(e, eventPhase, r.version(), nowMs);
            }
            return;
        }

        DecodeRequestState e = dStore.get(id);
        if (e == null) {
            if (rebuildMode) {
                dStore.adoptEngineOwned(id, (int) ep.endpointId(), ep.generation(), nowMs,
                        DecodePhase.fromEnginePhase(r.enginePhase()), r.kvTokens(), r.version());
                // 收养即引擎事实：相位 ≥ D_LOADING 时同样触发跨侧收缩（重放中 D 已确认 ⇒ P 已完成）
                if (DecodePhase.fromEnginePhase(r.enginePhase()).ordinal() >= DecodePhase.D_LOADING.ordinal()) {
                    shrinkPrefillOnDecodeConfirmed(id, nowMs);
                }
            } else if (dStore.isTombstoned(id)) {
                dStore.absorbLateEvent();
            } else {
                unknownRunningEvents.increment();
            }
            return;
        }
        DecodePhase eventPhase = DecodePhase.fromEnginePhase(r.enginePhase());
        PhaseVerdict v = DecodeLattice.arbitrate(e.phase(), e.lastVersion(),
                eventPhase, r.version(), false, bindingMatches(e.binding(), ep));
        countVerdict(v);
        if (v == PhaseVerdict.REJECT_GENERATION) {
            generations.recordCrossGenerationReject();
            return;
        }
        // C1 双记：DISPATCHED 相位收到 RECEIVED——预占保持（不动账）+ 引擎已见（观察账更新）
        if (r.version() >= e.lastVersion()) {
            dStore.noteEngineObserved(e, round, r.kvTokens(), r.version());
        }
        if (v == PhaseVerdict.ACCEPT_ADVANCE) {
            boolean winner = dStore.advance(e, eventPhase, r.version(), nowMs);
            // 跨侧规则 C7/L4：D 确认点即 P 释放点——同 tick 收缩 P 条目（仅 CAS 胜者触发一次）
            if (winner && eventPhase == DecodePhase.D_LOADING) {
                shrinkPrefillOnDecodeConfirmed(id, nowMs);
            }
        }
    }

    private void dispatchFinished(EngineObservation.FinishedObservation f, StateEndpointRef ep,
                                  long nowMs, boolean rebuildMode) {
        long id = f.requestId();
        boolean success = f.errorCode() == 0;
        if (f.side() == StateRole.PREFILL) {
            PrefillRequestState e = pStore.get(id);
            if (e == null) {
                if (rebuildMode) {
                    return; // 历史序列中条目未开或已收——重建基线不含
                }
                if (pStore.isTombstoned(id)) {
                    pStore.absorbLateEvent(); // 迟到 finished 被墓碑吸收
                } else {
                    unknownFinishedEvents.increment();
                }
                return;
            }
            PhaseVerdict v = PrefillLattice.arbitrate(e.phase(), e.lastVersion(),
                    PrefillPhase.PREFILL_DONE, f.version(), true, bindingMatches(e.binding(), ep));
            countVerdict(v);
            if (v == PhaseVerdict.REJECT_GENERATION) {
                generations.recordCrossGenerationReject();
                return;
            }
            if (v == PhaseVerdict.ACCEPT_TERMINAL || v == PhaseVerdict.WARN_FINISH_PRIORITY) {
                // P 侧 finish 只终局 P 账（P 完成 ≠ 请求完成，D 侧继续）
                pStore.settleRemove(e, finishedOutcome(success, f.errorCode()), nowMs);
            }
            return;
        }

        DecodeRequestState e = dStore.get(id);
        if (e == null) {
            if (rebuildMode) {
                return;
            }
            if (dStore.isTombstoned(id)) {
                dStore.absorbLateEvent();
            } else {
                unknownFinishedEvents.increment();
            }
            return;
        }
        PhaseVerdict v = DecodeLattice.arbitrate(e.phase(), e.lastVersion(),
                DecodePhase.D_RUNNING, f.version(), true, bindingMatches(e.binding(), ep));
        countVerdict(v);
        if (v == PhaseVerdict.REJECT_GENERATION) {
            generations.recordCrossGenerationReject();
            return;
        }
        if (v == PhaseVerdict.ACCEPT_TERMINAL || v == PhaseVerdict.WARN_FINISH_PRIORITY) {
            boolean settled = dStore.settleRemove(e, finishedOutcome(success, f.errorCode()), nowMs);
            if (settled && success) {
                // F1 因果闭包：D 成功完成 ⇒ 整请求完成 ⇒ 同 tick 收缩未终局的 P 条目
                PrefillRequestState p = pStore.get(id);
                if (p != null) {
                    pStore.settleRemove(p,
                            new TerminalOutcome(TerminalState.COMPLETED, TerminalReason.SUCCEEDED,
                                    "causal-closure:decode-finished"),
                            nowMs);
                }
            }
        }
    }

    /**
     * C7/L4 跨侧收缩：P 条目处 P_RECEIVED..P_WAITING_LOADED 时闭包推进到 PREFILL_DONE
     * （P 账释放点 = D 确认点，无中间窗口）。P_RUNNING 不收缩（传输重叠窗口）；
     * P 侧本地相位（&lt; P_RECEIVED）不收缩（状态滞后由后续观察对齐）。
     */
    private void shrinkPrefillOnDecodeConfirmed(long requestId, long nowMs) {
        PrefillRequestState p = pStore.get(requestId);
        if (p == null) {
            return;
        }
        int ord = p.phase().ordinal();
        if (ord >= PrefillPhase.P_RECEIVED.ordinal() && ord <= PrefillPhase.P_WAITING_LOADED.ordinal()) {
            pStore.advance(p, PrefillPhase.PREFILL_DONE, -1L, nowMs);
        }
    }

    // ---- settle（CAS 单出口 + cancel 双清）----

    private boolean settlePrefill(long requestId, TerminalOutcome outcome, boolean propagate) {
        PrefillRequestState e = pStore.get(requestId);
        if (e == null) {
            absorbLateSettlePrefill(requestId, outcome);
            return false;
        }
        long now = System.currentTimeMillis();
        boolean ok = pStore.settleRemove(e, outcome, now);
        if (ok && propagate && outcome.state() == TerminalState.CANCELLED) {
            // cancel 双清：同 tick 清对侧账（各自计数独立减）
            DecodeRequestState d = dStore.get(requestId);
            if (d != null) {
                dStore.settleRemove(d, outcome, now);
            }
        }
        return ok;
    }

    private boolean settleDecode(long requestId, TerminalOutcome outcome, boolean propagate) {
        DecodeRequestState e = dStore.get(requestId);
        if (e == null) {
            if (dStore.isTombstoned(requestId)) {
                if (outcome.state() == TerminalState.CANCELLED) {
                    dStore.absorbLateCancel();
                } else {
                    dStore.absorbLateEvent();
                }
            }
            return false;
        }
        long now = System.currentTimeMillis();
        boolean ok = dStore.settleRemove(e, outcome, now);
        if (ok && propagate && outcome.state() == TerminalState.CANCELLED) {
            PrefillRequestState p = pStore.get(requestId);
            if (p != null) {
                pStore.settleRemove(p, outcome, now);
            }
        }
        return ok;
    }

    private void absorbLateSettlePrefill(long requestId, TerminalOutcome outcome) {
        if (pStore.isTombstoned(requestId)) {
            if (outcome.state() == TerminalState.CANCELLED) {
                pStore.absorbLateCancel();
            } else {
                pStore.absorbLateEvent();
            }
        }
    }

    // ---- 视图 ----

    private static PrefillRequestStateView toView(PrefillRequestState e) {
        synchronized (e) {
            return new PrefillRequestStateView(
                    e.requestId(), e.createdAtMs(), e.phase().ordinal(), e.phase().name(),
                    e.batchId(), e.pendingCancel(), e.binding(), e.kvTokensReported(),
                    e.lastSeenRound(), e.engineOwned(), e.dispatchedAtMs(), e.lastVersion(),
                    e.traceSnapshot());
        }
    }

    private static DecodeRequestStateView toView(DecodeRequestState e) {
        synchronized (e) {
            return new DecodeRequestStateView(
                    e.requestId(), e.createdAtMs(), e.phase().ordinal(), e.phase().name(),
                    e.pendingCancel(), e.binding(), e.reservedKv(), e.reservedExpectedKv(),
                    e.kvTokensReported(), e.lastSeenRound(), e.engineOwned(), e.lastVersion(),
                    e.traceSnapshot());
        }
    }

    private BatchShadowView buildBatchShadow(long batchId) {
        List<PrefillRequestStateView> members = new ArrayList<>();
        for (PrefillRequestState e : pStore.batchMembers(batchId)) {
            members.add(toView(e));
        }
        int max = -1;
        int min = -1;
        for (PrefillRequestStateView v : members) {
            max = Math.max(max, v.phaseOrdinal());
            min = min < 0 ? v.phaseOrdinal() : Math.min(min, v.phaseOrdinal());
        }
        return new BatchShadowView(batchId, members, max, min);
    }

    // ---- 杂项 ----

    private static boolean bindingMatches(GenerationTriple binding, StateEndpointRef ep) {
        return binding.endpointId() == (int) ep.endpointId()
                && binding.generation() == ep.generation();
    }

    private void countVerdict(PhaseVerdict v) {
        verdictCounts.get(v).increment();
    }

    private static TerminalOutcome finishedOutcome(boolean success, int errorCode) {
        return success
                ? new TerminalOutcome(TerminalState.COMPLETED, TerminalReason.SUCCEEDED, "")
                : new TerminalOutcome(TerminalState.FAILED, TerminalReason.ENGINE_FAILED,
                        "errorCode=" + errorCode);
    }

    // ---- 门面实现 ----

    private final class PrefillSideImpl implements PrefillSide {

        @Override
        public RegisterResult register(long requestId, long batchId) {
            return pStore.register(requestId, batchId);
        }

        @Override
        public void onQueued(long requestId) {
            PrefillRequestState e = pStore.get(requestId);
            if (e != null) {
                pStore.advance(e, PrefillPhase.QUEUED, -1L, System.currentTimeMillis());
            }
        }

        @Override
        public void onDispatching(long requestId, long batchId) {
            PrefillRequestState e = pStore.get(requestId);
            if (e == null) {
                return;
            }
            e.setBatchId(batchId);
            pStore.advance(e, PrefillPhase.DISPATCHING, -1L, System.currentTimeMillis());
        }

        @Override
        public boolean onDispatched(long requestId, GenerationTriple binding) {
            PrefillRequestState e = pStore.get(requestId);
            if (e == null) {
                return false;
            }
            e.setBindingOnce(binding); // DISPATCHED 前可重绑；不可变后返回 false 保留原绑定
            return pStore.advance(e, PrefillPhase.DISPATCHED, -1L, System.currentTimeMillis());
        }

        @Override
        public boolean settle(long requestId, TerminalOutcome outcome, SettleReason reason) {
            Objects.requireNonNull(outcome, "outcome");
            Objects.requireNonNull(reason, "reason");
            return settlePrefill(requestId, outcome, true);
        }

        @Override
        public Optional<PrefillRequestStateView> get(long requestId) {
            PrefillRequestState e = pStore.get(requestId);
            return e == null ? Optional.empty() : Optional.of(toView(e));
        }

        @Override
        public PrefillCounterSnapshot snapshot() {
            return pStore.snapshot();
        }

        @Override
        public void refreshSnapshot() {
            pStore.refreshSnapshot();
        }

        @Override
        public BatchShadowView batchView(long batchId) {
            return buildBatchShadow(batchId);
        }

        @Override
        public void markPendingCancel(long requestId) {
            PrefillRequestState e = pStore.get(requestId);
            if (e != null) {
                e.markPendingCancel();
            }
        }
    }

    private final class DecodeSideImpl implements DecodeSide {

        @Override
        public ReserveResult reserve(long requestId, long seqLen, long expectedKv, GenerationTriple binding) {
            Objects.requireNonNull(binding, "binding");
            return dStore.reserve(requestId, seqLen, expectedKv, binding, System.currentTimeMillis());
        }

        @Override
        public boolean release(long requestId) {
            DecodeRequestState e = dStore.get(requestId);
            if (e == null) {
                return false;
            }
            // R4：fenced 条目驱逐前断言拒绝（release 属移除路径）
            fences.canEvict(requestId);
            return dStore.releaseRemove(e, System.currentTimeMillis());
        }

        @Override
        public boolean onDispatched(long requestId, GenerationTriple binding) {
            DecodeRequestState e = dStore.get(requestId);
            if (e == null) {
                return false;
            }
            e.setBindingOnce(binding);
            return dStore.advance(e, DecodePhase.DISPATCHED, -1L, System.currentTimeMillis());
        }

        @Override
        public boolean settle(long requestId, TerminalOutcome outcome, SettleReason reason) {
            Objects.requireNonNull(outcome, "outcome");
            Objects.requireNonNull(reason, "reason");
            return settleDecode(requestId, outcome, true);
        }

        @Override
        public Optional<DecodeRequestStateView> get(long requestId) {
            DecodeRequestState e = dStore.get(requestId);
            return e == null ? Optional.empty() : Optional.of(toView(e));
        }

        @Override
        public DecodeCounterSnapshot snapshot() {
            return dStore.snapshot();
        }

        @Override
        public void refreshSnapshot() {
            dStore.refreshSnapshot();
        }

        @Override
        public void markPendingCancel(long requestId) {
            DecodeRequestState e = dStore.get(requestId);
            if (e != null) {
                e.markPendingCancel();
            }
        }
    }
}

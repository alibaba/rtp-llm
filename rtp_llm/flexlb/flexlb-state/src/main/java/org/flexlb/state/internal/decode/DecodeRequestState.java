package org.flexlb.state.internal.decode;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.flexlb.state.GenerationTriple;
import org.flexlb.state.InternalApi;
import org.flexlb.state.TerminalState;
import org.flexlb.state.internal.EntryTraceRing;

/**
 * D 侧请求条目：影子预占双轨（reservedKv / reservedExpectedKv）与
 * 引擎事实账（kvTokensReported）的单条目载体（计费归属状态驱动）。
 *
 * <p>并发约定：相位推进/终局/绑定均为条目级 synchronized CAS 语义
 * （transitionTo 只进不退；finishTransition AtomicBoolean 一次性守卫）。</p>
 */
@InternalApi
public final class DecodeRequestState {

    /** trace 环内终态标记基址：200 + TerminalState.ordinal（200..204）。 */
    static final int TERMINAL_TRACE_BASE = 200;

    /** 未绑定占位。 */
    private static final GenerationTriple UNBOUND = new GenerationTriple(-1, -1, -1);

    private final long requestId;
    private final long createdAtMs;
    private final long seqLen;
    private final EntryTraceRing trace = new EntryTraceRing();

    // ---- 相位与裁决屏障 ----
    private volatile DecodePhase phase = DecodePhase.RESERVED;
    private volatile long lastVersion = -1L;

    // ---- 世代绑定 ----
    private volatile GenerationTriple binding = UNBOUND;

    // ---- 正交意图 ----
    private volatile boolean pendingCancel;

    // ---- D 侧 KV 账（双轨：影子预占 + 引擎事实）----
    /** 影子预占当前占用（KV_ALLOCATED 确认后清 0；归位读数见 reservedKv()）。 */
    private volatile long reservedKv;
    /** 预约时声明的期望 KV（历史记录，确认后保留）。 */
    private final long reservedExpectedKv;
    /** 引擎事实 KV（KV_ALLOCATED 起接管；0 = unknown，不更新）。 */
    private volatile long kvTokensReported;
    private volatile long lastSeenRound = -1L;
    private volatile boolean engineOwned;

    // ---- 终局 ----
    private final AtomicBoolean finished = new AtomicBoolean();
    private volatile TerminalState terminalState;

    public DecodeRequestState(long requestId, long seqLen, long expectedKv, long createdAtMs) {
        this.requestId = requestId;
        this.seqLen = seqLen;
        this.reservedKv = expectedKv;
        this.reservedExpectedKv = expectedKv;
        this.createdAtMs = createdAtMs;
        trace.append(DecodePhase.RESERVED.ordinal(), 0L);
    }

    /**
     * 相位 CAS 前进：只进不退（target ≤ 当前相位返回 false——供超车计数）。
     * 成功时按格闭包补记沿途相位进入历史。
     *
     * @param version 事件版本（本地决策事件传 -1：不更新 lastVersion）
     */
    public synchronized boolean transitionTo(DecodePhase target, long version) {
        return transitionTo(target, version, System.currentTimeMillis());
    }

    /** 可注入时刻的重载（确定性测试用）。 */
    public synchronized boolean transitionTo(DecodePhase target, long version, long nowMs) {
        if (finished.get()) {
            return false;
        }
        if (target.ordinal() <= phase.ordinal()) {
            return false;
        }
        DecodePhase from = phase;
        long dt = Math.max(nowMs - createdAtMs, 0L);
        for (DecodePhase p : DecodeLattice.closureBetween(from, target)) {
            if (p != from) {
                trace.append(p.ordinal(), dt);
            }
        }
        phase = target;
        if (version > lastVersion) {
            lastVersion = version;
        }
        return true;
    }

    /**
     * 终态一次性转换（AtomicBoolean 守卫）：成功后条目不可再推进，
     * 由 Store 侧负责移除入墓碑。trace 记终态标记（200 + ordinal）。
     */
    public synchronized boolean finishTransition(TerminalState state) {
        return finishTransition(state, System.currentTimeMillis());
    }

    /** 可注入时刻的重载。 */
    public synchronized boolean finishTransition(TerminalState state, long nowMs) {
        if (!finished.compareAndSet(false, true)) {
            return false;
        }
        terminalState = state;
        trace.append(TERMINAL_TRACE_BASE + state.ordinal(), Math.max(nowMs - createdAtMs, 0L));
        return true;
    }

    /** 世代绑定（setBindingOnce 语义）：首次允许；已绑定且 phase ≥ DISPATCHED 后不可变。 */
    public synchronized boolean setBindingOnce(GenerationTriple triple) {
        if (binding != UNBOUND && phase.ordinal() >= DecodePhase.DISPATCHED.ordinal()) {
            return false;
        }
        binding = triple;
        return true;
    }

    /** 引擎上报观察入账：引擎已见 + 上轮次 + KV（kvTokens=0 表示 unknown，不更新）。 */
    public synchronized void markEngineObserved(long round, long kvTokens, long version) {
        engineOwned = true;
        lastSeenRound = round;
        if (kvTokens > 0) {
            kvTokensReported = kvTokens;
        }
        if (version > lastVersion) {
            lastVersion = version;
        }
    }

    /**
     * 计费归属移交撤预占：KV_ALLOCATED 确认后影子预占清 0（调用方先读旧值入计数再清）。
     * reservedExpectedKv 保留（历史记录）。
     */
    public synchronized void clearReservation() {
        reservedKv = 0;
    }

    /** 正交取消意图标记。 */
    public void markPendingCancel() {
        this.pendingCancel = true;
    }

    // ---- 只读 ----

    public long requestId() {
        return requestId;
    }

    public long createdAtMs() {
        return createdAtMs;
    }

    /** 序列长度（M2 记账字段，供上层容量推导）。 */
    public long seqLen() {
        return seqLen;
    }

    public DecodePhase phase() {
        return phase;
    }

    public long lastVersion() {
        return lastVersion;
    }

    public GenerationTriple binding() {
        return binding;
    }

    public boolean isBound() {
        return binding != UNBOUND;
    }

    public boolean pendingCancel() {
        return pendingCancel;
    }

    public long reservedKv() {
        return reservedKv;
    }

    public long reservedExpectedKv() {
        return reservedExpectedKv;
    }

    public long kvTokensReported() {
        return kvTokensReported;
    }

    public long lastSeenRound() {
        return lastSeenRound;
    }

    public boolean engineOwned() {
        return engineOwned;
    }

    /** 终态（未终局为 null）。 */
    public TerminalState terminalState() {
        return terminalState;
    }

    public boolean isFinished() {
        return finished.get();
    }

    /** trace 快照（人类可读，最旧→最新）。 */
    public List<String> traceSnapshot() {
        return trace.drain();
    }
}

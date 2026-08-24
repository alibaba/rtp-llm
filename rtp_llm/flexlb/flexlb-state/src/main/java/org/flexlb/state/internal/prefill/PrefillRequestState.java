package org.flexlb.state.internal.prefill;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.flexlb.state.GenerationTriple;
import org.flexlb.state.InternalApi;
import org.flexlb.state.TerminalState;
import org.flexlb.state.internal.EntryTraceRing;

/**
 * P 侧请求条目：A 道（派发流水线）与 B 道（引擎观察账）字段的单条目载体。
 *
 * <p>并发约定：相位推进/终局/绑定均为条目级 synchronized CAS 语义
 * （transitionTo 只进不退；finishTransition AtomicBoolean 一次性守卫）。</p>
 */
@InternalApi
public final class PrefillRequestState {

    /** 散请求批次 ID。 */
    public static final long NO_BATCH = -1L;

    /** trace 环内终态标记基址：200 + TerminalState.ordinal（200..204）。 */
    static final int TERMINAL_TRACE_BASE = 200;

    /** 未绑定占位。 */
    private static final GenerationTriple UNBOUND = new GenerationTriple(-1, -1, -1);

    private final long requestId;
    private final long createdAtMs;
    private final EntryTraceRing trace = new EntryTraceRing();

    // ---- 相位与裁决屏障 ----
    private volatile PrefillPhase phase = PrefillPhase.INIT;
    private volatile long lastVersion = -1L;

    // ---- 世代绑定（发送前可重绑=新记录，发送后不可变）----
    private volatile GenerationTriple binding = UNBOUND;

    // ---- S3 正交意图 ----
    private volatile boolean pendingCancel;

    // ---- A 道区段（派发流水线）----
    private volatile long batchId;
    private volatile long dispatchedAtMs = -1L;

    // ---- B 道区段（引擎观察账）----
    private volatile long kvTokensReported;
    private volatile long lastSeenRound = -1L;
    private volatile boolean engineOwned;

    // ---- 终局 ----
    private final AtomicBoolean finished = new AtomicBoolean();
    private volatile TerminalState terminalState;

    public PrefillRequestState(long requestId, long batchId, long createdAtMs) {
        this.requestId = requestId;
        this.batchId = batchId;
        this.createdAtMs = createdAtMs;
        trace.append(PrefillPhase.INIT.ordinal(), 0L);
    }

    /**
     * 相位 CAS 前进：只进不退（target ≤ 当前相位返回 false——供超车计数）。
     * 成功时按格闭包补记沿途相位进入历史（L9：越级推进的 enteredAt 用当前时刻近似）。
     *
     * @param version 事件版本（本地决策事件传 -1：不更新 lastVersion）
     */
    public synchronized boolean transitionTo(PrefillPhase target, long version) {
        return transitionTo(target, version, System.currentTimeMillis());
    }

    /** 可注入时刻的重载（确定性测试用）。 */
    public synchronized boolean transitionTo(PrefillPhase target, long version, long nowMs) {
        if (finished.get()) {
            return false;
        }
        if (target.ordinal() <= phase.ordinal()) {
            return false;
        }
        PrefillPhase from = phase;
        long dt = Math.max(nowMs - createdAtMs, 0L);
        for (PrefillPhase p : PrefillLattice.closureBetween(from, target)) {
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

    /**
     * 世代绑定（setBindingOnce 语义）：首次绑定任何时候允许；
     * 已绑定且 phase ≥ DISPATCHED（已发送）后不可变——返回 false。
     */
    public synchronized boolean setBindingOnce(GenerationTriple triple) {
        if (binding != UNBOUND && phase.ordinal() >= PrefillPhase.DISPATCHED.ordinal()) {
            return false;
        }
        binding = triple;
        return true;
    }

    /** B 道观察入账：引擎已见 + 上轮次 + KV（E1：kvTokens=0 表示 unknown 不更新）。 */
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

    /** S3 正交取消意图标记。 */
    public void markPendingCancel() {
        this.pendingCancel = true;
    }

    /** A 道：更新批次外键（B0/B1 可空外键；onDispatching 时随相位一起写）。 */
    public void setBatchId(long batchId) {
        this.batchId = batchId;
    }

    /** A 道：派发时刻。 */
    public void noteDispatched(long nowMs) {
        this.dispatchedAtMs = nowMs;
    }

    // ---- 只读 ----

    public long requestId() {
        return requestId;
    }

    public long createdAtMs() {
        return createdAtMs;
    }

    public PrefillPhase phase() {
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

    public long batchId() {
        return batchId;
    }

    public long dispatchedAtMs() {
        return dispatchedAtMs;
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

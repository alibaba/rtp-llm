package org.flexlb.state;

import java.util.Optional;

/**
 * P 侧类型化门面 API（实例经 {@link StateLedger#prefill()} 获取）。
 *
 * <p>引擎事件方法（onEngineRunning/onEngineFinished）<b>不在本接口暴露</b>——
 * 统一走 {@link StateLedger#observe} 分发（跨侧规则单一入口）。</p>
 */
public interface PrefillSide {

    /** 登记 P 侧条目（INIT 起步）：判重（存活 → DUPLICATE_ALIVE / 墓碑 → DUPLICATE_TOMBSTONE）。 */
    RegisterResult register(long requestId, long batchId);

    /** 本地决策事件：推进到 QUEUED（幂等，已在更高相位时静默）。 */
    void onQueued(long requestId);

    /** 本地决策事件：推进到 DISPATCHING 并更新批次外键。 */
    void onDispatching(long requestId, long batchId);

    /**
     * 派发完成：绑定世代三元组（发送前可重绑；DISPATCHED 后不可变，见 setBindingOnce 语义）
     * 并推进到 DISPATCHED。
     *
     * @return 是否成功推进到 DISPATCHED（已在 DISPATCHED 及以上返回 false）
     */
    boolean onDispatched(long requestId, GenerationTriple binding);

    /** 终局：CAS 单出口（已终态/不存在返回 false）；CANCELLED 触发两侧双清。 */
    boolean settle(long requestId, TerminalOutcome outcome, SettleReason reason);

    /** 只读视图（终态后条目移入墓碑，返回 empty）。 */
    Optional<PrefillRequestStateView> get(long requestId);

    /** 已发布派生快照（零锁 volatile 读；精确值用 refreshSnapshot + snapshot）。 */
    PrefillCounterSnapshot snapshot();

    /** 强制重算并发布快照。 */
    void refreshSnapshot();

    /** 批次影子视图（B6 双视图：maxPhase/minPhase）。 */
    BatchShadowView batchView(long batchId);

    /** 正交取消意图标记（S3：只标记，终局走 settle）。 */
    void markPendingCancel(long requestId);
}

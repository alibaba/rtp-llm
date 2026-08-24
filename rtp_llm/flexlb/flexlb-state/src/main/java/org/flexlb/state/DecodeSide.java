package org.flexlb.state;

import java.util.Optional;

/**
 * D 侧类型化门面 API（实例经 {@link StateLedger#decode()} 获取）。
 *
 * <p>引擎事件方法不在本接口暴露——统一走 {@link StateLedger#observe} 分发。</p>
 */
public interface DecodeSide {

    /**
     * 预约（RESERVED 起步）：登记影子预占（reservedKv = expectedKv）并绑定世代三元组。
     *
     * @param seqLen     序列长度（M2 记账字段，供上层容量推导）
     * @param expectedKv 期望 KV（预占量）
     */
    ReserveResult reserve(long requestId, long seqLen, long expectedKv, GenerationTriple binding);

    /** 释放预约（未终态主动放弃）：撤预占账并移除条目（不进墓碑，可重新 reserve）。 */
    boolean release(long requestId);

    /** 派发完成：绑定世代（发送前可重绑）并推进到 DISPATCHED。 */
    boolean onDispatched(long requestId, GenerationTriple binding);

    /** 终局：CAS 单出口；CANCELLED 触发两侧双清。 */
    boolean settle(long requestId, TerminalOutcome outcome, SettleReason reason);

    /** 只读视图。 */
    Optional<DecodeRequestStateView> get(long requestId);

    /** 已发布派生快照（零锁 volatile 读）。 */
    DecodeCounterSnapshot snapshot();

    /** 强制重算并发布快照。 */
    void refreshSnapshot();

    /** 正交取消意图标记。 */
    void markPendingCancel(long requestId);
}

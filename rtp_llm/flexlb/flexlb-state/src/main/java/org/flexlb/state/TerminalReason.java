package org.flexlb.state;

/**
 * 终态受控原因（O2：取代自由文本 reason 字符串），值域对应设计 F1-F5 终局语义。
 */
public enum TerminalReason {

    /** 正常完成：引擎 finished 且错误码为 0。 */
    SUCCEEDED,

    /** 显式取消被确认：引擎/下游对取消请求给出 ack。 */
    CANCELLED_ACK,

    /** 隐式取消：取消后未收到显式 ack，按缺席/闭包推定取消成立。 */
    CANCELLED_IMPLICIT,

    /** 取消时请求从未到达引擎（本地取消直接成立，无需引擎证据）。 */
    CANCELLED_NEVER_ARRIVED,

    /** SLO 预算（TTFT/总时长）耗尽，由 SLO 通道判死。 */
    SLO_BUDGET_EXHAUSTED,

    /** 引擎侧失败：finished 报文携带非 0 错误码，或引擎健康通道判定失败。 */
    ENGINE_FAILED,

    /** 凭空消失：inflight 泄漏/僵尸任务——连续 N 轮观察缺席且无任何终局证据。 */
    VANISHED,

    /** 陈旧驱逐：世代/批次闭包判定该条目属于已被取代的陈旧路径。 */
    STALE_EVICTED,

    /** 端点排空：端点被运维/缩容排空，其上未终局条目统一收尾。 */
    EP_DRAINED,

    /** TTL 到期：条目存活时间超过受控上限。 */
    TTL_EXPIRED,

    /** 硬上限：条目总量/单端点配额触顶，强制收尾最旧条目。 */
    HARD_CAP,

    /** 被抢占（回边，对应 {@link TerminalState#PREEMPTED}，非吸收）。 */
    PREEMPTED
}

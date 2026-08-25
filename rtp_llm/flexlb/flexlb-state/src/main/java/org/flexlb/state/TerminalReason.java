package org.flexlb.state;

/**
 * 终态受控原因（受控枚举取代自由文本 reason 字符串），值域覆盖当前已实现
 * 的终局语义（完成/取消/超时/失败）。
 *
 * <p>完备性契约（观测层守护）：每个枚举值必须至少被一处产出路径使用——
 * {@code ReasonCompletenessTest} 反射遍历断言；新增值时须同步接线产出点，
 * 无实现路径的预留值不进枚举（历史裁剪：CANCELLED_ACK / STALE_EVICTED /
 * EP_DRAINED / PREEMPTED 曾为预留值，因无产出路径移除；显式取消 ack、
 * 世代闭包驱逐、端点排空与抢占回边通道落地时按需恢复并接线）。</p>
 */
public enum TerminalReason {

    /** 正常完成：引擎 finished 且错误码为 0。 */
    SUCCEEDED,

    /**
     * 隐式取消：取消已派发（引擎可能已见），未收到显式 ack，按推定取消成立。
     */
    CANCELLED_IMPLICIT,

    /** 取消时请求从未到达引擎（本地取消直接成立，无需引擎证据——未派发即取消）。 */
    CANCELLED_NEVER_ARRIVED,

    /** SLO 预算（TTFT/总时长）耗尽，由 SLO 通道判死。 */
    SLO_BUDGET_EXHAUSTED,

    /** 引擎侧失败：finished 报文携带非 0 错误码，或引擎健康通道判定失败。 */
    ENGINE_FAILED,

    /** 凭空消失：inflight 泄漏/僵尸任务——连续 N 轮观察缺席且无任何终局证据。 */
    VANISHED,

    /** TTL 到期：条目存活时间超过受控上限。 */
    TTL_EXPIRED,

    /** 硬上限：条目总量/单端点配额触顶，强制收尾最旧条目。 */
    HARD_CAP
}

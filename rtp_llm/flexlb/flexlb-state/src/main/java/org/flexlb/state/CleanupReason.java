package org.flexlb.state;

/**
 * 清理受控原因（三类受控枚举之一）：描述条目从活跃态被移除的通道。
 */
public enum CleanupReason {

    /** 引擎已上报 finished，正常清理。 */
    FINISHED_REPORTED,

    /** 连续 N 轮观察缺席，推定死亡后清理。 */
    ABSENT_N_ROUNDS,

    /** 截断上报中被排除：该轮上报不完整（detailCount 超出），缺席证据不可信，不清理。 */
    TRUNCATED_REPORT_EXCLUDED,

    /** fence 持有期间不清：跨世代 fence 未解除时冻结清理。 */
    FENCE_HOLD,

    /** TTL 到期清理。 */
    TTL,

    /** 硬上限（容量护栏）触顶清理。 */
    HARD_CAP
}

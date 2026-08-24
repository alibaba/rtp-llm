package org.flexlb.state;

/**
 * 请求终态（吸收态体系）。
 *
 * <p>设计 §8：PREEMPTED 是<b>回边态而非吸收态</b>——现语义 = 被抢占后可重试回"已决策"相位，
 * 因此 {@link #isAbsorbing()} 对 PREEMPTED 返回 false，其余四个终态均为吸收态，
 * 一旦进入不可逆转、不可重试。</p>
 */
public enum TerminalState {

    /** 请求正常完成（证据：引擎 finished 报文，错误码 0）。吸收态。 */
    COMPLETED,

    /** 请求被取消。吸收态。 */
    CANCELLED,

    /** SLO 预算耗尽超时。吸收态。 */
    SLO_TIMEOUT,

    /** 引擎/基础设施失败。吸收态。 */
    FAILED,

    /**
     * 被抢占（回边态，非吸收态）：现语义 = 可重试回已决策相位，
     * 终局是否成立由上层按策略裁决。
     */
    PREEMPTED;

    /**
     * 是否吸收态：前四个终态（COMPLETED / CANCELLED / SLO_TIMEOUT / FAILED）为 true，
     * PREEMPTED 为 false（回边可重试）。
     */
    public boolean isAbsorbing() {
        return this != PREEMPTED;
    }
}

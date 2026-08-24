package org.flexlb.state;

/**
 * 相位转换受控原因（三类受控枚举之一）：描述驱动一次相位格转换的事件来源。
 */
public enum TransitionReason {

    /** 调度器主动决策驱动的转换（派发、重路由、抢占等）。 */
    SCHEDULER_DECISION,

    /** 引擎观察上报驱动的转换（status 报文中的 running/finished 明细）。 */
    ENGINE_OBSERVATION,

    /** KV/负载传输进度驱动的转换（D 侧 LOAD 传输期相位横跨）。 */
    LOAD_TRANSFER,

    /** 取消请求传播驱动的转换。 */
    CANCEL_REQUEST,

    /** 回边重试驱动的转换（PREEMPTED 后重入已决策相位）。 */
    RETRY,

    /** 监督通道驱动的强制转换（超时、预算、容量护栏）。 */
    SUPERVISION
}

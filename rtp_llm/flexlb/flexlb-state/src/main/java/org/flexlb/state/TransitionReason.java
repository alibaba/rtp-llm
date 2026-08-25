package org.flexlb.state;

/**
 * 相位转换受控原因（三类受控枚举之一）：描述驱动一次相位格转换的事件来源。
 *
 * <p>完备性契约（观测层守护）：每个枚举值必须至少被一处产出路径使用——
 * {@code ReasonCompletenessTest} 反射遍历断言；新增值时须同步接线产出点，
 * 无实现路径的预留值不进枚举（历史裁剪：CANCEL_REQUEST / RETRY /
 * SUPERVISION 曾为预留值，因取消不产生中间相位转换（直接终局）、
 * PREEMPTED 回边与监督强制转换通道尚未落地而移除；对应通道落地时
 * 按需恢复并接线）。</p>
 */
public enum TransitionReason {

    /** 调度器主动决策驱动的转换（本地生命周期点：排队、派发流水线推进）。 */
    SCHEDULER_DECISION,

    /** 引擎观察上报驱动的转换（status 报文中的 running 明细裁决接受）。 */
    ENGINE_OBSERVATION,

    /** KV/负载传输进度驱动的转换（D 侧确认点即 P 释放点的跨侧收缩）。 */
    LOAD_TRANSFER
}

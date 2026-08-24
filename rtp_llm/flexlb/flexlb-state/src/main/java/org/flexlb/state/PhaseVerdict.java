package org.flexlb.state;

/**
 * 相位事件裁决结论（相位裁决矩阵输出）。
 */
public enum PhaseVerdict {

    /** 接受推进：事件相位高于当前相位，按格闭包越级推进（上层补记沿途 enteredAt）。 */
    ACCEPT_ADVANCE,

    /** 接受终局：finish 事件与当前相位一致，进入终态判定入口（推到格顶或由上层 settle）。 */
    ACCEPT_TERMINAL,

    /** 丢弃迟到中间态：事件相位低于当前已知相位（乱序裁决，迟到中间态丢弃）。 */
    DROP_LATE,

    /** 丢弃重复：版本陈旧（version 单调假设破坏时的丢弃语义）或同相位无推进事件。 */
    DROP_DUP,

    /** 警告但接受 finish 推进：finish 事件携带相位超前于当前已知相位——优先级倒挂告警，仍接受终局。 */
    WARN_FINISH_PRIORITY,

    /** 世代拒绝：事件世代三元组与条目世代不匹配（世代屏障），整报拒绝。 */
    REJECT_GENERATION
}

package org.flexlb.state.spi;

/**
 * 状态侧角色：Prefill 侧 / Decode 侧（PD 分离双格）。
 */
public enum StateRole {

    /** Prefill 侧（相位格见 org.flexlb.state.internal.prefill.PrefillPhase）。 */
    PREFILL,

    /** Decode 侧（相位格见 org.flexlb.state.internal.decode.DecodePhase）。 */
    DECODE
}

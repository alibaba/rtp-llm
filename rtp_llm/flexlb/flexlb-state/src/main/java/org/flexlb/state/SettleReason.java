package org.flexlb.state;

/**
 * 终局判定（settle）受控原因（O2 三类受控枚举之一）：描述一个请求被判定终局所走的证据通道。
 */
public enum SettleReason {

    /** 引擎 finished 报文——最强证据通道。 */
    ENGINE_FINISHED,

    /** 因果闭包推导终局（世代/批次闭包，请求随闭包整体收尾）。 */
    CAUSAL_CLOSURE,

    /** 证据通道：非 finished 直接证据但证据链充分（如错误码 + 缺席组合）。 */
    EVIDENCE_CHANNEL,

    /** TTL 通道：存活时间上限触发终局。 */
    TTL_CHANNEL,

    /** 强制通道：运维/熔断等外力强制终局。 */
    FORCE_CHANNEL,

    /** 本地取消：上游取消请求在本地直接成立终局。 */
    LOCAL_CANCEL
}

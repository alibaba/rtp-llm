package org.flexlb.balance.schedulev2.request;

/**
 * 请求在新链路里所处的阶段。
 * <p>
 * 正常顺序是 MAIN_WAITING → PREFILL_WAITING → DECODE_WAITING → DECODE_RESERVED → PUBLISHED。
 * 发布之前的任何阶段都可以进入 FAILED 或 CANCELLED。
 * 选好的 Prefill 下线时，PREFILL_WAITING 的请求退回 MAIN_WAITING 重新选；
 * reserve 的 Decode 下线或 reserve 丢失时，DECODE_RESERVED 的请求退回 DECODE_WAITING 重新选。
 * 没部署 Decode（PD 合一）时，DECODE_WAITING 直接进入 PUBLISHED。
 */
public enum Stage {
    /**
     * 在主队列里，还没选 Prefill。
     */
    MAIN_WAITING,
    /**
     * 已选好 Prefill，在这台 Prefill 的队列里等 dispatch。
     */
    PREFILL_WAITING,
    /**
     * Prefill 已 dispatch，在全局 Decode 队列里等选 Decode 和 reserve。
     */
    DECODE_WAITING,
    /**
     * 已在某台 Decode 上 reserve，等 permit。
     */
    DECODE_RESERVED,
    /**
     * 结果已发布，future 已完成。
     */
    PUBLISHED,
    /**
     * 发布前失败，future 已用错误完成。
     */
    FAILED,
    /**
     * 发布前被取消或超时，future 已用错误完成。
     */
    CANCELLED;

    /**
     * 是否已经结束，结束后不会再进入其他阶段。
     */
    public boolean isEnded() {
        return this == FAILED || this == CANCELLED;
    }

    /**
     * 是否还在等待发布。
     */
    public boolean isWaiting() {
        return this == MAIN_WAITING || this == PREFILL_WAITING || this == DECODE_WAITING || this == DECODE_RESERVED;
    }

    /**
     * 能否从当前阶段进入 next。
     */
    public boolean canMoveTo(Stage next) {
        return switch (this) {
            case MAIN_WAITING -> next == PREFILL_WAITING || next.isEnded();
            case PREFILL_WAITING -> next == DECODE_WAITING || next == MAIN_WAITING || next.isEnded();
            case DECODE_WAITING -> next == DECODE_RESERVED || next == PUBLISHED || next.isEnded();
            case DECODE_RESERVED -> next == PUBLISHED || next == DECODE_WAITING || next.isEnded();
            case PUBLISHED, FAILED, CANCELLED -> false;
        };
    }
}

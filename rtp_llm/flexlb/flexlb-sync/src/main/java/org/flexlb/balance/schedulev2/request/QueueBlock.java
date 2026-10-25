package org.flexlb.balance.schedulev2.request;

/**
 * 最近一次无法出队的原因。progress 对象标识本次排队阶段，阶段变化后旧原因不再有效。
 */
public record QueueBlock(QueueProgress progress, String reason, String endpoint) {
}

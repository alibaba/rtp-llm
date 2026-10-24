package org.flexlb.balance.delivery;

import org.flexlb.balance.scheduler.ScheduledRequest;

/**
 * prepare 时被卡住的请求：它是候选请求里排在可投递部分之后的第一个请求，
 * reason 说明它没能被覆盖的原因（容量不足、归属丢失、投递不可用等）。
 */
public record BlockedRequest(ScheduledRequest item, DeliveryBlockReason reason) {
}

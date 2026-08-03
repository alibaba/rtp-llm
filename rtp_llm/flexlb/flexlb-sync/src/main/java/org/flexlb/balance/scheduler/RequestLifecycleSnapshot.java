package org.flexlb.balance.scheduler;

/**
 * Immutable lifecycle view returned by request-state and reconciliation APIs.
 *
 * @deprecated v2 调度器重构：被 InflightItem/AbstractScheduler/InflightStore 替代。
 */
@Deprecated
public record RequestLifecycleSnapshot(long requestId,
                                       RequestLifecycleState state,
                                       long batchId,
                                       long createdAtMs,
                                       long updatedAtMs,
                                       String detail) {
}

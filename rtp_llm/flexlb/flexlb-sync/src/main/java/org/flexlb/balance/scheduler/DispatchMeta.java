package org.flexlb.balance.scheduler;

/**
 * Metadata describing why and how a batch was dispatched.
 *
 * <p>Extracted from {@link FlexlbBatchScheduler} to reduce coupling
 * with {@link WorkerBatcher}.
 */
public record DispatchMeta(String reason, double fillRatio, int queueDepth) {
}

package org.flexlb.balance.scheduler;

/**
 * Metadata describing why and how a batch was dispatched.
 *
 * <p>Extracted from {@link BatchScheduler} to reduce coupling
 * with {@link WorkerBatcher}.
 */
public record DispatchMeta(String reason, int queueDepth) {
}

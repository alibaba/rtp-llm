package org.flexlb.balance.scheduler;

/**
 * Metadata describing why and how a priority decision group was released.
 *
 * <p>Extracted from {@link PriorityScheduler} to reduce coupling
 * with {@link WorkerBatcher}.
 */
public record DecisionGroupMetadata(String reason, int queueDepth) {
}

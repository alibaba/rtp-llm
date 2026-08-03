package org.flexlb.balance.scheduler;

/**
 * v2 sealed interface for inflight tracking.
 *
 * <p>A sealed interface permitting only {@link InflightItem} (single-request
 * tracking) and {@link Batch} (multi-request batch tracking), so that
 * exhaustive switch expressions over inflight entries are compile-time safe.
 *
 * <p>Placed in the {@code scheduler} package (rather than {@code endpoint})
 * because the Java Language Specification requires sealed-interface permits
 * to reside in the same package when the code is compiled in the unnamed
 * module (no {@code module-info.java}).
 *
 * <p>Note: {@code FlexlbBatchScheduler} has a private inner class with the same
 * simple name. This is intentional — the inner class will be removed when the
 * v2 migration completes.
 */
public sealed interface InflightEntry permits InflightItem, Batch {
}

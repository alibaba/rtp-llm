package org.flexlb.balance.scheduler;

/**
 * Marker interface for entries tracked by inflight stores.
 *
 * <p>A sealed interface permitting only {@link InflightItem} (single-request
 * tracking) and {@link Batch} (multi-request batch tracking), so that
 * exhaustive switch expressions over inflight entries are compile-time safe.
 *
 * <p>Placed in the {@code scheduler} package (rather than {@code endpoint})
 * because the Java Language Specification requires sealed-interface permits
 * to reside in the same package when the code is compiled in the unnamed
 * module (no {@code module-info.java}).
 */
public sealed interface InflightEntry permits InflightItem, Batch {
}

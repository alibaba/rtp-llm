package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;

import java.util.List;

/**
 * Read-only view of batcher state for {@link FixedWindowBatcherAlgorithm}
 * decision making.
 *
 * <p>Deliberately excludes every mutating capability of
 * {@link BatcherContext} ({@code remove}, {@code drainTo}, {@code dispatch},
 * {@code dropHead}, {@code rejectForBatchTokenCapacity}) as well as direct
 * endpoint / reporter access, so the algorithm cannot produce side effects.
 */
public interface BatcherReadView {

    /** Worker key (ip:port) this batcher serves. */
    String key();

    /** FlexLB runtime configuration. */
    FlexlbConfig cfg();

    /** Current wall-clock time in milliseconds. */
    long now();

    // ---- queue inspection (read-only) ----

    /** Head of the queue, or {@code null} if empty. */
    BatchItem peek();

    boolean isEmpty();

    int size();

    /**
     * Snapshot of queued items sorted by {@link BatchItem#sortKey()},
     * suitable for greedy-fill iteration.
     */
    List<BatchItem> sortedItems();

    // ---- capacity queries ----

    /** Effective strict padded-token limit for one FlexLB batch. */
    long batchTokenCapacity();

    /** Latest worker-reported KV budget. */
    long batchKvCapacity();

    // ---- engine state (read-only) ----

    /** Current inflight batch count on the prefill worker, for backpressure. */
    int currentInflightCount();

    /** Prefill-time predictor for predictor-based early dispatch (read-only). */
    PrefillTimePredictor predictor();
}

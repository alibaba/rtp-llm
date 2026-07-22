package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;

import java.util.List;

/**
 * Dispatches pre-assembled batches of items to the prefill engine via gRPC.
 * <p>
 * The dispatcher is a pure network layer — it does NOT manage inflight
 * state or rollback. Those concerns belong to the scheduler.
 * Per-item results are reported through {@link DispatchCallback}.
 *
 * <h3>Contract</h3>
 * <ul>
 *   <li>All items in {@code items} have incomplete futures when
 *       {@code dispatch} is called.</li>
 *   <li>On batch-level failure (build error, network error), the
 *       dispatcher releases the PrefillEndpoint batch before calling
 *       {@link DispatchCallback#onFailure} for each item.</li>
 *   <li>Exactly one callback method is invoked per item.</li>
 * </ul>
 */
public interface BatchDispatcher {

    /**
     * Dispatch a batch of active items to the prefill engine.
     *
     * @param items    non-empty list of items
     * @param prefillEp  target prefill endpoint (already resolved by caller)
     * @param batchId  pre-allocated unique batch identifier
     * @param predMs   predicted batch execution time (for logging)
     * @param reason   dispatch trigger reason (for logging): "fill_rate", "emergency",
     *                 "batch_size_max", etc.
     * @param callback receives per-item success or failure
     */
    void dispatch(List<BatchItem> items,
                  PrefillEndpoint prefillEp,
                  long batchId,
                  long predMs,
                  String reason,
                  DispatchCallback callback);
}

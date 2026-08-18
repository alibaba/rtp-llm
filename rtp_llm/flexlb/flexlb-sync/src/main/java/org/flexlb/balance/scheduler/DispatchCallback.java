package org.flexlb.balance.scheduler;

/**
 * Receives per-item EnqueueBatch transport outcomes from {@link BatchDispatcher}.
 *
 * <p>This lower-level callback deliberately carries the batch id required to
 * reject stale transport callbacks. {@link BatchEnqueueDelivery} adapts it to
 * the batch-agnostic {@link DecisionDelivery.Callback} used by the scheduler.
 */
public interface DispatchCallback {

    /**
     * The engine acknowledged this item in the given batch.
     *
     * @param item    the dispatched item
     * @param batchId positive EnqueueBatch id
     */
    void onSuccess(BatchItem item, long batchId);

    /**
     * EnqueueBatch failed before ownership became ambiguous. Possible causes
     * include:
     * <ul>
     *   <li>gRPC request build failure (protobuf parsing)</li>
     *   <li>Engine rejected via error list in response</li>
     *   <li>A local executor rejection before the RPC invocation</li>
     * </ul>
     * When called due to a batch-level failure, the dispatcher has
     * already released the PrefillEndpoint batch before calling this.
     *
     * @param item  item whose delivery failed
     * @param error underlying error
     */
    void onFailure(BatchItem item, Throwable error);

    /** Batch dispatch deadline elapsed before an acknowledgement could be reconciled. */
    default void onTimeout(BatchItem item, Throwable error) {
        onFailure(item, error);
    }

    /**
     * EnqueueBatch completed without a trustworthy per-item acknowledgement
     * after dispatch started (transport failure, missing ACK, or malformed
     * response).
     * The engine may already own the request, so this is not a failure proof
     * and the Prefill batch ledger must remain held until reconciliation.
     */
    default void onDispatchUncertain(BatchItem item, long batchId, Throwable error) {
        onTimeout(item, error);
    }
}

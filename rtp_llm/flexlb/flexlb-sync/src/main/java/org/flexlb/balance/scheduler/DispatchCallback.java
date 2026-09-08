package org.flexlb.balance.scheduler;

/**
 * Receives per-item dispatch results from {@link BatchDispatcher}.
 * <p>
 * Implemented by the scheduler to manage inflight state in response to
 * engine acknowledgements. The dispatcher guarantees exactly one terminal
 * callback per item.
 */
public interface DispatchCallback {

    /**
     * Engine successfully accepted this item.
     * Called once per item that appears in the gRPC success list.
     *
     * @param item    the dispatched item
     * @param batchId the batch it was dispatched in
     */
    void onSuccess(BatchItem item, long batchId);

    /**
     * Item definitely failed to be enqueued. Possible causes:
     * <ul>
     *   <li>gRPC request build failure (protobuf parsing)</li>
     *   <li>Engine rejected via error list in response</li>
     *   <li>A local executor rejection before the RPC invocation</li>
     * </ul>
     * When called due to a batch-level failure, the dispatcher has
     * already released the PrefillEndpoint batch before calling this.
     *
     * @param item  the failed item
     * @param error the underlying error
     */
    void onFailure(BatchItem item, Throwable error);

    /** Dispatch deadline elapsed before an acknowledgement could be reconciled. */
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

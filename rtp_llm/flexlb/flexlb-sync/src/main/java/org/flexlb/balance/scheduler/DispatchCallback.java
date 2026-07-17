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
     * Item failed to be enqueued. Possible causes:
     * <ul>
     *   <li>gRPC request build failure (protobuf parsing)</li>
     *   <li>Engine rejected via error list in response</li>
     *   <li>Item missing from ack response (protocol error)</li>
     *   <li>Network error on the entire batch call</li>
     * </ul>
     * When called due to a batch-level failure, the dispatcher has
     * already released the PrefillEndpoint batch before calling this.
     *
     * @param item  the failed item
     * @param error the underlying error
     */
    void onFailure(BatchItem item, Throwable error);
}

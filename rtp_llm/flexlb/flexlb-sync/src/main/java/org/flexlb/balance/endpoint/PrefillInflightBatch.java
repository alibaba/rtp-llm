package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;

import java.util.List;

/**
 * Batch-path inflight entry: a real batchId with its member requests.
 *
 * @param batchId     engine-visible batch identifier
 * @param predictMs   predicted batch execution time in milliseconds
 * @param requests    batch members (immutable snapshot)
 * @param createdAtMs epoch millis when the batch was committed
 */
public record PrefillInflightBatch(
        long batchId, long predictMs, List<BatchItem> requests,
        long createdAtMs) implements PrefillInflightEntry {

    public PrefillInflightBatch {
        requests = List.copyOf(requests);
    }

    @Override
    public int requestCount() {
        return requests.size();
    }

    /**
     * Shrink the batch after members finished or failed: survivors keep the
     * original batchId and creation timestamp, prediction is recomputed by
     * the caller.
     */
    public PrefillInflightBatch repack(long newPredictMs, List<BatchItem> survivors) {
        return new PrefillInflightBatch(batchId, newPredictMs, survivors, createdAtMs);
    }
}

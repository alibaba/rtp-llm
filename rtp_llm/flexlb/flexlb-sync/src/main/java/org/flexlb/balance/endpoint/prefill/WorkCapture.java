package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.estimate.WorkView;

import java.util.Comparator;
import java.util.List;

/**
 * Immutable leaves copied under the ownership lock; load view is built outside it.
 */
public final class WorkCapture {
    public final long capturedAtMs;
    private final List<WorkView.RequestWork> requests;
    private final List<WorkView.BatchWork> batches;
    private final long unknownRequestCount;
    private volatile WorkView materialized;

    public WorkCapture(long capturedAtMs,
                       List<WorkView.RequestWork> requests,
                       List<WorkView.BatchWork> batches,
                       long unknownRequestCount) {
        this.capturedAtMs = capturedAtMs;
        this.requests = List.copyOf(requests);
        this.batches = List.copyOf(batches);
        this.unknownRequestCount = unknownRequestCount;
    }

    /**
     * Shared lazy load view; callers never hold the endpoint ownership lock.
     */
    public WorkView materialize() {
        WorkView snapshot = materialized;
        if (snapshot != null) {
            return snapshot;
        }
        synchronized (this) {
            if (materialized == null) {
                List<WorkView.RequestWork> orderedRequests =
                        requests.stream().sorted(Comparator.comparing(WorkView.RequestWork::requestId)).toList();
                List<WorkView.BatchWork> orderedBatches = batches.stream()
                        .sorted(Comparator.comparingLong(WorkView.BatchWork::batchId))
                        .map(batch -> new WorkView.BatchWork(batch.batchId(),
                                batch.requestIds().stream().sorted().toList(), batch.phase(), batch.remainingWorkMs()))
                        .toList();
                materialized = new WorkView(capturedAtMs, orderedRequests, orderedBatches, unknownRequestCount);
            }
            return materialized;
        }
    }
}

package org.flexlb.balance.endpoint.prefill;

public record Stats(long inflightRequests,
                    int locallyOwnedRequests,
                    int workerStatusUnconfirmedRequests,
                    int individuallyOwnedRequests,
                    int batchCount,
                    long maxObservedAgeMs) {
    public Stats {
        if (inflightRequests < 0L || locallyOwnedRequests < 0 || workerStatusUnconfirmedRequests < 0
                || individuallyOwnedRequests < 0 || batchCount < 0 || maxObservedAgeMs < 0L) {
            throw new IllegalArgumentException("Prefill state stats must be non-negative");
        }
    }
}

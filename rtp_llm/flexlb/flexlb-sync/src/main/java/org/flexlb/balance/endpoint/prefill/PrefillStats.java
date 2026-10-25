package org.flexlb.balance.endpoint.prefill;

public record PrefillStats(long inflightRequests,
                           int committedRequests,
                           int waitingForWorkerStatusRequests,
                           int singleRequests,
                           int batchCount,
                           long maxStatusAgeMs) {
    public PrefillStats {
        if (inflightRequests < 0L || committedRequests < 0 || waitingForWorkerStatusRequests < 0 || singleRequests < 0
                || batchCount < 0 || maxStatusAgeMs < 0L) {
            throw new IllegalArgumentException("Prefill state stats must be non-negative");
        }
    }
}

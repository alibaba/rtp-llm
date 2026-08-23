package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class BatchDispatcherConfig implements DispatcherConfig {

    /** Schema-v1 compatibility field; scheduler.decision takes precedence. */
    private int maxRequests = 8;
    /** Schema-v1 compatibility field; scheduler.decision takes precedence. */
    private long maxCollectionWaitMs = 300;
    /** Schema-v1 compatibility field; scheduler.capacity takes precedence. */
    private int maxWaitingRequestsPerPrefillWorker = 1024;
    /**
     * Schema-v1 early-dispatch trigger. Group growth stops before an additional
     * member whose resulting prediction is greater than or equal to this value.
     * This differs from the explicit strict maximum, which allows equality.
     */
    private Long earlyDispatchPredictedExecutionMs;
    private Integer maxInflightBatchesPerPrefillWorker;
    private long enqueueRpcTimeoutMs = 5000;
}

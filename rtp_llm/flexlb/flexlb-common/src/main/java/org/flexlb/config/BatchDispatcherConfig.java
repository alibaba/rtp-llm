package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class BatchDispatcherConfig implements DispatcherConfig {

    private int maxRequests = 8;
    private long maxCollectionWaitMs = 300;
    private int maxWaitingRequestsPerPrefillWorker = 1024;
    private Long earlyDispatchPredictedExecutionMs;
    private Integer maxInflightBatchesPerPrefillWorker;
    private long enqueueRpcTimeoutMs = 5000;
}

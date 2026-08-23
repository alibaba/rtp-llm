package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Controls Master-side EnqueueBatch delivery, not decision-group formation. */
@Getter
@Setter
public final class BatchDispatcherConfig implements DispatcherConfig {

    private Integer maxInflightBatchesPerPrefillWorker;
    private long enqueueRpcTimeoutMs = 5000;
}

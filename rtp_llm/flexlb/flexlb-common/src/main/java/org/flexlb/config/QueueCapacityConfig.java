package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class QueueCapacityConfig {

    /** Maximum requests currently owned by the QUEUE scheduler, cluster-wide. */
    private int maxOutstandingRequestsGlobal = 100_000;

    /**
     * Hard waiting-queue bound for each Prefill worker. Omission asks the
     * scheduler to use the schema-v1 dispatcher/runtime fallback.
     */
    private Integer maxWaitingRequestsPerPrefillWorker;
}

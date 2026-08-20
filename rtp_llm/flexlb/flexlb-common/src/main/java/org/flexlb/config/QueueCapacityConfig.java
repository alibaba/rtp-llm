package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class QueueCapacityConfig {

    /** Maximum requests currently owned by the QUEUE scheduler, cluster-wide. */
    private int maxOutstandingRequestsGlobal = 100_000;
}

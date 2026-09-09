package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class RequestLifecycleConfig {

    /** Maximum silence since registration or the latest matching Engine request status. */
    private long staleInflightTimeoutMs = 300_000;
    private long deliveredNotAcceptedTimeoutMs = 30_000;
    private int maxDeliveredNotAcceptedRequestsGlobal = 200;
}

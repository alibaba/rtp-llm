package org.flexlb.balance.resource;

import org.flexlb.config.ConfigService;
import org.springframework.stereotype.Component;

/** Canonical Prefill pending-request capacity predicate. */
@Component
public class PrefillResourceMeasure {
    private final long maxPendingRequests;

    public PrefillResourceMeasure(ConfigService configService) {
        this.maxPendingRequests = configService.loadBalanceConfig()
                .getRouter().getRoles().getPrefill()
                .getAvailability().getMaxPendingRequests();
    }

    public boolean isResourceAvailable(long pendingRequests) {
        if (pendingRequests < 0L) {
            throw new IllegalArgumentException(
                    "pendingRequests must be non-negative");
        }
        return pendingRequests < maxPendingRequests;
    }
}

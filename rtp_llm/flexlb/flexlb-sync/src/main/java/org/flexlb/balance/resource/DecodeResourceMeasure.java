package org.flexlb.balance.resource;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.springframework.stereotype.Component;

/** Canonical Decode KV and engine-concurrency capacity predicate. */
@Component
public class DecodeResourceMeasure {
    private final long availableThreshold;
    private final long concurrencyLimit;

    public DecodeResourceMeasure(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.availableThreshold = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxKvUsagePercent();
        Long configuredLimit = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests();
        this.concurrencyLimit = configuredLimit == null ? 0 : configuredLimit;
    }

    public boolean isResourceAvailable(DecodeEndpoint.DecodeRoutingView view) {
        return view != null && isAvailable(
                view.engineLoad(), view.realKvUsed(), view.totalKv());
    }

    public boolean isEngineDispatchAvailable(
            DecodeEndpoint.DecodeRoutingView view) {
        return view != null && isAvailable(
                view.engineCapacityUsed(),
                view.engineFacingKvUsed(),
                view.totalKv());
    }

    private boolean isAvailable(long engineLoad, long used, long total) {
        if (concurrencyLimit > 0 && engineLoad >= concurrencyLimit) {
            return false;
        }
        return total == 0
                || used * 100.0 / total < availableThreshold;
    }
}

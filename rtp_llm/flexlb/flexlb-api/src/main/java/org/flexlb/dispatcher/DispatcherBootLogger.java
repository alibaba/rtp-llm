package org.flexlb.dispatcher;

import javax.annotation.PostConstruct;
import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

/**
 * Emits the boot WARN line surfacing dispatcher footprint. WARN so the line survives default
 * {@code LOG_LEVEL=null} gating — operators need this exact line to verify which FE pool,
 * batchSpecs count, and timeouts the dispatcher came up with.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "enabled", havingValue = "true")
public class DispatcherBootLogger {

    private final DispatchConfig cfg;
    private final DispatcherFePoolRefresher refresher;

    public DispatcherBootLogger(DispatchConfig cfg, DispatcherFePoolRefresher refresher) {
        this.cfg = cfg;
        this.refresher = refresher;
    }

    @PostConstruct
    void logBoot() {
        Logger.warn("dispatcher enabled: fePoolServiceId={}, seedHosts={}, subBatch={}, batchSpecs={}, "
                        + "batchTimeoutMs={}, feMaxConnectionsPerHost={}, feMaxPendingAcquirePerHost={}, "
                        + "probePath={}, preAssignBe={}",
                cfg.getFePoolServiceId(), refresher.currentSize(), cfg.getSubBatch(),
                BatchEndpointSpec.SPECS.size(), cfg.getBatchTimeoutMs(),
                cfg.getFeMaxConnectionsPerHost(), cfg.getFeMaxPendingAcquirePerHost(),
                cfg.getProbePath(), cfg.isPreAssignBe());
    }
}

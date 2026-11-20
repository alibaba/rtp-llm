package org.flexlb.api.server;

import org.flexlb.cache.CacheMatchQuery;
import org.flexlb.cache.CacheMatchResult;
import org.flexlb.cache.CacheMatchSource;
import org.flexlb.cache.HostCacheMatch;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.config.scheduler.QueueOrderingConfig;
import org.flexlb.config.service.RoleType;
import org.junit.jupiter.api.BeforeEach;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Exercises configurable request ordering and Prefill selection through 750P/750D Schedule RPCs.
 * Three hot-cache densities and cold requests share the existing latency and resource limits.
 */
class TokenBalanceRequestOrderRoutePerformanceTest extends AbstractRoutePerformanceTest {
    private final List<CacheMatchResult> hotMatches = new ArrayList<>();
    private volatile boolean capacityProbe;

    @Override
    protected String mode() {
        return "QUEUE";
    }

    @Override
    protected FlexlbConfig createConfig() {
        var config = super.createConfig();
        config.getScheduler()
                .getOrdering()
                .getPolicy()
                .setType(
                        QueueOrderingConfig.PolicyType.valueOf(
                                System.getProperty("flexlb.perf.order", "PREFILL_COST")));
        config.getScheduler()
                .getOrdering()
                .getPriority()
                .setEnabled(
                        Boolean.parseBoolean(System.getProperty("flexlb.perf.priority", "true")));
        config.getRouter()
                .getRoles()
                .getPrefill()
                .getSelection()
                .setType(
                        RoutingConfig.PrefillSelectionType.valueOf(
                                System.getProperty("flexlb.perf.selection", "TOKEN_BALANCE")));
        if (Boolean.getBoolean("flexlb.perf.cacheAffinity")) {
            config.getRouter()
                    .getRoles()
                    .getPrefill()
                    .getSelection()
                    .setCacheAffinity(new RoutingConfig.CacheAffinityConfig());
        }
        return config;
    }

    @Override
    protected CacheMatchResult cacheMatches(CacheMatchQuery query) {
        long id = Long.parseLong(query.requestId());
        // A low-rate capacity probe uses distinct cache leaders; measured traffic keeps hot
        // cohorts.
        if (capacityProbe) return hotMatches.get((int) Math.floorMod(id, 750) * 3);
        int density = (int) Math.floorMod(id, 4);
        if (density == 0) return CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC);
        int worker = (int) Math.floorMod(id / 4, 750);
        return hotMatches.get(worker * 3 + density - 1);
    }

    @Override
    protected int requestPriority(long requestId) {
        return requestId % 2 == 0 ? 75 : 25;
    }

    @Override
    protected void prepareTraffic(int qps) {
        capacityProbe = qps == 64;
    }

    @BeforeEach
    void prepareCacheMatches() {
        var workers = endpointRegistry.endpoints(RoleType.PREFILL, null);
        org.junit.jupiter.api.Assertions.assertEquals(750, workers.size());
        for (var worker : workers) {
            for (int density = 1; density <= 3; density++) {
                long blocks = density * 16L;
                hotMatches.add(
                        new CacheMatchResult(
                                Map.of(worker.ipPort(), new HostCacheMatch(blocks, blocks)),
                                CacheMatchSource.LOCAL_SYNC,
                                0,
                                16));
            }
        }
    }
}

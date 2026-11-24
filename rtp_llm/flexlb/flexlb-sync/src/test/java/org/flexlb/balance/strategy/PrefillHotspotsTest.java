package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry.PrefillRoutingEntry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.RoutingConfig.HotspotReleaseConfig;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PrefillHotspotsTest {
    private long nowMs;
    private final PrefillHotspots hotspots = new PrefillHotspots(() -> TimeUnit.MILLISECONDS.toNanos(nowMs));
    private final HotspotReleaseConfig config = new HotspotReleaseConfig();

    @Test
    void thresholdUsesEachSignalsOwnMedianAndFloor() {
        config.setOutlierFactor(2);
        var requestOutlier = entry("requests", 27, 1000);
        var computeOutlier = entry("tokens", 1, 4001);
        var pool = List.of(requestOutlier, computeOutlier, entry("median", 13, 2000),
                entry("on-threshold", 18, 4000), entry("idle", 0, 0));
        // The medians are 13 requests and 2000 uncached tokens.
        assertEquals(Set.of(requestOutlier.endpoint(), computeOutlier.endpoint()), sample(pool, null, 0));
        config.setComputeTokenFloor(5000);
        config.setSetpointFloor(30);
        assertEquals(Set.of(), sample(pool, null, 100));
    }

    @Test
    void groupAndFleetCanDisagreeAboutTheSameEndpointInTheSameWindow() {
        var leader = entry("leader", 30, 0);
        var idle = entry("idle", 0, 0);
        var group = List.of(leader, idle);
        var fleet = List.of(leader, idle, entry("busy", 30, 0), entry("also-busy", 30, 0));
        assertEquals(Set.of(leader.endpoint()), sample(group, "a", 0));
        assertEquals(Set.of(), sample(fleet, null, 0));
        assertEquals(Set.of(leader.endpoint()), sample(group, "a", 1));
        assertEquals(Set.of(), sample(fleet, null, 1));
    }

    @Test
    void startsDwellAgainAfterAnIdleGapOrEndpointReplacement() {
        config.setSustainedPressureMs(500);
        var leader = entry("same-address", 30, 0);
        var idle = entry("idle", 0, 0);
        var group = List.of(leader, idle);
        assertEquals(Set.of(), sample(group, null, 0));
        assertEquals(Set.of(leader.endpoint()), sample(group, null, 500));
        assertEquals(Set.of(), sample(group, null, 2000));
        var replacement = entry("same-address", 30, 0);
        assertEquals(Set.of(), sample(List.of(replacement, idle), null, 2500));
        assertEquals(Set.of(replacement.endpoint()), sample(List.of(replacement, idle), null, 3000));
        when(idle.endpoint().observedRequestCount()).thenReturn(1L);
        assertEquals(Set.of(), sample(List.of(replacement, idle), null, 3100));
    }

    private Set<PrefillEndpoint> sample(List<PrefillRoutingEntry> pool, String group, long atMs) {
        nowMs = atMs;
        return hotspots.sample(RoleType.PREFILL, group, pool, config);
    }

    private static PrefillRoutingEntry entry(String address, long requests, long tokens) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.observedRequestCount()).thenReturn(requests);
        when(endpoint.pendingTokenWork()).thenReturn(tokens);
        return new PrefillRoutingEntry(address, endpoint);
    }
}

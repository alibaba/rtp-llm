package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry.PrefillRoutingEntry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.RoutingConfig.HotspotReleaseConfig;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
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
        var snapshot = sample(pool, null, 0);
        assertTrue(snapshot.isHot(requestOutlier.endpoint()));
        assertTrue(snapshot.isHot(computeOutlier.endpoint()));
        assertFalse(snapshot.isHot(pool.get(2).endpoint()));
        assertFalse(snapshot.isHot(pool.get(3).endpoint()));
        config.setComputeTokenFloor(5000);
        config.setSetpointFloor(30);
        var after = sample(pool, null, 100);
        assertFalse(after.isHot(requestOutlier.endpoint()));
        assertFalse(after.isHot(computeOutlier.endpoint()));
    }

    @Test
    void groupAndFleetCanDisagreeAboutTheSameEndpointInTheSameWindow() {
        var leader = entry("leader", 30, 0);
        var idle = entry("idle", 0, 0);
        var group = List.of(leader, idle);
        var fleet = List.of(leader, idle, entry("busy", 30, 0), entry("also-busy", 30, 0));
        assertTrue(sample(group, "a", 0).isHot(leader.endpoint()));
        assertFalse(sample(fleet, null, 0).isHot(leader.endpoint()));
        assertTrue(sample(group, "a", 1).isHot(leader.endpoint()));
        assertFalse(sample(fleet, null, 1).isHot(leader.endpoint()));
    }

    @Test
    void startsDwellAgainAfterAnIdleGapOrEndpointReplacement() {
        config.setSustainedPressureMs(500);
        var leader = entry("same-address", 30, 0);
        var idle = entry("idle", 0, 0);
        var group = List.of(leader, idle);
        assertFalse(sample(group, null, 0).isHot(leader.endpoint()));
        assertTrue(sample(group, null, 500).isHot(leader.endpoint()));
        assertFalse(sample(group, null, 2000).isHot(leader.endpoint()));
        var replacement = entry("same-address", 30, 0);
        assertFalse(sample(List.of(replacement, idle), null, 2500).isHot(replacement.endpoint()));
        assertTrue(sample(List.of(replacement, idle), null, 3000).isHot(replacement.endpoint()));
        when(idle.endpoint().observedRequestCount()).thenReturn(1L);
        assertTrue(sample(List.of(replacement, idle), null, 3100).mayRedirect(replacement.endpoint(), idle.endpoint()));
    }

    @Test
    void targetMustBeBelowTheMeanOnTheSignalThatMadeTheSourceHot() {
        // Request mean is exactly 20 and compute mean is exactly 2000.
        var requestHot = entry("request-hot", 60, 1000);
        var computeHot = entry("compute-hot", 10, 6000);
        var requestLight = entry("request-light", 10, 2200);
        var computeLight = entry("compute-light", 20, 500);
        var atBothMeans = entry("at-mean", 20, 2000);
        var peer = entry("peer", 0, 300);
        var remaining = entry("remaining", 20, 2000);
        var pool = List.of(requestHot, computeHot, requestLight, computeLight, atBothMeans, peer, remaining);
        var snapshot = sample(pool, null, 0);
        assertTrue(snapshot.mayRedirect(requestHot.endpoint(), requestLight.endpoint()));
        assertTrue(snapshot.mayRedirect(computeHot.endpoint(), computeLight.endpoint()));
        assertFalse(snapshot.mayRedirect(requestHot.endpoint(), computeLight.endpoint()));
        assertFalse(snapshot.mayRedirect(computeHot.endpoint(), requestLight.endpoint()));
        assertFalse(snapshot.mayRedirect(requestHot.endpoint(), atBothMeans.endpoint()));
        assertFalse(snapshot.mayRedirect(computeHot.endpoint(), atBothMeans.endpoint()));
        assertFalse(snapshot.mayRedirect(requestHot.endpoint(), computeHot.endpoint()));
        assertFalse(snapshot.mayRedirect(computeLight.endpoint(), requestLight.endpoint()));
    }

    @Test
    void workerAtTheIntegerPartOfAFractionalMeanCanReceiveWork() {
        var source = entry("source", 30, 0);
        var target = entry("target", 16, 0);
        var snapshot = sample(List.of(source, target, entry("a", 11, 0),
                entry("b", 12, 0), entry("c", 12, 0)), null, 0);
        assertTrue(snapshot.mayRedirect(source.endpoint(), target.endpoint()), "16 is below 16.2");
    }

    @Test
    void nonidlePeerWithPositiveInflightCanReceiveAHotspotsWork() {
        var hot = entry("hot", 100, 300);
        var lighter = entry("still-running", 15, 200);
        when(lighter.endpoint().getInflightBatchCount()).thenReturn(2);
        var snapshot = sample(List.of(hot, lighter), null, 0);
        assertTrue(snapshot.isHot(hot.endpoint()));
        assertTrue(snapshot.mayRedirect(hot.endpoint(), lighter.endpoint()));
    }

    @Test
    void noBelowMeanTargetWhenWorkersHaveIdenticalLoadOrPoolIsEmpty() {
        var first = entry("first", 30, 2500);
        var second = entry("second", 30, 2500);
        var snapshot = sample(List.of(first, second), null, 0);
        assertFalse(snapshot.isHot(first.endpoint()));
        assertFalse(snapshot.mayRedirect(first.endpoint(), second.endpoint()));
        assertFalse(sample(List.of(), null, 100).mayRedirect(first.endpoint(), second.endpoint()));
    }

    private PrefillHotspots.Snapshot sample(List<PrefillRoutingEntry> pool, String group, long atMs) {
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

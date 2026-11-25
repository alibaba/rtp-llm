package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry.PrefillRoutingEntry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.RoutingConfig.HotspotReleaseConfig;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.OptionalLong;
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
        assertTrue(mayRedirect(sample(List.of(replacement, idle), null, 3100), replacement, idle));
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
        assertTrue(mayRedirect(snapshot, requestHot, requestLight));
        assertTrue(mayRedirect(snapshot, computeHot, computeLight));
        assertFalse(mayRedirect(snapshot, requestHot, computeLight));
        assertFalse(mayRedirect(snapshot, computeHot, requestLight));
        assertFalse(mayRedirect(snapshot, requestHot, atBothMeans));
        assertFalse(mayRedirect(snapshot, computeHot, atBothMeans));
        assertFalse(mayRedirect(snapshot, requestHot, computeHot));
        assertFalse(mayRedirect(snapshot, computeLight, requestLight));
    }

    @Test
    void workerAtTheIntegerPartOfAFractionalMeanCanReceiveWork() {
        var source = entry("source", 30, 0);
        var target = entry("target", 16, 0);
        var snapshot = sample(List.of(source, target, entry("a", 11, 0),
                entry("b", 12, 0), entry("c", 12, 0)), null, 0);
        assertTrue(mayRedirect(snapshot, source, target), "16 is below 16.2");
    }

    @Test
    void nonidlePeerWithPositiveInflightCanReceiveAHotspotsWork() {
        var hot = entry("hot", 100, 300);
        var lighter = entry("still-running", 15, 200);
        when(lighter.endpoint().getInflightBatchCount()).thenReturn(2);
        var snapshot = sample(List.of(hot, lighter), null, 0);
        assertTrue(snapshot.isHot(hot.endpoint()));
        assertTrue(mayRedirect(snapshot, hot, lighter));
    }

    @Test
    void noBelowMeanTargetWhenWorkersHaveIdenticalLoadOrPoolIsEmpty() {
        var first = entry("first", 30, 2500);
        var second = entry("second", 30, 2500);
        var snapshot = sample(List.of(first, second), null, 0);
        assertFalse(snapshot.isHot(first.endpoint()));
        assertFalse(mayRedirect(snapshot, first, second));
        assertFalse(mayRedirect(sample(List.of(), null, 100), first, second));
    }

    @Test
    void stopsRedirectingToTargetThatFilledAfterPoolWasSampled() {
        config.setOutlierFactor(1.0);
        var source = entry("hot", 30, 5000);
        var target = entry("light", 10, 3000);
        var sampled = sample(List.of(source, target), null, 0);
        assertTrue(mayRedirect(sampled, source, target));

        var withPlans = new PrefillTokenWorkload.Snapshot(3000, 1000, 10, 100, 10);
        assertFalse(sampled.mayRedirect(source.endpoint(), target.endpoint(), withPlans));
        var afterAdmission = new PrefillTokenWorkload.Snapshot(4000, 0, 0, 0, 20);
        assertFalse(sampled.mayRedirect(source.endpoint(), target.endpoint(), afterAdmission));
    }

    @Test
    void receiverMustStayBelowBothHotspotLimitsBetweenPoolSamples() {
        var requestHot = entry("request-hot", 60, 100);
        var computeHot = entry("compute-hot", 1, 6000);
        var target = entry("target", 1, 100);
        var pool = List.of(requestHot, computeHot, target, entry("d", 1, 100), entry("e", 1, 100));
        var snapshot = sample(pool, null, 0);
        assertTrue(mayRedirect(snapshot, requestHot, target));
        assertTrue(mayRedirect(snapshot, computeHot, target));

        // The floors set the limits to 10 requests and 1000 tokens. Equal is still eligible.
        var atComputeLimit = new PrefillTokenWorkload.Snapshot(100, 900, 1, 0, 1);
        var aboveComputeLimit = new PrefillTokenWorkload.Snapshot(100, 901, 1, 0, 1);
        assertTrue(snapshot.mayRedirect(requestHot.endpoint(), target.endpoint(), atComputeLimit));
        assertFalse(snapshot.mayRedirect(requestHot.endpoint(), target.endpoint(), aboveComputeLimit));

        var atRequestLimit = new PrefillTokenWorkload.Snapshot(100, 0, 9, 0, 1);
        var aboveRequestLimit = new PrefillTokenWorkload.Snapshot(100, 0, 10, 0, 1);
        assertTrue(snapshot.mayRedirect(computeHot.endpoint(), target.endpoint(), atRequestLimit));
        assertFalse(snapshot.mayRedirect(computeHot.endpoint(), target.endpoint(), aboveRequestLimit));
    }

    @Test
    void windowRanksQueuedComputeAndKeepsOnlyItsTopTwoSources() {
        config.setSourceTopK(2);
        var first = entry("a", 60, 5000);
        var second = entry("b", 60, 6000);
        var third = entry("c", 1, 9000);
        var idle = entry("idle", 0, 0);
        var pool = List.of(third, second, first, idle, entry("d", 0, 0), entry("e", 0, 0), entry("f", 0, 0));
        when(first.endpoint().queuedComputeTokens()).thenReturn(OptionalLong.of(5000));
        when(second.endpoint().queuedComputeTokens()).thenReturn(OptionalLong.empty()); // falls back to 6000
        when(third.endpoint().queuedComputeTokens()).thenReturn(OptionalLong.of(4500));
        var initial = sample(pool, null, 0);
        assertTrue(mayRedirect(initial, first, idle));
        assertTrue(mayRedirect(initial, second, idle));
        assertFalse(mayRedirect(initial, third, idle));
        // The third worker is light on requests but hot on compute. It cannot receive work.
        assertFalse(mayRedirect(initial, first, third));

        when(first.endpoint().observedRequestCount()).thenReturn(0L);
        when(first.endpoint().pendingTokenWork()).thenReturn(0L);
        when(third.endpoint().queuedComputeTokens()).thenReturn(OptionalLong.of(9000));
        var mid = sample(pool, null, 900);
        assertFalse(mayRedirect(mid, first, idle)); // cooled, even though selected for this window
        assertFalse(mayRedirect(mid, third, idle)); // no mid-window promotion
        assertTrue(mayRedirect(mid, second, idle));
        var next = sample(pool, null, 1000);
        assertTrue(mayRedirect(next, third, idle));
        assertTrue(mayRedirect(next, second, idle));
    }

    @Test
    void tiedSourcesUseStableAddressesAndSeparateWindowsPerGroup() {
        config.setSourceTopK(1);
        var a = entry("a", 30, 5000);
        var b = entry("b", 30, 5000);
        var idle = entry("idle", 0, 0);
        var pool = List.of(b, a, idle, entry("d", 0, 0), entry("e", 0, 0));
        assertTrue(sample(pool, "g1", 0).isRedirectSource(a.endpoint()));
        assertTrue(sample(List.of(b, idle), "g2", 10).isRedirectSource(b.endpoint()));
        var replacement = entry("a", 30, 5000);
        var changed = List.of(replacement, b, idle, pool.get(3), pool.get(4));
        assertFalse(sample(changed, "g1", 100).isRedirectSource(replacement.endpoint()));
        assertTrue(sample(changed, "g1", 1000).isRedirectSource(replacement.endpoint()));
    }

    private static boolean mayRedirect(PrefillHotspots.Snapshot sample, PrefillRoutingEntry source,
                                       PrefillRoutingEntry target) {
        return sample.mayRedirect(source.endpoint(), target.endpoint(), target.endpoint().tokenWorkloadSnapshot());
    }

    private PrefillHotspots.Snapshot sample(List<PrefillRoutingEntry> pool, String group, long atMs) {
        nowMs = atMs;
        return hotspots.sample(RoleType.PREFILL, group, pool, config);
    }

    private static PrefillRoutingEntry entry(String address, long requests, long tokens) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.observedRequestCount()).thenReturn(requests);
        when(endpoint.pendingTokenWork()).thenReturn(tokens);
        when(endpoint.tokenWorkloadSnapshot()).thenReturn(new PrefillTokenWorkload.Snapshot(tokens, 0, 0, 0, requests));
        return new PrefillRoutingEntry(address, endpoint);
    }
}

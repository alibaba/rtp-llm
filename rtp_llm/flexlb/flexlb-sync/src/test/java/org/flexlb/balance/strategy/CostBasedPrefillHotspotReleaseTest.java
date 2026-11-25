package org.flexlb.balance.strategy;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.balance.strategy.HotspotRedirectLimiter.Ticket;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static org.flexlb.balance.strategy.HotspotRedirectLimiter.Rejection.BUDGET_EXHAUSTED;
import static org.flexlb.balance.strategy.HotspotRedirectLimiter.Rejection.PACING;
import static org.flexlb.config.RoutingConfig.HotspotTargetPolicy.MIN_PENDING_COMPUTE_WITHIN_ONE_BLOCK;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Shared redirect budget and hotspot selection for both Prefill ranking policies. */
class CostBasedPrefillHotspotReleaseTest {

    private final EngineHealthReporter reporter = mock(EngineHealthReporter.class);
    private final AtomicLong nowNanos = new AtomicLong();
    private final HotspotRedirectLimiter budget = new HotspotRedirectLimiter(nowNanos::get);

    private void recordColdArrival(long inputTokens) {
        var context = tokenContext(new FlexlbConfig());
        context.getRequest().setSeqLen(inputTokens);
        budget.recordPlacement(context, 0L);
    }

    @Test
    void bestHitBaselineIsPublishedOncePerRequestEvenAfterItsSampleExpires() {
        var request = tokenContext(new FlexlbConfig());
        java.util.stream.IntStream.range(0, 32).parallel()
                .forEach(i -> budget.recordPlacement(request, 900));
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(101, 1000));
        assertInstanceOf(Ticket.class, budget.tryAcquire(100, 1000));

        nowNanos.set(MILLISECONDS.toNanos(1000));
        budget.recordPlacement(request, 0L);
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(1, 1000), "a new cache estimate on retry cannot refill the window");
    }

    @Test
    void chargeOnlyAdditionalComputeTokensEvenWhenTheBudgetIsZero() {
        assertInstanceOf(Ticket.class, budget.tryAcquire(0, 0));
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(1, 10));
        recordColdArrival(100_099);
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(1001, 10));
        assertInstanceOf(Ticket.class, budget.tryAcquire(600, 10));
        assertInstanceOf(Ticket.class, budget.tryAcquire(0, 0));
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(401, 10), "both budget and pacing would reject");
        assertEquals(PACING, budget.tryAcquire(400, 10), "600 of 1000 tokens consumed 600 ms of pacing");
        nowNanos.set(MILLISECONDS.toNanos(600));
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(401, 10));
        assertInstanceOf(Ticket.class, budget.tryAcquire(400, 10));
        assertInstanceOf(Ticket.class, budget.tryAcquire(0, 10));
    }

    @Test
    void baselineAndChargedTokensExpireOnTheirOwnSlidingTimestamps() {
        nowNanos.set(MILLISECONDS.toNanos(900));
        recordColdArrival(200_000);
        nowNanos.set(MILLISECONDS.toNanos(1050));
        assertInstanceOf(Ticket.class, budget.tryAcquire(1000, 10));
        nowNanos.set(MILLISECONDS.toNanos(1750));
        recordColdArrival(100_000);
        nowNanos.set(MILLISECONDS.toNanos(1900));
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(1, 10), "remaining 1000-token allowance was already spent");
        nowNanos.set(MILLISECONDS.toNanos(2050));
        assertInstanceOf(Ticket.class, budget.tryAcquire(1000, 10));
        nowNanos.set(MILLISECONDS.toNanos(3000));
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(1, 10));
    }

    @Test
    void onlyAbandonedRedirectsRefundBudgetAndPacing() {
        recordColdArrival(1000);
        var first = assertInstanceOf(Ticket.class, budget.tryAcquire(400, 1000));
        first.commit();
        first.close();
        assertEquals(PACING, budget.tryAcquire(1, 1000));

        nowNanos.set(MILLISECONDS.toNanos(400));
        var provisional = assertInstanceOf(Ticket.class, budget.tryAcquire(600, 1000));
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(1, 1000));
        provisional.close();
        provisional.close();
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(601, 1000), "the committed first charge still counts");
        assertInstanceOf(Ticket.class, budget.tryAcquire(600, 1000), "refunding the latest charge restores its pacing slot");
    }

    @Test
    void lateRollbackCannotRefundAnotherTicketsAllowanceOrPacing() {
        recordColdArrival(1000);
        var first = assertInstanceOf(Ticket.class, budget.tryAcquire(400, 1000));
        nowNanos.set(MILLISECONDS.toNanos(400));
        var next = assertInstanceOf(Ticket.class, budget.tryAcquire(400, 1000));
        next.commit();
        first.close();
        assertEquals(PACING, budget.tryAcquire(100, 1000), "the later committed ticket still controls pacing");

        nowNanos.set(MILLISECONDS.toNanos(800));
        var abandoned = assertInstanceOf(Ticket.class, budget.tryAcquire(500, 1000));
        nowNanos.set(MILLISECONDS.toNanos(1800));
        recordColdArrival(1000);
        var current = assertInstanceOf(Ticket.class, budget.tryAcquire(1000, 1000));
        abandoned.close();
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(1, 1000), "closing an expired ticket must leave the new window intact");
    }

    @Test
    void releasedTokenBalanceLeaderDriftsToLowerUncachedWork() {
        FlexlbConfig config = configWithHotspotRelease();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(1000);
        recordColdArrival(1000);
        PrefillEndpoint leader = tokenEndpoint("leader");
        PrefillEndpoint other = tokenEndpoint("other");
        when(leader.observedRequestCount()).thenReturn(30L);
        when(leader.tokenWorkloadSnapshot()).thenReturn(new PrefillTokenWorkload.Snapshot(800));
        CostBasedPrefillStrategy strategy = tokenStrategy(leader, other);

        try (SelectedRole selected = strategy.select(
                tokenContext(config), RoleType.PREFILL, "g").value()) {
            assertEquals("other", selected.serverStatus().getServerIp(),
                    "a released leader must lose to lower uncached work");
            verify(other).tryPlanTokenWork(eq(0L), anyLong(), anyLong());
        }

        verify(reporter).reportPrefillHotspotDecision("leader", "FAILED", "ROUTE_NOT_COMMITTED");
        verify(reporter, never()).reportPrefillHotspotDecision(any(), eq("SUCCESS"), any());

        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(0);
        try (SelectedRole selected = strategy.select(tokenContext(config), RoleType.PREFILL, "g").value()) {
            assertEquals("leader", selected.serverStatus().getServerIp(), "exhausted redirect quota keeps the pin");
        }
        verify(reporter, times(2)).reportPrefillHotspotDecision("leader", "NEEDED", "HOTSPOT");
        verify(reporter).reportPrefillHotspotDecision("leader", "FAILED", "BUDGET_EXHAUSTED");
        verify(reporter, never()).reportPrefillHotspotDecision(eq("other"), any(), any());
    }

    @Test
    void placementRetriesDoNotIncreaseTheBestHitComputeBudget() {
        var config = configWithHotspotRelease();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(1000);
        PrefillEndpoint leader = tokenEndpoint("leader");
        PrefillEndpoint other = tokenEndpoint("other");
        when(leader.observedRequestCount()).thenReturn(30L);
        when(leader.tokenWorkloadSnapshot()).thenReturn(new PrefillTokenWorkload.Snapshot(800));
        var strategy = tokenStrategy(leader, other);
        var request = tokenContext(config);
        // Each 1000-token request has 900 effective hits at the leader and 500 at the peer.
        for (int attempt = 0; attempt < 5; attempt++) {
            try (var selected = strategy.select(request, RoleType.PREFILL, "g").value()) {
                assertEquals("leader", selected.serverStatus().getServerIp());
            }
        }
        for (int id = 2; id <= 4; id++) {
            var next = tokenContext(config);
            next.getRequest().setRequestId(id);
            try (var selected = strategy.select(next, RoleType.PREFILL, "g").value()) {
                assertEquals(id == 4 ? "other" : "leader", selected.serverStatus().getServerIp());
            }
        }
    }

    @Test
    void unreleasedTokenBalanceLeaderKeepsThePin() {
        FlexlbConfig config = configWithHotspotRelease();
        PrefillEndpoint leader = tokenEndpoint("leader");
        PrefillEndpoint other = tokenEndpoint("other");
        when(leader.tokenWorkloadSnapshot()).thenReturn(new PrefillTokenWorkload.Snapshot(800));
        CostBasedPrefillStrategy strategy = tokenStrategy(leader, other);

        try (SelectedRole selected = strategy.select(
                tokenContext(config), RoleType.PREFILL, "g").value()) {
            assertEquals("leader", selected.serverStatus().getServerIp(),
                    "a non-hotspot leader keeps its cache affinity");
            verify(leader).tryPlanTokenWork(eq(org.flexlb.balance.AdmissionVersion.ANY), anyLong(), anyLong());
        }
        verify(reporter).reportPrefillHotspotDecision("leader", "NOT_NEEDED", "ORDINARY_PLACEMENT");
        verify(reporter, never()).reportPrefillHotspotDecision(any(), eq("NEEDED"), any());
    }

    @Test
    void disabledSwitchKeepsTokenBalancePinningUnchanged() {
        PrefillEndpoint leader = tokenEndpoint("leader");
        PrefillEndpoint other = tokenEndpoint("other");
        when(leader.tokenWorkloadSnapshot()).thenReturn(new PrefillTokenWorkload.Snapshot(800));
        FlexlbConfig config = tokenBalanceConfig(false);
        CostBasedPrefillStrategy strategy = tokenStrategy(leader, other);

        try (SelectedRole selected = strategy.select(
                tokenContext(config), RoleType.PREFILL, "g").value()) {
            assertEquals("leader", selected.serverStatus().getServerIp());
        }
        verify(leader, never()).observedRequestCount();
        verify(reporter, never()).reportPrefillHotspotDecision(any(), any(), any());
    }

    @Test
    void hotLeaderKeepsTheRequestWhenTheOnlyPeerCannotAcceptIt() {
        var leader = tokenEndpoint("leader");
        var peer = tokenEndpoint("other");
        when(leader.observedRequestCount()).thenReturn(30L);
        when(peer.canAcceptRequest()).thenReturn(false);
        var strategy = tokenStrategy(leader, peer);

        try (var selected = strategy.select(tokenContext(configWithHotspotRelease()), RoleType.PREFILL, "g").value()) {
            assertEquals("leader", selected.serverStatus().getServerIp());
        }
        verify(reporter).reportPrefillHotspotDecision("leader", "NEEDED", "HOTSPOT");
        verify(reporter).reportPrefillHotspotDecision("leader", "FAILED", "NO_ELIGIBLE_TARGET");
        verify(reporter, never()).reportPrefillHotspotDecision(any(), eq("NOT_NEEDED"), any());
    }

    @Test
    void stalePeerRetryReportsSuccessOnlyWhenTheRouteOwnerCommitsAndPacesTheNextAttempt() {
        var config = configWithHotspotRelease();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(1000);
        recordColdArrival(10_000);
        var leader = tokenEndpoint("leader");
        var peer = tokenEndpoint("other");
        when(leader.observedRequestCount()).thenReturn(30L);
        var work = mock(PrefillTokenWorkload.Reservation.class);
        when(peer.tryPlanTokenWork(anyLong(), anyLong(), anyLong())).thenReturn(null).thenReturn(work);
        var strategy = tokenStrategy(leader, peer);

        try (var selected = strategy.select(tokenContext(config), RoleType.PREFILL, "g").value();
             var redirect = selected.takeHotspotRedirect()) {
            assertEquals("other", selected.serverStatus().getServerIp());
            verify(reporter).reportPrefillHotspotDecision("leader", "FAILED", "TARGET_VERSION_CHANGED");
            verify(reporter, never()).reportPrefillHotspotDecision(any(), eq("SUCCESS"), any());
            redirect.commit();
        }
        verify(reporter).reportPrefillHotspotDecision("leader", "SUCCESS", "ROUTE_COMMITTED");

        try (var selected = strategy.select(tokenContext(config), RoleType.PREFILL, "g").value()) {
            assertEquals("leader", selected.serverStatus().getServerIp());
        }
        verify(reporter).reportPrefillHotspotDecision("leader", "FAILED", "PACING");
        verify(reporter, never()).reportPrefillHotspotDecision(any(), eq("FAILED"), eq("ROUTE_NOT_COMMITTED"));
        verify(reporter, never()).reportPrefillHotspotDecision(eq("other"), any(), any());
    }

    @Test
    void enabledSwitchReplacesTheAffinityTtftCap() {
        FlexlbConfig config = affinityConfig();
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        var ranking = new PrefillCandidateRanking(reporter);
        var request = tokenContext(config);

        config.getRouter().getRoles().getPrefill().getHotspotRelease().setEnabled(false);
        assertEquals(0, ranking.select(
                candidates(100L, 1000L, 500L, 600L), RoleType.PREFILL, null, request).index(),
                "gate off: the TTFT cap must keep rejecting the leader");
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setEnabled(true);
        var enabled = ranking.select(candidates(100L, 1000L, 500L, 600L), RoleType.PREFILL, null, request);
        assertEquals(1, enabled.index(), "gate on: the healthy leader retains affinity");
        assertEquals("CACHE_LEADER", enabled.reason());
    }

    @Test
    void releasedAffinityLeaderFallsBackToBaseline() {
        FlexlbConfig config = affinityConfig();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(1000);
        recordColdArrival(1000);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        var ranking = new PrefillCandidateRanking(reporter);
        var request = tokenContext(config);
        PrefillCandidateSet survivors = candidates(100L, 1000L, 500L, 600L);
        sampleRequestLoads(survivors, 4, 30);

        var choice = ranking.select(survivors, RoleType.PREFILL, null, request);
        assertEquals(0, choice.index());
        assertEquals("HOTSPOT_RELEASE", choice.reason());
        assertEquals(1, choice.rateLimited().index());
        assertEquals("HOTSPOT_RATE_LIMIT", choice.rateLimited().reason());
    }

    @Test
    void redirectsHotMinimumTtftBaselineAtZeroCostWhenCacheHitsTie() {
        FlexlbConfig config = affinityConfig();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(0);
        var reporter = mock(EngineHealthReporter.class);
        var ranking = new PrefillCandidateRanking(reporter);
        var survivors = candidates(100, 200, 500, 500);
        sampleRequestLoads(survivors, 30, 4);

        var redirect = ranking.select(survivors, RoleType.PREFILL, "g", tokenContext(config));
        assertEquals(1, redirect.index());
    }

    @Test
    void rankingProposesTargetAndPreservesFastestFallbackWithoutSpendingBudget() {
        FlexlbConfig config = affinityConfig();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(10);
        var reporter = mock(EngineHealthReporter.class);
        var ranking = new PrefillCandidateRanking(reporter);
        var survivors = candidates(100, 200, 900, 500);
        sampleRequestLoads(survivors, 30, 4);
        var request = tokenContext(config);

        var proposed = ranking.select(survivors, RoleType.PREFILL, "g", request);
        assertEquals(1, proposed.index());
        assertEquals(0, proposed.fallback());
        assertEquals(BUDGET_EXHAUSTED, budget.tryAcquire(400, 10));

        recordColdArrival(40_000);
        assertEquals(proposed, ranking.select(survivors, RoleType.PREFILL, "g", request));
        assertInstanceOf(Ticket.class, budget.tryAcquire(400, 10), "ranking must leave budget available for registration");
    }

    @Test
    void retainsBaselineWithoutAHealthyAlternativeOrWhenReleaseIsDisabled() {
        FlexlbConfig config = affinityConfig();
        var reporter = mock(EngineHealthReporter.class);
        var ranking = new PrefillCandidateRanking(reporter);
        var request = tokenContext(config);
        var survivors = candidates(100, 200, 500, 500);

        assertEquals(0, ranking.select(survivors, RoleType.PREFILL, "g", request).index(), "healthy baseline");
        sampleRequestLoads(survivors, 30, 30, 4, 4, 4);
        assertEquals(0, ranking.select(survivors, RoleType.PREFILL, "g", request).index(), "all candidates hot");

        var single = new PrefillCandidateSet();
        single.reset(1);
        addCandidate(single, 1, "10.1.0.1", 100, 100, 500);
        sampleRequestLoads(single, 30, 4);
        assertEquals(0, ranking.select(single, RoleType.PREFILL, "g", request).index(), "one candidate");

        sampleRequestLoads(survivors, 30, 4);
        when(survivors.endpoint(1).engineQueueAdmissionBlocked()).thenReturn(true);
        assertEquals(0, ranking.select(survivors, RoleType.PREFILL, "g", request).index(), "peer cannot admit");
        when(survivors.endpoint(1).engineQueueAdmissionBlocked()).thenReturn(false);

        config.getRouter().getRoles().getPrefill().getHotspotRelease().setEnabled(false);
        assertEquals(0, ranking.select(survivors, RoleType.PREFILL, "g", request).index(), "release disabled");

    }

    @Test
    void fastestCoolPeerMustAlsoBeBelowMeanLoadForTtftAndTokenRanking() {
        var survivors = new PrefillCandidateSet();
        survivors.reset(3);
        addCandidate(survivors, 1, "10.1.0.1", 100, 100, 900);
        addCandidate(survivors, 2, "10.1.0.2", 110, 110, 800);
        addCandidate(survivors, 3, "10.1.0.3", 200, 200, 700);
        // 25 is above the pool mean (22.375) but below the source threshold (37.5).
        sampleRequestLoads(survivors, 70, 25, 2, 2, 2, 25, 26, 27);
        for (int i = 0; i < survivors.size(); i++) {
            when(survivors.endpoint(i).canAcceptRequest()).thenReturn(true);
        }
        when(survivors.workload(1).score()).thenReturn(1.0);
        when(survivors.workload(2).score()).thenReturn(50.0);
        for (FlexlbConfig config : List.of(affinityConfig(), tokenBalanceConfig(true))) {
            config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(1000);
            var ranking = new PrefillCandidateRanking(mock(EngineHealthReporter.class));
            assertEquals(2, ranking.select(survivors, RoleType.PREFILL, "g", tokenContext(config)).index());
        }
    }

    @Test
    void oneBlockPolicyRanksByPendingAndPlannedComputeThenHitThenRotationForBothStrategies() {
        var candidates = candidates(10, 20, 900, 850);
        addCandidate(candidates, 3, "10.1.0.3", 30, 30, 800);
        addCandidate(candidates, 4, "10.1.0.4", 15, 15, 799);
        addCandidate(candidates, 5, "10.1.0.5", 40, 40, 850);
        sampleRequestLoads(candidates, 60, 4, 4, 4, 4);
        for (int i = 0; i < candidates.size(); i++) {
            when(candidates.endpoint(i).canAcceptRequest()).thenReturn(true);
        }
        when(candidates.workload(2).pendingTokens()).thenReturn(45L);
        for (var config : List.of(affinityConfig(), tokenBalanceConfig(true))) {
            var ranking = new PrefillCandidateRanking(reporter);
            var request = tokenContext(config);
            var policy = config.getRouter().getRoles().getPrefill().getHotspotRelease();
            when(candidates.workload(1).plannedTokens()).thenReturn(80L);
            when(candidates.workload(4).pendingTokens()).thenReturn(80L);
            assertEquals(1, ranking.select(candidates, RoleType.PREFILL, "g", request).index());
            policy.setTargetPolicy(org.flexlb.config.RoutingConfig.HotspotTargetPolicy.MIN_PENDING_COMPUTE);
            assertEquals(3, ranking.select(candidates, RoleType.PREFILL, "g", request).index(),
                    "the lightest eligible receiver may lose more than one block");
            policy.setTargetPolicy(MIN_PENDING_COMPUTE_WITHIN_ONE_BLOCK);
            var selected = ranking.select(candidates, RoleType.PREFILL, "g", request);
            assertEquals(2, selected.index(), "a loss of exactly one block is allowed, one more token is excluded");
            assertEquals(0, selected.rateLimited().index());

            when(candidates.workload(1).plannedTokens()).thenReturn(45L);
            when(candidates.workload(4).pendingTokens()).thenReturn(45L);
            assertEquals(4, ranking.select(candidates, RoleType.PREFILL, "g", request).index());
            assertEquals(1, ranking.select(candidates, RoleType.PREFILL, "g", request).index());
        }
    }

    @Test
    void unknownRequestBlockSizeRequiresZeroLossAndDistinguishesMissingReceivers() {
        for (var config : List.of(affinityConfig(), tokenBalanceConfig(true))) {
            config.getRouter().getRoles().getPrefill().getHotspotRelease().setTargetPolicy(MIN_PENDING_COMPUTE_WITHIN_ONE_BLOCK);
            var request = tokenContext(config);
            request.getRequest().setCacheKeyBlockSize(0L);
            var candidates = candidates(10, 20, 900, 899);
            sampleRequestLoads(candidates, 40, 1);
            when(candidates.endpoint(0).canAcceptRequest()).thenReturn(true);
            when(candidates.endpoint(1).canAcceptRequest()).thenReturn(true);
            var ranking = new PrefillCandidateRanking(reporter);
            var rejected = ranking.select(candidates, RoleType.PREFILL, "g", request);
            assertEquals(0, rejected.index());
            assertEquals("CACHE_LOSS_LIMIT", rejected.noTargetReason());
            HotspotRedirect.observe(reporter, candidates, rejected);

            when(candidates.endpoint(1).engineQueueAdmissionBlocked()).thenReturn(true);
            var blocked = ranking.select(candidates, RoleType.PREFILL, "g", request);
            assertEquals("NO_ELIGIBLE_TARGET", blocked.noTargetReason());
            HotspotRedirect.observe(reporter, candidates, blocked);

            var equalHits = candidates(10, 20, 900, 900);
            sampleRequestLoads(equalHits, 40, 1);
            when(equalHits.endpoint(0).canAcceptRequest()).thenReturn(true);
            when(equalHits.endpoint(1).canAcceptRequest()).thenReturn(true);
            var zeroLoss = new PrefillCandidateRanking(reporter).select(equalHits, RoleType.PREFILL, "g", request);
            assertTrue(zeroLoss.redirect());
            assertEquals(1, zeroLoss.index());
        }
        verify(reporter, times(2)).reportPrefillHotspotDecision("10.1.0.1", "FAILED", "CACHE_LOSS_LIMIT");
        verify(reporter, times(2)).reportPrefillHotspotDecision("10.1.0.1", "FAILED", "NO_ELIGIBLE_TARGET");
    }

    @Test
    void oneBlockLimitUsesPoolMaximumEvenWhenOrdinaryWinnerHasLessCache() {
        var config = new FlexlbConfig();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setTargetPolicy(MIN_PENDING_COMPUTE_WITHIN_ONE_BLOCK);
        var candidates = candidates(10, 20, 800, 750);
        addCandidate(candidates, 3, "10.1.0.3", 30, 30, 900);
        sampleRequestLoads(candidates, 60, 1, 1);
        when(candidates.endpoint(2).engineQueueAdmissionBlocked()).thenReturn(true);
        for (int i = 0; i < candidates.size(); i++) {
            when(candidates.endpoint(i).canAcceptRequest()).thenReturn(true);
        }
        var selected = new PrefillCandidateRanking(reporter).select(candidates, RoleType.PREFILL, "g", tokenContext(config));
        assertEquals(0, selected.index());
        assertEquals("CACHE_LOSS_LIMIT", selected.noTargetReason());
    }

    @Test
    void twoHotGroupsCompeteForOnePacedStreamOfRedirects() {
        var config = affinityConfig();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(10);
        var ranking = new PrefillCandidateRanking(mock(EngineHealthReporter.class));
        var request = tokenContext(config);
        request.getRequest().setSeqLen(2000L);
        var a = candidates(100, 1000, 0, 1000);
        var b = candidates(100, 1000, 100, 1100);
        sampleRequestLoads(a, 4, 30);
        sampleRequestLoads(b, 4, 30);
        recordColdArrival(500_000);
        for (int ms : new int[] {0, 199, 200, 399, 400, 600, 800, 999}) {
            nowNanos.set(MILLISECONDS.toNanos(ms));
            long grants = java.util.stream.IntStream.range(0, 32).parallel().filter(i -> {
                boolean first = i % 2 == 0;
                var candidates = first ? a : b;
                var choice = ranking.select(candidates, RoleType.PREFILL, first ? "a" : "b", request);
                long extraTokens = candidates.maximumCacheHit - candidates.cacheHit(choice.index());
                return choice.redirect() && budget.tryAcquire(extraTokens, 10) instanceof Ticket;
            }).count();
            assertEquals(ms % 200 == 0 ? 1 : 0, grants, "redirects at " + ms + "ms");
        }
    }

    /** The pool may include endpoints that did not survive this request's projection. */
    private static void sampleRequestLoads(PrefillCandidateSet survivors, long... loads) {
        var pool = new ArrayList<EndpointRegistry.PrefillRoutingEntry>();
        for (int i = 0; i < loads.length; i++) {
            var endpoint = i < survivors.size() ? survivors.endpoint(i) : mock(PrefillEndpoint.class);
            when(endpoint.observedRequestCount()).thenReturn(loads[i]);
            if (i < survivors.size()) {
                when(survivors.workload(i).observedRequests()).thenReturn(loads[i]);
            }
            pool.add(new EndpointRegistry.PrefillRoutingEntry("peer-" + i, endpoint));
        }
        survivors.setHotspots(new PrefillHotspots(() -> 0L).sample(
                RoleType.PREFILL, "g", pool, new RoutingConfig.HotspotReleaseConfig()));
    }

    private static FlexlbConfig tokenBalanceConfig(boolean hotspotRelease) {
        FlexlbConfig config = new FlexlbConfig();
        config.getScheduler().setType(SchedulerConfig.Type.DIRECT);
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getRouter().getRoles().getPrefill()
                .setTokenBalance(new RoutingConfig.TokenBalanceConfig());
        config.getRouter().getRoles().getPrefill()
                .getHotspotRelease().setEnabled(hotspotRelease);
        return config;
    }

    private static FlexlbConfig configWithHotspotRelease() {
        return tokenBalanceConfig(true);
    }

    private CostBasedPrefillStrategy tokenStrategy(PrefillEndpoint leader, PrefillEndpoint other) {
        var fleet = List.of(new EndpointRegistry.PrefillRoutingEntry("leader:8080", leader),
                new EndpointRegistry.PrefillRoutingEntry("other:8080", other));
        WorkerDirectory directory = mock(WorkerDirectory.class);
        when(directory.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(fleet);
        for (EndpointRegistry.PrefillRoutingEntry entry : fleet) {
            PrefillEndpoint endpoint = entry.endpoint();
            long generationId = endpoint.getStatus().getGenerationId();
            var pin = mock(WorkerEndpoint.GenerationPin.class);
            when(pin.endpoint()).thenReturn(endpoint);
            when(pin.generationId()).thenReturn(generationId);
            when(directory.captureEndpoint(RoleType.PREFILL, entry.address())).thenReturn(pin);
        }
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), eq(RoleType.PREFILL), any())).thenReturn(
                Map.of("leader:8080", 9, "other:8080", 5));
        return new CostBasedPrefillStrategy(
                directory, cache, reporter, budget, new PrefillHotspots(nowNanos::get));
    }

    private static BalanceContext tokenContext(FlexlbConfig config) {
        Request request = new Request();
        request.setRequestId(1L);
        request.setSeqLen(1_000L);
        request.setPriority(50);
        request.setCacheKeyBlockSize(100L);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
        return context;
    }

    /** Mocked endpoint with production-shaped status and projection inputs. */
    private static PrefillEndpoint tokenEndpoint(String ip) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "g", ip, 8080, 8090, "test"));
        when(endpoint.getIp()).thenReturn(ip);
        when(endpoint.getHttpPort()).thenReturn(8080);
        when(endpoint.canAcceptRequest()).thenReturn(true);
        var predictor = mock(PrefillTimePredictor.class);
        var evaluator = mock(PrefillTimePredictor.Evaluator.class);
        when(evaluator.estimateMs(anyLong(), anyLong())).thenReturn(10L);
        when(predictor.evaluator()).thenReturn(evaluator);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.captureRouteProjectionInputs()).thenReturn(new RouteProjection.Inputs(
                new QueueSnapshot(0L, false, null, null, List.of(), null),
                new WorkSnapshot(0L, List.of(), List.of(), 0L)));
        when(endpoint.deliveryProjection()).thenReturn(
                new RouteDeliveryStrategy(
                        mock(RequestRegistry.class), mock(DeliveryMetrics.class))
                        .projectionPolicy());
        when(endpoint.engineQueueAdmissionBlocked()).thenReturn(false);
        when(endpoint.tokenWorkloadSnapshot()).thenReturn(new PrefillTokenWorkload.Snapshot(0));
        when(endpoint.tryPlanTokenWork(anyLong(), anyLong(), anyLong()))
                .thenReturn(mock(PrefillTokenWorkload.Reservation.class));
        return endpoint;
    }

    private static FlexlbConfig affinityConfig() {
        FlexlbConfig config = new FlexlbConfig();
        RoutingConfig.CacheAffinityConfig affinity =
                new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(10L);
        affinity.setMinPrefixHitPercent(0.0);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        return config;
    }

    private static PrefillCandidateSet candidates(
            long firstTtftMs, long secondTtftMs, long firstHit, long secondHit) {
        PrefillCandidateSet candidates = new PrefillCandidateSet();
        candidates.reset(2);
        addCandidate(candidates, 1, "10.1.0.1", firstTtftMs, firstTtftMs, firstHit);
        addCandidate(candidates, 2, "10.1.0.2", secondTtftMs, secondTtftMs, secondHit);
        return candidates;
    }

    private static void addCandidate(
            PrefillCandidateSet candidates, int index, String ip,
            long ttftMs, long executionMs, long hitTokens) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn(ip);
        when(endpoint.tokenWorkloadSnapshot()).thenReturn(mock(PrefillTokenWorkload.Snapshot.class));
        candidates.addCandidate(
                ip + ":8080",
                endpoint,
                new RouteProjection.Candidate(
                        ttftMs,
                        executionMs,
                        RouteProjection.Candidate.InitialHeadDisposition.NONE,
                        "",
                        hitTokens,
                        hitTokens),
                index);
    }
}

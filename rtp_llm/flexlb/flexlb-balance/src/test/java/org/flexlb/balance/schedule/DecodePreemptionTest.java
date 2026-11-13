package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.schedule.DecodePreemptionPolicy.DecodePreemptionPlan;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.scheduler.PreemptionConfig;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.status.DecodeTaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.EnumSet;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

/**
 * Admission and preemption contracts for {@link DecodePreemption#tryPreempt}.
 *
 * <p>Requirement: every early decline is side-effect free (returns null
 * without reserving a permit, touching any port, or emitting telemetry).
 * Corner cases are derived from the domain requirements, not by echoing
 * if-branches.
 */
@DisplayName("DecodePreemption.tryReclaim contracts")
class DecodePreemptionTest {
    private RequestSchedulerReporter reporter;
    private AbstractRequestScheduler requests;
    private DecodeEndpoint blockedEndpoint;
    private DefaultDecodePreemption acquirer;

    @BeforeEach
    void setUp() {
        reporter = mock(RequestSchedulerReporter.class);
        org.mockito.Mockito.doCallRealMethod()
                .when(reporter)
                .reportPreemptionCancelRequests(any(), any());
        requests = mock(AbstractRequestScheduler.class);
        blockedEndpoint = mock(DecodeEndpoint.class);
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        config.priorityOrdering().setPreemption(new PreemptionConfig());
        configureAdmission(config);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void capacityDecisionReportsOnlyInfeasibleEviction(boolean capacityBlocked) {
        var context = ctx(70);
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        var preemption = new PreemptionConfig();
        preemption.setAllowedPreemptionStages(EnumSet.of(PreemptionStage.DECODE_ENGINE_OWNED));
        config.priorityOrdering().setPreemption(preemption);
        config.getRouter()
                .getRoles()
                .getDecode()
                .getAvailability()
                .setMaxEngineRequests(capacityBlocked ? 1L : 4L);
        configureAdmission(config);
        when(context.getFuture()).thenReturn(new CompletableFuture<>());
        var endpoint = org.flexlb.balance.schedule.SchedulerTestSupport.mockDecodeEndpoint();
        org.mockito.Mockito.doReturn(true).when(endpoint).acceptsNewRequests();
        var snapshot = mock(DecodeResources.ResourceSnapshot.class);
        var routing = mock(DecodeEndpoint.SelectionSnapshot.class);
        when(endpoint.resourceSnapshot()).thenReturn(snapshot);
        when(snapshot.selection()).thenReturn(routing);
        when(routing.address()).thenReturn("decode-a");
        when(routing.usage())
                .thenReturn(new DecodeResources.CapacityUsage(1L, 1_000L, 1_000L, 0L, 0L));
        when(snapshot.requests())
                .thenReturn(
                        Map.of(
                                "901",
                                new DecodeResources.DecodeRequestView(
                                        String.valueOf(901L),
                                        70,
                                        128L,
                                        128L,
                                        DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                                        new DecodeEndpoint.RequestRegistration(
                                                1L, String.valueOf(901L)))));

        assertNull(acquirer.tryPreempt(context, endpoint));
        verify(endpoint, org.mockito.Mockito.times(1)).resourceSnapshot();
        verifyNoInteractions(requests);
        if (capacityBlocked) {
            verify(reporter)
                    .reportPreemption(
                            RequestSchedulerReporter.PreemptionEvent.PLAN,
                            70,
                            DecodePreemptionPlan.CASE_SLOT,
                            "infeasible");
            verifyNoMoreInteractions(reporter);
        } else {
            verifyNoInteractions(reporter);
        }
    }

    // ─── Shutdown ────────────────────────────────────────────────────────
    @Test
    @DisplayName("A shut-down acquirer declines without side effects")
    void shutdownDeclines() {
        acquirer.stopAdmission();
        org.junit.jupiter.api.Assertions.assertTrue(acquirer.awaitIdle(System.nanoTime()));
        assertNull(tryPreempt(ctx(70), new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    @Test
    void failedSetupDoesNotStrandDrain() {
        org.junit.jupiter.api.Assertions.assertThrows(
                NullPointerException.class, () -> acquirer.tryPreempt(null, blockedEndpoint));
        org.junit.jupiter.api.Assertions.assertTrue(acquirer.awaitIdle(System.nanoTime()));
        acquirer.close();
    }

    // ─── Future states ──────────────────────────────────────────────────
    @Test
    @DisplayName("An already-completed future declines without side effects")
    void completedFutureDeclines() {
        CompletableFuture<Response> done = new CompletableFuture<>();
        done.complete(null);
        assertNull(tryPreempt(ctx(70), done));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("An exceptionally-completed future declines without side effects")
    void exceptionalFutureDeclines() {
        CompletableFuture<Response> failed = new CompletableFuture<>();
        failed.completeExceptionally(new RuntimeException("test"));
        assertNull(tryPreempt(ctx(70), failed));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("A cancelled future declines without side effects")
    void cancelledFutureDeclines() {
        CompletableFuture<Response> cancelled = new CompletableFuture<>();
        cancelled.cancel(false);
        assertNull(tryPreempt(ctx(70), cancelled));
        assertZeroSideEffect();
    }

    // ─── Expiration ─────────────────────────────────────────────────────
    @Test
    @DisplayName("An expired request declines without side effects")
    void expiredRequestDeclines() {
        RequestContext expired = ctx(70);
        when(expired.getRequestExpiresAtMs()).thenReturn(1L);
        assertNull(tryPreempt(expired, new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    // Priority boundaries
    @Test
    @DisplayName("Priority 0 (NO_PRIORITY sentinel) declines without side effects")
    void noPriorityDeclines() {
        assertNull(tryPreempt(ctx(0), new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("Priority 1 reaches endpoint selection when preemption is enabled")
    void minimumValidPriorityPassesGuard() {
        var context = ctx(1);
        when(context.getFuture()).thenReturn(new CompletableFuture<>());
        var endpoint = org.flexlb.balance.schedule.SchedulerTestSupport.mockDecodeEndpoint();
        org.mockito.Mockito.doReturn(false).when(endpoint).acceptsNewRequests();
        assertNull(acquirer.tryPreempt(context, endpoint));
        verify(endpoint).resourceSnapshot();
        verify(endpoint).acceptsNewRequests();
        verifyNoInteractions(requests, reporter);
    }

    // ─── Scheduler mode ─────────────────────────────────────────────────
    @Test
    @DisplayName("FIFO ordering (no preemption policy) never evicts")
    void fifoOrderingNeverEvicts() {
        RequestContext ctx = ctx(50);
        configureAdmission(SchedulingTestConfig.newConfig());
        assertNull(tryPreempt(ctx, new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("DIRECT scheduler mode declines without side effects")
    void directSchedulerDeclines() {
        RequestContext ctx = ctx(50);
        FlexlbConfig directConfig = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(directConfig);
        directConfig.priorityOrdering().setPreemption(new PreemptionConfig());
        directConfig.setScheduler(ConfigTestSupport.direct());
        configureAdmission(directConfig);
        assertNull(tryPreempt(ctx, new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    private void configureAdmission(FlexlbConfig config) {
        acquirer =
                org.mockito.Mockito.spy(
                        new DefaultDecodePreemption(
                                org.flexlb.balance.schedule.SchedulerTestSupport.repository(
                                        requests),
                                reporter,
                                SchedulingTestConfig.configService(config),
                                SchedulingTestConfig.capacity(config),
                                new PriorityDecodePreemptionPolicy()));
    }

    private void assertZeroSideEffect() {
        verifyNoInteractions(requests);
        verifyNoInteractions(blockedEndpoint);
        verifyNoInteractions(reporter);
    }

    private CompletableFuture<RequestResult> tryPreempt(
            RequestContext context, CompletableFuture<Response> future) {
        when(context.getFuture()).thenReturn(future);
        return acquirer.tryPreempt(context, blockedEndpoint);
    }

    // ─── Helpers ────────────────────────────────────────────────────────
    private static RequestContext ctx(int priority) {
        RequestContext ctx = mock(RequestContext.class, org.mockito.Mockito.RETURNS_DEEP_STUBS);
        when(ctx.getRequest()).thenReturn(Request.builder().build());
        when(ctx.getPriority()).thenReturn(priority);
        when(ctx.getSchedulingMetadata())
                .thenReturn(
                        org.flexlb.balance.schedule.SchedulingMetadata.explicit(
                                priority, Long.MAX_VALUE));
        when(ctx.getRequestExpiresAtMs()).thenReturn(Long.MAX_VALUE);
        return ctx;
    }
}

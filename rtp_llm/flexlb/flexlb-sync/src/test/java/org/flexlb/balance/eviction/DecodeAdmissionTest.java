package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter.CancelEvent;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;

import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

/**
 * Admission and preemption contracts for {@link DecodeAdmission#tryReclaim}.
 *
 * <p>Requirement: every early decline is side-effect free (returns null
 * without reserving a permit, touching any port, or emitting telemetry).
 * Corner cases are derived from the domain requirements, not by echoing
 * if-branches.
 */
@DisplayName("DecodeAdmission.tryReclaim contracts")
class DecodeAdmissionTest {

    private RequestSchedulerReporter reporter;

    private DeliveryDispatcher dispatcher;

    private AbstractRequestScheduler requests;

    private DecodeEndpoint blockedEndpoint;

    private DecodeAdmission acquirer;

    @BeforeEach
    void setUp() {
        reporter = mock(RequestSchedulerReporter.class);
        dispatcher = mock(DeliveryDispatcher.class);
        requests = mock(AbstractRequestScheduler.class);
        blockedEndpoint = mock(DecodeEndpoint.class);
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        config.priorityOrdering().setPreemption(new PreemptionConfig());
        configureAdmission(config);
    }

    private void configureAdmission(FlexlbConfig config) {
        acquirer =
                org.mockito.Mockito.spy(
                        new DecodeAdmission(
                                dispatcher,
                                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                        requests),
                                mock(org.flexlb.balance.scheduler.SchedulerRuntime.class),
                                reporter,
                                SchedulingTestConfig.configService(config),
                                SchedulingTestConfig.capacity(config)));
    }

    @ParameterizedTest
    @CsvSource({
        "1000,false,true",
        "1000,true,true",
        "2750,false,true",
        "2750,true,true",
        "1000,false,false"
    })
    void enginePreemptionUsesFrozenBindingAndRequiresOpenAdmission(
            long completionTimeoutMs, boolean metricsFail, boolean admissionOpen) {
        if (metricsFail) {
            var failure = new IllegalStateException("metrics unavailable");
            doThrow(failure)
                    .when(reporter)
                    .reportEviction(
                            org.mockito.ArgumentMatchers.eq(
                                    RequestSchedulerReporter.EvictionEvent.PLAN),
                            anyInt(),
                            any(),
                            any());
            doThrow(failure)
                    .when(reporter)
                    .reportEngineCancel(
                            org.mockito.ArgumentMatchers.eq(CancelEvent.REQUEST), any(), anyInt());
            doThrow(failure).when(reporter).reportVictim(anyInt(), anyInt(), any(), any());
        }
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        PreemptionConfig preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_ENGINE_OWNED));
        if (completionTimeoutMs != 1_000L) {
            preemption.setTimeoutMs(completionTimeoutMs);
        }
        config.priorityOrdering().setPreemption(preemption);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);

        var endpoint = org.flexlb.balance.scheduler.SchedulerTestSupport.mockDecodeEndpoint();
        var routing = mock(DecodeResources.DecodeRoutingView.class);
        var view = mock(DecodeResources.ResourceSnapshot.class);
        var victim =
                new DecodeResources.DecodeRequestView(
                        901L,
                        30,
                        128L,
                        128L,
                        DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        true,
                        11L,
                        false);
        when(endpoint.ipPort()).thenReturn("127.0.0.1:8080");
        when(endpoint.resourceSnapshot()).thenReturn(view);
        when(view.routing()).thenReturn(routing);
        when(routing.address()).thenReturn("127.0.0.1:8080");
        when(view.requests()).thenReturn(Map.of(victim.requestId(), victim));
        when(routing.placementUsage())
                .thenReturn(new DecodeResources.CapacityUsage(1L, 20_000L, 10_000L, 0L, 10_000L));

        var completion = new CompletableFuture<DecodeAdmission.PreemptionResult>();
        var incoming = Request.builder().requestId(902L).seqLen(128L).rawPriority(70).build();
        var context = new RequestContext(config);
        context.data().setRequest(incoming);
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(70, System.currentTimeMillis() + 60_000L));
        var future =
                new org.flexlb.balance.scheduler.RequestFuture((kind, response, failure) -> false);
        org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                context.data(), "register", requests, future);
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindOwner(context, requests);
        var frozenRequest = context.data().getRequest();
        configureAdmission(config);
        org.mockito.Mockito.doReturn(completion).when(acquirer).preempt(any(), any());
        when(requests.isAdmissionOpen(902L, future)).thenReturn(admissionOpen);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(99L);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxKvUsagePercent(1L);
        var changed = incoming.toBuilder().seqLen(4_096L).maxNewTokens(1_024).build();
        assertEquals(4_096L, changed.getSeqLen());

        var outcome = acquirer.tryReclaim(context, endpoint);
        verify(requests).isAdmissionOpen(902L, future);
        verifyNoMoreInteractions(requests);
        if (!admissionOpen) {
            assertNull(outcome);
            assertFalse(
                    future.isDone(),
                    "accepted cancellation or close can precede future completion");
            verify(acquirer, never()).preempt(any(), any());
            return;
        }
        assertNotNull(outcome);

        var command = ArgumentCaptor.forClass(DecodeAdmission.PreemptionCommand.class);
        verify(acquirer).preempt(org.mockito.ArgumentMatchers.same(context), command.capture());
        assertEquals(completionTimeoutMs, command.getValue().preemptionTimeoutMs());
        assertEquals(50L, command.getValue().cancelAckTimeoutMs());
        assertSame(endpoint, command.getValue().endpoint());
        assertSame(frozenRequest, command.getValue().request());
        assertEquals(List.of(victim), command.getValue().victims());

        var exact = new DecodeResources.ReservationHandle(7L, 902L, 31L);
        completion.complete(new DecodeAdmission.PreemptionResult(exact, false, "committed"));
        assertSame(exact, outcome.join().reservation());
        assertNull(
                context.data().getResponse(),
                "reservation preparation does not publish a route response");
        verify(endpoint, org.mockito.Mockito.times(1)).resourceSnapshot();
        if (metricsFail) {
            verify(reporter)
                    .reportEviction(
                            org.mockito.ArgumentMatchers.eq(
                                    RequestSchedulerReporter.EvictionEvent.PLAN),
                            anyInt(),
                            any(),
                            any());
            verify(reporter)
                    .reportEngineCancel(
                            org.mockito.ArgumentMatchers.eq(CancelEvent.REQUEST), any(), anyInt());
            verify(reporter).reportVictim(anyInt(), anyInt(), any(), any());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void capacityDecisionReportsOnlyInfeasibleEviction(boolean capacityBlocked) {
        var context = ctx(70);
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        var preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_ENGINE_OWNED));
        config.priorityOrdering().setPreemption(preemption);
        config.getRouter()
                .getRoles()
                .getDecode()
                .getAvailability()
                .setMaxEngineRequests(capacityBlocked ? 1L : 4L);
        configureAdmission(config);
        when(context.data().getFuture()).thenReturn(new CompletableFuture<>());
        var endpoint = org.flexlb.balance.scheduler.SchedulerTestSupport.mockDecodeEndpoint();
        var snapshot = mock(DecodeResources.ResourceSnapshot.class);
        var routing = mock(DecodeResources.DecodeRoutingView.class);
        when(endpoint.resourceSnapshot()).thenReturn(snapshot);
        when(snapshot.routing()).thenReturn(routing);
        when(routing.address()).thenReturn("decode-a");
        when(routing.placementUsage())
                .thenReturn(new DecodeResources.CapacityUsage(1L, 1_000L, 1_000L, 0L, 0L));
        when(snapshot.requests())
                .thenReturn(
                        Map.of(
                                901L,
                                new DecodeResources.DecodeRequestView(
                                        901L,
                                        70,
                                        128L,
                                        128L,
                                        DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                                        true,
                                        11L,
                                        false)));

        assertNull(acquirer.tryReclaim(context, endpoint));
        verify(endpoint, org.mockito.Mockito.times(1)).resourceSnapshot();
        verifyNoInteractions(dispatcher, requests);
        verify(acquirer, never()).preempt(any(), any());
        if (capacityBlocked) {
            verify(reporter)
                    .reportEviction(
                            RequestSchedulerReporter.EvictionEvent.PLAN,
                            70,
                            DecodeEvictionProposal.CASE_SLOT,
                            "infeasible");
            verifyNoMoreInteractions(reporter);
        } else {
            verifyNoInteractions(reporter);
        }
    }

    private void assertZeroSideEffect() {
        verifyNoInteractions(dispatcher);
        verify(acquirer, never()).preempt(any(), any());
        verifyNoInteractions(requests);
        verifyNoInteractions(blockedEndpoint);
        verifyNoInteractions(reporter);
    }

    private CompletableFuture<DecodeAdmission.PreemptionResult> tryReclaim(
            RequestContext context, CompletableFuture<Response> future) {
        when(context.data().getFuture()).thenReturn(future);
        return acquirer.tryReclaim(context, blockedEndpoint);
    }

    // ─── Shutdown ────────────────────────────────────────────────────────
    @Test
    @DisplayName("A shut-down acquirer declines without side effects")
    void shutdownDeclines() {
        acquirer.shutdown();
        assertNull(tryReclaim(ctx(70), new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    // ─── Future states ──────────────────────────────────────────────────
    @Test
    @DisplayName("An already-completed future declines without side effects")
    void completedFutureDeclines() {
        CompletableFuture<Response> done = new CompletableFuture<>();
        done.complete(null);
        assertNull(tryReclaim(ctx(70), done));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("An exceptionally-completed future declines without side effects")
    void exceptionalFutureDeclines() {
        CompletableFuture<Response> failed = new CompletableFuture<>();
        failed.completeExceptionally(new RuntimeException("test"));
        assertNull(tryReclaim(ctx(70), failed));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("A cancelled future declines without side effects")
    void cancelledFutureDeclines() {
        CompletableFuture<Response> cancelled = new CompletableFuture<>();
        cancelled.cancel(false);
        assertNull(tryReclaim(ctx(70), cancelled));
        assertZeroSideEffect();
    }

    // ─── Expiration ─────────────────────────────────────────────────────
    @Test
    @DisplayName("An expired request declines without side effects")
    void expiredRequestDeclines() {
        RequestContext expired = ctx(70);
        when(expired.data().getRequestExpiresAtMs()).thenReturn(1L);
        assertNull(tryReclaim(expired, new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    // Priority boundaries
    @Test
    @DisplayName("Priority 0 (NO_PRIORITY sentinel) declines without side effects")
    void noPriorityDeclines() {
        assertNull(tryReclaim(ctx(0), new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    @Test
    @DisplayName("Priority 1 reaches endpoint selection when preemption is enabled")
    void minimumValidPriorityPassesGuard() {
        var context = ctx(1);
        when(context.data().getFuture()).thenReturn(new CompletableFuture<>());
        var endpoint = org.flexlb.balance.scheduler.SchedulerTestSupport.mockDecodeEndpoint();
        when(endpoint.isRetired()).thenReturn(true);
        assertNull(acquirer.tryReclaim(context, endpoint));
        verify(endpoint).resourceSnapshot();
        verify(endpoint).isRetired();
        verifyNoInteractions(dispatcher, requests, reporter);
    }

    // ─── Scheduler mode ─────────────────────────────────────────────────
    @Test
    @DisplayName("FIFO ordering (no preemption policy) never evicts")
    void fifoOrderingNeverEvicts() {
        RequestContext ctx = ctx(50);
        configureAdmission(SchedulingTestConfig.newConfig());
        assertNull(tryReclaim(ctx, new CompletableFuture<>()));
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
        assertNull(tryReclaim(ctx, new CompletableFuture<>()));
        assertZeroSideEffect();
    }

    // ─── Helpers ────────────────────────────────────────────────────────
    private static RequestContext ctx(int priority) {
        RequestContext ctx = mock(RequestContext.class, org.mockito.Mockito.RETURNS_DEEP_STUBS);
        when(ctx.data().getRequest()).thenReturn(Request.builder().build());
        when(ctx.data().getPriority()).thenReturn(priority);
        when(ctx.data().getSchedulingMetadata())
                .thenReturn(org.flexlb.dao.SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        when(ctx.data().getRequestExpiresAtMs()).thenReturn(Long.MAX_VALUE);
        return ctx;
    }
}

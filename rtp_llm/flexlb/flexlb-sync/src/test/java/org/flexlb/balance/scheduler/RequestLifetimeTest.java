package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RequestLifetimeTest {

    @Test
    void deadlineUsesTheActualDeliveryClockAndPreservesZeroAndSaturation() {
        WorkSnapshot preceding = emptyWork(1_000L);
        assertEquals(13_100L, visibilityDeadline(preceding, 1_000L, 2, 1_100L).orElseThrow());
        assertEquals(11_000L, visibilityDeadline(preceding, 0L, 2, 1_000L).orElseThrow());
        assertEquals(11_002L, visibilityDeadline(emptyWork(900L), 1L, 2, 1_000L).orElseThrow());
        assertEquals(Long.MAX_VALUE, visibilityDeadline(preceding, Long.MAX_VALUE, 2, 1_000L).orElseThrow());
        assertThrows(IllegalArgumentException.class, () -> visibilityDeadline(preceding, 1L, Double.NaN, 1_000L));
        assertThrows(IllegalArgumentException.class, () -> visibilityDeadline(preceding, -1L, 2, 1_000L));
    }

    @Test
    void slowPreparationCannotConsumeTheRequestsOwnWork() {
        assertEquals(56_000L, visibilityDeadline(emptyWork(1_000L), 20_000L, 1.5, 16_000L).orElseThrow());
        assertEquals(100_000L, visibilityDeadline(emptyWork(1_000L), 20_000L, 1.5, 60_000L).orElseThrow());
        WorkSnapshot preceding = new WorkSnapshot(1_000L, List.of(
                new WorkSnapshot.RequestWork(1L, WorkSnapshot.Phase.ENGINE_RUNNING, 20_000L),
                new WorkSnapshot.RequestWork(2L, WorkSnapshot.Phase.ENGINE_QUEUED, 3_000L)), List.of(), 0L);
        assertEquals(68_000L, visibilityDeadline(preceding, 20_000L, 1.5, 16_000L).orElseThrow());
    }

    @Test
    void onlyRunningPredecessorsAgeBeforeDelivery() {
        WorkSnapshot preceding = new WorkSnapshot(1_000L, List.of(
                new WorkSnapshot.RequestWork(1L, WorkSnapshot.Phase.ENGINE_RUNNING, 20_000L),
                new WorkSnapshot.RequestWork(2L, WorkSnapshot.Phase.ENGINE_QUEUED, 7_000L),
                new WorkSnapshot.RequestWork(3L, WorkSnapshot.Phase.COMMITTED, 3_000L)), List.of(
                new WorkSnapshot.BatchWork(List.of(4L), WorkSnapshot.Phase.ENGINE_RUNNING, OptionalLong.of(8_000L)),
                new WorkSnapshot.BatchWork(List.of(5L), WorkSnapshot.Phase.ENGINE_QUEUED, OptionalLong.of(4_000L))), 0L);
        assertEquals(63_500L, visibilityDeadline(preceding, 11_000L, 1, 500L).orElseThrow());
        assertEquals(56_000L, visibilityDeadline(preceding, 11_000L, 1, 11_000L).orElseThrow());
        assertEquals(66_000L, visibilityDeadline(preceding, 11_000L, 1, 31_000L).orElseThrow());
    }

    @Test
    void unknownPredecessorsNeverBecomeACompleteEstimate() {
        WorkSnapshot unknownBatch = new WorkSnapshot(1_000L, List.of(), List.of(
                new WorkSnapshot.BatchWork(List.of(1L), WorkSnapshot.Phase.COMMITTED,
                        OptionalLong.empty())), 0L);
        for (WorkSnapshot preceding : List.of(unknownWork(1_000L), unknownBatch)) {
            assertTrue(visibilityDeadline(preceding, 20_000L, 2, 60_000L).isEmpty());
        }
    }

    @Test
    void combinedWorkSaturatesWithoutWrapping() {
        WorkSnapshot preceding = new WorkSnapshot(1_000L, List.of(
                new WorkSnapshot.RequestWork(1L, WorkSnapshot.Phase.COMMITTED, Long.MAX_VALUE)), List.of(), 0L);
        assertEquals(Long.MAX_VALUE, visibilityDeadline(preceding, 1L, 2, 60_000L).orElseThrow());
    }

    @Test
    void latePrefillCompletionStartsOneFreshHandoffWindow() {
        Fixture fixture = fixture(true);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.inspect(fixture.scheduler, fixture.requestContext, "applyDeliveryPredictionLocked", emptyWork(1_000L), 100L, 1_000L);
            assertEquals(11_200L, fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
            recordPrefillStatusAt(fixture, false, 1_100L);
            assertTrue(fixture.requestContext.decisionDeadlineAtMs().isEmpty());
            recordPrefillStatusAt(fixture, true, 5_000L);
            assertEquals(15_000L, fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
            recordPrefillStatusAt(fixture, true, 5_020L);
            assertEquals(15_000L, fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
            var timer = mock(ExpirationTimer.DecisionDeadline.class);
            when(timer.deadlineAtMs()).thenReturn(15_000L);
            assertTrue(fixture.requestContext.installDecisionDeadline(timer));
            fixture.requestContext.onDecisionVisibilityDeadline(timer);
            assertTrue(fixture.requestContext.snapshot().detail().startsWith("SUSPECTED_LOST"));
            fixture.requestContext.markDecodeAcceptedLocked();
            assertTrue(fixture.requestContext.decisionDeadlineAtMs().isEmpty());
            assertFalse(RequestProtocolTestSupport.<Boolean>inspect(fixture.scheduler, fixture.requestContext, "needsDecisionConfirmationLocked"));
        }
    }

    @Test
    void completionAndAcceptanceBeforeDeliveryRemainAuthoritative() {
        Fixture fixture = fixture(true);
        synchronized (fixture.requestContext) {
            recordPrefillStatusAt(fixture, true, 1_000L);
            RequestProtocolTestSupport.inspect(fixture.scheduler, fixture.requestContext, "applyDeliveryPredictionLocked", emptyWork(1_000L), 100L, 1_010L);
            assertEquals(11_000L, fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
            fixture.requestContext.markDecodeAcceptedLocked();
            recordPrefillStatusAt(fixture, false, 2_000L);
            recordPrefillStatusAt(fixture, true, 2_000L);
            assertTrue(fixture.requestContext.decodeAccepted());
            assertTrue(fixture.requestContext.decisionDeadlineAtMs().isEmpty());
            assertThrows(IllegalStateException.class, () -> RequestProtocolTestSupport.inspect(fixture.scheduler, fixture.requestContext, "applyDeliveryPredictionLocked", emptyWork(2_000L), 100L, 2_000L));
        }
    }

    @Test
    void unknownPredictionCanStillStartHandoffDetectionAfterPrefillCompletes() {
        Fixture fixture = fixture(true);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.inspect(fixture.scheduler, fixture.requestContext, "applyDeliveryPredictionLocked", unknownWork(1_000L), 100L, 1_000L);
            assertTrue(fixture.requestContext.decisionDeadlineAtMs().isEmpty());
            assertFalse(RequestProtocolTestSupport.<Boolean>inspect(fixture.scheduler, fixture.requestContext, "needsDecisionConfirmationLocked"));
            recordPrefillStatusAt(fixture, true, 2_000L);
            assertEquals(12_000L, fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
        }
    }

    @Test
    void longRunningPrefillStillReceivesTheFullHandoffWindow() {
        Fixture fixture = fixture(true);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.inspect(fixture.scheduler, fixture.requestContext, "applyDeliveryPredictionLocked", emptyWork(1_000L), 100L, 1_000L);
            recordPrefillStatusAt(fixture, false, 1_100L);
            assertTrue(fixture.requestContext.decisionDeadlineAtMs().isEmpty());
            recordPrefillStatusAt(fixture, true, 3_601_000L);
            assertEquals(3_611_000L, fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
        }
    }

    @Test
    void inactivityWatchSurvivesRouteResponseAndDecodeAcceptance() {
        Fixture fixture = fixture(true);
        var exact = mock(ExpirationTimer.InactivityDeadline.class);
        synchronized (fixture.requestContext) {
            assertTrue(fixture.requestContext.installInactivityDeadline(exact));
            RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
            RequestProtocolTestSupport.markAcknowledged(fixture.requestContext);
            fixture.requestContext.markDecodeAcceptedLocked();
            assertTrue(fixture.requestContext.future().completeOwned(new Response()));
            assertTrue(fixture.requestContext.future().isDone());
            assertTrue(org.springframework.test.util.ReflectionTestUtils.<Boolean>invokeMethod(fixture.requestContext, "consumeInactivityDeadlineLocked", exact));
            assertFalse(org.springframework.test.util.ReflectionTestUtils.<Boolean>invokeMethod(fixture.requestContext, "consumeInactivityDeadlineLocked", exact));
        }
    }

    @Test
    void missingEngineEvidenceMarksSuspicionWithoutReleasingRequestOwnership() {
        Fixture fixture = fixture(true);
        var exact = mock(ExpirationTimer.DecisionDeadline.class);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
            startPrediction(fixture.scheduler, fixture.requestContext);
            when(exact.deadlineAtMs()).thenReturn(fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
            assertTrue(fixture.requestContext.installDecisionDeadline(exact));
            fixture.requestContext.onDecisionVisibilityDeadline(exact);
            assertSame(fixture.item, fixture.requestContext.activeRoute());
            assertTrue(RequestProtocolTestSupport.<Boolean>field(fixture.requestContext, "decisionExpired"));
            assertTrue(RequestProtocolTestSupport.<java.util.OptionalLong>field(fixture.requestContext, "decisionExpiresAtMs").isEmpty());
            assertNull(RequestProtocolTestSupport.field(fixture.requestContext, "decisionDeadline"));
            assertTrue(fixture.requestContext.isLiveGeneration());
            assertFalse(fixture.requestContext.snapshot().state().isTerminal());
            assertTrue(fixture.requestContext.snapshot().detail().contains("SUSPECTED_LOST"));
            assertStaleDecisionHasNoEffect(fixture, exact);
        }
    }

    @Test
    void runningPrefillMayExceedPredictionAndOnlyCompletedHandoffCanBecomeUnresolved() {
        Fixture fixture = fixture(true);
        var exact = mock(ExpirationTimer.DecisionDeadline.class);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
            startPrediction(fixture.scheduler, fixture.requestContext);
            when(exact.deadlineAtMs()).thenReturn(fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
            assertTrue(fixture.requestContext.installDecisionDeadline(exact));
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, false);
            assertStaleDecisionHasNoEffect(fixture, exact);
            assertFalse(RequestProtocolTestSupport.<Boolean>inspect(fixture.scheduler, fixture.requestContext, "needsDecisionConfirmationLocked"));
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, true);
            assertTrue(fixture.requestContext.decisionDeadlineAtMs().orElseThrow() > System.currentTimeMillis());
            fixture.requestContext.markDecodeAcceptedLocked();
            assertFalse(RequestProtocolTestSupport.<Boolean>inspect(fixture.scheduler, fixture.requestContext, "needsDecisionConfirmationLocked"));
        }
    }

    @Test
    void pdfusionEvidenceEndsMissingRequestDetectionButKeepsInactivityWatch() {
        Fixture fixture = fixture(false);
        var full = mock(ExpirationTimer.InactivityDeadline.class);
        var decision = mock(ExpirationTimer.DecisionDeadline.class);
        synchronized (fixture.requestContext) {
            fixture.requestContext.installInactivityDeadline(full);
            RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
            startPrediction(fixture.scheduler, fixture.requestContext);
            when(decision.deadlineAtMs()).thenReturn(fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
            fixture.requestContext.installDecisionDeadline(decision);
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, false);
            assertStaleDecisionHasNoEffect(fixture, decision);
            assertTrue(org.springframework.test.util.ReflectionTestUtils.<Boolean>invokeMethod(fixture.requestContext, "consumeInactivityDeadlineLocked", full));
        }
    }

    @Test
    void committedInactivityCancellationWinsOverLateHandoffEvidence() {
        Fixture fixture = fixture(true);
        var decision = mock(ExpirationTimer.DecisionDeadline.class);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
            startPrediction(fixture.scheduler, fixture.requestContext);
            RequestProtocolTestSupport.recordCancellation(fixture.scheduler, fixture.requestContext, CancelReason.DEADLINE_EXCEEDED, "request inactive");
            when(decision.deadlineAtMs()).thenReturn(fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
            fixture.requestContext.installDecisionDeadline(decision);
            fixture.requestContext.onDecisionVisibilityDeadline(decision);
            assertFalse(fixture.requestContext.snapshot().detail().startsWith("SUSPECTED_LOST"));
            assertEquals(CancelReason.DEADLINE_EXCEEDED, fixture.requestContext.cancellationReason());
            assertFalse(RequestProtocolTestSupport.<Boolean>inspect(fixture.scheduler, fixture.requestContext, "needsDecisionConfirmationLocked"));
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, true);
            assertFalse(RequestProtocolTestSupport.<Boolean>inspect(fixture.scheduler, fixture.requestContext, "needsDecisionConfirmationLocked"));
        }
    }

    @Test
    void latePrefillEvidenceInvalidatesTheExpiredDecision() {
        Fixture fixture = fixture(true);
        synchronized (fixture.requestContext) {
            expireDecision(fixture);
            assertTrue(fixture.requestContext.snapshot().detail().startsWith("SUSPECTED_LOST"));
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, false);
            fixture.requestContext.reconcileDecisionEvidenceLocked();
            assertFalse(RequestProtocolTestSupport.<Boolean>inspect(fixture.scheduler, fixture.requestContext, "needsDecisionConfirmationLocked"));
            assertFalse(fixture.requestContext.snapshot().detail().contains("SUSPECTED_LOST"));
            assertFalse(fixture.requestContext.decodeAccepted());
        }
    }

    @Test
    void delayedUncertaintyCannotReplaceMatchingEngineEvidence() {
        Fixture fixture = fixture(true);
        synchronized (fixture.requestContext) {
            expireDecision(fixture);
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, false);
            fixture.requestContext.markAwaitingConfirmationLocked("late transport uncertainty");
            assertFalse(fixture.requestContext.snapshot().detail().contains("SUSPECTED_LOST"));
            assertFalse((fixture.requestContext.cancellationReason() != null));
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, true);
            assertTrue(fixture.requestContext.decisionDeadlineAtMs().orElseThrow() > System.currentTimeMillis());
        }
    }

    @Test
    void uncertainDeliveryKeepsTheInactivityDeadlineAndDoesNotCancelTheRequest() {
        Fixture fixture = fixture(true);
        var deadline = mock(ExpirationTimer.InactivityDeadline.class);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
            startPrediction(fixture.scheduler, fixture.requestContext);
            assertTrue(fixture.requestContext.installInactivityDeadline(deadline));
            fixture.requestContext.markAwaitingConfirmationLocked("ambiguous transport");
            assertTrue(fixture.requestContext.snapshot().detail().contains("SUSPECTED_LOST"));
            assertFalse((fixture.requestContext.cancellationReason() != null));
            assertEquals(RequestState.Phase.DISPATCHING, fixture.requestContext.snapshot().state());
            assertTrue(org.springframework.test.util.ReflectionTestUtils.<Boolean>invokeMethod(fixture.requestContext, "consumeInactivityDeadlineLocked", deadline));
            assertTrue(fixture.requestContext.inactivityDeadlineAtMs().isPresent());
            assertSame(fixture.item, fixture.requestContext.activeRoute());
        }
    }

    @Test
    void cancellationFirstCauseCannotSuppressTheInactivityWatch() {
        Fixture fixture = fixture(true);
        var deadline = mock(ExpirationTimer.InactivityDeadline.class);
        var renewed = mock(ExpirationTimer.InactivityDeadline.class);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
            assertTrue(fixture.requestContext.installInactivityDeadline(deadline));
            RequestProtocolTestSupport.recordCancellation(fixture.scheduler, fixture.requestContext, CancelReason.CLIENT_CANCELLED, "client cancellation");
            assertTrue(org.springframework.test.util.ReflectionTestUtils.<Boolean>invokeMethod(fixture.requestContext, "consumeInactivityDeadlineLocked", deadline));
            assertTrue(fixture.requestContext.installInactivityDeadline(renewed));
            assertFalse(org.springframework.test.util.ReflectionTestUtils.<Boolean>invokeMethod(fixture.requestContext, "consumeInactivityDeadlineLocked", deadline));
            assertTrue(org.springframework.test.util.ReflectionTestUtils.<Boolean>invokeMethod(fixture.requestContext, "consumeInactivityDeadlineLocked", renewed));
            assertEquals(CancelReason.CLIENT_CANCELLED, RequestProtocolTestSupport.<CancelReason>inspect(fixture.scheduler, fixture.requestContext, "requireCancellationFirstCauseLocked"));
        }
    }

    @ParameterizedTest
    @CsvSource({ "PREFILL,true", "DECODE,true", "PREFILL,false", "DECODE,false" })
    void uncertaintyAllowsActualAckIndependentlyOfMatchingEngineEvidence(RoleType evidenceSource, boolean ackBeforeEvidence) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        try {
            RequestContext context = RequestProtocolTestSupport.context(config, 202L);
            var future = RequestProtocolTestSupport.register(registry, context);
            RequestContext requestContext = registry.findRequestContext(202L);
            PrefillEndpoint prefill = mock(PrefillEndpoint.class);
            DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
            var reservation = new DecodeResources.ReservationHandle(1L, 202L, 1L);
            context.setFuture(future);
            RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), prefillServer(), null, prefill, decode, reservation, System.currentTimeMillis());
            RequestProtocolTestSupport.bind(registry, new RequestProtocolTestSupport.Registered(item, future));
            DeliveryClaim claim = RequestProtocolTestSupport.claimBatchWithoutPrediction(registry, item, 7L, () -> true);
            assertNotNull(claim);
            synchronized (requestContext) {
                startPrediction(registry, requestContext);
                var deadline = mock(ExpirationTimer.DecisionDeadline.class);
                when(deadline.deadlineAtMs()).thenReturn(requestContext.decisionDeadlineAtMs().orElseThrow());
                assertTrue(requestContext.installDecisionDeadline(deadline));
                requestContext.onDecisionVisibilityDeadline(deadline);
                assertTrue(requestContext.snapshot().detail().startsWith("SUSPECTED_LOST"));
                assertFalse((requestContext.cancellationReason() != null));
            }
            if (ackBeforeEvidence) {
                claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.delivered());
            }
            if (ackBeforeEvidence) {
                assertTrue(future.get(1L, TimeUnit.SECONDS).isSuccess(), "uncertainty cannot hold a real EnqueueBatch ACK");
            } else {
                assertFalse(future.isDone());
            }
            AbstractRequestScheduler projector = registry;
            if (evidenceSource == RoleType.PREFILL) {
                projector.onPrefillStatus(requestContext, prefill, RoleType.PREFILL, PrefillState.PrefillRequestStatus.active(item));
            } else {
                projector.onDecodeStatus(requestContext, decode, DecodeResources.DecodeRequestStatus.active(reservation));
            }
            if (!ackBeforeEvidence) {
                assertFalse(future.isDone(), "Engine activity cannot create an EnqueueBatch ACK");
                claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.delivered());
            }
            assertTrue(future.get(1L, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(() -> {
                synchronized (requestContext) {
                    return evidenceSource == RoleType.DECODE ? requestContext.decodeAccepted() : !RequestProtocolTestSupport.<Boolean>inspect(registry, requestContext, "needsDecisionConfirmationLocked");
                }
            });
            synchronized (requestContext) {
                assertEquals(evidenceSource == RoleType.DECODE, requestContext.decodeAccepted());
                assertTrue(requestContext.isLiveGeneration());
                preparePrefillStatusEffects(registry, requestContext, item, true);
                assertFalse(RequestProtocolTestSupport.<Boolean>inspect(registry, requestContext, "needsDecisionConfirmationLocked"));
            }
        } finally {
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
                registry.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
            }
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"install", "reject", "throw"})
    void earlyTimerFiresOnlyAfterSuccessfulInstallation(String outcome) throws Exception {
        AbstractRequestScheduler scheduler = mock(AbstractRequestScheduler.class);
        RequestContext context = org.mockito.Mockito.spy(RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 901L));
        when(SchedulerTestSupport.repository(scheduler).isCurrent(context)).thenReturn(true);
        context.bindScheduler(scheduler);
        var exact = new java.util.concurrent.atomic.AtomicReference<ExpirationTimer.RequestDeadline>();
        var failure = new IllegalStateException("installation failed");
        try (var timer = new ExpirationTimer(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(scheduler))) {
            var executor = (java.util.concurrent.ScheduledThreadPoolExecutor)
                    org.springframework.test.util.ReflectionTestUtils.getField(timer, "executor");
            doAnswer(call -> {
                exact.set(call.getArgument(0));
                // The zero-delay task has run, but it cannot expire an uninstalled capability.
                executor.submit(() -> { }).get(5, TimeUnit.SECONDS);
                verify(scheduler, never()).onSchedulingDeadline(any(), any());
                if ("throw".equals(outcome)) { throw failure; }
                return "install".equals(outcome);
            }).when(context).installRequestDeadline(any());

            if ("throw".equals(outcome)) {
                assertSame(failure, assertThrows(IllegalStateException.class,
                        () -> timer.scheduleRequestDeadline(context, 0L)));
            } else {
                var installed = timer.scheduleRequestDeadline(context, 0L);
                if ("install".equals(outcome)) {
                    assertSame(exact.get(), installed);
                    verify(scheduler).onSchedulingDeadline(context, installed);
                } else {
                    assertNull(installed);
                }
            }
            assertNotNull(exact.get());
            assertFalse(exact.get().publishAfterInstall(), "consumed or canceled registration cannot fire again");
            assertFalse(exact.get().consume());
            assertFalse(exact.get().cancel());
            if (!"install".equals(outcome)) {
                verify(scheduler, never()).onSchedulingDeadline(any(), any());
            }
            assertTrue(executor.getQueue().isEmpty());
        }
    }

    @Test
    void timerChecksInactivityAfterThePublicFutureHasCompleted() throws Exception {
        Fixture fixture = fixture(true);
        AbstractRequestScheduler registry = fixture.scheduler;
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(fixture.config);
        doAnswer(invocation -> {
            synchronized (fixture.requestContext) {
                assertTrue(org.springframework.test.util.ReflectionTestUtils.<Boolean>invokeMethod(fixture.requestContext, "consumeInactivityDeadlineLocked", invocation.getArgument(1, ExpirationTimer.InactivityDeadline.class)));
                assertTrue(RequestProtocolTestSupport.<Boolean>inspect(fixture.scheduler, fixture.requestContext, "requestInactiveLocked", invocation.getArgument(2, Long.class)));
                finishInactivity(fixture.scheduler, fixture.requestContext);
            }
            invocation.getArgument(3, Runnable.class).run();
            return null;
        }).when(registry).enqueueInactivityDeadline(any(), any(), anyLong(), any());
        try (var timer = new ExpirationTimer(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry))) {
            synchronized (fixture.requestContext) {
                RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
                RequestProtocolTestSupport.markAcknowledged(fixture.requestContext);
                fixture.requestContext.future().completeOwned(new Response());
                // This test expires an already delivered request; shortening
                // the timeout before claim would instead reject the handoff.
                org.springframework.test.util.ReflectionTestUtils.setField(fixture.requestContext, "inactivityTimeoutMs", 20L);
            }
            assertNotNull(timer.scheduleInactivityDeadline(fixture.requestContext));
            verify(registry, timeout(1000L).times(1)).enqueueInactivityDeadline(any(), any(), anyLong(), any());
            verify(registry, never()).cancel(anyLong(), anyLong(), any());
        }
    }

    @Test
    void timerCloseWaitsForAnAlreadyStartedInactivityHandoff() throws Exception {
        Fixture fixture = fixture(true);
        AbstractRequestScheduler registry = fixture.scheduler;
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(fixture.config);
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        doAnswer(invocation -> {
            entered.countDown();
            boolean interrupted = false;
            while (true) {
                try {
                    release.await();
                    break;
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
            return null;
        }).when(registry).enqueueInactivityDeadline(any(), any(), anyLong(), any());
        ExpirationTimer timer = new ExpirationTimer(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry));
        Thread closer = null;
        try {
            org.springframework.test.util.ReflectionTestUtils.setField(fixture.requestContext, "inactivityTimeoutMs", 20L);
            assertNotNull(timer.scheduleInactivityDeadline(fixture.requestContext));
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            closer = new Thread(timer::close);
            closer.start();
            assertTrue(closer.isAlive(), "close must wait for the running timer producer");
        } finally {
            release.countDown();
            timer.close();
            if (closer != null) {
                closer.join(5_000);
            }
        }
        assertFalse(closer.isAlive());
    }

    @Test
    void renewedActivityRearmsTheTimerUntilTheNewInactivityDeadline() throws Exception {
        Fixture fixture = fixture(true);
        AbstractRequestScheduler registry = fixture.scheduler;
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(fixture.config);
        long start = fixture.requestContext.createdAtMs();
        AtomicLong now = new AtomicLong(start + 100L);
        AtomicInteger checks = new AtomicInteger();
        CountDownLatch expired = new CountDownLatch(1);
        doAnswer(invocation -> {
            synchronized (fixture.requestContext) {
                assertTrue(org.springframework.test.util.ReflectionTestUtils.<Boolean>invokeMethod(fixture.requestContext, "consumeInactivityDeadlineLocked", invocation.getArgument(1, ExpirationTimer.InactivityDeadline.class)));
                if (checks.incrementAndGet() == 1) {
                    // A matching status wins after the old timer fired but before cancellation.
                    fixture.requestContext.scheduler().acceptPrefillStatus(fixture.requestContext, fixture.item.prefillEp(), RoleType.PREFILL, PrefillState.PrefillRequestStatus.active(fixture.item), start + 50L);
                    assertFalse(RequestProtocolTestSupport.<Boolean>inspect(fixture.scheduler, fixture.requestContext, "requestInactiveLocked", invocation.getArgument(2, Long.class)));
                    now.set(start + 150L);
                } else {
                    assertTrue(RequestProtocolTestSupport.<Boolean>inspect(fixture.scheduler, fixture.requestContext, "requestInactiveLocked", invocation.getArgument(2, Long.class)));
                    finishInactivity(fixture.scheduler, fixture.requestContext);
                    expired.countDown();
                }
            }
            invocation.getArgument(3, Runnable.class).run();
            return null;
        }).when(registry).enqueueInactivityDeadline(any(), any(), anyLong(), any());
        try (var timer = new ExpirationTimer(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry), now::get)) {
            synchronized (fixture.requestContext) {
                org.springframework.test.util.ReflectionTestUtils.setField(fixture.requestContext, "inactivityTimeoutMs", 100L);
                RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
                RequestProtocolTestSupport.markAcknowledged(fixture.requestContext);
            }
            assertNotNull(timer.scheduleInactivityDeadline(fixture.requestContext));
            assertTrue(expired.await(1L, TimeUnit.SECONDS));
            assertEquals(2, checks.get(), "the renewed request must retain a timer for later silence");
        }
    }

    @Test
    void oldVisibilityTimerCannotInstallAfterPrefillCompletion() {
        Fixture fixture = fixture(true);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
            startPrediction(fixture.scheduler, fixture.requestContext);
            long oldDeadline = fixture.requestContext.decisionDeadlineAtMs().orElseThrow();
            var oldTimer = mock(ExpirationTimer.DecisionDeadline.class);
            when(oldTimer.deadlineAtMs()).thenReturn(oldDeadline);
            fixture.requestContext.scheduler().acceptPrefillStatus(fixture.requestContext, fixture.item.prefillEp(), RoleType.PREFILL, PrefillState.PrefillRequestStatus.terminal(fixture.item, PrefillState.PrefillRequestStatus.Kind.COMPLETED, 0L), oldDeadline + 100L);
            assertFalse(fixture.requestContext.installDecisionDeadline(oldTimer));
            var handoffTimer = mock(ExpirationTimer.DecisionDeadline.class);
            when(handoffTimer.deadlineAtMs()).thenReturn(fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
            assertTrue(fixture.requestContext.installDecisionDeadline(handoffTimer));
            assertStaleDecisionHasNoEffect(fixture, oldTimer);
            fixture.requestContext.onDecisionVisibilityDeadline(handoffTimer);
            assertTrue(fixture.requestContext.snapshot().detail().startsWith("SUSPECTED_LOST"));
        }
    }

    @Test
    void decodeAcceptanceDetachesDecisionTimerButKeepsTheCancellationAndInactivityWatch() {
        Fixture fixture = fixture(true);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
            startPrediction(fixture.scheduler, fixture.requestContext);
            var timer = mock(ExpirationTimer.DecisionDeadline.class);
            when(timer.deadlineAtMs()).thenReturn(fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
            assertTrue(fixture.requestContext.installDecisionDeadline(timer));
            RequestProtocolTestSupport.recordCancellation(fixture.scheduler, fixture.requestContext, CancelReason.CLIENT_CANCELLED, "client cancellation");
            var acceptance = fixture.requestContext.markDecodeAcceptedLocked();
            assertSame(timer, acceptance);
            assertStaleDecisionHasNoEffect(fixture, timer);
            assertEquals(CancelReason.CLIENT_CANCELLED, RequestProtocolTestSupport.<CancelReason>inspect(fixture.scheduler, fixture.requestContext, "requireCancellationFirstCauseLocked"));
            assertTrue(fixture.requestContext.inactivityDeadlineAtMs().isPresent());
        }
    }

    private static void finishInactivity(AbstractRequestScheduler scheduler, RequestContext requestContext) {
        TerminalAction terminal = RequestProtocolTestSupport.claimTerminal(scheduler, requestContext, TerminalOutcome.timeout("request inactive"), null, false);
        assertNotNull(terminal);
        scheduler.commitTerminalRecord(requestContext, terminal);
        assertEquals(RequestContext.RequestStage.FINISHED, requestContext.stage());
    }

    private static WorkSnapshot emptyWork(long capturedAtMs) {
        return new WorkSnapshot(capturedAtMs, List.of(), List.of(), 0L);
    }

    private static WorkSnapshot unknownWork(long capturedAtMs) {
        return new WorkSnapshot(capturedAtMs, List.of(), List.of(), 1L);
    }

    private static OptionalLong visibilityDeadline(WorkSnapshot preceding, long unstartedWorkMs, double lifetime, long deliveredAtMs) {
        Fixture fixture = fixture(true, lifetime);
        synchronized (fixture.requestContext) {
            RequestProtocolTestSupport.inspect(fixture.scheduler, fixture.requestContext, "applyDeliveryPredictionLocked", preceding, unstartedWorkMs, deliveredAtMs);
            return fixture.requestContext.decisionDeadlineAtMs();
        }
    }

    private static void recordPrefillStatusAt(Fixture fixture, boolean completed, long nowMs) {
        fixture.requestContext.scheduler().acceptPrefillStatus(fixture.requestContext, fixture.item.prefillEp(), RoleType.PREFILL, completed ? PrefillState.PrefillRequestStatus.terminal(fixture.item, PrefillState.PrefillRequestStatus.Kind.COMPLETED, 0L) : PrefillState.PrefillRequestStatus.active(fixture.item), nowMs);
    }

    private static Fixture fixture(boolean separateDecode) {
        return fixture(separateDecode, 2.0);
    }

    private static Fixture fixture(boolean separateDecode, double lifetime) {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        config.getRequestLifecycle().getDecision().setLifetime(lifetime);
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        RequestContext context = RequestProtocolTestSupport.context(config, 101L);
        AbstractRequestScheduler requestOwner = RequestProtocolTestSupport.initialize(mock(ResponseCompletionExecutor.class), context, mock(ExpirationTimer.class));
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = separateDecode ? RequestProtocolTestSupport.decodeEndpoint() : null;
        DecodeResources.ReservationHandle reservation = separateDecode ? new DecodeResources.ReservationHandle(1L, 101L, 1L) : null;
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), prefillServer(), null, prefill, decode, reservation, System.currentTimeMillis());
        AdmissionHandle mutation;
        synchronized (context) {
            mutation = RequestProtocolTestSupport.beginAdmission(requestOwner, context);
            assertNotNull(mutation);
        }
        assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS, requestOwner.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
        mutation.finish();
        return new Fixture(requestOwner, config, context, item);
    }

    private static Runnable preparePrefillStatusEffects(AbstractRequestScheduler scheduler, RequestContext requestContext, RequestRoute item, boolean completed) {
        return requestContext.scheduler().acceptPrefillStatus(requestContext, item.prefillEp(), RoleType.PREFILL, completed ? PrefillState.PrefillRequestStatus.terminal(item, PrefillState.PrefillRequestStatus.Kind.COMPLETED, 0L) : PrefillState.PrefillRequestStatus.active(item), System.currentTimeMillis());
    }

    private static void startPrediction(AbstractRequestScheduler requestOwner, RequestContext requestContext) {
        RequestProtocolTestSupport.inspect(requestContext.scheduler(), requestContext, "applyDeliveryPredictionLocked", emptyWork(System.currentTimeMillis()), 100L, System.currentTimeMillis());
    }

    private static void assertStaleDecisionHasNoEffect(Fixture fixture, ExpirationTimer.DecisionDeadline stale) {
        var before = fixture.requestContext.snapshot();
        ExpirationTimer.DecisionDeadline installed = RequestProtocolTestSupport.field(fixture.requestContext, "decisionDeadline");
        var deadline = RequestProtocolTestSupport.<java.util.OptionalLong>field(fixture.requestContext, "decisionExpiresAtMs");
        boolean expired = RequestProtocolTestSupport.<Boolean>field(fixture.requestContext, "decisionExpired");
        fixture.requestContext.onDecisionVisibilityDeadline(stale);
        assertEquals(before, fixture.requestContext.snapshot());
        assertSame(installed, RequestProtocolTestSupport.field(fixture.requestContext, "decisionDeadline"));
        assertEquals(deadline, RequestProtocolTestSupport.<java.util.OptionalLong>field(fixture.requestContext, "decisionExpiresAtMs"));
        assertEquals(expired, RequestProtocolTestSupport.<Boolean>field(fixture.requestContext, "decisionExpired"));
        assertSame(fixture.item, fixture.requestContext.activeRoute());
    }

    private static void expireDecision(Fixture fixture) {
        RequestProtocolTestSupport.startRouteDelivery(fixture.scheduler, fixture.requestContext);
        startPrediction(fixture.scheduler, fixture.requestContext);
        var deadline = mock(ExpirationTimer.DecisionDeadline.class);
        when(deadline.deadlineAtMs()).thenReturn(fixture.requestContext.decisionDeadlineAtMs().orElseThrow());
        assertTrue(fixture.requestContext.installDecisionDeadline(deadline));
        fixture.requestContext.onDecisionVisibilityDeadline(deadline);
    }

    private static ServerStatus prefillServer() {
        ServerStatus server = new ServerStatus();
        server.setRole(RoleType.PREFILL);
        server.setServerIp("127.0.0.1");
        server.setGrpcPort(8081);
        return server;
    }

    private record Fixture(AbstractRequestScheduler scheduler, FlexlbConfig config, RequestContext requestContext, RequestRoute item) {
    }
}

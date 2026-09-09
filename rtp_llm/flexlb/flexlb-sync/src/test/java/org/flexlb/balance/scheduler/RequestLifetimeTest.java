package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
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
                new WorkSnapshot.BatchWork(4L, List.of(4L), WorkSnapshot.Phase.ENGINE_RUNNING, 8_000L),
                new WorkSnapshot.BatchWork(5L, List.of(5L), WorkSnapshot.Phase.ENGINE_QUEUED, 4_000L)), 0L);
        assertEquals(63_500L, visibilityDeadline(preceding, 11_000L, 1, 500L).orElseThrow());
        assertEquals(56_000L, visibilityDeadline(preceding, 11_000L, 1, 11_000L).orElseThrow());
        assertEquals(66_000L, visibilityDeadline(preceding, 11_000L, 1, 31_000L).orElseThrow());
    }

    @Test
    void unknownPredecessorsNeverBecomeACompleteEstimate() {
        WorkSnapshot unknownBatch = new WorkSnapshot(1_000L, List.of(), List.of(
                new WorkSnapshot.BatchWork(1L, List.of(1L), WorkSnapshot.Phase.COMMITTED,
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
        synchronized (fixture.slot) {
            fixture.slot.startDecisionTracking(emptyWork(1_000L), 100L, 1_000L);
            assertEquals(11_200L, fixture.slot.decisionDeadlineAtMs().orElseThrow());
            observePrefillAt(fixture, false, 1_100L);
            assertTrue(fixture.slot.decisionDeadlineAtMs().isEmpty());
            observePrefillAt(fixture, true, 5_000L);
            assertEquals(15_000L, fixture.slot.decisionDeadlineAtMs().orElseThrow());
            observePrefillAt(fixture, true, 5_020L);
            assertEquals(15_000L, fixture.slot.decisionDeadlineAtMs().orElseThrow());
            var timer = mock(ExpirationTimer.DecisionDeadline.class);
            when(timer.deadlineAtMs()).thenReturn(15_000L);
            assertTrue(fixture.slot.installDecisionDeadline(timer));
            assertTrue(fixture.slot.expireDecisionDeadline(timer).needsConfirmation());
            fixture.slot.markDecodeAccepted();
            assertTrue(fixture.slot.decisionDeadlineAtMs().isEmpty());
            assertFalse(fixture.slot.needsDecisionConfirmation());
        }
    }

    @Test
    void completionAndAcceptanceBeforeDeliveryRemainAuthoritative() {
        Fixture fixture = fixture(true);
        synchronized (fixture.slot) {
            observePrefillAt(fixture, true, 1_000L);
            fixture.slot.startDecisionTracking(emptyWork(1_000L), 100L, 1_010L);
            assertEquals(11_000L, fixture.slot.decisionDeadlineAtMs().orElseThrow());
            fixture.slot.markDecodeAccepted();
            observePrefillAt(fixture, false, 2_000L);
            observePrefillAt(fixture, true, 2_000L);
            assertTrue(fixture.slot.decodeOwnsRequest());
            assertTrue(fixture.slot.decisionDeadlineAtMs().isEmpty());
            assertThrows(IllegalStateException.class,
                    () -> fixture.slot.startDecisionTracking(emptyWork(2_000L), 100L, 2_000L));
        }
    }

    @Test
    void unknownPredictionCanStillStartHandoffDetectionAfterPrefillCompletes() {
        Fixture fixture = fixture(true);
        synchronized (fixture.slot) {
            fixture.slot.startDecisionTracking(unknownWork(1_000L), 100L, 1_000L);
            assertTrue(fixture.slot.decisionDeadlineAtMs().isEmpty());
            assertFalse(fixture.slot.needsDecisionConfirmation());
            observePrefillAt(fixture, true, 2_000L);
            assertEquals(12_000L, fixture.slot.decisionDeadlineAtMs().orElseThrow());
        }
    }

    @Test
    void longRunningPrefillStillReceivesTheFullHandoffWindow() {
        Fixture fixture = fixture(true);
        synchronized (fixture.slot) {
            fixture.slot.startDecisionTracking(emptyWork(1_000L), 100L, 1_000L);
            observePrefillAt(fixture, false, 1_100L);
            assertTrue(fixture.slot.decisionDeadlineAtMs().isEmpty());
            observePrefillAt(fixture, true, 3_601_000L);
            assertEquals(3_611_000L, fixture.slot.decisionDeadlineAtMs().orElseThrow());
        }
    }

    @Test
    void inactivityWatchSurvivesRouteResponseAndDecodeAcceptance() {
        Fixture fixture = fixture(true);
        var exact = mock(ExpirationTimer.InactivityDeadline.class);
        synchronized (fixture.slot) {
            assertTrue(fixture.slot.installInactivityDeadline(exact));
            fixture.slot.startRouteDecisionDelivery();
            fixture.slot.markDeliveryConfirmed();
            fixture.slot.markDecodeAccepted();
            assertTrue(fixture.slot.future().completeOwned(new Response()));
            assertTrue(fixture.slot.future().isDone());
            assertTrue(fixture.slot.expireInactivityDeadline(exact));
            assertFalse(fixture.slot.expireInactivityDeadline(exact));
        }
    }

    @Test
    void missingEngineEvidenceMarksSuspicionWithoutReleasingRequestOwnership() {
        Fixture fixture = fixture(true);
        var exact = mock(ExpirationTimer.DecisionDeadline.class);
        synchronized (fixture.slot) {
            fixture.slot.startRouteDecisionDelivery();
            startPrediction(fixture.slot);
            when(exact.deadlineAtMs()).thenReturn(fixture.slot.decisionDeadlineAtMs().orElseThrow());
            assertTrue(fixture.slot.installDecisionDeadline(exact));
            var expiry = fixture.slot.expireDecisionDeadline(exact);
            assertTrue(expiry.needsConfirmation());
            assertSame(fixture.item, expiry.item());
            assertTrue(fixture.slot.isLiveGeneration());
            assertFalse(fixture.slot.snapshot().state().isTerminal());
            assertTrue(fixture.slot.snapshot().detail().contains("SUSPECTED_LOST"));

            assertNull(fixture.slot.expireDecisionDeadline(exact), "timer is consumed once");
        }
    }

    @Test
    void runningPrefillMayExceedPredictionAndOnlyCompletedHandoffCanBecomeUnresolved() {
        Fixture fixture = fixture(true);
        var exact = mock(ExpirationTimer.DecisionDeadline.class);
        synchronized (fixture.slot) {
            fixture.slot.startRouteDecisionDelivery();
            startPrediction(fixture.slot);
            when(exact.deadlineAtMs()).thenReturn(fixture.slot.decisionDeadlineAtMs().orElseThrow());
            assertTrue(fixture.slot.installDecisionDeadline(exact));
            observePrefill(fixture.slot, fixture.item, false);
            assertNull(fixture.slot.expireDecisionDeadline(exact));
            assertFalse(fixture.slot.needsDecisionConfirmation());
            observePrefill(fixture.slot, fixture.item, true);
            assertTrue(fixture.slot.decisionDeadlineAtMs().orElseThrow() > System.currentTimeMillis());
            fixture.slot.markDecodeAccepted();
            assertFalse(fixture.slot.needsDecisionConfirmation());
        }
    }

    @Test
    void pdfusionEvidenceEndsMissingRequestDetectionButKeepsInactivityWatch() {
        Fixture fixture = fixture(false);
        var full = mock(ExpirationTimer.InactivityDeadline.class);
        var decision = mock(ExpirationTimer.DecisionDeadline.class);
        synchronized (fixture.slot) {
            fixture.slot.installInactivityDeadline(full);
            fixture.slot.startRouteDecisionDelivery();
            startPrediction(fixture.slot);
            when(decision.deadlineAtMs()).thenReturn(fixture.slot.decisionDeadlineAtMs().orElseThrow());
            fixture.slot.installDecisionDeadline(decision);
            observePrefill(fixture.slot, fixture.item, false);
            assertNull(fixture.slot.expireDecisionDeadline(decision));
            assertTrue(fixture.slot.expireInactivityDeadline(full));
        }
    }

    @Test
    void committedInactivityCancellationWinsOverLateHandoffEvidence() {
        Fixture fixture = fixture(true);
        var decision = mock(ExpirationTimer.DecisionDeadline.class);
        synchronized (fixture.slot) {
            fixture.slot.startRouteDecisionDelivery();
            startPrediction(fixture.slot);
            fixture.slot.markCancellationRequested(CancelReason.DEADLINE_EXCEEDED, "request inactive");
            when(decision.deadlineAtMs()).thenReturn(fixture.slot.decisionDeadlineAtMs().orElseThrow());
            fixture.slot.installDecisionDeadline(decision);
            assertFalse(fixture.slot.expireDecisionDeadline(decision).needsConfirmation());
            observePrefill(fixture.slot, fixture.item, true);
            assertFalse(fixture.slot.needsDecisionConfirmation());
        }
    }

    @Test
    void latePrefillEvidenceInvalidatesTheExpiredDecision() {
        Fixture fixture = fixture(true);
        synchronized (fixture.slot) {
            assertTrue(expireDecision(fixture).needsConfirmation());
            observePrefill(fixture.slot, fixture.item, false);
            fixture.slot.reconcileDecisionEvidence();
            assertFalse(fixture.slot.needsDecisionConfirmation());
            assertFalse(fixture.slot.snapshot().detail().contains("SUSPECTED_LOST"));
            assertFalse(fixture.slot.decodeOwnsRequest());
        }
    }

    @Test
    void delayedUncertaintyCannotReplaceMatchingEngineEvidence() {
        Fixture fixture = fixture(true);
        synchronized (fixture.slot) {
            expireDecision(fixture);
            observePrefill(fixture.slot, fixture.item, false);
            assertFalse(fixture.slot.markAwaitingConfirmation("late transport uncertainty"));
            assertFalse(fixture.slot.snapshot().detail().contains("SUSPECTED_LOST"));
            assertFalse(fixture.slot.hasCancellationFirstCause());
            observePrefill(fixture.slot, fixture.item, true);
            assertTrue(fixture.slot.decisionDeadlineAtMs().orElseThrow() > System.currentTimeMillis());
        }
    }

    @Test
    void uncertainDeliveryKeepsTheInactivityDeadlineAndDoesNotCancelTheRequest() {
        Fixture fixture = fixture(true);
        var deadline = mock(ExpirationTimer.InactivityDeadline.class);
        synchronized (fixture.slot) {
            fixture.slot.startRouteDecisionDelivery();
            startPrediction(fixture.slot);
            assertTrue(fixture.slot.installInactivityDeadline(deadline));
            assertTrue(fixture.slot.markAwaitingConfirmation("ambiguous transport"));
            assertFalse(fixture.slot.hasCancellationFirstCause());
            assertEquals(RequestState.Phase.DISPATCHING, fixture.slot.snapshot().state());
            assertTrue(fixture.slot.expireInactivityDeadline(deadline));
            assertTrue(fixture.slot.inactivityDeadlineAtMs().isPresent());
            assertSame(fixture.item, fixture.slot.activeItem());
        }
    }

    @Test
    void cancellationFirstCauseCannotSuppressTheInactivityWatch() {
        Fixture fixture = fixture(true);
        var deadline = mock(ExpirationTimer.InactivityDeadline.class);
        var renewed = mock(ExpirationTimer.InactivityDeadline.class);
        synchronized (fixture.slot) {
            fixture.slot.startRouteDecisionDelivery();
            assertTrue(fixture.slot.installInactivityDeadline(deadline));
            fixture.slot.markCancellationRequested(CancelReason.CLIENT_CANCELLED, "client cancellation");
            assertTrue(fixture.slot.expireInactivityDeadline(deadline));
            assertTrue(fixture.slot.installInactivityDeadline(renewed));
            assertFalse(fixture.slot.expireInactivityDeadline(deadline));
            assertTrue(fixture.slot.expireInactivityDeadline(renewed));
            assertEquals(CancelReason.CLIENT_CANCELLED, fixture.slot.requireCancellationFirstCause());
        }
    }

    @ParameterizedTest
    @CsvSource({"PREFILL,true", "DECODE,true", "PREFILL,false", "DECODE,false"})
    void uncertaintyAllowsActualAckIndependentlyOfMatchingEngineEvidence(
            RoleType evidenceSource, boolean ackBeforeEvidence) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestRegistry registry = new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
        try {
            BalanceContext context = RequestLifecycleTestSupport.context(config, 202L);
            var future = registry.register(context);
            RequestSlot slot = registry.requestSlot(202L);
            PrefillEndpoint prefill = mock(PrefillEndpoint.class);
            DecodeEndpoint decode = mock(DecodeEndpoint.class);
            var reservation = new DecodeEndpoint.ReservationHandle(1L, 202L, 1L);
            ScheduledRequest item = new ScheduledRequest(context, future, new Response(), prefillServer(), null,
                    prefill, decode, reservation, System.currentTimeMillis());
            RequestLifecycleTestSupport.bind(registry, new RequestLifecycleTestSupport.Registered(item, future));
            RequestRegistry.DeliveryClaim claim = registry.tryClaimBatchDelivery(item, 7L, () -> true);
            assertNotNull(claim);
            synchronized (slot) {
                startPrediction(slot);
                var deadline = mock(ExpirationTimer.DecisionDeadline.class);
                when(deadline.deadlineAtMs()).thenReturn(slot.decisionDeadlineAtMs().orElseThrow());
                assertTrue(slot.installDecisionDeadline(deadline));
                assertTrue(slot.expireDecisionDeadline(deadline).needsConfirmation());
                assertFalse(slot.hasCancellationFirstCause());
            }
            if (ackBeforeEvidence) {
                registry.complete(claim, DeliveryResult.delivered());
            }
            if (ackBeforeEvidence) {
                assertTrue(future.get(1L, TimeUnit.SECONDS).isSuccess(),
                        "uncertainty cannot hold a real EnqueueBatch ACK");
            } else {
                assertFalse(future.isDone());
            }

            EndpointEventProjector projector = new EndpointEventProjector(registry);
            if (evidenceSource == RoleType.PREFILL) {
                projector.onPrefillStatus(prefill, RoleType.PREFILL,
                        List.of(PrefillState.WorkerStatusFact.active(item)));
            } else {
                projector.onDecodeStatus(decode, List.of(DecodeEndpoint.WorkerStatusFact.active(reservation)));
            }

            if (!ackBeforeEvidence) {
                assertFalse(future.isDone(), "Engine activity cannot create an EnqueueBatch ACK");
                registry.complete(claim, DeliveryResult.delivered());
            }
            assertTrue(future.get(1L, TimeUnit.SECONDS).isSuccess());
            synchronized (slot) {
                assertEquals(evidenceSource == RoleType.DECODE, slot.decodeOwnsRequest());
                assertTrue(slot.isLiveGeneration());
                observePrefill(slot, item, true);
                assertFalse(slot.needsDecisionConfirmation());
            }
        } finally {
            if (registry.closeAdmissionAndAwaitMutations()) {
                registry.closeOutstandingAndTerminalize();
                registry.closeExpiration();
                registry.closePublisher();
            }
        }
    }

    @Test
    void timerChecksInactivityAfterThePublicFutureHasCompleted() throws Exception {
        Fixture fixture = fixture(true);
        RequestRegistry registry = mock(RequestRegistry.class);
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(fixture.config);
        when(registry.isCurrentSlot(fixture.slot)).thenReturn(true);
        when(registry.snapshotSlots()).thenReturn(List.of(fixture.slot));
        doAnswer(invocation -> {
            synchronized (fixture.slot) {
                assertTrue(fixture.slot.requestInactive(invocation.getArgument(1)));
                finishInactivity(fixture.slot);
            }
            return null;
        }).when(registry).expireInactiveRequest(eq(fixture.slot), anyLong());
        try (var timer = new ExpirationTimer(registry, config, mock(BatchSchedulerReporter.class))) {
            synchronized (fixture.slot) {
                fixture.slot.configureInactivityTimeout(20L);
                fixture.slot.startRouteDecisionDelivery();
                fixture.slot.markDeliveryConfirmed();
                fixture.slot.future().completeOwned(new Response());
            }
            assertNotNull(timer.attachInactivityDeadline(fixture.slot));
            verify(registry, timeout(1000L).times(1)).expireInactiveRequest(eq(fixture.slot), anyLong());
            verify(registry, never()).cancelForDeadline(any());
        }
    }

    @Test
    void renewedActivityRearmsTheTimerUntilTheNewInactivityDeadline() throws Exception {
        Fixture fixture = fixture(true);
        RequestRegistry registry = mock(RequestRegistry.class);
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(fixture.config);
        when(registry.isCurrentSlot(fixture.slot)).thenReturn(true);
        when(registry.snapshotSlots()).thenReturn(List.of(fixture.slot));
        long start = fixture.slot.createdAtMs();
        AtomicLong now = new AtomicLong(start + 100L);
        AtomicInteger checks = new AtomicInteger();
        CountDownLatch expired = new CountDownLatch(1);
        doAnswer(invocation -> {
            synchronized (fixture.slot) {
                if (checks.incrementAndGet() == 1) {
                    // A matching status wins after the old timer fired but before cancellation.
                    fixture.slot.observePrefillFact(fixture.item.prefillEp(), RoleType.PREFILL,
                            PrefillState.WorkerStatusFact.active(fixture.item), start + 50L);
                    assertFalse(fixture.slot.requestInactive(invocation.getArgument(1)));
                    now.set(start + 150L);
                } else {
                    assertTrue(fixture.slot.requestInactive(invocation.getArgument(1)));
                    finishInactivity(fixture.slot);
                    expired.countDown();
                }
            }
            return null;
        }).when(registry).expireInactiveRequest(eq(fixture.slot), anyLong());
        try (var timer = new ExpirationTimer(registry, config, mock(BatchSchedulerReporter.class), now::get)) {
            synchronized (fixture.slot) {
                fixture.slot.configureInactivityTimeout(100L);
                fixture.slot.startRouteDecisionDelivery();
                fixture.slot.markDeliveryConfirmed();
            }
            assertNotNull(timer.attachInactivityDeadline(fixture.slot));
            assertTrue(expired.await(1L, TimeUnit.SECONDS));
            assertEquals(2, checks.get(), "the renewed request must retain a timer for later silence");
        }
    }

    @Test
    void oldVisibilityTimerCannotInstallAfterPrefillCompletion() {
        Fixture fixture = fixture(true);
        synchronized (fixture.slot) {
            fixture.slot.startRouteDecisionDelivery();
            startPrediction(fixture.slot);
            long oldDeadline = fixture.slot.decisionDeadlineAtMs().orElseThrow();
            var oldTimer = mock(ExpirationTimer.DecisionDeadline.class);
            when(oldTimer.deadlineAtMs()).thenReturn(oldDeadline);
            fixture.slot.observePrefillFact(fixture.item.prefillEp(), RoleType.PREFILL,
                    PrefillState.WorkerStatusFact.terminal(fixture.item,
                            PrefillState.WorkerStatusFact.Kind.COMPLETED, 0L), oldDeadline + 100L);
            assertFalse(fixture.slot.installDecisionDeadline(oldTimer));
            var handoffTimer = mock(ExpirationTimer.DecisionDeadline.class);
            when(handoffTimer.deadlineAtMs()).thenReturn(fixture.slot.decisionDeadlineAtMs().orElseThrow());
            assertTrue(fixture.slot.installDecisionDeadline(handoffTimer));
            assertNull(fixture.slot.expireDecisionDeadline(oldTimer));
            assertTrue(fixture.slot.expireDecisionDeadline(handoffTimer).needsConfirmation());
        }
    }

    @Test
    void decodeAcceptanceDetachesDecisionTimerButKeepsTheCancellationAndInactivityWatch() {
        Fixture fixture = fixture(true);
        synchronized (fixture.slot) {
            fixture.slot.startRouteDecisionDelivery();
            startPrediction(fixture.slot);
            var timer = mock(ExpirationTimer.DecisionDeadline.class);
            when(timer.deadlineAtMs()).thenReturn(fixture.slot.decisionDeadlineAtMs().orElseThrow());
            assertTrue(fixture.slot.installDecisionDeadline(timer));
            fixture.slot.markCancellationRequested(CancelReason.CLIENT_CANCELLED, "client cancellation");
            var acceptance = fixture.slot.markDecodeAccepted();
            assertSame(timer, acceptance.detachedDecisionDeadline());
            assertNull(fixture.slot.expireDecisionDeadline(timer));
            assertEquals(CancelReason.CLIENT_CANCELLED, fixture.slot.requireCancellationFirstCause());
            assertTrue(fixture.slot.inactivityDeadlineAtMs().isPresent());
        }
    }

    private static void finishInactivity(RequestSlot slot) {
        TerminalAction terminal = slot.beginTerminalizing(false, false, false, null,
                owner -> owner.timeout("request inactive"), null);
        assertNotNull(terminal);
        assertNotNull(slot.finishTombstone(terminal).terminal());
    }

    private static WorkSnapshot emptyWork(long capturedAtMs) {
        return new WorkSnapshot(capturedAtMs, List.of(), List.of(), 0L);
    }

    private static WorkSnapshot unknownWork(long capturedAtMs) {
        return new WorkSnapshot(capturedAtMs, List.of(), List.of(), 1L);
    }

    private static OptionalLong visibilityDeadline(WorkSnapshot preceding, long unstartedWorkMs,
                                                   double lifetime, long deliveredAtMs) {
        Fixture fixture = fixture(true);
        fixture.config.getRequestLifecycle().getDecision().setLifetime(lifetime);
        synchronized (fixture.slot) {
            fixture.slot.startDecisionTracking(preceding, unstartedWorkMs, deliveredAtMs);
            return fixture.slot.decisionDeadlineAtMs();
        }
    }

    private static void observePrefillAt(Fixture fixture, boolean completed, long nowMs) {
        fixture.slot.observePrefillFact(fixture.item.prefillEp(), RoleType.PREFILL,
                completed ? PrefillState.WorkerStatusFact.terminal(fixture.item,
                        PrefillState.WorkerStatusFact.Kind.COMPLETED, 0L)
                        : PrefillState.WorkerStatusFact.active(fixture.item), nowMs);
    }

    private static Fixture fixture(boolean separateDecode) {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useNonBatchDispatcher(config);
        BalanceContext context = RequestLifecycleTestSupport.context(config, 101L);
        RequestSlot slot = new RequestSlot(mock(RequestCompletionPublisher.class), 101L);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = separateDecode ? mock(DecodeEndpoint.class) : null;
        DecodeEndpoint.ReservationHandle reservation = separateDecode
                ? new DecodeEndpoint.ReservationHandle(1L, 101L, 1L) : null;
        ScheduledRequest item = new ScheduledRequest(context, slot.future(), new Response(), prefillServer(), null,
                prefill, decode, reservation, System.currentTimeMillis());
        synchronized (slot) {
            slot.configureInactivityTimeout(60_000L);
            AdmissionMutation mutation = slot.tryBeginAdmissionMutation((owner, response) -> { }, owner -> { });
            assertNotNull(mutation);
            assertTrue(slot.tryBindItemForPublication(item));
            slot.completeAdmissionMutation(mutation);
        }
        return new Fixture(config, slot, item);
    }

    private static RequestSlot.EngineObservation observePrefill(RequestSlot slot, ScheduledRequest item, boolean completed) {
        return slot.observePrefillFact(item.prefillEp(), RoleType.PREFILL,
                completed ? PrefillState.WorkerStatusFact.terminal(item, PrefillState.WorkerStatusFact.Kind.COMPLETED, 0L)
                        : PrefillState.WorkerStatusFact.active(item), System.currentTimeMillis());
    }

    private static void startPrediction(RequestSlot slot) {
        slot.startDecisionTracking(emptyWork(System.currentTimeMillis()), 100L, System.currentTimeMillis());
    }

    private static RequestSlot.DecisionExpiry expireDecision(Fixture fixture) {
        fixture.slot.startRouteDecisionDelivery();
        startPrediction(fixture.slot);
        var deadline = mock(ExpirationTimer.DecisionDeadline.class);
        when(deadline.deadlineAtMs()).thenReturn(fixture.slot.decisionDeadlineAtMs().orElseThrow());
        assertTrue(fixture.slot.installDecisionDeadline(deadline));
        return fixture.slot.expireDecisionDeadline(deadline);
    }

    private static ServerStatus prefillServer() {
        ServerStatus server = new ServerStatus();
        server.setRole(RoleType.PREFILL);
        server.setServerIp("127.0.0.1");
        server.setGrpcPort(8081);
        return server;
    }

    private record Fixture(FlexlbConfig config, RequestSlot slot, ScheduledRequest item) { }
}

package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.flexlb.balance.scheduler.ExpirationTimer.Kind.*;
import org.flexlb.balance.scheduler.AdmissionHandle;
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
            predictDelivery(fixture.requestContext, emptyWork(1_000L), 100L, 1_000L);
            assertEquals(11_200L, deadlineAt(fixture.requestContext, VISIBILITY).orElseThrow());
            recordPrefillStatusAt(fixture, false, 1_100L);
            assertTrue(deadlineAt(fixture.requestContext, VISIBILITY).isEmpty());
            recordPrefillStatusAt(fixture, true, 5_000L);
            assertEquals(15_000L, deadlineAt(fixture.requestContext, VISIBILITY).orElseThrow());
            recordPrefillStatusAt(fixture, true, 5_020L);
            assertEquals(15_000L, deadlineAt(fixture.requestContext, VISIBILITY).orElseThrow());
            var timer = fixture.requestContext.snapshot().deadlines().visibility();

            assertTrue(matches(fixture.requestContext, timer));
            expire(fixture.requestContext, timer);
            assertTrue(RequestState.from(fixture.requestContext.snapshot()).detail().startsWith("SUSPECTED_LOST"));
            decodeAccepted(fixture.requestContext);
            assertTrue(deadlineAt(fixture.requestContext, VISIBILITY).isEmpty());
            assertFalse(needsConfirmation(fixture.requestContext));
    }

    @Test
    void completionAndAcceptanceBeforeDeliveryRemainAuthoritative() {
        Fixture fixture = fixture(true);
            recordPrefillStatusAt(fixture, true, 1_000L);
            predictDelivery(fixture.requestContext, emptyWork(1_000L), 100L, 1_010L);
            assertEquals(11_000L, deadlineAt(fixture.requestContext, VISIBILITY).orElseThrow());
            decodeAccepted(fixture.requestContext);
            recordPrefillStatusAt(fixture, false, 2_000L);
            recordPrefillStatusAt(fixture, true, 2_000L);
            assertTrue(fixture.requestContext.snapshot().engine().decodeAccepted());
            assertTrue(deadlineAt(fixture.requestContext, VISIBILITY).isEmpty());
            assertThrows(IllegalStateException.class, () -> predictDelivery(fixture.requestContext, emptyWork(2_000L), 100L, 2_000L));
    }

    @Test
    void unknownPredictionCanStillStartHandoffDetectionAfterPrefillCompletes() {
        Fixture fixture = fixture(true);
            predictDelivery(fixture.requestContext, unknownWork(1_000L), 100L, 1_000L);
            assertTrue(deadlineAt(fixture.requestContext, VISIBILITY).isEmpty());
            assertFalse(needsConfirmation(fixture.requestContext));
            recordPrefillStatusAt(fixture, true, 2_000L);
            assertEquals(12_000L, deadlineAt(fixture.requestContext, VISIBILITY).orElseThrow());
    }

    @Test
    void longRunningPrefillStillReceivesTheFullHandoffWindow() {
        Fixture fixture = fixture(true);
            predictDelivery(fixture.requestContext, emptyWork(1_000L), 100L, 1_000L);
            recordPrefillStatusAt(fixture, false, 1_100L);
            assertTrue(deadlineAt(fixture.requestContext, VISIBILITY).isEmpty());
            recordPrefillStatusAt(fixture, true, 3_601_000L);
            assertEquals(3_611_000L, deadlineAt(fixture.requestContext, VISIBILITY).orElseThrow());
    }

    @Test
    void inactivityWatchSurvivesRouteResponseAndDecodeAcceptance() {
        Fixture fixture = fixture(true);
        var exact = fixture.requestContext.snapshot().deadlines().inactivity();
            assertTrue(matches(fixture.requestContext, exact));
            startDelivery(fixture.requestContext);
            acknowledge(fixture.requestContext);
            decodeAccepted(fixture.requestContext);
            assertTrue(((RequestFuture) fixture.requestContext.data().getFuture()).completeOwned(new Response()));
            assertTrue(((RequestFuture) fixture.requestContext.data().getFuture()).isDone());
            assertNotNull(checkBeforeInactivityExpiry(fixture.requestContext, exact));
            assertNull(checkBeforeInactivityExpiry(fixture.requestContext, exact));
    }

    @Test
    void missingEngineEvidenceMarksSuspicionWithoutReleasingRequestOwnership() {
        Fixture fixture = fixture(true);
        startDelivery(fixture.requestContext);
        startPrediction(fixture.scheduler, fixture.requestContext);
        var exact = fixture.requestContext.snapshot().deadlines().visibility();

            assertTrue(matches(fixture.requestContext, exact));
            expire(fixture.requestContext, exact);
            assertSame(fixture.item, fixture.requestContext.snapshot().route());
            assertTrue(needsConfirmation(fixture.requestContext));
            assertTrue(deadlineAt(fixture.requestContext, VISIBILITY).isEmpty());
            assertNull(fixture.requestContext.snapshot().deadlines().visibility());
            assertTrue(fixture.requestContext.snapshot().phase() != Phase.CLOSED);
            assertFalse(RequestState.from(fixture.requestContext.snapshot()).state().isTerminal());
            assertTrue(RequestState.from(fixture.requestContext.snapshot()).detail().contains("SUSPECTED_LOST"));
            assertStaleDecisionHasNoEffect(fixture, exact);
    }

    @Test
    void runningPrefillMayExceedPredictionAndOnlyCompletedHandoffCanBecomeUnresolved() {
        Fixture fixture = fixture(true);
        startDelivery(fixture.requestContext);
        startPrediction(fixture.scheduler, fixture.requestContext);
        var exact = fixture.requestContext.snapshot().deadlines().visibility();

            assertTrue(matches(fixture.requestContext, exact));
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, false);
            assertStaleDecisionHasNoEffect(fixture, exact);
            assertFalse(needsConfirmation(fixture.requestContext));
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, true);
            assertTrue(deadlineAt(fixture.requestContext, VISIBILITY).orElseThrow() > System.currentTimeMillis());
            decodeAccepted(fixture.requestContext);
            assertFalse(needsConfirmation(fixture.requestContext));
    }

    @Test
    void pdfusionEvidenceEndsMissingRequestDetectionButKeepsInactivityWatch() {
        Fixture fixture = fixture(false);
        var full = fixture.requestContext.snapshot().deadlines().inactivity();
        matches(fixture.requestContext, full);
        startDelivery(fixture.requestContext);
        startPrediction(fixture.scheduler, fixture.requestContext);
        var decision = fixture.requestContext.snapshot().deadlines().visibility();

            matches(fixture.requestContext, decision);
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, false);
            assertStaleDecisionHasNoEffect(fixture, decision);
            assertNotNull(checkBeforeInactivityExpiry(fixture.requestContext, full));
    }

    @Test
    void committedInactivityCancellationWinsOverLateHandoffEvidence() {
        Fixture fixture = fixture(true);
        startDelivery(fixture.requestContext);
        startPrediction(fixture.scheduler, fixture.requestContext);
        var decision = fixture.requestContext.snapshot().deadlines().visibility();
            cancel(fixture.requestContext, CancelReason.DEADLINE_EXCEEDED, "request inactive");

            matches(fixture.requestContext, decision);
            expire(fixture.requestContext, decision);
            assertFalse(RequestState.from(fixture.requestContext.snapshot()).detail().startsWith("SUSPECTED_LOST"));
            assertEquals(CancelReason.DEADLINE_EXCEEDED, fixture.requestContext.snapshot().cancellation() == null ? null : fixture.requestContext.snapshot().cancellation().reason());
            assertFalse(needsConfirmation(fixture.requestContext));
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, true);
            assertFalse(needsConfirmation(fixture.requestContext));
    }

    @Test
    void latePrefillEvidenceInvalidatesTheExpiredDecision() {
        Fixture fixture = fixture(true);
            expireDecision(fixture);
            assertTrue(RequestState.from(fixture.requestContext.snapshot()).detail().startsWith("SUSPECTED_LOST"));
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, false);
            assertFalse(needsConfirmation(fixture.requestContext));
            assertFalse(RequestState.from(fixture.requestContext.snapshot()).detail().contains("SUSPECTED_LOST"));
            assertFalse(fixture.requestContext.snapshot().engine().decodeAccepted());
    }

    @Test
    void delayedUncertaintyCannotReplaceMatchingEngineEvidence() {
        Fixture fixture = fixture(true);
            expireDecision(fixture);
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, false);
            uncertain(fixture.requestContext);
            assertFalse(RequestState.from(fixture.requestContext.snapshot()).detail().contains("SUSPECTED_LOST"));
            assertNull(fixture.requestContext.snapshot().cancellation());
            preparePrefillStatusEffects(fixture.scheduler, fixture.requestContext, fixture.item, true);
            assertTrue(deadlineAt(fixture.requestContext, VISIBILITY).orElseThrow() > System.currentTimeMillis());
    }

    @Test
    void uncertainDeliveryKeepsTheInactivityDeadlineAndDoesNotCancelTheRequest() {
        Fixture fixture = fixture(true);
        var deadline = fixture.requestContext.snapshot().deadlines().inactivity();
            startDelivery(fixture.requestContext);
            startPrediction(fixture.scheduler, fixture.requestContext);
            assertTrue(matches(fixture.requestContext, deadline));
            uncertain(fixture.requestContext);
            assertTrue(RequestState.from(fixture.requestContext.snapshot()).detail().contains("SUSPECTED_LOST"));
            assertNull(fixture.requestContext.snapshot().cancellation());
            assertEquals(RequestState.Phase.DISPATCHING, RequestState.from(fixture.requestContext.snapshot()).state());
            assertNotNull(checkBeforeInactivityExpiry(fixture.requestContext, deadline));
            assertTrue(deadlineAt(fixture.requestContext, INACTIVITY).isPresent());
            assertSame(fixture.item, fixture.requestContext.snapshot().route());
    }

    @Test
    void cancellationFirstCauseCannotSuppressTheInactivityWatch() {
        Fixture fixture = fixture(true);
        var deadline = fixture.requestContext.snapshot().deadlines().inactivity();
            startDelivery(fixture.requestContext);
            assertTrue(matches(fixture.requestContext, deadline));
            cancel(fixture.requestContext, CancelReason.CLIENT_CANCELLED, "client cancellation");
            assertNotNull(checkBeforeInactivityExpiry(fixture.requestContext, deadline));
            var renewed = fixture.requestContext.snapshot().deadlines().inactivity();
            assertTrue(matches(fixture.requestContext, renewed));
            assertNull(checkBeforeInactivityExpiry(fixture.requestContext, deadline));
            assertNotNull(checkBeforeInactivityExpiry(fixture.requestContext, renewed));
            assertEquals(CancelReason.CLIENT_CANCELLED, fixture.requestContext.snapshot().cancellation().reason());
    }

    @ParameterizedTest
    @CsvSource({ "PREFILL,true", "DECODE,true", "PREFILL,false", "DECODE,false" })
    void uncertaintyAllowsActualAckIndependentlyOfMatchingEngineEvidence(RoleType evidenceSource, boolean ackBeforeEvidence) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler registry = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
        try {
            RequestContext context = RequestProtocolTestSupport.context(config, 202L);
            var future = RequestProtocolTestSupport.register(registry, context);
            PrefillEndpoint prefill = mock(PrefillEndpoint.class);
            DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
            var reservation = new DecodeResources.ReservationHandle(1L, 202L, 1L);
            RequestRoute item = SchedulingTestConfig.createRoute(context, new Response(), prefillServer(), null,
                    prefill, decode, reservation, System.currentTimeMillis());
            RequestProtocolTestSupport.bind(registry, new RequestProtocolTestSupport.Registered(item, future));
            DeliveryClaim claim = RequestProtocolTestSupport.claimBatchWithoutPrediction(registry, item, 7L, () -> true);
            assertNotNull(claim);
            startPrediction(registry, context);
            expire(context, context.snapshot().deadlines().visibility());
            assertTrue(RequestState.from(context.snapshot()).detail().startsWith("SUSPECTED_LOST"));
            assertNull(context.snapshot().cancellation());
            if (ackBeforeEvidence) {
                registry.onDeliveryEvidence(claim, new TransportResult(claim, DeliveryResult.delivered(), System.currentTimeMillis()));
                assertTrue(future.get(1L, TimeUnit.SECONDS).isSuccess(), "uncertainty cannot hold a real EnqueueBatch ACK");
            } else { assertFalse(future.isDone()); }
            registry.onEngineEvidence(context, new EngineEvidence(item, evidenceSource == RoleType.DECODE ? reservation : null,
                    evidenceSource, EngineKind.ACTIVE, 0L, null, System.currentTimeMillis()));
            if (!ackBeforeEvidence) {
                assertFalse(future.isDone(), "Engine activity cannot create an EnqueueBatch ACK");
                registry.onDeliveryEvidence(claim, new TransportResult(claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            }
            assertTrue(future.get(1L, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(() -> evidenceSource == RoleType.DECODE
                    ? context.snapshot().engine().decodeAccepted() : !needsConfirmation(context));
            assertEquals(evidenceSource == RoleType.DECODE, context.snapshot().engine().decodeAccepted());
            assertTrue(context.snapshot().phase() != Phase.CLOSED);
            preparePrefillStatusEffects(registry, context, item, true);
            assertFalse(needsConfirmation(context));
            registry.onEngineEvidence(context, new EngineEvidence(item, reservation, RoleType.DECODE,
                    EngineKind.COMPLETED, 0L, null, System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
        } finally {
            registry.runtime.shutdown();
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"install", "reject", "throw"})
    void earlyTimerFiresOnlyAfterSuccessfulInstallation(String outcome) throws Exception {
        Fixture fixture = fixture(true);
        RequestContext context = fixture.requestContext;
        AbstractRequestScheduler scheduler = fixture.scheduler;
        var failure = new IllegalStateException("installation failed");
        try (var timer = new ExpirationTimer(scheduler.requests)) {
            var original = (java.util.concurrent.ScheduledThreadPoolExecutor)
                    org.springframework.test.util.ReflectionTestUtils.getField(timer, "executor");
            var executor = org.mockito.Mockito.spy(original);
            org.springframework.test.util.ReflectionTestUtils.setField(timer, "executor", executor);
            var token = new ExpirationTimer.Token(context, REQUEST, 10L, 1L);
            var interceptFirst = new java.util.concurrent.atomic.AtomicBoolean(true);
            var exactHandle = new java.util.concurrent.atomic.AtomicReference<Object>();
            doAnswer(call -> {
                var scheduled = call.callRealMethod();
                if (!interceptFirst.compareAndSet(true, false)) { return scheduled; }
                var slots = (java.util.Map<?, ?>) org.springframework.test.util.ReflectionTestUtils.getField(timer, "slots");
                var handles = (java.util.Map<?, ?>) org.springframework.test.util.ReflectionTestUtils.invokeMethod(slots.get(context), "handles");
                exactHandle.set(handles.get(REQUEST));
                // A zero-delay task has run but cannot publish until schedule returns successfully.
                executor.submit(() -> { }).get(5, TimeUnit.SECONDS);
                verify(scheduler, never()).onDeadline(any(), anyLong());
                if ("reject".equals(outcome)) { timer.update(context, new ExpirationTimer.Plan(11L, null, null, null)); }
                if ("throw".equals(outcome)) { throw failure; }
                return scheduled;
            }).when(executor).schedule(any(Runnable.class), anyLong(), any(TimeUnit.class));
            if ("throw".equals(outcome)) {
                assertSame(failure, assertThrows(IllegalStateException.class,
                        () -> timer.update(context, new ExpirationTimer.Plan(10L, token, null, null))));
            } else {
                timer.update(context, new ExpirationTimer.Plan(10L, token, null, null));
                if ("install".equals(outcome)) { verify(scheduler).onDeadline(org.mockito.ArgumentMatchers.eq(token), anyLong()); }
            }
            assertNotNull(exactHandle.get());
            org.springframework.test.util.ReflectionTestUtils.invokeMethod(exactHandle.get(), "fire");
            org.springframework.test.util.ReflectionTestUtils.invokeMethod(exactHandle.get(), "publish");
            org.springframework.test.util.ReflectionTestUtils.invokeMethod(exactHandle.get(), "cancel");
            if (!"install".equals(outcome)) { verify(scheduler, never()).onDeadline(any(), anyLong()); }
            else { verify(scheduler, org.mockito.Mockito.times(1)).onDeadline(org.mockito.ArgumentMatchers.eq(token), anyLong()); }
            timer.remove(context);
            timer.update(context, new ExpirationTimer.Plan(12L, null, null, null));
            assertTrue(executor.getQueue().isEmpty());
        }
    }

    @Test
    void timerChecksInactivityAfterThePublicFutureHasCompleted() throws Exception {
        Fixture fixture = fixture(true);
        AbstractRequestScheduler registry = fixture.scheduler;
        doAnswer(invocation -> {
            var token = invocation.getArgument(0, ExpirationTimer.Token.class);
            long nowMs = invocation.getArgument(1, Long.class);
            var update = fixture.requestContext.recordDeadline(token, nowMs, null);
            assertNotNull(update.after().pendingEnd());
            finishInactivity(fixture.requestContext, nowMs);
            return null;
        }).when(registry).onDeadline(any(), anyLong());
        try (var timer = new ExpirationTimer(registry.requests)) {
            startDelivery(fixture.requestContext);
            acknowledge(fixture.requestContext);
            ((RequestFuture) fixture.requestContext.data().getFuture()).completeOwned(new Response());
            shortenInactivity(fixture.requestContext, 20L);
            assertNotNull(fixture.requestContext.snapshot().deadlines().inactivity());
            timer.update(fixture.requestContext, fixture.requestContext.snapshot().deadlines());
            verify(registry, timeout(1000L).times(1)).onDeadline(any(), anyLong());
            verify(registry, never()).cancel(anyLong(), anyLong(), any());
        }
    }

    @Test
    void timerCloseWaitsForAnAlreadyStartedInactivityHandoff() throws Exception {
        Fixture fixture = fixture(true);
        AbstractRequestScheduler registry = fixture.scheduler;
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        doAnswer(invocation -> {
            entered.countDown();
            boolean interrupted = false;
            while (true) {
                try { release.await(); break; }
                catch (InterruptedException ignored) { interrupted = true; }
            }
            if (interrupted) { Thread.currentThread().interrupt(); }
            return null;
        }).when(registry).onDeadline(any(), anyLong());
        ExpirationTimer timer = new ExpirationTimer(registry.requests);
        Thread closer = null;
        try {
            shortenInactivity(fixture.requestContext, 20L);
            timer.update(fixture.requestContext, fixture.requestContext.snapshot().deadlines());
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            closer = new Thread(timer::close);
            closer.start();
            assertTrue(closer.isAlive(), "close must wait for the running timer producer");
        } finally {
            release.countDown();
            timer.close();
            if (closer != null) { closer.join(5_000); }
        }
        assertFalse(closer.isAlive());
    }

    @Test
    void renewedActivityRearmsTheTimerUntilTheNewInactivityDeadline() throws Exception {
        Fixture fixture = fixture(true);
        AbstractRequestScheduler registry = fixture.scheduler;
        long start = fixture.requestContext.snapshot().createdAtMs();
        AtomicLong now = new AtomicLong(start + 100L);
        AtomicInteger checks = new AtomicInteger();
        CountDownLatch expired = new CountDownLatch(1);
        try (var timer = new ExpirationTimer(registry.requests, now::get)) {
            doAnswer(invocation -> {
                var token = invocation.getArgument(0, ExpirationTimer.Token.class);
                long nowMs = invocation.getArgument(1, Long.class);
                boolean renewed = checks.incrementAndGet() == 1;
                if (renewed) { recordPrefill(fixture.requestContext, false, start + 50L); }
                var update = fixture.requestContext.recordDeadline(token, nowMs, null);
                if (renewed) {
                    assertNull(update.after().pendingEnd(), "renewed activity must keep the request alive");
                    now.set(start + 150L);
                    timer.update(fixture.requestContext, fixture.requestContext.snapshot().deadlines());
                } else {
                    assertNotNull(update.after().pendingEnd());
                    finishInactivity(fixture.requestContext, nowMs);
                    expired.countDown();
                }
                return null;
            }).when(registry).onDeadline(any(), anyLong());
            startDelivery(fixture.requestContext);
            acknowledge(fixture.requestContext);
            shortenInactivity(fixture.requestContext, 100L);
            timer.update(fixture.requestContext, fixture.requestContext.snapshot().deadlines());
            assertTrue(expired.await(1L, TimeUnit.SECONDS));
            assertEquals(2, checks.get(), "the renewed request must retain a timer for later silence");
        }
    }

    @Test
    void oldVisibilityTimerCannotInstallAfterPrefillCompletion() {
        Fixture fixture = fixture(true);
            startDelivery(fixture.requestContext);
            startPrediction(fixture.scheduler, fixture.requestContext);
            long oldDeadline = deadlineAt(fixture.requestContext, VISIBILITY).orElseThrow();
            var oldTimer = fixture.requestContext.snapshot().deadlines().visibility();

            recordPrefill(fixture.requestContext, true, oldDeadline + 100L);
            assertFalse(matches(fixture.requestContext, oldTimer));
            var handoffTimer = fixture.requestContext.snapshot().deadlines().visibility();

            assertTrue(matches(fixture.requestContext, handoffTimer));
            assertStaleDecisionHasNoEffect(fixture, oldTimer);
            expire(fixture.requestContext, handoffTimer);
            assertTrue(RequestState.from(fixture.requestContext.snapshot()).detail().startsWith("SUSPECTED_LOST"));
    }

    @Test
    void decodeAcceptanceDetachesDecisionTimerButKeepsTheCancellationAndInactivityWatch() {
        Fixture fixture = fixture(true);
            startDelivery(fixture.requestContext);
            startPrediction(fixture.scheduler, fixture.requestContext);
            var timer = fixture.requestContext.snapshot().deadlines().visibility();

            assertTrue(matches(fixture.requestContext, timer));
            cancel(fixture.requestContext, CancelReason.CLIENT_CANCELLED, "client cancellation");
            var acceptance = decodeAccepted(fixture.requestContext);
            assertSame(timer, acceptance);
            assertStaleDecisionHasNoEffect(fixture, timer);
            assertEquals(CancelReason.CLIENT_CANCELLED, fixture.requestContext.snapshot().cancellation().reason());
            assertTrue(deadlineAt(fixture.requestContext, INACTIVITY).isPresent());
    }

    private static Update checkBeforeInactivityExpiry(RequestContext context, ExpirationTimer.Token exact) {
        if (!matches(context, exact)) { return null; }
        // An old wake-up still consumes its exact version after matching activity renews the deadline.
        recordPrefill(context, false, exact.atMs() - 1L);
        var update = context.recordDeadline(exact, exact.atMs(), null);
        assertNull(update.after().pendingEnd(), "identity consumption must not expire the request");
        return update;
    }

    private static void finishInactivity(RequestContext context, long nowMs) {
        var terminal = (EndToken) ((Granted<?>) context.beginClose(null, nowMs)).value();
        SchedulerTestSupport.settleTerminalFixture(context, terminal);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
        assertEquals(RequestState.Phase.TIMED_OUT, RequestState.from(context.snapshot()).state());
    }

    private static OptionalLong deadlineAt(RequestContext context, ExpirationTimer.Kind kind) {
        var token = context.snapshot().deadlines().token(kind);
        return token == null ? OptionalLong.empty() : OptionalLong.of(token.atMs());
    }

    private static boolean matches(RequestContext context, ExpirationTimer.Token token) {
        return token != null && context.snapshot().deadlines().token(token.kind()) == token;
    }

    private static void expire(RequestContext context, ExpirationTimer.Token token) {
        if (token != null) { context.recordDeadline(token, token.atMs(), null); }
    }

    private static void startDelivery(RequestContext context) {
        if (context.snapshot().delivery() != null) { return; }
        long now = context.snapshot().lastWorkerAtMs();
        var claim = (DeliveryClaim) ((Granted<?>) context.beginDelivery(context.snapshot().route(),
                DeliveryClaimKind.BATCH_ENQUEUE, 1L, now)).value();
        context.commitDelivery(claim, true, now);
    }

    private static void acknowledge(RequestContext context) {
        var identity = context.snapshot().delivery().identity();
        long now = context.snapshot().lastWorkerAtMs();
        context.recordDeliveryResult(new TransportResult(identity, DeliveryResult.delivered(), now));
        assertTrue(context.selectResponse(new ResponseCandidate(context, ResponseKind.CONFIRMATION, null, now)) instanceof Granted<?>);
    }

    private static ExpirationTimer.Token decodeAccepted(RequestContext context) {
        var old = context.snapshot().deadlines().visibility();
        var route = context.snapshot().route();
        context.recordEngineEvidence(new EngineEvidence(route, route.decodeReservation(), RoleType.DECODE,
                EngineKind.ACTIVE, 0L, null, context.snapshot().lastWorkerAtMs()));
        return old;
    }

    private static void recordPrefill(RequestContext context, boolean completed, long nowMs) {
        var route = context.snapshot().route();
        context.recordEngineEvidence(new EngineEvidence(route, null,
                route.decodeEp() == null ? RoleType.PDFUSION : RoleType.PREFILL,
                completed ? EngineKind.COMPLETED : EngineKind.ACTIVE, 0L, null, nowMs));
    }

    private static boolean needsConfirmation(RequestContext context) {
        var state = context.snapshot();
        return state.cancellation() == null && state.outcome() == null && !state.engine().decodeAccepted()
                && !(state.engine().prefillObserved() && state.engine().prefillCompletedAtMs() == 0L)
                && (state.engine().visibilityOverdue() || state.delivery() != null && state.delivery().send() == SendState.UNKNOWN);
    }

    private static void cancel(RequestContext context, CancelReason reason, String detail) {
        if (reason == CancelReason.DEADLINE_EXCEEDED) {
            var deadline = context.snapshot().deadlines().inactivity();
            context.recordDeadline(deadline, deadline.atMs(), new CancellationFacts(reason,
                    org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED, detail, null));
        } else {
            context.requestCancel(new Cancellation(0L, reason, false, context.snapshot().lastWorkerAtMs(),
                    org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED, detail));
        }
    }

    private static void uncertain(RequestContext context) {
        context.recordDeliveryResult(new TransportResult(context.snapshot().delivery().identity(),
                DeliveryResult.uncertain(new IllegalStateException("sender result unknown")), context.snapshot().lastWorkerAtMs()));
    }

    private static void shortenInactivity(RequestContext context, long durationMs) {
        org.springframework.test.util.ReflectionTestUtils.setField(context, "inactivityMs", durationMs);
        var old = context.snapshot().deadlines().inactivity();
        // Recompute the immutable desired plan without consuming the previous wake-up.
        recordPrefill(context, false, context.snapshot().lastWorkerAtMs());
        assertTrue(context.snapshot().deadlines().inactivity().atMs() <= old.atMs());
    }

    private static WorkSnapshot emptyWork(long capturedAtMs) {
        return new WorkSnapshot(capturedAtMs, List.of(), List.of(), 0L);
    }

    private static WorkSnapshot unknownWork(long capturedAtMs) {
        return new WorkSnapshot(capturedAtMs, List.of(), List.of(), 1L);
    }

    private static OptionalLong visibilityDeadline(WorkSnapshot preceding, long unstartedWorkMs, double lifetime, long deliveredAtMs) {
        long at = PrefillPredictionBoundary.predictedVisibilityDeadlineMs(preceding, unstartedWorkMs, lifetime, deliveredAtMs);
        return at == 0L ? OptionalLong.empty() : OptionalLong.of(at);
    }

    private static void recordPrefillStatusAt(Fixture fixture, boolean completed, long nowMs) {
        recordPrefill(fixture.requestContext, completed, nowMs);
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
        AbstractRequestScheduler owner = mock(AbstractRequestScheduler.class);
        var repository = mock(RequestRepository.class);
        when(repository.isCurrent(context)).thenReturn(true);
        org.springframework.test.util.ReflectionTestUtils.setField(owner, "requests", repository);
        context.data().register(owner, new RequestFuture((kind, response, failure) -> false));
        SchedulerTestSupport.bindOwner(context, owner);
        context.register(101L, System.currentTimeMillis(), Long.MAX_VALUE, 60_000L);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = separateDecode ? RequestProtocolTestSupport.decodeEndpoint() : null;
        var reservation = separateDecode ? new DecodeResources.ReservationHandle(1L, 101L, 1L) : null;
        RequestRoute item = SchedulingTestConfig.createRoute(context, new Response(), prefillServer(), null,
                prefill, decode, reservation, System.currentTimeMillis());
        var placement = (PlacementToken) ((Granted<?>) context.beginPlacement(null)).value();
        context.bindPlacement(placement, item, PublicationResult.PREPARED);
        context.bindPlacement(placement, item, PublicationResult.PUBLISHED);
        context.completePlacement(placement, PublicationResult.PUBLISHED, System.currentTimeMillis());
        return new Fixture(owner, config, context, item);
    }

    private static void preparePrefillStatusEffects(AbstractRequestScheduler scheduler, RequestContext context, RequestRoute item, boolean completed) {
        recordPrefill(context, completed, System.currentTimeMillis());
    }

    private static void predictDelivery(RequestContext context, WorkSnapshot work, long predictedMs, long nowMs) {
        startDelivery(context);
        long at = PrefillPredictionBoundary.predictedVisibilityDeadlineMs(work, predictedMs,
                context.data().getConfig().getRequestLifecycle().getDecision().getLifetime(), nowMs);
        context.recordDeliveryResult(new Prediction(context.snapshot().delivery().identity(), at, nowMs));
    }

    private static void startPrediction(AbstractRequestScheduler requestOwner, RequestContext requestContext) {
        predictDelivery(requestContext, emptyWork(System.currentTimeMillis()), 100L, System.currentTimeMillis());
    }

    private static void assertStaleDecisionHasNoEffect(Fixture fixture, ExpirationTimer.Token stale) {
        var before = fixture.requestContext.snapshot();
        var installed = before.deadlines().visibility();
        var deadline = deadlineAt(fixture.requestContext, VISIBILITY);
        var visibility = before.engine();
        expire(fixture.requestContext, stale);
        assertSame(before, fixture.requestContext.snapshot());
        assertSame(installed, fixture.requestContext.snapshot().deadlines().visibility());
        assertEquals(deadline, deadlineAt(fixture.requestContext, VISIBILITY));
        assertSame(visibility, fixture.requestContext.snapshot().engine());
        assertSame(fixture.item, fixture.requestContext.snapshot().route());
    }

    private static void expireDecision(Fixture fixture) {
        startDelivery(fixture.requestContext);
        startPrediction(fixture.scheduler, fixture.requestContext);
        var deadline = fixture.requestContext.snapshot().deadlines().visibility();
        assertTrue(matches(fixture.requestContext, deadline));
        expire(fixture.requestContext, deadline);
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

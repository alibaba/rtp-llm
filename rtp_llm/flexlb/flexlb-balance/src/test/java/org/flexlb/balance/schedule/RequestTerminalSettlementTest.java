package org.flexlb.balance.schedule;

import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.schedule.SchedulerTestSupport.Stage;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.RequestEventType;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.flexlb.balance.schedule.RequestContext.EndKind;
import static org.flexlb.balance.schedule.RequestContext.Phase;
import static org.flexlb.balance.schedule.RequestContext.RequestEndEvent;
import static org.flexlb.balance.schedule.RequestContext.RequestEvent;
import static org.flexlb.balance.schedule.SchedulerTestSupport.stage;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks that terminal evidence retains its cause and exact resource ownership until all required effects settle.
 */
class RequestTerminalSettlementTest {
    private static final DecodeEndpoint.RequestRegistration RESERVATION =
            new DecodeEndpoint.RequestRegistration(1L, String.valueOf(2L));

    @Test
    void nextPlacementAttemptCannotReuseEarlierCapacityAsItsTimeoutCause() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 4203L);
        var publisher = mock(ResponseCompletionExecutor.class);
        when(publisher.register()).thenReturn(true);
        var owner =
                RequestProtocolTestSupport.initialize(
                        publisher, context, mock(ExpirationTimer.class));
        var first = RequestProtocolTestSupport.beginAdmission(owner, context);
        assertNotNull(first);
        context.recordRoutingDiagnostics(
                first, java.util.Map.of("cause", "D1 capacity exhausted", "capturedAtMs", 10L));
        assertEquals("D1 capacity exhausted", context.getSchedulingDiagnostics().get("cause"));
        first.requestContext().getRequestScheduler().finishSelectionWithoutCommit(first, null);
        var current = RequestProtocolTestSupport.beginAdmission(owner, context);
        assertNotNull(current);
        try {
            assertNull(context.getSchedulingDiagnostics());
            context.recordRoutingDiagnostics(first, java.util.Map.of("cause", "late D1 result"));
            assertNull(context.getSchedulingDiagnostics());
            context.recordRoutingDiagnostics(
                    current, java.util.Map.of("cause", "current observation"));
            context.recordRoutingDiagnostics(current, null);
            assertNull(context.getSchedulingDiagnostics());
            owner.cancel(context, CancelReason.DEADLINE_EXCEEDED);
            var timeout = context.getSchedulingDiagnostics();
            assertEquals("PLACEMENT", timeout.get("stage"));
            assertFalse(timeout.containsKey("lastPlacementObservation"));
            assertFalse(timeout.containsKey("queueObservation"));
            assertTrue(String.valueOf(timeout.get("cause")).contains("request deadline exceeded"));
            assertFalse(String.valueOf(timeout.get("cause")).contains("D1"));
            context.recordRoutingDiagnostics(
                    current, java.util.Map.of("cause", "late current result"));
            assertEquals(timeout, context.getSchedulingDiagnostics());
        } finally {
            current.requestContext()
                    .getRequestScheduler()
                    .finishSelectionWithoutCommit(current, null);
            publisher.close();
        }
    }

    @Test
    void latePlacementDiagnosticsCannotOverwriteFrozenTimeoutEvidence() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 42L);
        AbstractRequestScheduler requestOwner =
                RequestProtocolTestSupport.initialize(
                        mock(ResponseCompletionExecutor.class),
                        context,
                        mock(ExpirationTimer.class));
        var admission = RequestProtocolTestSupport.beginAdmission(requestOwner, context);
        assertNotNull(admission);
        admission
                .requestContext()
                .recordRoutingDiagnostics(
                        admission, java.util.Map.of("cause", "earlier placement"));
        assertEquals("earlier placement", context.getSchedulingDiagnostics().get("cause"));
        requestOwner.cancel(context, CancelReason.DEADLINE_EXCEEDED);
        var timeoutEvidence = context.getSchedulingDiagnostics();
        assertEquals("PLACEMENT", timeoutEvidence.get("stage"));
        assertEquals(
                "earlier placement",
                ((java.util.Map<?, ?>) timeoutEvidence.get("lastPlacementObservation"))
                        .get("cause"));
        assertTrue(
                String.valueOf(timeoutEvidence.get("cause")).contains("request deadline exceeded"));
        assertTrue(String.valueOf(timeoutEvidence.get("cause")).contains("earlier placement"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> timeoutEvidence.put("cause", "overwritten"));
        admission
                .requestContext()
                .recordRoutingDiagnostics(admission, java.util.Map.of("cause", "late placement"));
        assertEquals(timeoutEvidence, context.getSchedulingDiagnostics());
        assertFalse(
                context.getFuture().isDone(),
                "cancellation must still wait for the active admission");
    }

    @Test
    void oldDecodeRetirementCannotRemoveANewSchedulingGeneration() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 4202L);
        AbstractRequestScheduler requestOwner =
                RequestProtocolTestSupport.initialize(
                        mock(ResponseCompletionExecutor.class),
                        context,
                        mock(ExpirationTimer.class));
        RequestProtocolTestSupport.decodeRetired(
                requestOwner,
                context,
                RequestProtocolTestSupport.decodeEndpoint(),
                new DecodeEndpoint.RequestRegistration(1L, String.valueOf(4202L)));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(requestOwner));
        assertTrue((context.snapshot().phase() == Phase.OPEN));
        assertFalse(context.getFuture().isDone());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeRetirementRejectsAnotherSourceOrReservation(boolean wrongSource) {
        Fixture f = fixture(true, true);
        var source =
                wrongSource
                        ? RequestProtocolTestSupport.decodeEndpoint()
                        : f.item().decodeEndpoint();
        var reservation =
                wrongSource
                        ? RESERVATION
                        : new DecodeEndpoint.RequestRegistration(
                                RESERVATION.endpointGenerationId(),
                                String.valueOf(RESERVATION.requestId()));

        RequestProtocolTestSupport.decodeRetired(
                f.scheduler(), f.requestContext(), source, reservation);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.scheduler()));

        assertTrue((f.requestContext().snapshot().phase() == Phase.OPEN));
        assertTrue(f.requestContext().getFuture().isDone());
        assertSame(f.item(), f.requestContext().snapshot().route());
        verify(f.item().prefillEndpoint(), never()).releaseRequest(any());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeEndDuringAdmissionPreservesTheEarlierCancellationCause(boolean cleanup) {
        Fixture f = fixture(false);
        f.scheduler().cancel(f.requestContext(), CancelReason.CLIENT_CANCELLED);
        if (cleanup) {
            RequestProtocolTestSupport.decodeRetired(
                    f.scheduler(), f.requestContext(), f.item().decodeEndpoint(), RESERVATION);
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.scheduler()));
        } else {
            RequestProtocolTestSupport.applyDecodeStatus(
                    f.scheduler(),
                    f.requestContext(),
                    f.item().decodeEndpoint(),
                    DecodeResources.DecodeRequestStatus.terminal(RESERVATION, 0L));
        }
        assertFalse((stage(f.requestContext()) == Stage.FINISHED));
        assertFalse(f.requestContext().getFuture().isDone());
        f.admission()
                .requestContext()
                .getRequestScheduler()
                .finishSelectionWithoutCommit(f.admission(), null);
        assertEquals(
                RequestResult.State.CANCELLED,
                RequestResult.from(f.requestContext().snapshot()).state());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.scheduler()));
        assertTrue((stage(f.requestContext()) == Stage.FINISHED));
        assertFalse(
                RequestProtocolTestSupport.awaitResult(f.requestContext().getFuture()).isSuccess());
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 4, 5})
    void transientTerminalTimerFailureRetriesAfterActualResourceSettlement(int failures) {
        Fixture f = fixture(true, true);
        var context = f.requestContext();
        context.recordEvent(
                new RequestContext.RequestEvent(
                        f.item(),
                        null,
                        null,
                        null,
                        org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED,
                        0L,
                        null,
                        System.currentTimeMillis()));
        var timer =
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.scheduler()).deadlines();
        org.mockito.Mockito.clearInvocations(timer);
        var expected = new IllegalStateException("timer update failed");
        var attempts = new java.util.concurrent.atomic.AtomicInteger();
        var installedRevision =
                new java.util.concurrent.atomic.AtomicLong(
                        context.snapshot().deadlines().revision());
        when(timer.needsUpdate(any(), any()))
                .thenAnswer(
                        call -> {
                            RequestContext owner = call.getArgument(0);
                            ExpirationTimer.Plan plan = call.getArgument(1);
                            return f.scheduler().repository.isCurrent(owner)
                                    && plan.revision() > installedRevision.get();
                        });
        doAnswer(
                        call -> {
                            if (attempts.incrementAndGet() <= failures) {
                                throw expected;
                            }
                            ExpirationTimer.Plan plan = call.getArgument(1);
                            installedRevision.accumulateAndGet(plan.revision(), Math::max);
                            return null;
                        })
                .when(timer)
                .update(any(), any());
        assertSame(
                expected,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                f.scheduler()
                                        .closeRequest(
                                                context,
                                                new RequestEndEvent(
                                                        f.item(),
                                                        EndKind.FAILURE,
                                                        false,
                                                        0L,
                                                        org.flexlb.worker.endpoint.StrategyErrorType
                                                                .DISPATCH_FAILED,
                                                        "original terminal failure"))));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.scheduler()));
        var original = context.snapshot().requestEndResult();
        assertEquals("original terminal failure", original.detail());
        assertTrue(RequestProtocolTestSupport.awaitResult(context.getFuture()).isSuccess());
        if (failures >= 3) {
            assertEquals(3, attempts.get(), "one finish wake-up performs at most three attempts");
            assertEquals(Phase.ENDING, context.snapshot().phase());
            assertFalse(context.snapshot().notificationsDone());
            assertNotNull(
                    context.snapshot().deadlines().inactivity(),
                    "real pending notification keeps its existing watch");
            assertSame(context, f.scheduler().repository.findActive(RESERVATION.requestId()));
            assertNull(f.scheduler().repository.findTerminal(RESERVATION.requestId()));
            verify(f.item().prefillEndpoint(), times(1)).releaseRequest(f.item());
            verify(f.item().decodeEndpoint(), times(1)).releaseRequest(RESERVATION);
            if (failures > 3) {
                assertSame(
                        expected,
                        assertThrows(
                                IllegalStateException.class,
                                () -> f.scheduler().closeRequest(context, null)));
            } else {
                f.scheduler().closeRequest(context, null);
            }
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.scheduler()));
        }
        assertEquals(
                Phase.CLOSED,
                context.snapshot().phase(),
                "recovering notifications finish their existing owner");
        assertSame(
                original,
                context.snapshot().requestEndResult(),
                "retries preserve the frozen outcome and first cause");
        assertTrue(context.snapshot().notificationsDone());
        assertNull(context.snapshot().deadlines().inactivity());
        verify(f.item().prefillEndpoint(), times(1)).releaseRequest(f.item());
        verify(f.item().decodeEndpoint(), times(1)).releaseRequest(RESERVATION);
        verify(timer, times(failures + 1)).update(any(), any());
    }

    private static RequestEvent prefillFailure(Fixture fixture, RoleType role, long observedAtMs) {
        return new RequestEvent(
                fixture.item(),
                RESERVATION,
                role,
                null,
                RequestEventType.FAILED,
                9L,
                null,
                observedAtMs);
    }

    private static boolean contextFinished(RequestContext context) {
        return context.snapshot().phase() == Phase.CLOSED;
    }

    private static Fixture fixture(boolean finishAdmission) {
        return fixture(finishAdmission, false);
    }

    private static Fixture fixture(boolean finishAdmission, boolean queueScheduling) {
        var config = SchedulingTestConfig.newConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        RequestContext context =
                RequestProtocolTestSupport.context(config, Long.parseLong(RESERVATION.requestId()));
        var publisher = mock(ResponseCompletionExecutor.class);
        var timer = mock(ExpirationTimer.class);
        AbstractRequestScheduler requestOwner =
                RequestProtocolTestSupport.initialize(publisher, context, timer);
        when(SchedulerTestSupport.engineClient(requestOwner)
                        .cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenReturn(
                        java.util.concurrent.CompletableFuture.completedFuture(
                                EngineGrpcClient.CancelReply.REQUEST_CLEANED));
        when(publisher.register()).thenReturn(true);
        doAnswer(
                        call -> {
                            call.<Runnable>getArgument(1).run();
                            return null;
                        })
                .when(publisher)
                .submit(any(), any());
        when(publisher.invoke(any()))
                .thenAnswer(
                        call ->
                                call.<java.util.function.BooleanSupplier>getArgument(0)
                                        .getAsBoolean());
        var prefillServer = new org.flexlb.worker.status.ServerStatus();
        prefillServer.setServerIp("127.0.0.1");
        prefillServer.setGrpcPort(8090);
        RequestResult item =
                org.flexlb.balance.schedule.SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        prefillServer,
                        null,
                        mock(PrefillEndpoint.class),
                        RequestProtocolTestSupport.decodeEndpoint(),
                        RESERVATION,
                        System.currentTimeMillis());
        SelectionAttempt admission =
                RequestProtocolTestSupport.beginAdmission(requestOwner, context);
        assertNotNull(admission);
        if (finishAdmission) {
            assertEquals(
                    org.flexlb.balance.strategy.RoutingResult.Status.SUCCESS,
                    RequestProtocolTestSupport.commitRoute(
                            requestOwner,
                            item,
                            RequestProtocolTestSupport.publication(() -> true)));
        } else {
            context.registerSelectedRoute(admission, item);
        }
        return new Fixture(requestOwner, context, item, admission);
    }

    private record Fixture(
            AbstractRequestScheduler scheduler,
            RequestContext requestContext,
            RequestResult item,
            SelectionAttempt admission) {}
}

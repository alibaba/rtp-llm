package org.flexlb.balance.schedule;

import org.flexlb.balance.request.Request;
import org.flexlb.balance.strategy.SelectionResult;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Checks immutable observations, frozen request inputs and exact selection attempts on one context.
 */
class RequestContextConcurrencyTest {
    @Test
    void cancellationSelectsItsReservedResponseInTheSameCall() {
        var context = request();
        var candidate =
                SchedulerTestSupport.responseCandidate(
                        context, RequestContext.ResponseKind.TERMINAL);
        assertTrue(
                context.requestCancel(
                        CancelReason.CLIENT_CANCELLED,
                        false,
                        1100L,
                        StrategyErrorType.REQUEST_CANCELLED,
                        "client cancelled",
                        candidate));
        assertEquals(RequestState.State.CANCELLED, context.snapshot().endState());
        assertSame(candidate, context.snapshot().responseForFuture());
    }

    @Test
    void cancellationWithoutSelectionImmediatelyDeterminesTheResult() {
        var context = request();
        assertTrue(
                context.requestCancel(
                        CancelReason.CLIENT_CANCELLED,
                        false,
                        1100L,
                        StrategyErrorType.REQUEST_CANCELLED,
                        "client cancelled",
                        null));
        assertEquals(RequestContext.Phase.ENDING, context.snapshot().phase());
        assertEquals(RequestState.State.CANCELLED, context.snapshot().endState());
        assertNull(context.snapshot().responseForFuture());
        assertNull(
                org.springframework.test.util.ReflectionTestUtils.getField(
                        context, "pendingEndEvent"));
    }

    @Test
    void selectionExitDeterminesPendingCancellationWithoutAnotherCloseCall() {
        var context = request();
        var attempt = context.beginRouteSelection();
        context.requestCancel(
                CancelReason.CLIENT_CANCELLED,
                false,
                1100L,
                StrategyErrorType.REQUEST_CANCELLED,
                "client cancelled",
                null);
        assertNull(context.snapshot().endState());
        var ended = context.finishSelectionWithoutCommit(attempt, null, 1200L);
        assertEquals(RequestState.State.CANCELLED, ended.endState());
        assertNull(
                org.springframework.test.util.ReflectionTestUtils.getField(
                        context, "pendingEndEvent"));
    }

    @Test
    void capturedViewDoesNotChangeWhenSelectionStartsOrEnds() {
        var context = request();
        var initial = context.snapshot();
        var selecting = context.beginRouteSelection();
        var active = context.snapshot();
        assertEquals(0L, initial.selectionId());
        assertEquals(RequestContext.RoutingPhase.NONE, initial.routingPhase());
        assertEquals(selecting.id(), active.selectionId());
        assertEquals(RequestContext.RoutingPhase.SELECTING, active.routingPhase());
        context.finishSelectionWithoutCommit(selecting, null, 1100L);
        assertEquals(selecting.id(), active.selectionId());
        assertEquals(0L, context.snapshot().selectionId());
        assertEquals(RequestContext.RoutingPhase.NONE, context.snapshot().routingPhase());
    }

    @Test
    void obsoleteSelectionCannotCompleteTheCurrentAttemptOrAnotherRequest() {
        var context = request();
        var first = context.beginRouteSelection();
        context.finishSelectionWithoutCommit(first, null, 1100L);
        var second = context.beginRouteSelection();
        var before = context.snapshot();
        assertTrue(second.id() > first.id());
        assertNull(context.finishSelectionWithoutCommit(first, null, 1200L));
        assertNull(
                context.finishSelectionWithoutCommit(
                        first,
                        org.flexlb.balance.request.Response.buildErrorResponse(
                                org.flexlb.worker.endpoint.StrategyErrorType.DISPATCH_FAILED,
                                "selection failed"),
                        1200L));
        var foreign = new RequestContext.SelectionAttempt(request(), second.id());
        assertNull(context.finishSelectionWithoutCommit(foreign, null, 1200L));
        assertEquals(before, context.snapshot());
    }

    @Test
    void registrationFreezesInputsOnTheSameRequestObject() {
        var context = new RequestContext(new FlexlbConfig());
        context.setRequest(Request.builder().requestId("request").seqLen(100L).build());
        context.setSchedulingMetadata(SchedulingMetadata.of(50, null, Long.MAX_VALUE, 50));
        var input = context.getRequest();
        var metadata = context.getSchedulingMetadata();
        var future =
                new ResponseCompletionExecutor.RequestFuture((kind, response, failure) -> false);
        context.register(
                mock(AbstractRequestScheduler.class),
                future,
                System.currentTimeMillis(),
                60000L,
                StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        assertSame(input, context.getRequest());
        assertSame(metadata, context.getSchedulingMetadata());
        assertSame(future, context.getFuture());
        assertEquals("request", context.getRequestId());
        assertThrows(
                IllegalStateException.class,
                () -> context.setRequest(input.toBuilder().requestId("other").build()));
        assertThrows(
                IllegalStateException.class,
                () -> context.setSchedulingMetadata(SchedulingMetadata.of(70, null, 60000L, 50)));
        assertThrows(
                IllegalStateException.class,
                () -> context.setPhase(org.flexlb.balance.schedule.RequestPhase.ENCODER));
        assertSame(input, context.getRequest());
    }

    @Test
    void cancellationBeforeCommitRejectsTheRouteAndRollbackReleasesIt() {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRoute(context, attempt);
        assertTrue(
                context.requestCancel(
                        CancelReason.CLIENT_CANCELLED,
                        false,
                        1100L,
                        StrategyErrorType.REQUEST_CANCELLED,
                        "client cancelled",
                        null));
        assertFalse(
                context.tryCommitRoute(
                        attempt,
                        route,
                        SchedulerTestSupport.responseCandidate(
                                context, RequestContext.ResponseKind.ACK),
                        1200L));
        assertNull(context.snapshot().responseForFuture());
        assertSame(route, context.snapshot().route());
        context.recordCleanupProgress(route, true, true, false, 1201L);
        context.finishSelectionWithoutCommit(attempt, null, 1202L);
        assertNull(context.snapshot().route());
        var ended =
                context.endRequest(
                        null,
                        1203L,
                        SchedulerTestSupport.responseCandidate(
                                context, RequestContext.ResponseKind.TERMINAL));
        assertEquals(RequestState.State.CANCELLED, ended.endState());
        context.recordCleanupProgress(context.snapshot().route(), false, false, true, 1204L);
        assertEquals(RequestContext.Phase.CLOSED, context.snapshot().phase());
    }

    @Test
    void committedResponseSurvivesCancellationBeforeItsCallbackRuns() throws Exception {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRoute(context, attempt);
        var future = new ResponseCompletionExecutor.RequestFuture((kind, value, error) -> false);
        try (var executor = new ResponseCompletionExecutor(1)) {
            var completion =
                    new ResponseCompletionExecutor.Registration(
                            executor, context, future, RequestContext.ResponseKind.ACK);
            synchronized (context) {
                assertTrue(context.tryCommitRoute(attempt, route, completion, 1100L));
                completion.prepare(
                        ResponseCompletionExecutor.RequestFuture.Completion.RESPONSE,
                        new org.flexlb.balance.request.Response(),
                        null);
            }
            assertFalse(future.isDone());
            assertNull(context.currentSelectionAttempt());
            assertNull(context.snapshot().deadlines().request());
            assertTrue(
                    context.requestCancel(
                            CancelReason.CLIENT_CANCELLED,
                            false,
                            1200L,
                            StrategyErrorType.REQUEST_CANCELLED,
                            "cancel execution",
                            null));
            assertSame(
                    completion,
                    context.endRequest(
                                    null,
                                    1200L,
                                    SchedulerTestSupport.responseCandidate(
                                            context, RequestContext.ResponseKind.CANCELLATION))
                            .responseForFuture());
            future.thenRun(() -> assertFalse(Thread.holdsLock(context)));
            completion.publish(false);
            future.get(5, java.util.concurrent.TimeUnit.SECONDS);
            assertSame(completion, context.snapshot().responseForFuture());
            assertSame(route, context.snapshot().committedRoute());
        }
    }

    @Test
    void partialRollbackKeepsTheSameRouteUntilRemainingResourcesAreReleased() {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRoute(context, attempt);
        context.recordCleanupProgress(route, true, false, false, 1100L);
        assertSame(route, context.snapshot().route());
        context.finishSelectionWithoutCommit(
                attempt,
                org.flexlb.balance.request.Response.buildErrorResponse(
                        org.flexlb.worker.endpoint.StrategyErrorType.DISPATCH_FAILED,
                        "selection failed"),
                1200L);
        context.endRequest(
                RequestContext.RequestEndEvent.failure(
                        route,
                        RequestContext.EndKind.FAILURE,
                        StrategyErrorType.DISPATCH_FAILED,
                        "rollback failed"),
                1201L,
                SchedulerTestSupport.responseCandidate(
                        context, RequestContext.ResponseKind.TERMINAL));
        context.recordCleanupProgress(context.snapshot().route(), false, false, true, 1202L);
        assertEquals(RequestContext.Phase.ENDING, context.snapshot().phase());
        context.recordCleanupProgress(route, true, true, false, 1300L);
        assertEquals(RequestContext.Phase.CLOSED, context.snapshot().phase());
        var finished = context.snapshot();
        context.recordCleanupProgress(route, true, true, false, 1400L);
        assertEquals(finished, context.snapshot());
    }

    @Test
    void expiredSelectionCannotChooseASuccessResponse() {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRoute(context, attempt);
        assertFalse(
                context.tryCommitRoute(
                        attempt,
                        route,
                        SchedulerTestSupport.responseCandidate(
                                context, RequestContext.ResponseKind.ACK),
                        61000L));
        assertNull(context.snapshot().responseForFuture());
        assertEquals(CancelReason.DEADLINE_EXCEEDED, context.snapshot().cancellationReason());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void schedulerCommitsOrRollsBackThroughOneResponseDecision(boolean cancelBeforeCommit)
            throws Exception {
        var repository = new RequestRepository();
        try (var completions = new ResponseCompletionExecutor(1)) {
            var scheduler =
                    mock(
                            AbstractRequestScheduler.class,
                            org.mockito.Mockito.withSettings()
                                    .useConstructor(
                                            repository,
                                            mock(RequestTaskPool.class),
                                            mock(ExpirationTimer.class),
                                            mock(EngineCancellation.class),
                                            completions,
                                            mock(
                                                    org.flexlb.balance.metrics
                                                            .DeliveryMetricsReporter.class),
                                            mock(
                                                    org.flexlb.balance.requestlog
                                                            .TheoryCacheHitReporter.class),
                                            mock(EndpointRegistry.class))
                                    .defaultAnswer(org.mockito.Mockito.CALLS_REAL_METHODS));
            var context = org.mockito.Mockito.spy(new RequestContext(new FlexlbConfig()));
            context.setRequest(Request.builder().requestId("commit-test").seqLen(100L).build());
            context.setSchedulingMetadata(SchedulingMetadata.of(50, null, Long.MAX_VALUE, 50));
            var future =
                    new ResponseCompletionExecutor.RequestFuture(
                            (kind, response, failure) ->
                                    scheduler.tryCompleteFromExternal(
                                            context, kind, response, failure, true));
            assertEquals(
                    RequestRepository.RegistrationResult.REGISTERED,
                    repository.register(
                            context,
                            scheduler,
                            future,
                            60000L,
                            StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED));
            var attempt = context.beginRouteSelection();
            var route = mock(RequestRoute.class);
            var prefill = mock(PrefillEndpoint.class);
            var response = new org.flexlb.balance.request.Response();
            response.setSuccess(true);
            org.mockito.Mockito.when(route.ctx()).thenReturn(context);
            org.mockito.Mockito.when(route.requestId()).thenReturn("commit-test");
            org.mockito.Mockito.when(route.prefillEndpoint()).thenReturn(prefill);
            org.mockito.Mockito.when(route.selectedEndpointsAvailable()).thenReturn(true);
            org.mockito.Mockito.when(route.successResponse()).thenReturn(response);
            org.mockito.Mockito.when(route.validateEndpoints(org.mockito.ArgumentMatchers.any()))
                    .thenAnswer(call -> call.getArgument(0));
            org.mockito.Mockito.when(prefill.commitSelection(route))
                    .thenAnswer(
                            call -> {
                                if (cancelBeforeCommit)
                                    context.requestCancel(
                                            CancelReason.CLIENT_CANCELLED,
                                            false,
                                            System.currentTimeMillis(),
                                            StrategyErrorType.REQUEST_CANCELLED,
                                            "cancel before commit",
                                            null);
                                return true;
                            });
            var submitted =
                    scheduler.executeRouteSelection(attempt, () -> SelectionResult.success(route));
            org.mockito.Mockito.verify(context).registerSelectedRoute(attempt, route);
            assertEquals(
                    cancelBeforeCommit
                            ? org.flexlb.balance.strategy.SelectionResult.Status.CLOSED
                            : org.flexlb.balance.strategy.SelectionResult.Status.SUCCESS,
                    submitted.status());
            var received = future.get(5, java.util.concurrent.TimeUnit.SECONDS);
            assertEquals(!cancelBeforeCommit, received.isSuccess());
            assertNull(context.currentSelectionAttempt());
            if (cancelBeforeCommit) {
                org.mockito.Mockito.verify(prefill).releaseRequest(route.prefillRegistration());
                assertNull(repository.findActive("commit-test"));
            } else {
                assertSame(response, received);
                assertSame(route, context.snapshot().committedRoute());
                org.mockito.Mockito.verify(prefill, org.mockito.Mockito.never())
                        .releaseRequest(route.prefillRegistration());
            }
        }
    }

    @Test
    void workerFeedbackAtTheSameTimeStillChangesTheVisibleDeadlinePlan() {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRoute(context, attempt);
        org.mockito.Mockito.when(route.decodeEndpoint()).thenReturn(mock(DecodeEndpoint.class));
        assertTrue(
                context.tryCommitRoute(
                        attempt,
                        route,
                        SchedulerTestSupport.responseCandidate(
                                context, RequestContext.ResponseKind.ACK),
                        1000L));
        var before = context.snapshot();
        var activePrefill =
                new RequestContext.RequestEvent(
                        route,
                        null,
                        RoleType.PREFILL,
                        null,
                        org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                        0L,
                        "prefill active",
                        1000L);
        assertFalse(context.recordEvent(activePrefill));
        assertFalse(context.recordEvent(activePrefill));
        assertEquals(before, context.snapshot());
        var needsFollowUpprefillFinished =
                context.recordEvent(
                        new RequestContext.RequestEvent(
                                route,
                                null,
                                org.flexlb.config.service.RoleType.PREFILL,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.COMPLETED,
                                0L,
                                "prefill finished",
                                1000L));
        assertTrue(needsFollowUpprefillFinished);
        var prefillFinished = context.snapshot();
        assertNull(before.deadlines().visibility());
        assertEquals(11000L, prefillFinished.deadlines().visibility().atMs());
        assertTrue(prefillFinished.prefillCleanupCompleted());
        assertFalse(prefillFinished.decodeCleanupCompleted());
        assertFalse(
                context.recordEvent(
                        new RequestContext.RequestEvent(
                                route,
                                null,
                                RoleType.PREFILL,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.COMPLETED,
                                0L,
                                "prefill finished",
                                1000L)));
        assertEquals(prefillFinished, context.snapshot());
        var needsFollowUpdecodeActive =
                context.recordEvent(
                        new RequestContext.RequestEvent(
                                route,
                                route.decodeRegistration(),
                                org.flexlb.config.service.RoleType.DECODE,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.ALLOCATED,
                                0L,
                                "decode allocated",
                                1000L));
        assertTrue(needsFollowUpdecodeActive);
        var decodeActive = context.snapshot();
        assertFalse(
                context.recordEvent(
                        new RequestContext.RequestEvent(
                                route,
                                route.decodeRegistration(),
                                org.flexlb.config.service.RoleType.DECODE,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.ALLOCATED,
                                0L,
                                "decode allocated",
                                1000L)));
        assertEquals(decodeActive, context.snapshot());
        assertTrue(decodeActive.decodeAccepted());
        assertNull(decodeActive.deadlines().visibility());
        assertEquals(11000L, prefillFinished.deadlines().visibility().atMs());
        assertEquals(before.updatedAtMs(), decodeActive.updatedAtMs());
    }

    @Test
    void endpointClosureRequiresItsSourceAndDoesNotRenewActivity() {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRoute(context, attempt);
        var endpoint = mock(DecodeEndpoint.class);
        org.mockito.Mockito.when(route.decodeEndpoint()).thenReturn(endpoint);
        assertTrue(
                context.tryCommitRoute(
                        attempt,
                        route,
                        SchedulerTestSupport.responseCandidate(
                                context, RequestContext.ResponseKind.ACK),
                        1000L));
        var before = context.snapshot();
        assertFalse(
                context.recordEvent(
                        new RequestContext.RequestEvent(
                                route,
                                route.decodeRegistration(),
                                org.flexlb.config.service.RoleType.DECODE,
                                mock(DecodeEndpoint.class),
                                org.flexlb.worker.endpoint.RequestEventType.ENDPOINT_CLOSED,
                                0L,
                                "closed",
                                2000L)));
        assertEquals(before, context.snapshot());
        var closed =
                new RequestContext.RequestEvent(
                        route,
                        route.decodeRegistration(),
                        org.flexlb.config.service.RoleType.DECODE,
                        endpoint,
                        org.flexlb.worker.endpoint.RequestEventType.ENDPOINT_CLOSED,
                        0L,
                        "closed",
                        2000L);
        assertTrue(context.recordEvent(closed));
        var after = context.snapshot();
        assertEquals(before.inactivityExpiresAtMs(), after.inactivityExpiresAtMs());
        assertEquals(RequestContext.EndKind.ENDPOINT_CLOSED, after.endKind());
        assertFalse(context.recordEvent(closed));
        assertEquals(after, context.snapshot());
    }

    @Test
    void localCleanupAndWorkerReleaseRemainSeparateUntilTheRequestCanBeArchived() {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRoute(context, attempt);
        var response =
                SchedulerTestSupport.responseCandidate(context, RequestContext.ResponseKind.ACK);
        assertTrue(context.tryCommitRoute(attempt, route, response, 1000L));
        assertTrue(
                context.requestCancel(
                        CancelReason.CLIENT_CANCELLED,
                        false,
                        1000L,
                        StrategyErrorType.REQUEST_CANCELLED,
                        "client cancelled",
                        null));
        var ending = context.endRequest(null, 1000L, null);
        assertEquals(RequestState.State.CANCELLED, ending.endState());
        assertSame(response, ending.responseForFuture());
        context.recordCleanupProgress(route, true, true, false, 1000L);
        var locallyCleaned = context.snapshot();
        assertTrue(locallyCleaned.decodeCleanupCompleted());
        assertFalse(locallyCleaned.decodeWorkerReleaseConfirmed());
        assertNull(locallyCleaned.requestRecordForArchive());
        assertFalse(
                context.recordEvent(
                        new RequestContext.RequestEvent(
                                mock(RequestRoute.class),
                                null,
                                null,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED,
                                0L,
                                null,
                                2000L)));
        assertEquals(locallyCleaned, context.snapshot());
        var fenced =
                new RequestContext.RequestEvent(
                        route,
                        null,
                        null,
                        null,
                        org.flexlb.worker.endpoint.RequestEventType.REMOTE_FENCED,
                        0L,
                        null,
                        2000L);
        assertTrue(context.recordEvent(fenced));
        var afterFence = context.snapshot();
        assertTrue(afterFence.prefillWorkerReleaseConfirmed());
        assertFalse(afterFence.decodeWorkerReleaseConfirmed());
        assertEquals(locallyCleaned.inactivityExpiresAtMs(), afterFence.inactivityExpiresAtMs());
        assertFalse(context.recordEvent(fenced));
        assertEquals(afterFence, context.snapshot());
        var cleanupNeedsFollowUp =
                context.recordEvent(
                        new RequestContext.RequestEvent(
                                route,
                                null,
                                null,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED,
                                0L,
                                null,
                                1000L));
        assertTrue(cleanupNeedsFollowUp);
        var confirmed = context.snapshot();
        assertTrue(confirmed.decodeWorkerReleaseConfirmed());
        assertFalse(locallyCleaned.decodeWorkerReleaseConfirmed());
        context.recordCleanupProgress(context.snapshot().route(), false, false, true, 1000L);
        var finished = context.snapshot();
        assertEquals(RequestContext.Phase.CLOSED, finished.phase());
        assertEquals(RequestState.State.CANCELLED, finished.requestRecordForArchive().state());
        assertNull(finished.route());
        assertSame(route, locallyCleaned.route());
        assertSame(response, finished.responseForFuture());
    }

    @Test
    void remoteCleanupReportsFollowUpWithoutCreatingASnapshot() {
        var context = org.mockito.Mockito.spy(request());
        var attempt = context.beginRouteSelection();
        var route = acquiredRoute(context, attempt);
        assertTrue(
                context.tryCommitRoute(
                        attempt,
                        route,
                        SchedulerTestSupport.responseCandidate(
                                context, RequestContext.ResponseKind.ACK),
                        1000L));
        assertTrue(
                context.requestCancel(
                        CancelReason.CLIENT_CANCELLED,
                        false,
                        1000L,
                        StrategyErrorType.REQUEST_CANCELLED,
                        "client cancelled",
                        null));
        context.endRequest(null, 1000L, null);
        org.mockito.Mockito.clearInvocations(context);
        var cleanup =
                new RequestContext.RequestEvent(
                        route,
                        null,
                        null,
                        null,
                        org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED,
                        0L,
                        null,
                        1000L);

        assertTrue(
                context.recordEvent(cleanup),
                "new release evidence must request follow-up actions");
        assertFalse(
                context.recordEvent(cleanup),
                "duplicate evidence must not repeat follow-up actions");
        org.mockito.Mockito.verify(context, org.mockito.Mockito.never()).snapshot();
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(
            value = org.flexlb.worker.endpoint.RequestEventType.class,
            names = {"REMOTE_CLEANED", "REMOTE_FENCED"})
    void remoteCleanupOnlyConfirmsResourcesSupportedByItsEvidence(
            org.flexlb.worker.endpoint.RequestEventType type) {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRoute(context, attempt);
        var response =
                SchedulerTestSupport.responseCandidate(context, RequestContext.ResponseKind.ACK);
        assertTrue(context.tryCommitRoute(attempt, route, response, 1000L));
        var before = context.snapshot();
        var cleanup =
                new RequestContext.RequestEvent(route, null, null, null, type, 0L, null, 2000L);

        assertFalse(context.recordEvent(cleanup));

        var after = context.snapshot();
        assertTrue(after.prefillWorkerReleaseConfirmed());
        assertEquals(
                type == org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED,
                after.decodeWorkerReleaseConfirmed());
        assertFalse(after.prefillCleanupCompleted());
        assertFalse(after.decodeCleanupCompleted());
        assertNull(after.endState());
        assertSame(response, after.responseForFuture());
        assertSame(route, after.committedRoute());
        assertSame(before.deadlines(), after.deadlines());
        assertEquals(before.inactivityExpiresAtMs(), after.inactivityExpiresAtMs());
        assertFalse(context.recordEvent(cleanup));
        assertEquals(after, context.snapshot());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void remoteCleanupIgnoresUncommittedAndForeignRoutes(boolean committed) {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRoute(context, attempt);
        var foreign = mock(RequestRoute.class);
        org.mockito.Mockito.when(foreign.ctx()).thenReturn(context);
        org.mockito.Mockito.when(foreign.requestId()).thenReturn(context.getRequestId());
        if (committed) {
            assertTrue(
                    context.tryCommitRoute(
                            attempt,
                            route,
                            SchedulerTestSupport.responseCandidate(
                                    context, RequestContext.ResponseKind.ACK),
                            1000L));
        }
        var before = context.snapshot();
        assertFalse(
                context.recordEvent(
                        new RequestContext.RequestEvent(
                                committed ? foreign : route,
                                null,
                                null,
                                null,
                                org.flexlb.worker.endpoint.RequestEventType.REMOTE_CLEANED,
                                0L,
                                null,
                                1000L)));
        assertEquals(before, context.snapshot());
    }

    @ParameterizedTest
    @CsvSource({
        "PREFILL, FAILED", "PREFILL, ENDPOINT_CLOSED",
        "DECODE, FAILED", "DECODE, ENDPOINT_CLOSED",
        "PDFUSION, FAILED", "PDFUSION, ENDPOINT_CLOSED",
        "ENCODER, FAILED", "ENCODER, ENDPOINT_CLOSED"
    })
    void unpublishedRollbackRetriesClosureButPreservesWorkerFailure(
            RoleType role, org.flexlb.worker.endpoint.RequestEventType type) {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRouteForEvent(context, attempt, role);
        var endpoint =
                role == RoleType.DECODE
                        ? route.decodeEndpoint()
                        : role == RoleType.ENCODER
                                ? route.encoderEndpoint()
                                : route.prefillEndpoint();
        var event =
                new RequestContext.RequestEvent(
                        route,
                        route.decodeRegistration(),
                        role,
                        endpoint,
                        type,
                        42L,
                        "source ended",
                        1100L);
        assertTrue(context.recordEvent(event));
        var waiting = context.endRequest(null, 1100L, null);
        assertNull(waiting.endState());
        assertEquals(attempt.id(), waiting.selectionId());
        assertSame(route, waiting.route());

        context.recordCleanupProgress(route, true, true, false, 1200L);
        context.finishSelectionWithoutCommit(attempt, null, 1200L);
        var ended = context.endRequest(null, 1200L, null);
        assertNull(ended.route());
        assertTrue(ended.prefillCleanupCompleted());
        assertTrue(ended.decodeCleanupCompleted());
        assertNull(ended.responseForFuture());
        if (type == org.flexlb.worker.endpoint.RequestEventType.ENDPOINT_CLOSED) {
            assertNull(ended.endState());
            assertEquals(RequestContext.Phase.OPEN, ended.phase());
            var retry = context.beginRouteSelection();
            assertTrue(retry != null && retry.id() != attempt.id());
        } else {
            assertEquals(RequestState.State.FAILED, ended.endState());
            assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED, ended.endError());
            assertEquals(RequestContext.EndKind.WORKER, ended.endKind());
            assertNull(context.beginRouteSelection());
        }
        var beforeDuplicate = context.snapshot();
        assertFalse(context.recordEvent(event));
        assertEquals(beforeDuplicate, context.snapshot());
    }

    @ParameterizedTest
    @CsvSource({
        "PDFUSION, FAILED", "PDFUSION, ENDPOINT_CLOSED",
        "ENCODER, FAILED", "ENCODER, ENDPOINT_CLOSED"
    })
    void committedSingleStageRequestEndsOnWorkerFailureOrEndpointClosure(
            RoleType role, org.flexlb.worker.endpoint.RequestEventType type) {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRouteForEvent(context, attempt, role);
        var response =
                SchedulerTestSupport.responseCandidate(context, RequestContext.ResponseKind.ACK);
        assertTrue(context.tryCommitRoute(attempt, route, response, 1000L));
        var endpoint = role == RoleType.ENCODER ? route.encoderEndpoint() : route.prefillEndpoint();

        context.recordEvent(
                new RequestContext.RequestEvent(
                        route, null, role, endpoint, type, 42L, "source ended", 1100L));
        var ended = context.endRequest(null, 1100L, null);

        org.junit.jupiter.api.Assertions.assertNotNull(
                ended.endState(),
                "a committed single-stage request cannot continue after its execution endpoint ends");
        assertEquals(RequestState.State.FAILED, ended.endState());
        assertEquals(
                type == org.flexlb.worker.endpoint.RequestEventType.ENDPOINT_CLOSED
                        ? RequestContext.EndKind.ENDPOINT_CLOSED
                        : RequestContext.EndKind.WORKER,
                ended.endKind());
        assertSame(response, ended.responseForFuture());
        assertTrue(ended.prefillCleanupCompleted());
        assertTrue(ended.decodeCleanupCompleted());
        assertNull(context.beginRouteSelection());
    }

    @ParameterizedTest
    @EnumSource(
            value = org.flexlb.worker.endpoint.RequestEventType.class,
            names = {"COMPLETED", "FAILED"})
    void committedPrefillCompletionWaitsForDecodeButFailureEndsTheRequest(
            org.flexlb.worker.endpoint.RequestEventType type) {
        var context = request();
        var attempt = context.beginRouteSelection();
        var route = acquiredRouteForEvent(context, attempt, RoleType.PREFILL);
        assertTrue(
                context.tryCommitRoute(
                        attempt,
                        route,
                        SchedulerTestSupport.responseCandidate(
                                context, RequestContext.ResponseKind.ACK),
                        1000L));

        context.recordEvent(
                new RequestContext.RequestEvent(
                        route,
                        null,
                        RoleType.PREFILL,
                        route.prefillEndpoint(),
                        type,
                        type == org.flexlb.worker.endpoint.RequestEventType.FAILED ? 42L : 0L,
                        "prefill ended",
                        1100L));
        var afterPrefill = context.endRequest(null, 1100L, null);
        assertTrue(afterPrefill.prefillCleanupCompleted());
        assertFalse(afterPrefill.decodeCleanupCompleted());
        if (type == org.flexlb.worker.endpoint.RequestEventType.FAILED) {
            assertEquals(RequestState.State.FAILED, afterPrefill.endState());
        } else {
            assertNull(afterPrefill.endState());
            assertEquals(RequestContext.Phase.OPEN, afterPrefill.phase());
            context.recordEvent(
                    new RequestContext.RequestEvent(
                            route,
                            route.decodeRegistration(),
                            RoleType.DECODE,
                            route.decodeEndpoint(),
                            org.flexlb.worker.endpoint.RequestEventType.COMPLETED,
                            0L,
                            "decode finished",
                            1200L));
            assertEquals(
                    RequestState.State.COMPLETED, context.endRequest(null, 1200L, null).endState());
        }
    }

    private static RequestRoute acquiredRouteForEvent(
            RequestContext context, RequestContext.SelectionAttempt attempt, RoleType role) {
        if (role == RoleType.PREFILL || role == RoleType.DECODE) {
            var route = acquiredRoute(context, attempt);
            org.mockito.Mockito.when(route.decodeEndpoint()).thenReturn(mock(DecodeEndpoint.class));
            return route;
        }
        var route = mock(RequestRoute.class);
        org.mockito.Mockito.when(route.ctx()).thenReturn(context);
        org.mockito.Mockito.when(route.requestId()).thenReturn(context.getRequestId());
        if (role == RoleType.ENCODER) {
            org.mockito.Mockito.when(route.encoderEndpoint())
                    .thenReturn(mock(EncoderEndpoint.class));
        } else {
            org.mockito.Mockito.when(route.prefillEndpoint())
                    .thenReturn(mock(PrefillEndpoint.class));
        }
        context.registerSelectedRoute(attempt, route);
        return route;
    }

    private static RequestRoute acquiredRoute(
            RequestContext context, RequestContext.SelectionAttempt attempt) {
        var requestId = context.getRequestId();
        var route = mock(RequestRoute.class);
        org.mockito.Mockito.when(route.ctx()).thenReturn(context);
        org.mockito.Mockito.when(route.requestId()).thenReturn(requestId);
        org.mockito.Mockito.when(route.prefillEndpoint()).thenReturn(mock(PrefillEndpoint.class));
        org.mockito.Mockito.when(route.decodeRegistration())
                .thenReturn(new DecodeEndpoint.RequestRegistration(1L, requestId));
        context.registerSelectedRoute(attempt, route);
        return route;
    }

    private static RequestContext request() {
        var context = new RequestContext(new FlexlbConfig());
        context.setRequest(Request.builder().requestId("request").seqLen(100L).build());
        context.setSchedulingMetadata(SchedulingMetadata.of(50, null, Long.MAX_VALUE, 50));
        context.setFuture(new java.util.concurrent.CompletableFuture<>());
        RequestProtocolTestSupport.register(
                context,
                "request",
                1000L,
                Long.MAX_VALUE,
                60000L,
                StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        return context;
    }
}

package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.RequestEventType;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerRequestEvent;
import org.junit.jupiter.api.Test;

import static org.flexlb.balance.schedule.RequestContext.EndCause;
import static org.flexlb.balance.schedule.RequestContext.Phase;
import static org.flexlb.balance.schedule.RequestContext.RequestEndEvent;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Cleanup passes preserve partial resource release and archive only after resources and notifications complete.
 */
class RequestCleanupPassTest {
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
        "WORKER_REPORT_TIMEOUT, TIMED_OUT",
        "REQUEST_DEADLINE_EXCEEDED, TIMED_OUT",
        "ROUTE_PUBLICATION_FAILED, FAILED",
        "EXTERNAL_RESPONSE_FAILED, FAILED",
        "SCHEDULER_SHUTDOWN, FAILED"
    })
    void endingCauseDeterminesResultWithoutASuccessFlag(
            EndCause cause, RequestState.State expected) {
        var context = openContext(7400L);
        var ended =
                context.endRequest(
                        new RequestEndEvent(context.snapshot().route(), cause, null, "ended"),
                        1001L,
                        null);
        assertEquals(cause, ended.endCause());
        assertEquals(expected, ended.endState());
    }

    static RequestContext openContext(long id) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        RequestProtocolTestSupport.register(
                context,
                String.valueOf(id),
                1000L,
                100_000L,
                10_000L,
                org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        var route = mock(RequestRoute.class);
        when(route.ctx()).thenReturn(context);
        when(route.requestId()).thenReturn(String.valueOf(id));
        var decode = SchedulerTestSupport.mockDecodeEndpoint();
        when(route.decodeEndpoint()).thenReturn(decode);
        when(route.decodeRegistration())
                .thenReturn(new DecodeEndpoint.RequestRegistration(1L, String.valueOf(id)));
        var placement = requireValue(context.beginRouteSelection());
        org.junit.jupiter.api.Assertions.assertTrue(
                RequestProtocolTestSupport.commitSelectedRoute(context, placement, route));
        org.flexlb.balance.schedule.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.finishSelectionWithoutCommit(placement, null, 1000L));
        return context;
    }

    static <T> T requireValue(T value) {
        assertNotNull(value);
        return value;
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
        "COMPLETED, COMPLETED",
        "FAILED, FAILED",
        "PRIORITY_CANCELLED, CANCELLED",
        "ENDPOINT_CLOSED, FAILED"
    })
    void workerEventDeterminesResult(RequestEventType type, RequestState.State expected) {
        var context = openContext(7402L);
        reportWorkerEnd(context, type, null);
        var ended = context.snapshot();
        assertEquals(WorkerEventTestSupport.endCause(type), ended.endCause());
        assertEquals(expected, ended.endState());
        if (type == RequestEventType.FAILED) {
            assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED, ended.endError());
            assertEquals("worker error code 42", ended.endDetail());
        }
    }

    @Test
    void cancellationKeepsItsResultWhenWorkerCompletionArrives() {
        var context = openContext(7401L);
        context.requestCancel(
                CancelReason.CLIENT_CANCELLED,
                false,
                1001L,
                StrategyErrorType.REQUEST_CANCELLED,
                "client cancelled",
                null);
        reportWorkerEnd(context, RequestEventType.COMPLETED, "worker completed");
        var ended = context.snapshot();
        assertEquals(RequestState.State.CANCELLED, ended.endState());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED, ended.endError());
    }

    @Test
    void responseSelectedAfterCleanupImmediatelyArchivesTheRequest() {
        var context = new RequestContext(new FlexlbConfig());
        context.setRequest(
                org.flexlb.balance.request.Request.builder().requestId("late-response").build());
        context.setSchedulingMetadata(
                org.flexlb.balance.schedule.SchedulingMetadata.of(0, null, 100_000L, 50));
        context.register(
                mock(AbstractRequestScheduler.class),
                new ResponseCompletionExecutor.RequestFuture((kind, response, error) -> false),
                1000L,
                10_000L,
                StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        context.requestCancel(
                CancelReason.CLIENT_CANCELLED,
                false,
                1001L,
                StrategyErrorType.REQUEST_CANCELLED,
                "cancelled",
                null);
        context.recordCleanupProgress(null, true, true, true, 1002L);
        assertEquals(Phase.ENDING, context.snapshot().phase());
        assertNotNull(context.snapshot().deadlines().inactivity());

        var response =
                SchedulerTestSupport.responseCandidate(
                        context, RequestContext.ResponseKind.TERMINAL);
        var closed = context.endRequest(null, 1003L, response);
        assertEquals(Phase.CLOSED, closed.phase());
        assertSame(response, closed.responseForFuture());
        assertEquals(RequestState.State.CANCELLED, closed.requestRecordForArchive().state());
        assertNull(closed.deadlines().inactivity());
        assertNull(closed.deadlines().request());
        assertNull(closed.deadlines().visibility());
    }

    @Test
    void combinedProgressArchivesOnlyAfterEveryCleanupObligationCompletes() {
        var context = openContext(7302L);
        var route = context.snapshot().route();
        ended(
                context.endRequest(
                        RequestEndEvent.failure(
                                route,
                                EndCause.ROUTE_PUBLICATION_FAILED,
                                StrategyErrorType.DISPATCH_FAILED,
                                "ended"),
                        1001L,
                        null));
        context.recordCleanupProgress(route, true, false, true, 1002L);
        var partial = context.snapshot();
        assertTrue(partial.prefillCleanupCompleted());
        assertTrue(partial.notificationsDone());
        assertFalse(partial.decodeCleanupCompleted());
        assertEquals(Phase.ENDING, partial.phase());
        context.recordCleanupProgress(mock(RequestRoute.class), false, true, true, 1003L);
        assertEquals(partial, context.snapshot());
        context.recordCleanupProgress(route, false, true, false, 1004L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
        assertNotNull(context.snapshot().requestRecordForArchive());
    }

    static RequestState.State ended(RequestContext.Snapshot snapshot) {
        assertNotNull(snapshot.endState());
        return snapshot.endState();
    }

    @Test
    void partialCompletionRetainsProofAndCannotRewriteAnArchivedRequest() {
        RequestContext context = endingContext(7301L);
        var route = context.snapshot().route();
        assertEquals(Phase.ENDING, context.snapshot().phase());
        var beforeForeignRelease = context.snapshot();
        context.recordCleanupProgress(mock(RequestRoute.class), true, true, false, 1002L);
        assertEquals(
                beforeForeignRelease,
                context.snapshot(),
                "another route cannot contribute resource release proof");
        context.recordCleanupProgress(route, true, false, false, 1002L);
        assertTrue(context.snapshot().prefillCleanupCompleted());
        assertFalse(context.snapshot().decodeCleanupCompleted());
        context.recordCleanupProgress(route, false, false, false, 1004L);
        assertTrue(
                context.snapshot().prefillCleanupCompleted(),
                "completed physical proof is monotonic");
        assertFalse(context.snapshot().decodeCleanupCompleted());
        context.recordCleanupProgress(route, true, true, false, 1005L);
        var closed = context.snapshot();
        assertEquals(Phase.CLOSED, closed.phase());
        assertNotNull(closed.requestRecordForArchive());
        assertNull(context.snapshot().route());
        context.recordCleanupProgress(route, true, true, false, 1006L);
        assertEquals(closed, context.snapshot());
        assertSame(closed.requestRecordForArchive(), context.snapshot().requestRecordForArchive());
    }

    static RequestContext endingContext(long id) {
        var context = openContext(id);
        ended(
                context.endRequest(
                        new RequestEndEvent(
                                context.snapshot().route(),
                                EndCause.ROUTE_PUBLICATION_FAILED,
                                StrategyErrorType.DISPATCH_FAILED,
                                "cleanup required"),
                        1001L,
                        null));
        context.recordCleanupProgress(context.snapshot().route(), false, false, true, 1001L);
        return context;
    }

    @Test
    void cancellationBeforePublicationLeavesCandidateWithProducer() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        RequestProtocolTestSupport.register(
                context,
                "991",
                1000L,
                100_000L,
                10_000L,
                org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        var operation = requireValue(context.beginRouteSelection());
        var candidate = route(context, 991L);
        context.requestCancel(
                CancelReason.CLIENT_CANCELLED,
                false,
                1001L,
                StrategyErrorType.REQUEST_CANCELLED,
                "original cancellation",
                null);
        assertFalse(RequestProtocolTestSupport.commitSelectedRoute(context, operation, candidate));
        assertNull(context.snapshot().route());
        assertNull(context.snapshot().endState());
        assertEquals("original cancellation", context.snapshot().cancellationDetail());
        context.finishSelectionWithoutCommit(
                operation,
                org.flexlb.balance.request.Response.buildErrorResponse(
                        org.flexlb.worker.endpoint.StrategyErrorType.DISPATCH_FAILED,
                        "selection failed"),
                1003L);
        assertEquals("original cancellation", context.snapshot().endDetail());
        ended(context.snapshot());
        assertEquals(RequestState.State.CANCELLED, context.snapshot().endState());
        context.endRequest(
                null,
                1005L,
                SchedulerTestSupport.responseCandidate(
                        context, RequestContext.ResponseKind.TERMINAL));
        context.recordCleanupProgress(context.snapshot().route(), false, false, true, 1005L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
    }

    static RequestRoute route(RequestContext context, long id) {
        var route = mock(RequestRoute.class);
        when(route.ctx()).thenReturn(context);
        when(route.requestId()).thenReturn(String.valueOf(id));
        var decode = SchedulerTestSupport.mockDecodeEndpoint();
        when(route.decodeEndpoint()).thenReturn(decode);
        when(route.decodeRegistration())
                .thenReturn(new DecodeEndpoint.RequestRegistration(1L, String.valueOf(id)));
        return route;
    }

    @Test
    void realAdmissionFailureSupersedesUnclaimedShutdownEvidence() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        RequestProtocolTestSupport.register(
                context,
                "993",
                1000L,
                100_000L,
                10_000L,
                org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        var placement = requireValue(context.beginRouteSelection());
        var route = route(context, 993L);
        context.registerSelectedRoute(placement, route);
        assertNull(
                context.endRequest(
                                new RequestEndEvent(
                                        route,
                                        EndCause.SCHEDULER_SHUTDOWN,
                                        StrategyErrorType.DISPATCH_FAILED,
                                        "shutdown"),
                                1001L,
                                null)
                        .endState());
        assertNull(context.snapshot().endState());
        org.flexlb.balance.schedule.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.finishSelectionWithoutCommit(
                                placement,
                                org.flexlb.balance.request.Response.buildErrorResponse(
                                        StrategyErrorType.RESOURCE_EXHAUSTED,
                                        "KV capacity unavailable"),
                                1002L));
        ended(context.snapshot());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED, context.snapshot().endError());
        assertEquals(
                org.flexlb.balance.request.Response.buildErrorResponse(
                                StrategyErrorType.RESOURCE_EXHAUSTED, "KV capacity unavailable")
                        .getErrorMessage(),
                context.snapshot().endDetail());
    }

    private static void reportWorkerEnd(
            RequestContext context, RequestEventType type, String detail) {
        var route = context.snapshot().route();
        var role = mock(RoleSelection.class);
        var registration = route.decodeRegistration();
        var endpoint = route.decodeEndpoint();
        when(role.registration()).thenReturn(registration);
        when(role.endpoint()).thenReturn(endpoint);
        when(route.role(RoleType.DECODE)).thenReturn(role);
        assertTrue(
                AbstractRequestScheduler.handleWorkerEvent(
                        route,
                        new WorkerRequestEvent(
                                route.decodeRegistration(),
                                RoleType.DECODE,
                                route.decodeEndpoint(),
                                type,
                                42L,
                                detail,
                                1002L)));
    }
}

package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEvent;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEventType;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Closing records physical proof monotonically; execution is serialized by Scheduler.
 */
class RequestCleanupPassTest {
    @Test
    void partialCompletionRetainsProofAndCannotRewriteAnArchivedRequest() {
        RequestContext context = endingContext(7301L);
        var route = context.snapshot().route();
        assertNotNull(context.releaseReason());
        var beforeForeignRelease = context.snapshot();
        context.recordResourcesReleased(mock(RouteResult.class), true, true, 1002L);
        assertSame(
                beforeForeignRelease,
                context.snapshot(),
                "another route cannot contribute resource release proof");
        context.recordResourcesReleased(route, true, false, 1002L);
        assertTrue(context.snapshot().close().prefillSettled());
        assertFalse(context.snapshot().close().decodeSettled());
        context.recordResourcesReleased(route, false, false, 1004L);
        assertTrue(
                context.snapshot().close().prefillSettled(),
                "completed physical proof is monotonic");
        assertFalse(context.snapshot().close().decodeSettled());
        context.recordResourcesReleased(route, true, true, 1005L);
        var closed = context.snapshot();
        assertEquals(Phase.CLOSED, closed.phase());
        assertNotNull(closed.terminalRecord());
        assertNull(context.snapshot().route());
        context.recordResourcesReleased(route, true, true, 1006L);
        assertSame(closed, context.snapshot());
        assertSame(closed.terminalRecord(), context.snapshot().terminalRecord());
    }

    static RequestContext endingContext(long id) {
        var context = openContext(id);
        ended(
                context.beginClose(
                        new EndEvidence(
                                context.snapshot().route(),
                                EndKind.FAILURE,
                                null,
                                false,
                                0L,
                                StrategyErrorType.DISPATCH_FAILED,
                                "cleanup required",
                                false,
                                false),
                        1001L));
        context.recordCloseNotifications(true, true, 1001L);
        return context;
    }

    static RequestContext openContext(long id) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(
                String.valueOf(id),
                1000L,
                100_000L,
                10_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        var route = mock(RouteResult.class);
        when(route.ctx()).thenReturn(context);
        when(route.requestId()).thenReturn(String.valueOf(id));
        var decode = SchedulerTestSupport.mockDecodeEndpoint();
        when(route.decodeEndpoint()).thenReturn(decode);
        when(route.decodeReservation())
                .thenReturn(new DecodeResources.RequestReservation(1L, String.valueOf(id), 1L));
        var placement = requireValue(context.beginPlacement(null));
        org.junit.jupiter.api.Assertions.assertTrue(context.publishRoute(placement, route));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completePlacement(
                                placement, PublicationResult.PUBLISHED, 1000L));
        return context;
    }

    static <T> T requireValue(T value) {
        assertNotNull(value);
        return value;
    }

    static ContextResult.End ended(ContextResult.Snapshot snapshot) {
        assertNotNull(snapshot.outcome());
        return snapshot.outcome();
    }

    @Test
    void onlyPublishedRouteEntersContext() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(
                "990",
                1000L,
                100_000L,
                10_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        var operation = requireValue(context.beginPlacement(null));
        var first = route(context, 990L);
        var alternate = route(context, 990L);
        assertNull(context.snapshot().route());
        assertTrue(context.canPublishRoute(operation));
        assertTrue(context.publishRoute(operation, alternate));
        assertFalse(context.publishRoute(operation, first));
        context.completePlacement(operation, PublicationResult.PUBLISHED, 1002L);
        assertSame(alternate, context.snapshot().route());
        assertTrue(context.snapshot().preparedDelivery());
        assertFalse(context.snapshot().close().prefillSettled());
        assertFalse(context.snapshot().close().decodeSettled());
    }

    static RouteResult route(RequestContext context, long id) {
        var route = mock(RouteResult.class);
        when(route.ctx()).thenReturn(context);
        when(route.requestId()).thenReturn(String.valueOf(id));
        var decode = SchedulerTestSupport.mockDecodeEndpoint();
        when(route.decodeEndpoint()).thenReturn(decode);
        when(route.decodeReservation())
                .thenReturn(new DecodeResources.RequestReservation(1L, String.valueOf(id), 1L));
        return route;
    }

    @Test
    void cancellationBeforePublicationLeavesCandidateWithProducer() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(
                "991",
                1000L,
                100_000L,
                10_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        var operation = requireValue(context.beginPlacement(null));
        var candidate = route(context, 991L);
        context.requestCancel(
                new Cancellation(
                        0L,
                        CancelReason.CLIENT_CANCELLED,
                        false,
                        1001L,
                        StrategyErrorType.REQUEST_CANCELLED,
                        "original cancellation"));
        assertFalse(context.publishRoute(operation, candidate));
        assertNull(context.snapshot().route());
        assertNull(context.snapshot().pendingEnd().route());
        context.completePlacement(operation, PublicationResult.FAILED, 1003L);
        assertEquals("original cancellation", context.snapshot().pendingEnd().detail());
        ended(context.beginClose(context.snapshot().pendingEnd(), 1004L));
        assertEquals(RequestState.Phase.CANCELLED, context.snapshot().outcome().phase());
        context.recordCloseNotifications(true, true, 1005L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
    }

    @Test
    void knownWorkerFailureCannotGrantAnotherDeliveryBeforePublisherOwnership() {
        var context = openContext(992L);
        var route = context.snapshot().route();
        assertTrue(context.snapshot().preparedDelivery());
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordWorkerRequestEvent(
                                new WorkerRequestEvent(
                                        route,
                                        null,
                                        org.flexlb.dao.route.RoleType.PREFILL,
                                        WorkerRequestEventType.FAILED,
                                        42L,
                                        "worker failed",
                                        1001L)));
        assertNull(context.snapshot().outcome(), "publication capacity precedes outcome ownership");
        assertNotNull(context.snapshot().pendingEnd());
        assertFalse(context.snapshot().preparedDelivery());
        synchronized (context) {
            assertFalse(context.validateDelivery(route, 1002L));
        }
        assertNull(context.snapshot().delivery());
    }

    @Test
    void realAdmissionFailureSupersedesUnclaimedShutdownEvidence() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(
                "993",
                1000L,
                100_000L,
                10_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        var placement = requireValue(context.beginPlacement(null));
        var route = route(context, 993L);
        org.junit.jupiter.api.Assertions.assertTrue(context.publishRoute(placement, route));
        assertNull(
                context.beginClose(
                                new EndEvidence(
                                        route,
                                        EndKind.SHUTDOWN,
                                        null,
                                        false,
                                        0L,
                                        StrategyErrorType.DISPATCH_FAILED,
                                        "shutdown",
                                        false,
                                        false),
                                1001L)
                        .outcome());
        assertNull(context.snapshot().outcome());
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completePlacement(
                                placement, PublicationResult.FAILED, 1002L));
        assertNull(context.snapshot().pendingEnd());
        ended(
                context.beginClose(
                        new EndEvidence(
                                route,
                                EndKind.FAILURE,
                                null,
                                false,
                                0L,
                                StrategyErrorType.RESOURCE_EXHAUSTED,
                                "KV capacity unavailable",
                                false,
                                false),
                        1003L));
        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED, context.snapshot().outcome().errorType());
        assertEquals("KV capacity unavailable", context.snapshot().outcome().detail());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void rejectedLatePublicationCannotReplacePublishedRoute(boolean notificationsFinished) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(
                "994",
                1000L,
                100_000L,
                10_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        var placement = requireValue(context.beginPlacement(null));
        var originalRoute = route(context, 994L);
        assertTrue(context.publishRoute(placement, originalRoute));
        ended(
                context.beginClose(
                        new EndEvidence(
                                originalRoute,
                                EndKind.DELIVERY_FAILURE,
                                null,
                                false,
                                0L,
                                StrategyErrorType.DISPATCH_FAILED,
                                "original delivery failure",
                                false,
                                false),
                        1001L));
        var outcome = context.snapshot().outcome();
        context.recordCloseNotifications(true, notificationsFinished, 1002L);
        var before = context.snapshot();
        assertFalse(context.publishRoute(placement, route(context, 994L)));
        assertSame(before, context.snapshot());
        assertSame(originalRoute, context.snapshot().route());
        assertNull(context.releaseReason());
        context.completePlacement(placement, PublicationResult.FAILED, 1003L);
        if (!notificationsFinished) {
            assertSame(outcome, context.beginClose(null, 1004L).outcome());
            context.recordCloseNotifications(true, true, 1004L);
        }
        context.recordResourcesReleased(originalRoute, true, true, 1005L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
        assertSame(outcome, context.snapshot().outcome());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preparationFailureWinsPendingWorkerEndBeforePlacementFinishes(boolean workerFirst) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(
                "995",
                1000L,
                100_000L,
                10_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        var placement = requireValue(context.beginPlacement(null));
        var route = route(context, 995L);
        assertTrue(context.publishRoute(placement, route));
        Runnable worker =
                () ->
                        context.recordWorkerRequestEvent(
                                new WorkerRequestEvent(
                                        route,
                                        route.decodeReservation(),
                                        org.flexlb.dao.route.RoleType.DECODE,
                                        WorkerRequestEventType.COMPLETED,
                                        0L,
                                        "worker complete",
                                        1001L));
        Runnable failed =
                () ->
                        context.recordDeliveryResult(
                                new ContextResult.PreparationFailure(
                                        null, route, "preparation failed", 1002L));
        if (workerFirst) worker.run();
        failed.run();
        if (!workerFirst) worker.run();
        var outcome = ended(context.beginClose(null, 1003L));
        assertEquals(EndKind.DELIVERY_FAILURE, outcome.kind());
        assertEquals(RequestState.Phase.FAILED, outcome.phase());
        context.completePlacement(placement, PublicationResult.FAILED, 1004L);
        assertSame(outcome, context.snapshot().outcome());
    }
}

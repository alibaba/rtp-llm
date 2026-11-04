package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.CloseProgress;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.EngineEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EngineKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Closing records physical proof monotonically; execution is serialized by Scheduler. */
class RequestCleanupPassTest {
    @Test
    void partialCompletionRetainsProofAndCannotRewriteAnArchivedRequest() {
        RequestContext context = endingContext(7301L);
        assertNotNull(context.releaseReason());
        context.completeClose(released(true, false), 1002L);
        assertTrue(context.snapshot().close().prefillSettled());
        assertFalse(context.snapshot().close().decodeSettled());
        context.completeClose(released(false, false), 1004L);
        assertTrue(
                context.snapshot().close().prefillSettled(),
                "completed physical proof is monotonic");
        assertFalse(context.snapshot().close().decodeSettled());
        context.completeClose(released(true, true), 1005L);
        var closed = context.snapshot();
        assertEquals(Phase.CLOSED, closed.phase());
        assertNotNull(closed.terminalRecord());
        assertNull(context.snapshot().route());
        context.completeClose(released(true, true), 1006L);
        assertSame(closed, context.snapshot());
        assertSame(closed.terminalRecord(), context.snapshot().terminalRecord());
    }

    static <T> T requireValue(T value) {
        assertNotNull(value);
        return value;
    }

    static ContextResult.End ended(ContextResult.Snapshot snapshot) {
        assertNotNull(snapshot.outcome());
        return snapshot.outcome();
    }

    static CloseProgress released(boolean prefill, boolean decode) {
        return new CloseProgress(false, prefill, decode, false);
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
        context.completeClose(new CloseProgress(true, false, false, true), 1001L);
        return context;
    }

    @Test
    void completedRollbackAllowsAnotherRouteInTheSamePlacementScope() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register("990", 1000L, 100_000L, 10_000L);
        var operation = requireValue(context.beginPlacement(null));
        var first = route(context, 990L);
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(operation, first, PublicationResult.PREPARED));
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(operation, first, PublicationResult.NOT_PUBLISHED));
        var alternate = route(context, 990L);
        org.junit.jupiter.api.Assertions.assertFalse(
                context.bindPlacement(operation, alternate, PublicationResult.PREPARED),
                "unreleased acquisition cannot be overwritten");
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(operation, first, PublicationResult.ROLLBACK_COMPLETED));
        assertNull(context.snapshot().route());
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(operation, alternate, PublicationResult.PREPARED));
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(operation, alternate, PublicationResult.PUBLISHED));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completePlacement(
                                operation, PublicationResult.PUBLISHED, 1002L));
        assertSame(alternate, context.snapshot().route());
        assertTrue(context.snapshot().preparedDelivery());
        assertFalse(context.snapshot().close().prefillSettled());
        assertFalse(context.snapshot().close().decodeSettled());
    }

    @Test
    void samePlacementRollbackKeepsCancellationWithoutTheDiscardedRouteIdentity() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register("991", 1000L, 100_000L, 10_000L);
        var operation = requireValue(context.beginPlacement(null));
        var route = route(context, 991L);
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(operation, route, PublicationResult.PREPARED));
        context.requestCancel(
                new Cancellation(
                        0L,
                        CancelReason.CLIENT_CANCELLED,
                        false,
                        1001L,
                        StrategyErrorType.REQUEST_CANCELLED,
                        "original cancellation"));
        assertSame(route, context.snapshot().pendingEnd().route());
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(operation, route, PublicationResult.ROLLBACK_COMPLETED));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completePlacement(
                                operation, PublicationResult.PUBLISHED, 1003L));
        assertNull(
                context.snapshot().pendingEnd().route(),
                "completed rollback detaches only this scope's old identity");
        assertEquals("original cancellation", context.snapshot().pendingEnd().detail());
        ended(context.beginClose(context.snapshot().pendingEnd(), 1004L));
        assertEquals(RequestState.Phase.CANCELLED, context.snapshot().outcome().phase());
        context.completeClose(new CloseProgress(true, false, false, true), 1005L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
        assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellation().reason());
    }

    @Test
    void knownWorkerFailureCannotGrantAnotherDeliveryBeforePublisherOwnership() {
        var context = openContext(992L);
        var route = context.snapshot().route();
        assertTrue(context.snapshot().preparedDelivery());
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordEngineEvidence(
                                new EngineEvidence(
                                        route,
                                        null,
                                        org.flexlb.dao.route.RoleType.PREFILL,
                                        EngineKind.FAILED,
                                        42L,
                                        "worker failed",
                                        1001L)));
        assertNull(context.snapshot().outcome(), "publication capacity precedes outcome ownership");
        assertNotNull(context.snapshot().pendingEnd());
        assertFalse(context.snapshot().preparedDelivery());
        org.junit.jupiter.api.Assertions.assertNull(
                context.prepareDelivery(
                        new org.flexlb.balance.scheduler.Delivery.Member(route, null),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        1L,
                        1002L));
        assertNull(context.snapshot().delivery());
    }

    @Test
    void realAdmissionFailureSupersedesUnclaimedShutdownEvidence() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register("993", 1000L, 100_000L, 10_000L);
        var placement = requireValue(context.beginPlacement(null));
        var route = route(context, 993L);
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(placement, route, PublicationResult.PREPARED));
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
    void lateAcquisitionPreservesNotificationsAndRequiresItsOwnReleaseProof(
            boolean notificationsFinished) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register("994", 1000L, 100_000L, 10_000L);
        var placement = requireValue(context.beginPlacement(null));
        var originalRoute = route(context, 994L);
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(placement, originalRoute, PublicationResult.PREPARED));
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
        context.completeClose(new CloseProgress(true, false, false, notificationsFinished), 1002L);
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(
                        placement, originalRoute, PublicationResult.ROLLBACK_COMPLETED));
        var acquired = route(context, 994L);
        org.junit.jupiter.api.Assertions.assertFalse(
                context.bindPlacement(placement, acquired, PublicationResult.PREPARED));
        var afterAcquisition = context.snapshot();
        assertSame(
                acquired,
                afterAcquisition.route(),
                "real acquisition remains owned despite denying publication");
        assertSame(outcome, afterAcquisition.outcome());
        assertEquals(notificationsFinished, afterAcquisition.close().notificationsDone());
        assertTrue(afterAcquisition.close().notificationsAttempted());
        assertFalse(
                afterAcquisition.close().prefillSettled(),
                "new acquisition reopens its physical proof");
        assertFalse(afterAcquisition.close().decodeSettled());
        assertNull(context.releaseReason(), "the producer still owns placement completion");
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completePlacement(
                                placement, PublicationResult.FAILED, 1003L));
        if (!notificationsFinished) {
            assertSame(outcome, context.beginClose(null, 1004L).outcome());
            context.completeClose(new CloseProgress(true, false, false, true), 1004L);
        }
        assertEquals(
                Phase.ENDING,
                context.snapshot().phase(),
                "old rollback cannot prove new resources released");
        assertNotNull(context.releaseReason());
        assertSame(acquired, context.snapshot().route());
        assertFalse(context.snapshot().close().prefillSettled());
        assertFalse(context.snapshot().close().decodeSettled());
        context.completeClose(released(true, true), 1005L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
        assertSame(outcome, context.snapshot().outcome());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preparationFailureWinsPendingWorkerEndBeforePlacementFinishes(boolean workerFirst) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register("995", 1000L, 100_000L, 10_000L);
        var placement = requireValue(context.beginPlacement(null));
        var route = route(context, 995L);
        assertTrue(context.bindPlacement(placement, route, PublicationResult.PREPARED));
        Runnable worker =
                () ->
                        context.recordEngineEvidence(
                                new EngineEvidence(
                                        route,
                                        route.decodeReservation(),
                                        org.flexlb.dao.route.RoleType.DECODE,
                                        EngineKind.COMPLETED,
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

    static RequestRoute route(RequestContext context, long id) {
        var route = mock(RequestRoute.class);
        when(route.ctx()).thenReturn(context);
        when(route.requestId()).thenReturn(String.valueOf(id));
        var decode = SchedulerTestSupport.mockDecodeEndpoint();
        when(route.decodeEp()).thenReturn(decode);
        when(route.decodeReservation())
                .thenReturn(new DecodeResources.ReservationHandle(1L, String.valueOf(id), 1L));
        return route;
    }

    static RequestContext openContext(long id) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(String.valueOf(id), 1000L, 100_000L, 10_000L);
        var route = mock(RequestRoute.class);
        when(route.ctx()).thenReturn(context);
        when(route.requestId()).thenReturn(String.valueOf(id));
        var decode = SchedulerTestSupport.mockDecodeEndpoint();
        when(route.decodeEp()).thenReturn(decode);
        when(route.decodeReservation())
                .thenReturn(new DecodeResources.ReservationHandle(1L, String.valueOf(id), 1L));
        var placement = requireValue(context.beginPlacement(null));
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(placement, route, PublicationResult.PREPARED));
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(placement, route, PublicationResult.PUBLISHED));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completePlacement(
                                placement, PublicationResult.PUBLISHED, 1000L));
        return context;
    }
}

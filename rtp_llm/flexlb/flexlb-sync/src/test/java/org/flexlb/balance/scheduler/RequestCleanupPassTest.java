package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.Claim;
import static org.flexlb.balance.scheduler.ContextResult.CloseProgress;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.EngineEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EngineKind;
import static org.flexlb.balance.scheduler.ContextResult.GrantValue;
import static org.flexlb.balance.scheduler.ContextResult.Granted;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.flexlb.balance.scheduler.ContextResult.Rejected;
import static org.flexlb.balance.scheduler.ContextResult.UpdateKind;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
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
        assertEquals(
                UpdateKind.APPLIED, context.completeClose(released(true, false), 1002L).kind());
        assertTrue(context.snapshot().close().prefillSettled());
        assertFalse(context.snapshot().close().decodeSettled());
        context.completeClose(released(false, false), 1004L);
        assertTrue(
                context.snapshot().close().prefillSettled(),
                "completed physical proof is monotonic");
        assertFalse(context.snapshot().close().decodeSettled());
        var closed = context.completeClose(released(true, true), 1005L);
        assertEquals(Phase.CLOSED, closed.after().phase());
        assertNotNull(closed.after().terminalRecord());
        assertNull(context.snapshot().route());
        assertEquals(UpdateKind.STALE, context.completeClose(released(true, true), 1006L).kind());
        assertSame(closed.after().terminalRecord(), context.snapshot().terminalRecord());
    }

    static <T extends GrantValue> T granted(Claim<T> claim) {
        return ((Granted<T>) assertInstanceOf(Granted.class, claim)).value();
    }

    static ContextResult.End ended(ContextResult.Update update) {
        assertNotNull(update.after().outcome());
        return update.after().outcome();
    }

    static CloseProgress released(boolean prefill, boolean decode) {
        return new CloseProgress(false, prefill, decode, false, false);
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
        context.completeClose(new CloseProgress(true, false, false, true, true), 1001L);
        return context;
    }

    @Test
    void completedRollbackAllowsAnotherRouteInTheSamePlacementScope() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(990L, 1000L, 100_000L, 10_000L);
        var operation = granted(context.beginPlacement(null));
        var first = route(context, 990L);
        granted(context.bindPlacement(operation, first, PublicationResult.PREPARED));
        granted(context.bindPlacement(operation, first, PublicationResult.NOT_PUBLISHED));
        var alternate = route(context, 990L);
        assertInstanceOf(
                Rejected.class,
                context.bindPlacement(operation, alternate, PublicationResult.PREPARED),
                "unreleased acquisition cannot be overwritten");
        granted(context.bindPlacement(operation, first, PublicationResult.ROLLBACK_COMPLETED));
        assertNull(context.snapshot().route());
        assertNull(context.snapshot().placement().acquired());
        granted(context.bindPlacement(operation, alternate, PublicationResult.PREPARED));
        granted(context.bindPlacement(operation, alternate, PublicationResult.PUBLISHED));
        context.completePlacement(operation, PublicationResult.PUBLISHED, 1002L);
        assertSame(alternate, context.snapshot().route());
        assertTrue(context.snapshot().preparedDelivery());
        assertFalse(context.snapshot().close().prefillSettled());
        assertFalse(context.snapshot().close().decodeSettled());
    }

    @Test
    void samePlacementRollbackKeepsCancellationWithoutTheDiscardedRouteIdentity() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(991L, 1000L, 100_000L, 10_000L);
        var operation = granted(context.beginPlacement(null));
        var route = route(context, 991L);
        granted(context.bindPlacement(operation, route, PublicationResult.PREPARED));
        context.requestCancel(
                new Cancellation(
                        0L,
                        CancelReason.CLIENT_CANCELLED,
                        false,
                        1001L,
                        StrategyErrorType.REQUEST_CANCELLED,
                        "original cancellation"));
        assertSame(route, context.snapshot().placement().pendingTerminal().route());
        granted(context.bindPlacement(operation, route, PublicationResult.ROLLBACK_COMPLETED));
        context.completePlacement(operation, PublicationResult.PUBLISHED, 1003L);
        assertNull(
                context.snapshot().pendingEnd().route(),
                "completed rollback detaches only this scope's old identity");
        assertEquals("original cancellation", context.snapshot().pendingEnd().detail());
        ended(context.beginClose(context.snapshot().pendingEnd(), 1004L));
        assertEquals(RequestState.Phase.CANCELLED, context.snapshot().outcome().outcome().phase());
        context.completeClose(new CloseProgress(true, false, false, true, true), 1005L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
        assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellation().reason());
    }

    @Test
    void knownWorkerFailureCannotGrantAnotherDeliveryBeforePublisherOwnership() {
        var context = openContext(992L);
        var route = context.snapshot().route();
        assertTrue(context.snapshot().preparedDelivery());
        context.recordEngineEvidence(
                new EngineEvidence(
                        route,
                        null,
                        org.flexlb.dao.route.RoleType.PREFILL,
                        EngineKind.FAILED,
                        42L,
                        "worker failed",
                        1001L));
        assertNull(context.snapshot().outcome(), "publication capacity precedes outcome ownership");
        assertNotNull(context.snapshot().pendingEnd());
        assertFalse(context.snapshot().preparedDelivery());
        assertInstanceOf(
                Rejected.class,
                context.prepareDelivery(route, DeliveryClaimKind.BATCH_ENQUEUE, 1L, 1002L));
        assertNull(context.snapshot().delivery());
    }

    @Test
    void realAdmissionFailureSupersedesUnclaimedShutdownEvidence() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(993L, 1000L, 100_000L, 10_000L);
        var placement = granted(context.beginPlacement(null));
        var route = route(context, 993L);
        granted(context.bindPlacement(placement, route, PublicationResult.PREPARED));
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
                        .after()
                        .outcome());
        assertNull(context.snapshot().outcome());
        context.completePlacement(placement, PublicationResult.FAILED, 1002L);
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
        context.register(994L, 1000L, 100_000L, 10_000L);
        var placement = granted(context.beginPlacement(null));
        var originalRoute = route(context, 994L);
        granted(context.bindPlacement(placement, originalRoute, PublicationResult.PREPARED));
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
        context.completeClose(
                new CloseProgress(true, false, false, notificationsFinished, true), 1002L);
        granted(
                context.bindPlacement(
                        placement, originalRoute, PublicationResult.ROLLBACK_COMPLETED));
        var acquired = route(context, 994L);
        assertInstanceOf(
                Rejected.class,
                context.bindPlacement(placement, acquired, PublicationResult.PREPARED));
        var afterAcquisition = context.snapshot();
        assertSame(
                acquired,
                afterAcquisition.route(),
                "real acquisition remains owned despite denying publication");
        assertSame(acquired, afterAcquisition.placement().acquired());
        assertSame(outcome, afterAcquisition.outcome());
        assertEquals(notificationsFinished, afterAcquisition.close().notificationsDone());
        assertTrue(afterAcquisition.close().notificationsAttempted());
        assertTrue(afterAcquisition.close().timersDone());
        assertFalse(
                afterAcquisition.close().prefillSettled(),
                "new acquisition reopens its physical proof");
        assertFalse(afterAcquisition.close().decodeSettled());
        assertNull(context.releaseReason(), "the producer still owns placement completion");
        context.completePlacement(placement, PublicationResult.FAILED, 1003L);
        if (!notificationsFinished) {
            assertSame(outcome, context.beginClose(null, 1004L).after().outcome());
            context.completeClose(new CloseProgress(true, false, false, true, true), 1004L);
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

    static RequestRoute route(RequestContext context, long id) {
        var route = mock(RequestRoute.class);
        when(route.ctx()).thenReturn(context);
        when(route.requestId()).thenReturn(id);
        var decode = SchedulerTestSupport.mockDecodeEndpoint();
        when(route.decodeEp()).thenReturn(decode);
        when(route.decodeReservation())
                .thenReturn(new DecodeResources.ReservationHandle(1L, id, 1L));
        return route;
    }

    static RequestContext openContext(long id) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(id, 1000L, 100_000L, 10_000L);
        var route = mock(RequestRoute.class);
        when(route.ctx()).thenReturn(context);
        when(route.requestId()).thenReturn(id);
        var decode = SchedulerTestSupport.mockDecodeEndpoint();
        when(route.decodeEp()).thenReturn(decode);
        when(route.decodeReservation())
                .thenReturn(new DecodeResources.ReservationHandle(1L, id, 1L));
        var placement = granted(context.beginPlacement(null));
        granted(context.bindPlacement(placement, route, PublicationResult.PREPARED));
        granted(context.bindPlacement(placement, route, PublicationResult.PUBLISHED));
        context.completePlacement(placement, PublicationResult.PUBLISHED, 1000L);
        return context;
    }
}

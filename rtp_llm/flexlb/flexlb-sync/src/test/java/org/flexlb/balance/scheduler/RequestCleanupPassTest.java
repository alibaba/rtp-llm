package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/** Stale release results never contribute proof to a subsequent exact operation. */
class RequestCleanupPassTest {
    @Test
    void staleCompletionCannotSettleAnOutstandingRelease() {
        RequestContext context = endingContext(7301L);
        var first = granted(context.beginSettlement());
        assertEquals(UpdateKind.APPLIED, context.completeClose(first, released(true, false), 1002L).kind());
        var second = granted(context.beginSettlement());
        assertTrue(second.prefillSettled());
        assertFalse(second.decodeSettled());
        var stale = context.completeClose(first, released(true, true), 1003L);
        assertEquals(UpdateKind.STALE, stale.kind());
        assertSame(stale.before(), stale.after());
        var blocked = assertInstanceOf(Blocked.class, context.beginSettlement());
        assertEquals(Blocker.CLOSE_IN_PROGRESS, blocked.blocker());
        context.completeClose(second, released(true, false), 1004L);
        var third = granted(context.beginSettlement());
        assertFalse(third.decodeSettled(), "the stale operation cannot prove Decode release");
        var closed = context.completeClose(third, released(true, true), 1005L);
        assertEquals(Phase.CLOSED, closed.after().phase());
        assertNotNull(closed.after().terminalRecord());
        assertNull(context.snapshot().route());
        assertEquals(UpdateKind.STALE, context.completeClose(third, released(true, true), 1006L).kind());
        assertSame(closed.after().terminalRecord(), context.snapshot().terminalRecord());
    }

    static <T extends GrantValue> T granted(Claim<T> claim) {
        return ((Granted<T>) assertInstanceOf(Granted.class, claim)).value();
    }

    static CloseProgress released(boolean prefill, boolean decode) {
        return new CloseProgress(ProgressKind.RELEASE, prefill, decode, false, false);
    }

    static RequestContext endingContext(long id) {
        var context = openContext(id);
        var terminal = granted(context.beginClose(new EndEvidence(context.snapshot().route(), EndKind.FAILURE,
                null, false, 0L, StrategyErrorType.DISPATCH_FAILED, "cleanup required", false, false), 1001L));
        context.completeClose(terminal, new CloseProgress(ProgressKind.NOTIFICATIONS, false, false, true, true), 1001L);
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
        assertInstanceOf(Rejected.class, context.bindPlacement(operation, alternate, PublicationResult.PREPARED),
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
        var cancellation = granted(context.requestCancel(new Cancellation(0L, CancelReason.CLIENT_CANCELLED,
                false, 1001L, StrategyErrorType.REQUEST_CANCELLED, "original cancellation")));
        context.completeCancel(cancellation, new CancelEvidence(CancelEvidenceKind.LOCAL_WITHDRAWN, false, false), 1002L);
        assertSame(route, context.snapshot().placement().pendingTerminal().route());
        granted(context.bindPlacement(operation, route, PublicationResult.ROLLBACK_COMPLETED));
        context.completePlacement(operation, PublicationResult.PUBLISHED, 1003L);
        assertNull(context.snapshot().pendingEnd().route(), "completed rollback detaches only this scope's old identity");
        assertEquals("original cancellation", context.snapshot().pendingEnd().detail());
        var end = granted(context.beginClose(context.snapshot().pendingEnd(), 1004L));
        assertEquals(RequestState.Phase.CANCELLED, context.snapshot().outcome().outcome().phase());
        context.completeClose(end, new CloseProgress(ProgressKind.NOTIFICATIONS, false, false, true, true), 1005L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
        assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellation().reason());
    }

    @Test
    void knownWorkerFailureCannotGrantAnotherDeliveryBeforePublisherOwnership() {
        var context = openContext(992L);
        var route = context.snapshot().route();
        assertTrue(context.snapshot().preparedDelivery());
        context.recordEngineEvidence(new EngineEvidence(route, null, org.flexlb.dao.route.RoleType.PREFILL,
                EngineKind.FAILED, 42L, "worker failed", 1001L));
        assertNull(context.snapshot().outcome(), "publication capacity precedes outcome ownership");
        assertNotNull(context.snapshot().pendingEnd());
        assertFalse(context.snapshot().preparedDelivery());
        assertInstanceOf(Rejected.class, context.beginDelivery(route, DeliveryClaimKind.BATCH_ENQUEUE, 1L, 1002L));
        assertNull(context.snapshot().delivery());
    }

    @Test
    void realAdmissionFailureSupersedesUnclaimedShutdownEvidence() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(993L, 1000L, 100_000L, 10_000L);
        var placement = granted(context.beginPlacement(null));
        var route = route(context, 993L);
        granted(context.bindPlacement(placement, route, PublicationResult.PREPARED));
        assertInstanceOf(Blocked.class, context.beginClose(new EndEvidence(route, EndKind.SHUTDOWN,
                null, false, 0L, StrategyErrorType.DISPATCH_FAILED, "shutdown", false, false), 1001L));
        assertNull(context.snapshot().outcome());
        context.completePlacement(placement, PublicationResult.FAILED, 1002L);
        assertNull(context.snapshot().pendingEnd());
        granted(context.beginClose(new EndEvidence(route, EndKind.FAILURE, null, false, 0L,
                StrategyErrorType.RESOURCE_EXHAUSTED, "KV capacity unavailable", false, false), 1003L));
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED, context.snapshot().outcome().errorType());
        assertEquals("KV capacity unavailable", context.snapshot().outcome().detail());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void lateAcquisitionInExactPlacementScopeKeepsTerminalNotificationOwnerAndProgress(boolean notificationsFinished) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(994L, 1000L, 100_000L, 10_000L);
        var placement = granted(context.beginPlacement(null));
        var originalRoute = route(context, 994L);
        granted(context.bindPlacement(placement, originalRoute, PublicationResult.PREPARED));
        var terminal = granted(context.beginClose(new EndEvidence(originalRoute, EndKind.DELIVERY_FAILURE,
                null, false, 0L, StrategyErrorType.DISPATCH_FAILED, "original delivery failure", false, false), 1001L));
        var outcome = context.snapshot().outcome();
        context.completeClose(terminal, new CloseProgress(ProgressKind.NOTIFICATIONS, false, false, notificationsFinished, true), 1002L);
        granted(context.bindPlacement(placement, originalRoute, PublicationResult.ROLLBACK_COMPLETED));
        var acquired = route(context, 994L);
        assertInstanceOf(Rejected.class, context.bindPlacement(placement, acquired, PublicationResult.PREPARED));
        var afterAcquisition = context.snapshot();
        assertSame(acquired, afterAcquisition.route(), "real acquisition remains owned despite denying publication");
        assertSame(acquired, afterAcquisition.placement().acquired());
        if (notificationsFinished) { assertNull(afterAcquisition.close().endOperation()); }
        else { assertSame(terminal, afterAcquisition.close().endOperation()); }
        assertSame(outcome, afterAcquisition.outcome());
        assertEquals(notificationsFinished, afterAcquisition.close().notificationsDone());
        assertTrue(afterAcquisition.close().notificationsAttempted());
        assertTrue(afterAcquisition.close().timersDone());
        assertFalse(afterAcquisition.close().prefillSettled(), "new acquisition reopens its physical proof");
        assertFalse(afterAcquisition.close().decodeSettled());
        assertInstanceOf(Blocked.class, context.beginSettlement(), "the producer still owns placement completion");
        context.completePlacement(placement, PublicationResult.FAILED, 1003L);
        if (!notificationsFinished) {
            context.completeClose(terminal, new CloseProgress(ProgressKind.NOTIFICATIONS, false, false, true, true), 1004L);
        }
        assertEquals(Phase.ENDING, context.snapshot().phase(), "old rollback cannot prove new resources released");
        var release = granted(context.beginSettlement());
        assertSame(acquired, release.route());
        assertFalse(release.prefillSettled());
        assertFalse(release.decodeSettled());
        context.completeClose(release, released(true, true), 1005L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
        assertSame(outcome, context.snapshot().outcome());
    }

    static RequestRoute route(RequestContext context, long id) {
        var route = mock(RequestRoute.class);
        when(route.ctx()).thenReturn(context);
        when(route.requestId()).thenReturn(id);
        var decode = SchedulerTestSupport.mockDecodeEndpoint();
        when(route.decodeEp()).thenReturn(decode);
        when(route.decodeReservation()).thenReturn(new DecodeResources.ReservationHandle(1L, id, 1L));
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
        when(route.decodeReservation()).thenReturn(new DecodeResources.ReservationHandle(1L, id, 1L));
        var placement = granted(context.beginPlacement(null));
        granted(context.bindPlacement(placement, route, PublicationResult.PREPARED));
        granted(context.bindPlacement(placement, route, PublicationResult.PUBLISHED));
        context.completePlacement(placement, PublicationResult.PUBLISHED, 1000L);
        return context;
    }
}

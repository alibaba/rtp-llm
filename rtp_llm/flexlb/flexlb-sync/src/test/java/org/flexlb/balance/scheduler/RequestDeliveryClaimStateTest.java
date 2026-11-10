package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.flexlb.balance.scheduler.ContextResult.CancelEvidenceKind;
import static org.flexlb.balance.scheduler.ContextResult.CleanupAcknowledgement;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.RoutingCompletion;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.flexlb.balance.scheduler.ContextResult.TransportResult;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEvent;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEventType;
import static org.flexlb.balance.scheduler.RequestChangeAssertions.UpdateKind;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks delivery transitions against exact identities, sender exit, deadlines, and retained Worker evidence.
 */
class RequestDeliveryClaimStateTest {
    @Test
    void predictionAndCleanupDoNotInventASenderResult() {
        var f = fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var ctx = f.context();
        var sending = ctx.snapshot().delivery().send();
        ctx.recordDeliveryPrediction(new ContextResult.Prediction(f.delivery(), 5000L, 1002L));
        assertEquals(sending, ctx.snapshot().delivery().send());
        assertEquals(5000L, ctx.snapshot().engine().visibilityAtMs());
        ctx.recordRemoteCleanup(
                new CleanupAcknowledgement(f.delivery(), CancelEvidenceKind.CLEANED, 1003L));
        assertEquals(sending, ctx.snapshot().delivery().send());
        assertTrue(ctx.snapshot().delivery().prefillProof());
        assertTrue(ctx.snapshot().delivery().decodeProof());
        assertFalse(ctx.snapshot().deliveryReleasable());
        assertEquals(5000L, ctx.snapshot().engine().visibilityAtMs());
        assertNull(ctx.snapshot().pendingEnd());
    }

    static Fixture fixture(Delivery.SendMode kind, boolean fusion) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(
                "7502",
                1000L,
                100_000L,
                10_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        var prefill = mock(PrefillEndpoint.class);
        var decode =
                fusion
                        ? null
                        : org.flexlb.balance.scheduler.SchedulerTestSupport.mockDecodeEndpoint();
        var reservation =
                fusion
                        ? null
                        : new DecodeResources.RequestReservation(1L, String.valueOf(7502L), 1L);
        var route = mock(RouteResult.class);
        when(route.selectedEndpointsAvailable()).thenReturn(true);
        when(route.requestId()).thenReturn(String.valueOf(7502L));
        when(route.ctx()).thenReturn(context);
        when(route.prefillEndpoint()).thenReturn(prefill);
        when(route.decodeEndpoint()).thenReturn(decode);
        when(route.decodeReservation()).thenReturn(reservation);
        var placement = RequestCleanupPassTest.requireValue(context.beginRouteSelection());
        org.junit.jupiter.api.Assertions.assertTrue(context.publishRoute(placement, route));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completeRouting(
                                placement, RoutingCompletion.FINISHED, 1000L));
        var member = new Delivery.Member(route, null);
        synchronized (context) {
            assertTrue(context.validateDelivery(route, null, 1001L));
            member.initialize(kind, kind == Delivery.SendMode.BATCH_ENQUEUE ? 11L : 0L);
            context.recordDeliveryStarted(member, 1001L);
        }
        assertEquals(
                kind == Delivery.SendMode.BATCH_ENQUEUE ? SendState.SENDING : SendState.NOT_STARTED,
                context.snapshot().delivery().send());
        return new Fixture(context, route, prefill, decode, reservation, member);
    }

    @Test
    void stalePredictionAndCleanupCannotChangeTheCurrentDelivery() {
        var f = fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var foreign = new Delivery.Member(f.route(), null);
        foreign.initialize(Delivery.SendMode.BATCH_ENQUEUE, 999L);
        var before = f.context().snapshot();
        assertNull(
                f.context()
                        .recordDeliveryPrediction(
                                new ContextResult.Prediction(foreign, 5000L, 1002L)));
        assertNull(
                f.context()
                        .recordRemoteCleanup(
                                new CleanupAcknowledgement(
                                        foreign, CancelEvidenceKind.CLEANED, 1003L)));
        assertSame(before, f.context().snapshot());
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(
            value = org.flexlb.dao.loadbalance.StrategyErrorType.class,
            names = {
                "BATCH_SLO_EXPIRED",
                "QUEUE_TIMEOUT",
                "RESOURCE_EXHAUSTED",
                "ADMISSION_UNAVAILABLE"
            })
    void deliveryExpiryUsesTheRegisteredSchedulerError(
            org.flexlb.dao.loadbalance.StrategyErrorType error) {
        for (boolean beforeSend : new boolean[] {true, false}) {
            var context = new RequestContext(mock(FlexlbConfig.class));
            context.register("7504", 1000L, 100_000L, 10_000L, error);
            var route = mock(RouteResult.class);
            when(route.selectedEndpointsAvailable()).thenReturn(true);
            when(route.ctx()).thenReturn(context);
            when(route.requestId()).thenReturn("7504");
            var placement = RequestCleanupPassTest.requireValue(context.beginRouteSelection());
            assertTrue(context.publishRoute(placement, route));
            context.completeRouting(placement, RoutingCompletion.FINISHED, 1000L);
            var member = new Delivery.Member(route, null);
            if (beforeSend) {
                synchronized (context) {
                    assertFalse(context.validateDelivery(route, null, 11_001L));
                }
            } else {
                synchronized (context) {
                    assertTrue(context.validateDelivery(route, null, 1001L));
                    member.initialize(Delivery.SendMode.ROUTE_DECISION, 0L);
                    context.recordDeliveryStarted(member, 1001L);
                }
                context.recordDeliveryResult(
                        new ContextResult.RouteDelivered(member, 2000L, 1002L));
                assertTrue(context.snapshot().canConfirmDelivery());
                assertFalse(context.confirmDelivery(11_001L));
            }
            assertEquals(error, context.snapshot().cancellation().errorType());
            assertEquals(error, context.snapshot().pendingEnd().errorType());
        }
    }

    @Test
    void deliveryPreparationDoesNotPublishUntransferredOwnership() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(
                "7503",
                1000L,
                100_000L,
                10_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        var route = mock(RouteResult.class);
        when(route.selectedEndpointsAvailable()).thenReturn(true);
        when(route.requestId()).thenReturn(String.valueOf(7503L));
        when(route.ctx()).thenReturn(context);
        var placement = RequestCleanupPassTest.requireValue(context.beginRouteSelection());
        org.junit.jupiter.api.Assertions.assertTrue(context.publishRoute(placement, route));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completeRouting(
                                placement, RoutingCompletion.FINISHED, 1000L));
        var before = context.snapshot();
        var member = new Delivery.Member(route, null);
        synchronized (context) {
            assertTrue(context.validateDelivery(route, null, 1001L));
            member.initialize(Delivery.SendMode.BATCH_ENQUEUE, 12L);
            assertSame(before, context.snapshot());
            assertNull(context.snapshot().delivery());
            context.recordDeliveryStarted(member, 1001L);
        }
        assertSame(member, context.snapshot().delivery().identity());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void engineCompletionCannotCloseRoutePublicationBeforeTerminalEffects(boolean fusion) {
        var fixture = fixture(Delivery.SendMode.ROUTE_DECISION, fusion);
        var context = fixture.context();
        var observed =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordWorkerRequestEvent(
                                        new WorkerRequestEvent(
                                                fixture.route(),
                                                fixture.reservation(),
                                                fusion ? RoleType.PDFUSION : RoleType.DECODE,
                                                WorkerRequestEventType.COMPLETED,
                                                0L,
                                                null,
                                                1002L)));
        assertEquals(UpdateKind.APPLIED, observed.kind());
        assertNotNull(observed.after().pendingEnd());
        assertFalse(
                context.snapshot().deliveryReleasable(),
                "execution evidence does not close the route sender");
        RequestCleanupPassTest.ended(context.beginClose(observed.after().pendingEnd(), 1003L));
        assertFalse(
                context.snapshot().deliveryReleasable(),
                "terminal ownership precedes sender closure");
        context.recordCloseNotifications(true, true, 1004L);
        assertTrue(context.snapshot().deliveryReleasable());
        assertNull(context.snapshot().delivery().abandonment());
        assertFalse(
                context.snapshot().delivery().acknowledged(),
                "terminal does not manufacture an ACK");
        assertEquals(
                SendState.ROUTE_CLOSED,
                context.snapshot().delivery().send(),
                "closing route publication is not batch NOT_SENT proof");
        if (context.snapshot().phase() != Phase.CLOSED) {
            assertNotNull(context.releaseReason());
            assertNotEquals(DecodeResources.ReleaseReason.NOT_SENT, context.releaseReason());
        } else {
            assertNotNull(
                    context.snapshot().terminalRecord(),
                    "already settled fusion ledger requires no extra release");
        }
    }

    @Test
    void committedBatchCanReportNotSentBeforeAnyBytesAreSent() {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var context = fixture.context();
        var rejected = Delivery.Result.notSent(new IllegalStateException("RPC preparation failed"));
        var update =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeliveryResult(
                                        new TransportResult(fixture.delivery(), rejected, 1002L)));
        assertNull(update.after().delivery().abandonment());
        assertEquals(EndKind.DELIVERY_FAILURE, update.after().pendingEnd().kind());
        assertEquals(
                "Delivery failed: RPC preparation failed", update.after().pendingEnd().detail());
        assertTrue(update.after().deliveryReleasable());
        assertNull(update.after().outcome(), "publisher capacity precedes freezing the failure");
        var terminal =
                RequestCleanupPassTest.ended(
                        context.beginClose(update.after().pendingEnd(), 1003L));
        assertEquals(DecodeResources.ReleaseReason.NOT_SENT, context.releaseReason());
        assertFalse(context.snapshot().delivery().acknowledged());
        assertFalse(context.validateDelivery(fixture.route(), null, 1004L));
        var duplicate =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeliveryResult(
                                        new TransportResult(fixture.delivery(), rejected, 1005L)));
        assertEquals(UpdateKind.NO_CHANGE, duplicate.kind());
        assertSame(duplicate.before(), duplicate.after());
        assertThrows(
                IllegalStateException.class,
                () ->
                        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                                context,
                                observedRequest ->
                                        observedRequest.recordDeliveryResult(
                                                new TransportResult(
                                                        fixture.delivery(),
                                                        Delivery.Result.delivered(),
                                                        1005L))));
        assertEquals(SendState.NOT_SENT, context.snapshot().delivery().send());
        assertNotNull(terminal);
    }

    @Test
    void cancellingUnpublishedRouteClosesItsSenderAndPreservesFirstReason() {
        var fixture = fixture(Delivery.SendMode.ROUTE_DECISION, false);
        var context = fixture.context();
        assertTrue(
                context.requestCancel(
                        new ContextResult.Cancellation(
                                0L,
                                CancelReason.CLIENT_CANCELLED,
                                false,
                                1002L,
                                org.flexlb.dao.loadbalance.StrategyErrorType.REQUEST_CANCELLED,
                                "client cancelled")));
        var cancelled = context.snapshot();
        assertTrue(cancelled.deliveryReleasable());
        assertEquals(SendState.ROUTE_CLOSED, cancelled.delivery().send());
        assertFalse(cancelled.delivery().acknowledged());
        assertFalse(
                context.requestCancel(
                        new ContextResult.Cancellation(
                                0L,
                                CancelReason.SHUTDOWN,
                                false,
                                1003L,
                                org.flexlb.dao.loadbalance.StrategyErrorType.REQUEST_CANCELLED,
                                "shutdown")));
        assertSame(cancelled, context.snapshot());
        assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().delivery().abandonment());
    }

    @Test
    void consumedSchedulingTimerDoesNotRevokeAnAlreadyCommittedBatch() {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var context = fixture.context();
        var deadline = context.snapshot().deadlines().request();
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordWorkerRequestEvent(
                                new WorkerRequestEvent(
                                        fixture.route(),
                                        fixture.reservation(),
                                        RoleType.DECODE,
                                        WorkerRequestEventType.ACTIVE,
                                        0L,
                                        null,
                                        99_999L)));
        var expired =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeadline(deadline, 100_000L, null));
        assertEquals(UpdateKind.APPLIED, expired.kind());
        assertNull(expired.after().deadlines().request());
        assertNull(expired.after().cancellation(), "handoff owns scheduling cancellation race");
        assertEquals(SendState.SENDING, context.snapshot().delivery().send());
        assertFalse(
                context.validateDelivery(fixture.route(), null, 100_001L),
                "a committed batch cannot be committed a second time");
        assertFalse(
                context.snapshot().deliveryReleasable(),
                "consuming the scheduling deadline does not invent sender exit proof");
    }

    @Test
    void lateSenderRejectionCannotInventCleanupAfterSuccessfulWorkerCompletion() {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var context = fixture.context();
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordWorkerRequestEvent(
                                new WorkerRequestEvent(
                                        fixture.route(),
                                        fixture.reservation(),
                                        RoleType.DECODE,
                                        WorkerRequestEventType.COMPLETED,
                                        0L,
                                        null,
                                        1003L)));
        RequestCleanupPassTest.ended(context.beginClose(context.snapshot().pendingEnd(), 1004L));
        assertFalse(
                context.snapshot().deliveryReleasable(),
                "worker completion still awaits sender exit");
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordDeliveryResult(
                                new TransportResult(
                                        fixture.delivery(),
                                        Delivery.Result.prefillRejected(
                                                new IllegalStateException("late rejection")),
                                        1005L)));
        assertEquals(RequestState.Phase.COMPLETED, context.snapshot().outcome().phase());
        assertTrue(context.snapshot().deliveryReleasable());
        assertNull(
                context.snapshot().delivery().abandonment(),
                "completed execution needs no new remote proof");
        assertEquals(DecodeResources.ReleaseReason.COUNTERPART_FINISHED, context.releaseReason());
    }

    @Test
    void life004DuplicateTerminalAndLateActiveEventsRemainMonotonic() {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var context = fixture.context();
        var terminal =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordWorkerRequestEvent(
                                        new WorkerRequestEvent(
                                                fixture.route(),
                                                fixture.reservation(),
                                                RoleType.DECODE,
                                                WorkerRequestEventType.COMPLETED,
                                                0L,
                                                null,
                                                1003L)));
        RequestCleanupPassTest.ended(context.beginClose(terminal.after().pendingEnd(), 1004L));
        var first =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeliveryResult(
                                        new TransportResult(
                                                fixture.delivery(),
                                                Delivery.Result.delivered(),
                                                1005L)));
        assertEquals(RequestState.Phase.COMPLETED, RequestState.from(context.snapshot()).state());
        assertTrue(first.after().deliveryReleasable());

        var duplicate =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeliveryResult(
                                        new TransportResult(
                                                fixture.delivery(),
                                                Delivery.Result.delivered(),
                                                1006L)));
        assertEquals(UpdateKind.NO_CHANGE, duplicate.kind());
        var lateActive =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordWorkerRequestEvent(
                                        new WorkerRequestEvent(
                                                fixture.route(),
                                                fixture.reservation(),
                                                RoleType.DECODE,
                                                WorkerRequestEventType.ACTIVE,
                                                0L,
                                                null,
                                                1007L)));
        assertEquals(UpdateKind.APPLIED, lateActive.kind());
        assertEquals(RequestState.Phase.COMPLETED, RequestState.from(context.snapshot()).state());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void allLedgerProofCannotArchiveBeforeTheBatchSenderExits(boolean failed) {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, true);
        var context = fixture.context();
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordWorkerRequestEvent(
                                new WorkerRequestEvent(
                                        fixture.route(),
                                        null,
                                        RoleType.PDFUSION,
                                        failed
                                                ? WorkerRequestEventType.FAILED
                                                : WorkerRequestEventType.COMPLETED,
                                        failed ? 42L : 0L,
                                        null,
                                        1003L)));
        RequestCleanupPassTest.ended(context.beginClose(context.snapshot().pendingEnd(), 1004L));
        context.recordCloseNotifications(true, true, 1005L);
        if (failed) {
            org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                    context,
                    observedRequest ->
                            observedRequest.recordRemoteCleanup(
                                    new CleanupAcknowledgement(
                                            fixture.delivery(),
                                            CancelEvidenceKind.CLEANED,
                                            1005L)));
        }
        assertEquals(
                Phase.ENDING,
                context.snapshot().phase(),
                "physical worker proof does not exit the sender");
        assertNull(context.snapshot().terminalRecord());
        assertFalse(context.snapshot().deliveryReleasable());
        assertNull(context.releaseReason());
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordDeliveryResult(
                                new TransportResult(
                                        fixture.delivery(), Delivery.Result.delivered(), 1006L)));
        assertTrue(context.snapshot().deliveryReleasable());
        assertNotNull(context.releaseReason());
        context.recordResourcesReleased(fixture.route(), true, true, 1007L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
        assertEquals(
                failed ? RequestState.Phase.FAILED : RequestState.Phase.COMPLETED,
                context.snapshot().terminalRecord().state());
    }

    @Test
    void retainedWorkerEndStillWaitsForTheCommittedBatchSender() {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, true);
        var context = fixture.context();
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordWorkerRequestEvent(
                                new WorkerRequestEvent(
                                        fixture.route(),
                                        null,
                                        RoleType.PDFUSION,
                                        WorkerRequestEventType.COMPLETED,
                                        0L,
                                        null,
                                        1002L)));
        assertNull(context.snapshot().outcome());
        assertNotNull(context.snapshot().pendingEnd());
        assertFalse(context.validateDelivery(fixture.route(), null, 1003L));
        assertEquals(SendState.SENDING, context.snapshot().delivery().send());
        assertFalse(
                context.snapshot().deliveryReleasable(),
                "worker completion does not finish the committed sender");
        assertEquals(EndKind.WORKER, context.snapshot().pendingEnd().kind());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void logicalPreparationFailureCannotRewriteAnExitedTransport(boolean uncertain) {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var context = fixture.context();
        var actual =
                uncertain
                        ? Delivery.Result.uncertain(new IllegalStateException("RPC reply lost"))
                        : Delivery.Result.delivered();
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordDeliveryResult(
                                new TransportResult(fixture.delivery(), actual, 1003L)));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordPreparationFailure(
                                new PreparationFailure(
                                        fixture.delivery(),
                                        fixture.route(),
                                        "logical failure",
                                        1004L)));
        assertEquals(
                uncertain ? SendState.UNKNOWN : SendState.DELIVERED,
                context.snapshot().delivery().send());
        assertEquals("logical failure", context.snapshot().pendingEnd().detail());
        RequestCleanupPassTest.ended(context.beginClose(context.snapshot().pendingEnd(), 1005L));
        assertNull(context.releaseReason(), "logical failure does not fabricate NOT_SENT proof");
    }

    record Fixture(
            RequestContext context,
            RouteResult route,
            PrefillEndpoint prefill,
            DecodeEndpoint decode,
            DecodeResources.RequestReservation reservation,
            Delivery.Member delivery) {}
}

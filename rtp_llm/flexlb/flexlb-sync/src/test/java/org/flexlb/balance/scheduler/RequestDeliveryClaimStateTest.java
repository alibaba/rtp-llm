package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.flexlb.balance.scheduler.ContextResult.Abandonment;
import static org.flexlb.balance.scheduler.ContextResult.CancelEvidenceKind;
import static org.flexlb.balance.scheduler.ContextResult.CleanupAcknowledgement;
import static org.flexlb.balance.scheduler.ContextResult.CloseProgress;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.EngineEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EngineKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.flexlb.balance.scheduler.ContextResult.TransportResult;
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

class RequestDeliveryClaimStateTest {
    @Test
    void deliveryPreparationDoesNotPublishUntransferredOwnership() {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register("7503", 1000L, 100_000L, 10_000L);
        var route = mock(RequestRoute.class);
        when(route.requestId()).thenReturn(String.valueOf(7503L));
        when(route.ctx()).thenReturn(context);
        var placement = RequestCleanupPassTest.requireValue(context.beginPlacement(null));
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(placement, route, PublicationResult.PREPARED));
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(placement, route, PublicationResult.PUBLISHED));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completePlacement(
                                placement, PublicationResult.PUBLISHED, 1000L));
        var before = context.snapshot();
        var delivery =
                RequestCleanupPassTest.requireValue(
                        context.prepareDelivery(
                                new org.flexlb.balance.scheduler.Delivery.Member(route, null),
                                Delivery.SendMode.BATCH_ENQUEUE,
                                12L,
                                1001L));
        assertSame(before, context.snapshot());
        assertNull(context.snapshot().delivery());
        context.publishDelivery(delivery);
        assertSame(delivery.identity(), context.snapshot().delivery().identity());
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
                                observedRequest.recordEngineEvidence(
                                        new EngineEvidence(
                                                fixture.route(),
                                                fixture.reservation(),
                                                fusion ? RoleType.PDFUSION : RoleType.DECODE,
                                                EngineKind.COMPLETED,
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
        context.completeClose(new CloseProgress(true, false, false, true), 1004L);
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
    void preparedBatchCanReportNotSentWithoutClaimingTheSenderSlot() {
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
        org.junit.jupiter.api.Assertions.assertEquals(
                RequestContext.SendResult.UNAVAILABLE,
                context.beginSend(fixture.delivery(), 1004L));
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
    void abandoningUnpublishedRouteClosesItsSenderWithoutInventingNotSentProof() {
        var fixture = fixture(Delivery.SendMode.ROUTE_DECISION, false);
        var context = fixture.context();
        var update =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeliveryResult(
                                        new Abandonment(
                                                fixture.delivery(),
                                                CancelReason.CLIENT_CANCELLED,
                                                1002L)));
        assertEquals(UpdateKind.APPLIED, update.kind());
        assertTrue(update.after().deliveryReleasable());
        assertEquals(SendState.ROUTE_CLOSED, update.after().delivery().send());
        assertFalse(update.after().delivery().acknowledged());
        var repeated =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeliveryResult(
                                        new Abandonment(
                                                fixture.delivery(), CancelReason.SHUTDOWN, 1003L)));
        assertEquals(UpdateKind.NO_CHANGE, repeated.kind());
        assertSame(repeated.before(), repeated.after());
        assertEquals(CancelReason.CLIENT_CANCELLED, repeated.after().delivery().abandonment());
    }

    @Test
    void consumedSchedulingTimerStillForbidsStartingAfterAbsoluteDeadline() {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var context = fixture.context();
        var deadline = context.snapshot().deadlines().request();
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordEngineEvidence(
                                new EngineEvidence(
                                        fixture.route(),
                                        fixture.reservation(),
                                        RoleType.DECODE,
                                        EngineKind.ACTIVE,
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
        org.junit.jupiter.api.Assertions.assertEquals(
                RequestContext.SendResult.CANCELLED,
                context.beginSend(fixture.delivery(), 100_001L));
        assertEquals(SendState.NOT_STARTED, context.snapshot().delivery().send());
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordDeliveryResult(
                                new TransportResult(
                                        fixture.delivery(),
                                        Delivery.Result.notSent(
                                                new IllegalStateException(
                                                        "sender deadline exceeded")),
                                        100_001L)));
        assertTrue(context.snapshot().deliveryReleasable());
    }

    @Test
    void lateSenderRejectionCannotInventCleanupAfterSuccessfulWorkerCompletion() {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var context = fixture.context();
        org.junit.jupiter.api.Assertions.assertEquals(
                RequestContext.SendResult.STARTED, context.beginSend(fixture.delivery(), 1002L));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordEngineEvidence(
                                new EngineEvidence(
                                        fixture.route(),
                                        fixture.reservation(),
                                        RoleType.DECODE,
                                        EngineKind.COMPLETED,
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
        org.junit.jupiter.api.Assertions.assertEquals(
                RequestContext.SendResult.STARTED, context.beginSend(fixture.delivery(), 1002L));
        var terminal =
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordEngineEvidence(
                                        new EngineEvidence(
                                                fixture.route(),
                                                fixture.reservation(),
                                                RoleType.DECODE,
                                                EngineKind.COMPLETED,
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
                                observedRequest.recordEngineEvidence(
                                        new EngineEvidence(
                                                fixture.route(),
                                                fixture.reservation(),
                                                RoleType.DECODE,
                                                EngineKind.ACTIVE,
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
        org.junit.jupiter.api.Assertions.assertEquals(
                RequestContext.SendResult.STARTED, context.beginSend(fixture.delivery(), 1002L));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordEngineEvidence(
                                new EngineEvidence(
                                        fixture.route(),
                                        null,
                                        RoleType.PDFUSION,
                                        failed ? EngineKind.FAILED : EngineKind.COMPLETED,
                                        failed ? 42L : 0L,
                                        null,
                                        1003L)));
        RequestCleanupPassTest.ended(context.beginClose(context.snapshot().pendingEnd(), 1004L));
        context.completeClose(new CloseProgress(true, false, false, true), 1005L);
        if (failed) {
            org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                    context,
                    observedRequest ->
                            observedRequest.recordDeliveryResult(
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
        context.completeClose(new CloseProgress(false, true, true, false), 1007L);
        assertEquals(Phase.CLOSED, context.snapshot().phase());
        assertEquals(
                failed ? RequestState.Phase.FAILED : RequestState.Phase.COMPLETED,
                context.snapshot().terminalRecord().state());
    }

    @Test
    void retainedWorkerEndCannotStartAPreparedBatchSender() {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, true);
        var context = fixture.context();
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordEngineEvidence(
                                new EngineEvidence(
                                        fixture.route(),
                                        null,
                                        RoleType.PDFUSION,
                                        EngineKind.COMPLETED,
                                        0L,
                                        null,
                                        1002L)));
        assertNull(context.snapshot().outcome());
        assertNotNull(context.snapshot().pendingEnd());
        org.junit.jupiter.api.Assertions.assertEquals(
                RequestContext.SendResult.CANCELLED, context.beginSend(fixture.delivery(), 1003L));
        assertEquals(SendState.NOT_STARTED, context.snapshot().delivery().send());
        assertEquals(EndKind.WORKER, context.snapshot().pendingEnd().kind());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void logicalPreparationFailureCannotRewriteAnExitedTransport(boolean uncertain) {
        var fixture = fixture(Delivery.SendMode.BATCH_ENQUEUE, false);
        var context = fixture.context();
        org.junit.jupiter.api.Assertions.assertEquals(
                RequestContext.SendResult.STARTED, context.beginSend(fixture.delivery(), 1002L));
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
                        observedRequest.recordDeliveryResult(
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

    static Fixture fixture(Delivery.SendMode kind, boolean fusion) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register("7502", 1000L, 100_000L, 10_000L);
        var prefill = mock(PrefillEndpoint.class);
        var decode =
                fusion
                        ? null
                        : org.flexlb.balance.scheduler.SchedulerTestSupport.mockDecodeEndpoint();
        var reservation =
                fusion
                        ? null
                        : new DecodeResources.ReservationHandle(1L, String.valueOf(7502L), 1L);
        var route = mock(RequestRoute.class);
        when(route.requestId()).thenReturn(String.valueOf(7502L));
        when(route.ctx()).thenReturn(context);
        when(route.prefillEp()).thenReturn(prefill);
        when(route.decodeEp()).thenReturn(decode);
        when(route.decodeReservation()).thenReturn(reservation);
        var placement = RequestCleanupPassTest.requireValue(context.beginPlacement(null));
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(placement, route, PublicationResult.PREPARED));
        org.junit.jupiter.api.Assertions.assertTrue(
                context.bindPlacement(placement, route, PublicationResult.PUBLISHED));
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completePlacement(
                                placement, PublicationResult.PUBLISHED, 1000L));
        var delivery =
                RequestCleanupPassTest.requireValue(
                        context.prepareDelivery(
                                new org.flexlb.balance.scheduler.Delivery.Member(route, null),
                                kind,
                                kind == Delivery.SendMode.BATCH_ENQUEUE ? 11L : 0L,
                                1001L));
        context.publishDelivery(delivery);
        return new Fixture(context, route, prefill, decode, reservation, delivery.identity());
    }

    record Fixture(
            RequestContext context,
            RequestRoute route,
            PrefillEndpoint prefill,
            DecodeEndpoint decode,
            DecodeResources.ReservationHandle reservation,
            Delivery.Member delivery) {}
}

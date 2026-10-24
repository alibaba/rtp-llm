package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class RequestDeliveryEventTest {
    @Test
    void staleGenerationCannotConsumeTheCurrentRoutesPrediction() {
        var current = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.ROUTE_DECISION, false);
        var previous = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.ROUTE_DECISION, false);
        var before = current.context().snapshot();
        var stalePrediction = current.context().recordDeliveryResult(new Prediction(previous.delivery(), 3000L, 1002L));
        var staleCompletion = current.context().recordDeliveryResult(new RouteDelivered(previous.delivery(), 3000L, 1002L));
        assertEquals(UpdateKind.STALE, stalePrediction.kind());
        assertEquals(UpdateKind.STALE, staleCompletion.kind());
        assertSame(before, current.context().snapshot());
        assertNull(current.context().snapshot().deadlines().visibility());
        verify(current.decode(), never()).ownership(any());
        verify(previous.decode(), never()).ownership(any());
        var update = current.context().recordDeliveryResult(new RouteDelivered(current.delivery(), 3000L, 1002L));
        assertEquals(UpdateKind.APPLIED, update.kind());
        assertSame(current.delivery(), update.after().delivery().identity());
        assertNotNull(update.after().deadlines().visibility());
    }

    @Test
    void wrongDeliveryKindIsRejectedBeforeReadingDecodeOrConsumingPrediction() {
        var fixture = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.BATCH_ENQUEUE, false);
        var before = fixture.context().snapshot();
        assertThrows(IllegalArgumentException.class,
                () -> fixture.context().recordDeliveryResult(new RouteDelivered(fixture.delivery(), 3000L, 1002L)));
        assertSame(before, fixture.context().snapshot());
        verify(fixture.decode(), never()).ownership(any());
        var prediction = fixture.context().recordDeliveryResult(new Prediction(fixture.delivery(), 3000L, 1002L));
        assertEquals(UpdateKind.APPLIED, prediction.kind());
        assertEquals(SendState.NOT_STARTED, prediction.after().delivery().send());
        assertFalse(prediction.after().deliveryReleasable());
        assertFalse(prediction.after().delivery().acknowledged());
        assertNotNull(prediction.after().deadlines().visibility());
        verify(fixture.decode(), never()).ownership(any());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void routeCompletionReturnsSettlementAndConfirmationWithoutAnotherStateQuery(boolean decodeObserved) {
        var fixture = RequestDeliveryClaimStateTest.fixture(DeliveryClaimKind.ROUTE_DECISION, false);
        if (decodeObserved) {
            fixture.context().recordEngineEvidence(new EngineEvidence(fixture.route(), fixture.reservation(),
                    RoleType.DECODE, EngineKind.ACTIVE, 0L, null, 1002L));
        }
        var update = fixture.context().recordDeliveryResult(new RouteDelivered(fixture.delivery(), 3000L, 1003L));
        assertEquals(UpdateKind.APPLIED, update.kind());
        assertSame(fixture.delivery(), update.after().delivery().identity());
        assertTrue(update.after().deliveryReleasable());
        assertNull(update.after().outcome());
        assertNull(update.after().selectedResponse());
        assertEquals(SendState.DELIVERED, update.after().delivery().send());
        assertFalse(update.after().delivery().acknowledged(), "recording transport does not select a frontend response");
        assertEquals(decodeObserved, update.after().engine().decodeAccepted());
        assertEquals(!decodeObserved, update.after().deadlines().visibility() != null);
        var confirmed = fixture.context().selectResponse(new ResponseCandidate(fixture.context(), ResponseKind.CONFIRMATION, null, 1004L));
        assertInstanceOf(Granted.class, confirmed);
        assertNull(confirmed.after().selectedResponse(), "Scheduler publishes outside this transition");
        verify(fixture.decode(), never()).ownership(any());
    }
}

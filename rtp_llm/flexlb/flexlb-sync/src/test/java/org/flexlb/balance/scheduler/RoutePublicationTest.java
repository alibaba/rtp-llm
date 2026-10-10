package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import static org.flexlb.balance.scheduler.ContextResult.*;
import org.flexlb.balance.endpoint.DecodeResources;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** NON_BATCH contracts run through the same real Worker preparation/commit path. */
@Timeout(15)
class RoutePublicationTest {
    @Test
    void removingFailedSuffixPublishesCapacityChangeOnceAndStillDeliversPrefix() {
        try (var f = new LinearDeliveryFixture(false)) {
            var first = f.item(1, 10);
            var failed = f.item(2, 20);
            var failure = new IllegalStateException("Decode preparation failed");
            doThrow(failure).when(f.decode).acquireDispatchPermit(eq(failed.decodeReservation()), any());
            when(f.handoff.removedFailedMember()).thenReturn(true);
            assertNull(f.process(List.of(first, failed), OptionalLong.empty()));
            assertEquals(List.of(first), f.committedItems);
            verify(f.owner).onDeliveryEvidence(isNull(), argThat(e -> e instanceof PreparationFailure evidence
                    && evidence.route() == failed && evidence.cause() == failure));
            verify(f.owner).onDeliveryEvidence(eq(f.claims.getFirst()), argThat(e -> e instanceof RouteDelivered delivered
                    && f.duration(delivered.predictedAtMs(), delivered.nowMs()) == 10L));
            verify(f.prefill, times(1)).signal(org.flexlb.balance.endpoint.PrefillEndpoint.Signal.PLACEMENT);
            verify(f.handoff, times(1)).close();
        }
    }

    @Test
    void wholeGroupIsClaimedBeforeFirstPublicationAndPredictionsAccumulateInOrder() {
        try (var f = new LinearDeliveryFixture(false)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            var published = new CopyOnWriteArrayList<Long>();
            doAnswer(call -> {
                DeliveryClaim claim = call.getArgument(0);
                assertEquals(List.of(1L, 2L), f.claimIds(), "publication must not race the next claim");
                assertFalse(Thread.holdsLock(claim.item().ctx()));
                assertFalse(f.queueLock.isHeldByCurrentThread());
                assertFalse(f.handoffClosed.get());
                published.add(claim.item().requestId());
                f.predictions.put(claim.item().requestId(), f.duration(((RouteDelivered) call.getArgument(1)).predictedAtMs(), ((RouteDelivered) call.getArgument(1)).nowMs()));
                return null;
            }).when(f.owner).onDeliveryEvidence(any(), isA(RouteDelivered.class));
            assertNull(f.process(List.of(first, second), OptionalLong.empty()));
            assertEquals(List.of(1L, 2L), published);
            assertEquals(10, f.predictions.get(1L).longValue());
            assertEquals(30, f.predictions.get(2L).longValue());
            assertArrayEquals(new long[]{10, 20}, f.committedRoutePredictions);
            verify(f.permits.get(1L)).release();
            verify(f.permits.get(2L)).release();
            verify(f.handoff, times(1)).close();
            verifyNoInteractions(f.grpc);
            verify(f.runtime.deliveries(), never()).tryReserveDelivery();
        }
    }

    @Test
    void unavailableSuffixCommitsLargestOrderedPrefixWithoutTouchingLaterMembers() {
        try (var f = new LinearDeliveryFixture(false)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            var third = f.item(3, 30);
            f.blockDecode(2);
            f.process(List.of(first, second, third), OptionalLong.empty());
            assertEquals(List.of(first), f.committedItems);
            assertArrayEquals(new long[]{10}, f.committedRoutePredictions);
            assertEquals(List.of(1L), f.claimIds());
            verify(f.owner).onDeliveryEvidence(eq(f.claims.getFirst()), argThat(e -> e instanceof RouteDelivered delivered
                    && f.duration(delivered.predictedAtMs(), delivered.nowMs()) == 10L));
            verify(f.decode, never()).acquireDispatchPermit(eq(third.decodeReservation()), any());
            verifyNoInteractions(f.grpc);
        }
    }

    @Test
    void unavailableHeadReturnsCapacityBoundaryWithoutPublishingAnyRoute() {
        try (var f = new LinearDeliveryFixture(false)) {
            var item = f.item(1, 10);
            f.blockDecode(1);
            var boundary = f.process(List.of(item), OptionalLong.empty());
            assertEquals(CapacityBoundary.Status.UNAVAILABLE, boundary.status());
            assertEquals(CapacityBoundary.Resource.DECODE_ENGINE, boundary.resource());
            assertTrue(f.committedItems.isEmpty());
            verify(f.owner, never()).claimDelivery(any(), any(), anyLong(), any());
            verify(f.owner, never()).onDeliveryEvidence(any(), isA(RouteDelivered.class));
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 2})
    void lostClaimDoesNotContributeToLaterRoutePrediction(int lostIndex) {
        try (var f = new LinearDeliveryFixture(false)) {
            var items = List.of(f.item(1, 10), f.item(2, 20), f.item(3, 30));
            f.lostClaims.add(items.get(lostIndex).requestId());
            var cumulative = new CopyOnWriteArrayList<Long>();
            doAnswer(call -> { cumulative.add(f.duration(((RouteDelivered) call.getArgument(1)).predictedAtMs(), ((RouteDelivered) call.getArgument(1)).nowMs())); return null; })
                    .when(f.owner).onDeliveryEvidence(any(), isA(RouteDelivered.class));
            f.process(items, OptionalLong.empty());
            long total = 0;
            var expected = new java.util.ArrayList<Long>();
            for (var item : items) {
                if (f.lostClaims.contains(item.requestId())) { continue; }
                total += item.seqLen();
                expected.add(total);
            }
            assertEquals(expected, cumulative);
            assertEquals(2, f.claims.size());
            verify(f.permits.get(items.get(lostIndex).requestId()), never()).dispatch();
            verify(f.permits.get(items.get(lostIndex).requestId())).release();
        }
    }

    @Test
    void routePublicationFailureDoesNotSuppressLaterMembersOrKillWorker() {
        try (var f = new LinearDeliveryFixture(false)) {
            var published = new CopyOnWriteArrayList<Long>();
            f.expectedShutdownFailure = new IllegalStateException("first owner failed");
            doAnswer(call -> {
                DeliveryClaim claim = call.getArgument(0);
                published.add(claim.item().requestId());
                if (claim.item().requestId() == 1) { throw f.expectedShutdownFailure; }
                return null;
            }).when(f.owner).onDeliveryEvidence(any(), isA(RouteDelivered.class));
            assertDoesNotThrow(() -> f.process(List.of(f.item(1, 10), f.item(2, 20)), OptionalLong.empty()));
            assertEquals(List.of(1L, 2L), published);
            verify(f.handoff).close();
            verify(f.permits.get(1L)).release();
            verify(f.permits.get(2L)).release();
        }
    }

    @Test
    void cleanupFailureStillClosesSiblingPermitAndGenerationHandoff() {
        try (var f = new LinearDeliveryFixture(false)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            f.expectedShutdownFailure = new IllegalStateException("first permit cleanup failed");
            doThrow(f.expectedShutdownFailure).when(f.permits.get(1L)).release();
            assertDoesNotThrow(() -> f.process(List.of(first, second), OptionalLong.empty()));
            verify(f.permits.get(1L)).release();
            verify(f.permits.get(2L)).release();
            verify(f.handoff).close();
            verify(f.owner, times(2)).onDeliveryEvidence(any(), isA(RouteDelivered.class));
        }
    }

    @Test
    void zeroPredictionStillClaimsAndPublishesEveryRoute() {
        try (var f = new LinearDeliveryFixture(false)) {
            f.process(List.of(f.item(1, 0), f.item(2, 0)), OptionalLong.empty());
            assertEquals(2, f.claims.size());
            verify(f.owner, times(2)).onDeliveryEvidence(any(), argThat(e -> e instanceof RouteDelivered delivered
                    && f.duration(delivered.predictedAtMs(), delivered.nowMs()) == 0L));
        }
    }

    @Test
    void transferredRealPermitsAreResolvedWithoutReleasingRequestResources() {
        try (var f = new LinearDeliveryFixture(false)) {
            var endpoint = f.realDecode();
            var first = f.realItem(endpoint, 1, 10);
            var second = f.realItem(endpoint, 2, 20);
            f.process(List.of(first, second), OptionalLong.empty());
            var snapshot = endpoint.resourceSnapshot();
            assertEquals(0, snapshot.activeDispatchPermits());
            assertEquals(0, snapshot.queuedCount());
            assertEquals(2, snapshot.requests().size(), "temporary permit close cannot release transferred requests");
            assertEquals(2, endpoint.routingView().engineCapacityUsed());
            assertEquals(200, endpoint.routingView().inflightHardKv());
            assertEquals(DecodeResources.ReservationReleaseResult.RELEASED,
                    endpoint.release(first.decodeReservation(), DecodeResources.ReleaseReason.NOT_SENT).status());
            assertEquals(1, endpoint.resourceSnapshot().requests().size());
            assertTrue(endpoint.resourceSnapshot().requests().containsKey(2L));
        }
    }

    @Test
    void lostRealClaimAbandonsOnlyItsPermitAndPreservesTransferredSibling() {
        try (var f = new LinearDeliveryFixture(false)) {
            var endpoint = f.realDecode();
            var first = f.realItem(endpoint, 1, 10);
            var second = f.realItem(endpoint, 2, 20);
            f.lostClaims.add(1L);
            var capacitySignals = new AtomicInteger();
            endpoint.watchEngineDispatchCapacity(capacitySignals::incrementAndGet);
            f.process(List.of(first, second), OptionalLong.empty());
            var snapshot = endpoint.resourceSnapshot();
            assertEquals(0, snapshot.activeDispatchPermits());
            assertEquals(1, snapshot.queuedCount());
            assertEquals(2, snapshot.requests().size());
            assertEquals(1, endpoint.routingView().engineCapacityUsed());
            assertTrue(org.flexlb.balance.endpoint.EndpointTestSupport.isQueued(snapshot, 1));
            assertFalse(org.flexlb.balance.endpoint.EndpointTestSupport.isQueued(snapshot, 2));
            assertEquals(List.of(2L), f.claimIds());
            assertEquals(1, capacitySignals.get(), "only the abandoned temporary permit releases dispatch capacity");
        }
    }

    @Test
    void failedQueueCommitReleasesRealPreparedPermitsWithoutPublishing() {
        try (var f = new LinearDeliveryFixture(false)) {
            var endpoint = f.realDecode();
            var first = f.realItem(endpoint, 1, 10);
            var second = f.realItem(endpoint, 2, 20);
            when(f.prefill.commitLocked(any(org.flexlb.balance.endpoint.PrefillState.CommitPlan.class))).thenReturn(null);
            f.process(List.of(first, second), OptionalLong.empty());
            var snapshot = endpoint.resourceSnapshot();
            assertEquals(0, snapshot.activeDispatchPermits());
            assertEquals(2, snapshot.queuedCount());
            assertEquals(0, endpoint.routingView().engineCapacityUsed());
            assertEquals(2, snapshot.requests().size());
            verify(f.owner, never()).claimDelivery(any(), any(), anyLong(), any());
            verify(f.owner, never()).onDeliveryEvidence(any(), isA(RouteDelivered.class));
            verify(f.handoff, never()).close();
        }
    }
}

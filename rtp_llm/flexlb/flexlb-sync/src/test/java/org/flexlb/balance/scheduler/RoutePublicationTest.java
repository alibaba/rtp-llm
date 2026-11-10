package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isA;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.argThat;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.isA;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * NON_BATCH contracts run through the same real Worker preparation/commit path.
 */
@Timeout(15)
class RoutePublicationTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void dequeueRejectionKeepsItsCapacityReasonAndReleasesPreparedPermits(boolean batch) {
        try (var f = new LinearDeliveryFixture(batch)) {
            var endpoint = f.realDecode();
            var first = f.realItem(endpoint, 1, 10);
            var second = f.realItem(endpoint, 2, 20);
            var boundary =
                    CapacityBoundary.unavailable(
                            batch
                                    ? CapacityBoundary.Resource.PREFILL_BATCH
                                    : CapacityBoundary.Resource.PREFILL_REQUEST);
            when(f.prefill.dequeueForDelivery(any()))
                    .thenReturn(CapacityBoundary.Attempt.rejected(boundary));
            assertSame(boundary, f.process(List.of(first, second), OptionalLong.empty()));
            assertEquals(0, endpoint.resourceSnapshot().activeDispatchPermits());
            assertEquals(2, endpoint.resourceSnapshot().queuedCount());
            assertTrue(f.claimIds().isEmpty());
        }
    }

    @Test
    void nonBatchBudgetUsesTheSumOfIndividualPredictions() {
        try (var f =
                new LinearDeliveryFixture(
                        false,
                        config -> {
                            config.decisionPolicy().setMaxRequests(2);
                            config.decisionPolicy().setMaxCollectionWaitMs(60_000L);
                            config.decisionPolicy().setMaxPredictedExecutionMs(25L);
                        })) {
            var first = f.item(1L, 10L);
            var second = f.item(2L, 20L);
            RequestExecutionTestSupport.schedule(f.worker);
            assertEquals(List.of(first), f.committedItems);
            assertEquals(List.of(1L), f.claimIds());
            assertArrayEquals(new long[] {10L}, f.committedRoutePredictions);
            verify(f.decode, never()).acquireDispatchPermit(eq(second.decodeReservation()), any());
            verify(f.evaluator, never()).predictBatchMs(any());
            verifyNoInteractions(f.grpc);
        }
    }

    @ParameterizedTest
    @ValueSource(longs = {0L, 1_000L})
    void invalidSuffixPredictionFailsThatRequestWithoutTerminatingTheValidPrefix(long budget) {
        try (var f =
                new LinearDeliveryFixture(
                        false,
                        config -> {
                            config.decisionPolicy().setMaxRequests(2);
                            config.decisionPolicy().setMaxCollectionWaitMs(0L);
                            config.decisionPolicy()
                                    .setMaxPredictedExecutionMs(budget == 0L ? null : budget);
                        })) {
            var first = f.item(1L, 10L);
            var second = f.item(2L, 20L);
            org.mockito.Mockito.doReturn(-1L).when(f.evaluator).estimateMs(20L, 0L);

            RequestExecutionTestSupport.schedule(f.worker);
            if (budget > 0L) {
                assertEquals(List.of(first), f.queueOwnership.captureQueue(10).items());
                RequestExecutionTestSupport.schedule(f.worker);
            }
            assertEquals(List.of(1L), f.claimIds());
            assertEquals(java.util.Set.of(2L), f.preparationFailures.keySet());
            verify(f.owner, times(1)).recordPreparationFailure(isA(PreparationFailure.class));
            verify(f.decode, never()).acquireDispatchPermit(eq(second.decodeReservation()), any());
            verify(f.permits.get(1L)).release();
        }
    }

    @Test
    void successfulPrefixIsPublishedBeforeHandlingAnUnrelatedPreparationFailure() {
        try (var f = new LinearDeliveryFixture(false)) {
            var first = f.item(1, 10);
            var failed = f.item(2, 20);
            doThrow(new IllegalStateException("Decode preparation failed"))
                    .when(f.decode)
                    .acquireDispatchPermit(eq(failed.decodeReservation()), any());

            f.expectedShutdownFailure = new IllegalStateException("failed member observer failed");
            var prefixPublished = new java.util.concurrent.atomic.AtomicBoolean();
            doAnswer(
                            call -> {
                                prefixPublished.set(f.claimIds().equals(List.of(1L)));
                                throw f.expectedShutdownFailure;
                            })
                    .when(f.owner)
                    .recordPreparationFailure(isA(PreparationFailure.class));

            f.process(List.of(first, failed), OptionalLong.empty());

            assertTrue(prefixPublished.get());
            assertEquals(List.of(1L), f.claimIds());
            verify(f.permits.get(1L)).release();
            assertFalse(
                    org.flexlb.balance.endpoint.EndpointTestSupport.isQueued(
                            f.queueOwnership, failed));
        }
    }

    @Test
    void fixedWindowGroupsRouteDecisionsWithoutCreatingAnEngineBatch() {
        try (var f =
                new LinearDeliveryFixture(
                        false,
                        config -> {
                            config.decisionPolicy().setMaxRequests(2);
                            config.decisionPolicy().setMaxCollectionWaitMs(60_000L);
                            config.decisionPolicy().setMaxPredictedExecutionMs(1_000L);
                        })) {
            var first = f.item(1L, 10L);
            RequestExecutionTestSupport.schedule(f.worker);
            assertTrue(f.claims.isEmpty(), "NON_BATCH must still honor its decision window");
            var second = f.item(2L, 20L);
            RequestExecutionTestSupport.schedule(f.worker);
            assertEquals(List.of(first, second), f.committedItems);
            assertEquals(List.of(1L, 2L), f.claimIds());
            assertArrayEquals(new long[] {10L, 20L}, f.committedRoutePredictions);
            assertEquals(10L, f.predictions.get(1L));
            assertEquals(30L, f.predictions.get(2L));
            verify(f.evaluator, never()).predictBatchMs(any());
            verify(f.prefill, never())
                    .batchCapacityAvailable(org.mockito.ArgumentMatchers.anyInt());
            verify(f.runtime.deliveries(), never()).tryReserveDelivery();
            verify(f.runtime.deliveries(), never()).nextBatchId();
            verifyNoInteractions(f.grpc);
        }
    }

    @Test
    void prefixDeliveryPublishesAndFailedSuffixReleasesItsQueueEntry() {
        try (var f = new LinearDeliveryFixture(false)) {
            var first = f.item(1, 10);
            var failed = f.item(2, 20);
            var failure = new IllegalStateException("Decode preparation failed");
            doThrow(failure)
                    .when(f.decode)
                    .acquireDispatchPermit(eq(failed.decodeReservation()), any());

            assertNull(f.process(List.of(first, failed), OptionalLong.empty()));
            assertEquals(List.of(first), f.committedItems);
            verify(f.owner)
                    .recordPreparationFailure(
                            argThat(
                                    e ->
                                            e instanceof PreparationFailure evidence
                                                    && evidence.route() == failed
                                                    && evidence.detail()
                                                            .equals(failure.getMessage())));
            verify(f.owner)
                    .onDeliveryResult(
                            argThat(
                                    e ->
                                            e instanceof RouteDelivered delivered
                                                    && f.duration(
                                                                    delivered.predictedAtMs(),
                                                                    delivered.nowMs())
                                                            == 10L));
            verify(f.prefill).signal(org.flexlb.balance.endpoint.PrefillEndpoint.Signal.PLACEMENT);
            verify(f.prefill).releaseQueuedRequest(failed);
            assertEquals(0, f.queueOwnership.queueDepth());
            verify(f.prefill, never()).close();
        }
    }

    @Test
    void wholeGroupIsPreparedBeforeOrderedCommitAndPublication() {
        try (var f = new LinearDeliveryFixture(false)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            var published = new CopyOnWriteArrayList<Long>();
            doAnswer(
                            call -> {
                                RouteDelivered delivered = call.getArgument(0);
                                Delivery.Member claim = delivered.delivery();
                                assertEquals(List.of(first, second), f.committedItems);
                                verify(f.decode)
                                        .acquireDispatchPermit(
                                                eq(first.decodeReservation()), any());
                                verify(f.decode)
                                        .acquireDispatchPermit(
                                                eq(second.decodeReservation()), any());
                                assertEquals(
                                        List.of(1L, 2L).subList(0, published.size() + 1),
                                        f.claimIds(),
                                        "each prepared member commits immediately before its publication");
                                assertSame(claim, claim.owner().snapshot().delivery().identity());
                                assertSame(
                                        f.permits.get(Long.parseLong(claim.route().requestId())),
                                        claim.decode());
                                assertFalse(Thread.holdsLock(claim.route().ctx()));
                                assertFalse(f.queueLock.isHeldByCurrentThread());
                                assertFalse(f.prefill.isRetired());
                                published.add(Long.parseLong(claim.route().requestId()));
                                f.predictions.put(
                                        Long.parseLong(claim.route().requestId()),
                                        f.duration(
                                                ((RouteDelivered) call.getArgument(0))
                                                        .predictedAtMs(),
                                                ((RouteDelivered) call.getArgument(0)).nowMs()));
                                return null;
                            })
                    .when(f.owner)
                    .onDeliveryResult(isA(RouteDelivered.class));
            assertNull(f.process(List.of(first, second), OptionalLong.empty()));
            assertEquals(List.of(1L, 2L), published);
            assertEquals(10, f.predictions.get(1L).longValue());
            assertEquals(30, f.predictions.get(2L).longValue());
            assertArrayEquals(new long[] {10, 20}, f.committedRoutePredictions);
            verify(f.permits.get(1L)).release();
            verify(f.permits.get(2L)).release();
            verify(f.prefill, never()).close();
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
            assertArrayEquals(new long[] {10}, f.committedRoutePredictions);
            assertEquals(List.of(1L), f.claimIds());
            verify(f.owner)
                    .onDeliveryResult(
                            argThat(
                                    e ->
                                            e instanceof RouteDelivered delivered
                                                    && f.duration(
                                                                    delivered.predictedAtMs(),
                                                                    delivered.nowMs())
                                                            == 10L));
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
            verify(
                            org.flexlb.balance.scheduler.DeliveryTestSupport.transport(
                                    f.owner.dispatcher),
                            never())
                    .commit(
                            any(Delivery.Member.class),
                            any(),
                            anyLong(),
                            org.mockito.ArgumentMatchers.isNull());
            verify(f.owner, never()).onDeliveryResult(isA(RouteDelivered.class));
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 2})
    void lostClaimDoesNotContributeToLaterRoutePrediction(int lostIndex) {
        try (var f = new LinearDeliveryFixture(false)) {
            var items = List.of(f.item(1, 10), f.item(2, 20), f.item(3, 30));
            f.lostClaims.add(Long.parseLong(items.get(lostIndex).requestId()));
            var cumulative = new CopyOnWriteArrayList<Long>();
            doAnswer(
                            call -> {
                                cumulative.add(
                                        f.duration(
                                                ((RouteDelivered) call.getArgument(0))
                                                        .predictedAtMs(),
                                                ((RouteDelivered) call.getArgument(0)).nowMs()));
                                return null;
                            })
                    .when(f.owner)
                    .onDeliveryResult(isA(RouteDelivered.class));
            f.process(items, OptionalLong.empty());
            long total = 0;
            var expected = new java.util.ArrayList<Long>();
            for (var item : items) {
                if (f.lostClaims.contains(Long.parseLong(item.requestId()))) {
                    continue;
                }
                total += item.seqLen();
                expected.add(total);
            }
            assertEquals(expected, cumulative);
            assertEquals(2, f.claims.size());
            verify(f.permits.get(Long.parseLong(items.get(lostIndex).requestId())), never())
                    .dispatch();
            verify(f.permits.get(Long.parseLong(items.get(lostIndex).requestId()))).release();
        }
    }

    @Test
    void routePublicationFailureDoesNotSuppressLaterMembersOrKillWorker() {
        try (var f = new LinearDeliveryFixture(false)) {
            var published = new CopyOnWriteArrayList<Long>();
            f.expectedShutdownFailure = new IllegalStateException("first owner failed");
            doAnswer(
                            call -> {
                                RouteDelivered delivered = call.getArgument(0);
                                Delivery.Member claim = delivered.delivery();
                                published.add(Long.parseLong(claim.route().requestId()));
                                if (claim.route().requestId().equals("1")) {
                                    throw f.expectedShutdownFailure;
                                }
                                return null;
                            })
                    .when(f.owner)
                    .onDeliveryResult(isA(RouteDelivered.class));
            assertDoesNotThrow(
                    () -> f.process(List.of(f.item(1, 10), f.item(2, 20)), OptionalLong.empty()));
            assertEquals(List.of(1L, 2L), published);
            verify(f.prefill, never()).close();
            verify(f.permits.get(1L)).release();
            verify(f.permits.get(2L)).release();
        }
    }

    @Test
    void cleanupFailureStillReleasesSiblingPermit() {
        try (var f = new LinearDeliveryFixture(false)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            f.expectedShutdownFailure = new IllegalStateException("first permit cleanup failed");
            doThrow(f.expectedShutdownFailure).when(f.permits.get(1L)).release();
            assertDoesNotThrow(() -> f.process(List.of(first, second), OptionalLong.empty()));
            verify(f.permits.get(1L)).release();
            verify(f.permits.get(2L)).release();
            verify(f.prefill, never()).close();
            verify(f.owner, times(2)).onDeliveryResult(isA(RouteDelivered.class));
        }
    }

    @Test
    void zeroPredictionStillClaimsAndPublishesEveryRoute() {
        try (var f = new LinearDeliveryFixture(false)) {
            f.process(List.of(f.item(1, 0), f.item(2, 0)), OptionalLong.empty());
            assertEquals(2, f.claims.size());
            verify(f.owner, times(2))
                    .onDeliveryResult(
                            argThat(
                                    e ->
                                            e instanceof RouteDelivered delivered
                                                    && f.duration(
                                                                    delivered.predictedAtMs(),
                                                                    delivered.nowMs())
                                                            == 0L));
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
            assertEquals(
                    2,
                    snapshot.requests().size(),
                    "temporary permit close cannot release transferred requests");
            assertEquals(2, endpoint.routingView().engineCapacityUsed());
            assertEquals(200, endpoint.routingView().inflightHardKv());
            assertEquals(
                    DecodeResources.ReservationReleaseResult.RELEASED,
                    endpoint.release(
                                    first.decodeReservation(),
                                    DecodeResources.ReleaseReason.NOT_SENT)
                            .status());
            assertEquals(1, endpoint.resourceSnapshot().requests().size());
            assertTrue(endpoint.resourceSnapshot().requests().containsKey(String.valueOf(2L)));
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
            assertTrue(
                    org.flexlb.balance.endpoint.EndpointTestSupport.isQueued(
                            snapshot, String.valueOf(1)));
            assertFalse(
                    org.flexlb.balance.endpoint.EndpointTestSupport.isQueued(
                            snapshot, String.valueOf(2)));
            assertEquals(List.of(2L), f.claimIds());
            assertEquals(
                    1,
                    capacitySignals.get(),
                    "only the abandoned temporary permit releases dispatch capacity");
        }
    }

    @Test
    void failedQueueCommitReleasesRealPreparedPermitsWithoutPublishing() {
        try (var f = new LinearDeliveryFixture(false)) {
            var endpoint = f.realDecode();
            var first = f.realItem(endpoint, 1, 10);
            var second = f.realItem(endpoint, 2, 20);
            when(f.prefill.dequeueForDelivery(
                            any(
                                    org.flexlb.balance.endpoint.PrefillEndpoint.DeliverySelection
                                            .class)))
                    .thenReturn(
                            org.flexlb.balance.delivery.CapacityBoundary.Attempt.rejected(
                                    org.flexlb.balance.delivery.CapacityBoundary.OWNERSHIP_LOST));
            f.process(List.of(first, second), OptionalLong.empty());
            var snapshot = endpoint.resourceSnapshot();
            assertEquals(0, snapshot.activeDispatchPermits());
            assertEquals(2, snapshot.queuedCount());
            assertEquals(0, endpoint.routingView().engineCapacityUsed());
            assertEquals(2, snapshot.requests().size());
            verify(
                            org.flexlb.balance.scheduler.DeliveryTestSupport.transport(
                                    f.owner.dispatcher),
                            never())
                    .commit(
                            any(Delivery.Member.class),
                            any(),
                            anyLong(),
                            org.mockito.ArgumentMatchers.isNull());
            verify(f.owner, never()).onDeliveryResult(isA(RouteDelivered.class));
            verify(f.prefill, never()).close();
        }
    }
}

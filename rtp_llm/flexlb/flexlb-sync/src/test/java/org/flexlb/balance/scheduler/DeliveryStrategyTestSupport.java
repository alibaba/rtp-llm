package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.dao.route.RoleType;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.function.BiConsumer;
import java.util.function.BooleanSupplier;

/** Exact-capability fakes for final delivery-strategy tests. */
final class DeliveryStrategyTestSupport {

    static final PrefillTimePredictor.Evaluator EVALUATOR =
            new PrefillTimePredictor.Evaluator() {
                @Override
                public long estimateMs(long totalTokens, long hitTokens) {
                    return totalTokens - hitTokens;
                }

                @Override
                public double predictBatchMs(PrefillBatchFeatures features) {
                    return features.items().stream()
                            .mapToLong(PrefillBatchFeatures.Item::seqLen)
                            .sum();
                }
            };

    private DeliveryStrategyTestSupport() {
    }

    static ScheduledRequest item(long requestId) {
        return item(requestId, 50, requestId, 100L, 10L);
    }

    static ScheduledRequest item(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long seqLen,
            long hitCache) {
        ScheduledRequest item = Mockito.mock(ScheduledRequest.class);
        Mockito.when(item.requestId()).thenReturn(requestId);
        Mockito.when(item.priority()).thenReturn(priority);
        Mockito.when(item.enqueuedAtMs()).thenReturn(enqueuedAtMs);
        Mockito.when(item.seqLen()).thenReturn(seqLen);
        Mockito.when(item.hitCache()).thenReturn(hitCache);
        return item;
    }

    record TestBoundary(ScheduledRequest item, CapacityBoundary result) {
    }

    static final class TestContext {

        private boolean commit = true;
        private DeliveryStrategy.Transaction preparedSelection;
        private TestBoundary committedBoundary;
        private TestBoundary emptyBoundary;

        String deliver(
                DeliveryStrategy strategy,
                List<ScheduledRequest> candidates,
                String decisionReason,
                int remainingQueueDepth,
                OptionalLong plannedPredictionMs) {
            try (DeliveryStrategy.Transaction transaction = strategy.prepare(
                    candidates, EVALUATOR, plannedPredictionMs)) {
                if (transaction.items().isEmpty()) {
                    emptyBoundary = new TestBoundary(
                            transaction.blockedItem(),
                            transaction.blockedResult());
                    return "BOUNDARY";
                }
                preparedSelection = transaction;
                if (transaction.blockedItem() != null) {
                    committedBoundary = new TestBoundary(
                            transaction.blockedItem(),
                            transaction.blockedResult());
                }
                if (!commit) {
                    return "NOT_COMMITTED";
                }
                transaction.commitUnderLock();
                transaction.handoff(decisionReason, remainingQueueDepth);
                return "COMMITTED";
            }
        }

        void commit(boolean value) {
            commit = value;
        }

        DeliveryStrategy.Transaction preparedSelection() {
            return preparedSelection;
        }

        TestBoundary committedBoundary() {
            return committedBoundary;
        }

        TestBoundary emptyBoundary() {
            return emptyBoundary;
        }

    }

    /** Concrete endpoint capabilities used by the real delivery transaction. */
    static final class TestEndpointCapabilities {
        private final PrefillEndpoint prefill = Mockito.mock(PrefillEndpoint.class);
        private final DecodeEndpoint decode = Mockito.mock(DecodeEndpoint.class);
        private final Map<ScheduledRequest, PrefillState.RouteReservation>
                routeReservations = new IdentityHashMap<>();
        private final Map<ScheduledRequest, DecodeEndpoint.EngineDispatchPermit>
                permits = new IdentityHashMap<>();
        private final Map<Long, ScheduledRequest> itemsByRequestId =
                new HashMap<>();
        private final List<PrefillState.CommittedHandoff> handoffs =
                new ArrayList<>();
        private PrefillState.BatchReservation batchReservation;
        private int routeAttempt;
        private int permitAttempt;
        private int rejectRouteAt = -1;
        private int rejectPermitAt = -1;

        TestEndpointCapabilities() {
            CapacityBoundary.Availability unavailable = unavailable().availability();
            Mockito.when(prefill.routeAdmissionAvailability(Mockito.anyInt()))
                    .thenReturn(unavailable);
            Mockito.when(prefill.batchAdmissionAvailability(Mockito.anyInt()))
                    .thenReturn(unavailable);
            Mockito.when(prefill.reserveRoute(
                            Mockito.any(), Mockito.anyLong(), Mockito.anyInt()))
                    .thenAnswer(invocation -> reserveRoute(invocation.getArgument(0)));
            Mockito.when(prefill.reserveBatch(
                            Mockito.any(), Mockito.anyLong(), Mockito.anyInt()))
                    .thenAnswer(invocation -> reserveBatch());
            Mockito.when(decode.acquireEngineDispatchPermit(
                            Mockito.anyLong(), Mockito.anyLong(), Mockito.anyLong()))
                    .thenAnswer(invocation -> acquirePermit(invocation.getArgument(0)));
        }

        void bind(ScheduledRequest... items) {
            for (ScheduledRequest item : items) {
                itemsByRequestId.put(item.requestId(), item);
                DecodeEndpoint.ReservationHandle reservation = Mockito.mock(
                        DecodeEndpoint.ReservationHandle.class);
                Mockito.when(item.prefillEp()).thenReturn(prefill);
                Mockito.when(item.decodeBinding()).thenReturn(
                        new ScheduledRequest.DecodeBinding(
                                null, decode, reservation));
                Mockito.when(item.selectDecodeForDispatch()).thenReturn(
                        PlacementResult.blocked(RoleType.DECODE));
            }
        }

        void rejectRouteAt(int preparedSize) {
            rejectRouteAt = preparedSize;
        }

        void rejectPermitAt(int preparedSize) {
            rejectPermitAt = preparedSize;
        }

        PrefillState.RouteReservation routeReservation(ScheduledRequest item) {
            return routeReservations.get(item);
        }

        PrefillEndpoint prefill() {
            return prefill;
        }

        DecodeEndpoint.EngineDispatchPermit permit(ScheduledRequest item) {
            return permits.get(item);
        }

        PrefillState.BatchReservation batchReservation() {
            return batchReservation;
        }

        List<PrefillState.CommittedHandoff> handoffs() {
            return List.copyOf(handoffs);
        }

        private PrefillState.ReservationResult<PrefillState.RouteReservation>
                reserveRoute(ScheduledRequest item) {
            if (routeAttempt++ == rejectRouteAt) {
                return new PrefillState.ReservationResult<>(
                        PrefillState.CapacityStatus.CAPACITY_FULL, null);
            }
            PrefillState.RouteReservation reservation = Mockito.mock(
                    PrefillState.RouteReservation.class);
            routeReservations.put(item, reservation);
            Mockito.when(reservation.commitGroup(
                            Mockito.anyList(), Mockito.anyList()))
                    .thenAnswer(invocation -> committedHandoffs(
                            ((List<?>) invocation.getArgument(0)).size()));
            return new PrefillState.ReservationResult<>(
                    PrefillState.CapacityStatus.ACQUIRED, reservation);
        }

        private PrefillState.ReservationResult<PrefillState.BatchReservation>
                reserveBatch() {
            batchReservation = Mockito.mock(PrefillState.BatchReservation.class);
            Mockito.when(batchReservation.commit(
                            Mockito.anyList(), Mockito.anyLong()))
                    .thenAnswer(invocation -> committedHandoffs(1).getFirst());
            return new PrefillState.ReservationResult<>(
                    PrefillState.CapacityStatus.ACQUIRED, batchReservation);
        }

        private DecodeEndpoint.EngineDispatchPermitAcquisition acquirePermit(
                long requestId) {
            if (permitAttempt++ == rejectPermitAt) {
                return new DecodeEndpoint.EngineDispatchPermitAcquisition(
                        DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                        null,
                        null);
            }
            ScheduledRequest item = itemsByRequestId.get(requestId);
            DecodeEndpoint.EngineDispatchPermit permit = Mockito.mock(
                    DecodeEndpoint.EngineDispatchPermit.class);
            Mockito.when(permit.transferToEngineLifecycle()).thenReturn(
                    DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED);
            if (item != null) {
                permits.put(item, permit);
            }
            return new DecodeEndpoint.EngineDispatchPermitAcquisition(
                    DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED,
                    null,
                    permit);
        }

        private List<PrefillState.CommittedHandoff> committedHandoffs(int count) {
            List<PrefillState.CommittedHandoff> committed = new ArrayList<>(count);
            for (int index = 0; index < count; index++) {
                PrefillState.CommittedHandoff handoff = Mockito.mock(
                        PrefillState.CommittedHandoff.class);
                handoffs.add(handoff);
                committed.add(handoff);
            }
            return committed;
        }
    }

    static final class TestRequestRegistry {

        private final RequestRegistry requests = Mockito.mock(RequestRegistry.class);
        private final List<ScheduledRequest> prepared = new ArrayList<>();
        private final List<ScheduledRequest> committed = new ArrayList<>();
        private final List<ClaimIdentity> identities = new ArrayList<>();
        private final List<CompletionEvent> completions = new ArrayList<>();
        private final List<ScheduledRequest> failedPrepared = new ArrayList<>();
        private final List<Throwable> preparedFailures = new ArrayList<>();
        private final List<String> events = new ArrayList<>();
        private ScheduledRequest preparationLostFor;
        private ScheduledRequest commitLostFor;
        private ScheduledRequest throwCommitFor;
        private ScheduledRequest throwCompletionFor;
        private Runnable beforeCompletion = () -> { };

        TestRequestRegistry() {
            Mockito.doAnswer(invocation -> prepareIfOwned(
                    invocation.getArgument(0), invocation.getArgument(1)))
                    .when(requests).prepareIfOwned(
                            Mockito.any(), Mockito.any());
            Mockito.doAnswer(invocation -> claim(
                    invocation.getArgument(0),
                    DeliveryClaimKind.ROUTE_DECISION,
                    0L,
                    invocation.getArgument(1)))
                    .when(requests).tryClaimRouteDelivery(
                            Mockito.any(), Mockito.any());
            Mockito.doAnswer(invocation -> claim(
                    invocation.getArgument(0),
                    DeliveryClaimKind.BATCH_ENQUEUE,
                    invocation.getArgument(1),
                    invocation.getArgument(2)))
                    .when(requests).tryClaimBatchDelivery(
                            Mockito.any(), Mockito.anyLong(), Mockito.any());
            Mockito.doAnswer(invocation -> {
                complete(invocation.getArgument(0), invocation.getArgument(1));
                return null;
            }).when(requests).complete(Mockito.any(), Mockito.any());
            Mockito.doAnswer(invocation -> {
                failPrepared(invocation.getArgument(0), invocation.getArgument(1));
                return null;
            }).when(requests).failPrepared(Mockito.any(), Mockito.any());
        }

        private <T> Optional<T> prepareIfOwned(
                ScheduledRequest exactItem,
                java.util.function.Supplier<T> preparation) {
            if (exactItem == preparationLostFor) {
                return Optional.empty();
            }
            prepared.add(exactItem);
            return Optional.of(preparation.get());
        }

        private RequestRegistry.DeliveryClaim claim(
                ScheduledRequest exactItem,
                DeliveryClaimKind kind,
                long correlationId,
                BooleanSupplier endpointHandoff) {
            committed.add(exactItem);
            identities.add(new ClaimIdentity(kind, correlationId));
            if (exactItem == throwCommitFor) {
                throw new IllegalStateException(
                        "synthetic slot commit failure "
                                + exactItem.requestId());
            }
            if (exactItem == commitLostFor) {
                return null;
            }
            if (!endpointHandoff.getAsBoolean()) {
                return null;
            }
            events.add("point-of-no-return-" + exactItem.requestId());
            RequestRegistry.DeliveryClaim claim =
                    Mockito.mock(RequestRegistry.DeliveryClaim.class);
            Mockito.when(claim.item()).thenReturn(exactItem);
            return claim;
        }

        private void complete(
                RequestRegistry.DeliveryClaim exactClaim,
                DeliveryResult completion) {
            beforeCompletion.run();
            completions.add(new CompletionEvent(exactClaim.item(), completion));
            events.add("complete-" + exactClaim.item().requestId());
            if (exactClaim.item() == throwCompletionFor) {
                throw new IllegalStateException(
                        "synthetic completion failure "
                                + exactClaim.item().requestId());
            }
        }

        private void failPrepared(ScheduledRequest exactItem, Throwable cause) {
            failedPrepared.add(exactItem);
            preparedFailures.add(cause);
        }

        void preparationLostFor(ScheduledRequest item) {
            preparationLostFor = item;
        }

        void commitLostFor(ScheduledRequest item) {
            commitLostFor = item;
        }

        void throwCommitFor(ScheduledRequest item) {
            throwCommitFor = item;
        }

        void throwCompletionFor(ScheduledRequest item) {
            throwCompletionFor = item;
        }

        void beforeCompletion(Runnable check) {
            beforeCompletion = check;
        }

        List<ScheduledRequest> prepared() {
            return List.copyOf(prepared);
        }

        List<ScheduledRequest> committed() {
            return List.copyOf(committed);
        }

        List<ClaimIdentity> identities() {
            return List.copyOf(identities);
        }

        List<CompletionEvent> completions() {
            return List.copyOf(completions);
        }

        List<ScheduledRequest> failedPrepared() {
            return List.copyOf(failedPrepared);
        }

        List<Throwable> preparedFailures() {
            return List.copyOf(preparedFailures);
        }

        List<String> events() {
            return List.copyOf(events);
        }

        RequestRegistry requests() {
            return requests;
        }
    }

    record ClaimIdentity(DeliveryClaimKind kind, long correlationId) {
    }

    record CompletionEvent(
            ScheduledRequest item,
            DeliveryResult completion) {
    }

    record SubmittedBatch(
            List<ScheduledRequest> exactItems,
            long batchId,
            long predictedMs,
            String decisionReason) {
    }

    static final class TestBatchSubmission {

        private CapacityBoundary prepareBoundary;
        private int prepareCount;
        private int closeCount;
        private int totalCloseCount;
        private SubmittedBatch command;
        private BiConsumer<ScheduledRequest, DeliveryResult> observer;
        private final List<CompletionEvent> synchronousCompletions =
                new ArrayList<>();
        private final List<String> events = new ArrayList<>();

        CapacityBoundary.Attempt<BatchDeliveryStrategy.PreparedSubmission>
                tryPrepareSubmission() {
            prepareCount++;
            if (prepareBoundary != null) {
                return CapacityBoundary.Attempt.rejected(prepareBoundary);
            }
            return CapacityBoundary.Attempt.accepted(
                    new BatchDeliveryStrategy.PreparedSubmission() {
                        private boolean submitted;

                        @Override
                        public void submitBatch(
                                List<ScheduledRequest> exactItems,
                                long batchId,
                                long predictedMs,
                                String decisionReason,
                                BiConsumer<ScheduledRequest, DeliveryResult> exactObserver) {
                            submitted = true;
                            command = new SubmittedBatch(
                                    exactItems,
                                    batchId,
                                    predictedMs,
                                    decisionReason);
                            observer = exactObserver;
                            events.add("submit");
                            for (CompletionEvent completion
                                    : synchronousCompletions) {
                                exactObserver.accept(
                                        completion.item(),
                                        completion.completion());
                            }
                        }

                        @Override
                        public void close() {
                            totalCloseCount++;
                            if (!submitted) {
                                closeCount++;
                            }
                            events.add("submission-close");
                        }
                    });
        }

        void prepareBoundary(CapacityBoundary value) {
            prepareBoundary = value;
        }

        void completeSynchronously(
                ScheduledRequest item,
                DeliveryResult completion) {
            synchronousCompletions.add(new CompletionEvent(item, completion));
        }

        void complete(
                ScheduledRequest item,
                DeliveryResult completion) {
            observer.accept(item, completion);
        }

        int prepareCount() {
            return prepareCount;
        }

        int closeCount() {
            return closeCount;
        }

        int totalCloseCount() {
            return totalCloseCount;
        }

        SubmittedBatch command() {
            return command;
        }

        BiConsumer<ScheduledRequest, DeliveryResult> observer() {
            return observer;
        }

        List<String> events() {
            return List.copyOf(events);
        }
    }

    static final class TestTelemetry {

        private final List<List<ScheduledRequest>> routes = new ArrayList<>();
        private final List<BatchTelemetry> batches = new ArrayList<>();
        private final DeliveryMetrics metrics =
                org.mockito.Mockito.mock(DeliveryMetrics.class);

        TestTelemetry() {
            org.mockito.Mockito.doAnswer(invocation -> {
                routesDelivered(invocation.getArgument(0), invocation.getArgument(1));
                return null;
            }).when(metrics).routesDelivered(
                    org.mockito.ArgumentMatchers.anyInt(),
                    org.mockito.ArgumentMatchers.anyList());
            org.mockito.Mockito.doAnswer(invocation -> {
                batchDispatched(
                        invocation.getArgument(0), invocation.getArgument(1),
                        invocation.getArgument(2), invocation.getArgument(3),
                        invocation.getArgument(4));
                return null;
            }).when(metrics).batchDispatched(
                    org.mockito.ArgumentMatchers.anyLong(),
                    org.mockito.ArgumentMatchers.anyString(),
                    org.mockito.ArgumentMatchers.anyInt(),
                    org.mockito.ArgumentMatchers.anyList(),
                    org.mockito.ArgumentMatchers.anyLong());
        }

        private void routesDelivered(
                int remainingQueueDepth,
                List<ScheduledRequest> exactItems) {
            routes.add(List.copyOf(exactItems));
        }

        private void batchDispatched(
                long batchId,
                String decisionReason,
                int remainingQueueDepth,
                List<ScheduledRequest> dispatched,
                long predictedMs) {
            batches.add(new BatchTelemetry(
                    batchId, decisionReason, remainingQueueDepth,
                    dispatched, predictedMs));
        }

        List<List<ScheduledRequest>> routes() {
            return List.copyOf(routes);
        }

        List<BatchTelemetry> batches() {
            return List.copyOf(batches);
        }

        DeliveryMetrics metrics() {
            return metrics;
        }
    }

    record BatchTelemetry(
            long batchId,
            String decisionReason,
            int remainingQueueDepth,
            List<ScheduledRequest> dispatched,
            long predictedMs) {
        BatchTelemetry {
            dispatched = List.copyOf(dispatched);
        }
    }

    static CapacityBoundary unavailable() {
        return CapacityBoundary.unavailable(
                new CapacityBoundary.Availability() {
                    @Override
                    public boolean isAvailable() {
                        return false;
                    }

                    @Override
                    public void addListener(Runnable listener) {
                    }

                    @Override
                    public void removeListener(Runnable listener) {
                    }
                },
                new RouteProjection.AdmissionBlockSemantics(
                        "TEST_BLOCK",
                        RouteProjection.AfterProbeAdmission.BLOCKED,
                        "TEST_BLOCK",
                        RoleType.PREFILL));
    }
}

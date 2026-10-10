package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.prediction.DecodeCostFormula;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.mockito.Mockito;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.function.BiConsumer;

/** Exact-capability fakes for final delivery-strategy tests. */
public final class DeliveryStrategyTestSupport {

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

    public static void stubRouteDelivery(AbstractRequestScheduler requests,
            java.util.function.BiConsumer<DeliveryClaim, DeliveryResult> completed) {
        Mockito.when(requests.ownsPreparedDeliveryLocked(Mockito.any(), Mockito.any())).thenReturn(true);
        Mockito.doAnswer(invocation -> {
            var permit = invocation.getArgument(3, DecodeEndpoint.EngineDispatchPermit.class);
            if (permit != null && permit.dispatch() != DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED) { return null; }
            DeliveryClaim claim = Mockito.mock(DeliveryClaim.class);
            ReflectionTestUtils.setField(claim, "item", invocation.getArgument(0));
            Mockito.doAnswer(inv -> { completed.accept(claim, DeliveryResult.delivered()); return null; })
                    .when(requests).publishRoute(Mockito.eq(claim), Mockito.any(), Mockito.anyLong());
            return claim;
        }).when(requests).claimDelivery(Mockito.any(), Mockito.eq(DeliveryClaimKind.ROUTE_DECISION), Mockito.eq(0L), Mockito.any());
    }

    private DeliveryStrategyTestSupport() {
    }

    static RequestRoute item(long requestId) {
        return item(requestId, 50, requestId, 100L, 10L);
    }

    static RequestRoute item(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long seqLen,
            long hitCache) {
        RequestRoute item = Mockito.mock(RequestRoute.class);
        Mockito.when(item.requestId()).thenReturn(requestId);
        Mockito.when(item.priority()).thenReturn(priority);
        Mockito.when(item.enqueuedAtMs()).thenReturn(enqueuedAtMs);
        Mockito.when(item.seqLen()).thenReturn(seqLen);
        Mockito.when(item.hitCache()).thenReturn(hitCache);
        return item;
    }

    record TestBoundary(RequestRoute item, CapacityBoundary result) {
    }

    static final class TestContext {

        private boolean commit = true;
        private DeliveryTransaction preparedSelection;
        private TestBoundary committedBoundary;
        private TestBoundary emptyBoundary;

        String deliver(
                DeliveryStrategy strategy,
                List<RequestRoute> candidates,
                String decisionReason,
                int remainingQueueDepth,
                OptionalLong plannedPredictionMs) {
            try (DeliveryTransaction transaction = strategy.prepare(
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
                WorkSnapshot precedingWork = transaction.commitSelectionLocked(System.currentTimeMillis()).precedingWork().materialize();
                strategy.deliver(transaction, decisionReason, remainingQueueDepth, precedingWork, EVALUATOR);
                return "COMMITTED";
            }
        }

        void commit(boolean value) {
            commit = value;
        }

        DeliveryTransaction preparedSelection() {
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
        private WorkSnapshot precedingWork = new WorkSnapshot(1_000L, List.of(), List.of(), 0L);
        private final PrefillEndpoint prefill = Mockito.mock(PrefillEndpoint.class);
        private final PrefillEndpoint.RouteCommitAdmission routeCommit =
                Mockito.mock(PrefillEndpoint.RouteCommitAdmission.class);
        private final DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        private final Map<RequestRoute, DecodeEndpoint.EngineDispatchPermit>
                permits = new IdentityHashMap<>();
        private final Map<Long, RequestRoute> itemsByRequestId =
                new HashMap<>();
        private final List<PrefillState.CommittedHandoff> handoffs =
                new ArrayList<>();
        private PrefillState.BatchReservation batchReservation;
        private int permitAttempt;
        private int rejectPermitAt = -1;

        TestEndpointCapabilities() {
            CapacityBoundary.Availability unavailable = unavailable().availability();
            Mockito.when(prefill.batchAdmissionAvailability(Mockito.anyInt()))
                    .thenReturn(unavailable);
            Mockito.when(prefill.tryBeginRouteCommitAdmission())
                    .thenReturn(routeCommit);
            Mockito.when(prefill.commitQueuedRoutesLocked(
                            Mockito.anyList(), Mockito.any(long[].class), Mockito.any(), Mockito.anyLong()))
                    .thenAnswer(invocation -> committedHandoffs(1).getFirst());
            Mockito.when(prefill.reserveBatch(
                            Mockito.any(), Mockito.anyLong(), Mockito.anyInt()))
                    .thenAnswer(invocation -> reserveBatch());
            Mockito.when(decode.acquireDispatchPermit(Mockito.any(), Mockito.any()))
                    .thenAnswer(invocation -> acquirePermit(
                            ((DecodeResources.ReservationHandle) invocation.getArgument(0)).requestId()));
        }

        void bind(RequestRoute... items) {
            for (RequestRoute item : items) {
                long requestId = item.requestId();
                itemsByRequestId.put(requestId, item);
                DecodeResources.ReservationHandle reservation = Mockito.mock(
                        DecodeResources.ReservationHandle.class);
                Mockito.when(reservation.requestId()).thenReturn(requestId);
                Mockito.when(item.prefillEp()).thenReturn(prefill);
                Mockito.when(item.requiresRouteReservation()).thenReturn(true);
                RequestRequirements binding = new RequestRequirements(
                        requestId, org.flexlb.dao.SchedulingMetadata.explicit(item.priority(), Long.MAX_VALUE), item.seqLen(),
                        new DecodeResources.AdmissionCapacity(0L, 100L),
                        RequestRequirements.DecodeMode.WAIT_AT_PLACEMENT,
                        DecodeCostFormula.parse("kvcache_used_ratio"), item.seqLen(), null, List.of(), 0L, true, 0);
                Mockito.when(item.requirements()).thenReturn(binding);
                Mockito.when(item.decodeEp()).thenReturn(decode);
                Mockito.when(item.decodeReservation()).thenReturn(reservation);
            }
        }

        void rejectPermitAt(int preparedSize) {
            rejectPermitAt = preparedSize;
        }

        void precedingWork(WorkSnapshot prediction) {
            precedingWork = prediction;
        }

        PrefillEndpoint prefill() {
            return prefill;
        }

        PrefillEndpoint.RouteCommitAdmission routeCommit() {
            return routeCommit;
        }

        DecodeEndpoint.EngineDispatchPermit permit(RequestRoute item) {
            return permits.get(item);
        }

        PrefillState.BatchReservation batchReservation() {
            return batchReservation;
        }

        List<PrefillState.CommittedHandoff> handoffs() {
            return List.copyOf(handoffs);
        }

        private PrefillState.ReservationResult<PrefillState.BatchReservation>
                reserveBatch() {
            batchReservation = Mockito.mock(PrefillState.BatchReservation.class);
            Mockito.when(batchReservation.commitLocked(
                            Mockito.anyList(), Mockito.anyLong(), Mockito.any(), Mockito.anyLong()))
                    .thenAnswer(invocation -> committedHandoffs(1).getFirst());
            return new PrefillState.ReservationResult<>(
                    PrefillState.CapacityStatus.ACQUIRED, batchReservation);
        }

        private DecodeEndpoint.EngineDispatchPermitAcquisition acquirePermit(
                long requestId) {
            if (permitAttempt++ == rejectPermitAt) {
                return new DecodeEndpoint.EngineDispatchPermitAcquisition(
                        DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                        null);
            }
            RequestRoute item = itemsByRequestId.get(requestId);
            DecodeEndpoint.EngineDispatchPermit permit = Mockito.mock(
                    DecodeEndpoint.EngineDispatchPermit.class);
            Mockito.when(permit.belongsTo(Mockito.any(), Mockito.any())).thenReturn(true);
            Mockito.when(permit.dispatch()).thenReturn(
                    DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED);
            if (item != null) {
                permits.put(item, permit);
            }
            return new DecodeEndpoint.EngineDispatchPermitAcquisition(
                    DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                    permit);
        }

        private List<PrefillState.CommittedHandoff> committedHandoffs(int count) {
            List<PrefillState.CommittedHandoff> committed = new ArrayList<>(count);
            for (int index = 0; index < count; index++) {
                PrefillState.CommittedHandoff handoff = Mockito.mock(
                        PrefillState.CommittedHandoff.class);
                var capture = Mockito.mock(PrefillState.WorkCapture.class);
                Mockito.when(capture.materialize()).thenReturn(precedingWork);
                Mockito.when(handoff.precedingWork()).thenReturn(capture);
                handoffs.add(handoff);
                committed.add(handoff);
            }
            return committed;
        }
    }

    static final class TestRequestScheduler {

        private final AbstractRequestScheduler scheduler = RequestProtocolTestSupport.schedulerMock();
        private final List<RequestRoute> prepared = new ArrayList<>();
        private final List<RequestRoute> committed = new ArrayList<>();
        private final List<ClaimIdentity> identities = new ArrayList<>();
        private final List<CompletionEvent> completions = new ArrayList<>();
        private final Map<RequestRoute, WorkSnapshot> precedingWork = new IdentityHashMap<>();
        private final Map<RequestRoute, Long> unstartedWorkMs = new IdentityHashMap<>();
        private final List<RequestRoute> failedPrepared = new ArrayList<>();
        private final List<Throwable> preparedFailures = new ArrayList<>();
        private final List<String> events = new ArrayList<>();
        private RequestRoute preparationLostFor;
        private RequestRoute commitLostFor;
        private RequestRoute throwCommitFor;
        private RequestRoute throwCompletionFor;
        private Runnable beforeCompletion = () -> { };

        TestRequestScheduler() {
            Mockito.doAnswer(invocation -> {
                RequestRoute item = invocation.getArgument(1);
                if (item == preparationLostFor) { return false; }
                prepared.add(item);
                return true;
            }).when(scheduler).ownsPreparedDeliveryLocked(Mockito.any(), Mockito.any());
            Mockito.doAnswer(invocation -> claim(invocation.getArgument(0), invocation.getArgument(1),
                    invocation.getArgument(2), invocation.getArgument(3)))
                    .when(scheduler).claimDelivery(Mockito.any(), Mockito.any(), Mockito.anyLong(), Mockito.any());
            Mockito.doAnswer(invocation -> {
                failDeliveryPreparation(invocation.getArgument(0), invocation.getArgument(1));
                return null;
            }).when(scheduler).failDeliveryPreparation(Mockito.any(), Mockito.any());
        }

        private DeliveryClaim claim(
                RequestRoute exactItem,
                DeliveryClaimKind kind,
                long correlationId,
                DecodeEndpoint.EngineDispatchPermit endpointHandoff) {
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
            if (endpointHandoff != null && endpointHandoff.dispatch() != DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED) {
                return null;
            }
            events.add("point-of-no-return-" + exactItem.requestId());
            DeliveryClaim claim =
                    Mockito.mock(DeliveryClaim.class);
            ReflectionTestUtils.setField(claim, "item", exactItem);
            Mockito.doAnswer(inv -> { complete(claim, inv.getArgument(1)); return null; })
                    .when(scheduler).completeDelivery(Mockito.eq(claim), Mockito.any());
            Mockito.doAnswer(inv -> {
                precedingWork.put(exactItem, inv.getArgument(1));
                unstartedWorkMs.put(exactItem, inv.getArgument(2));
                return null;
            }).when(scheduler).setDeliveryPrediction(Mockito.eq(claim), Mockito.any(), Mockito.anyLong());
            Mockito.doAnswer(inv -> {
                scheduler.setDeliveryPrediction(claim, inv.getArgument(1), inv.getArgument(2));
                complete(claim, DeliveryResult.delivered());
                return null;
            }).when(scheduler).publishRoute(Mockito.eq(claim), Mockito.any(), Mockito.anyLong());
            return claim;
        }

        private void complete(
                DeliveryClaim exactClaim,
                DeliveryResult completion) {
            RequestRoute item = (RequestRoute) ReflectionTestUtils.getField(exactClaim, "item");
            beforeCompletion.run();
            completions.add(new CompletionEvent(item, completion));
            events.add("complete-" + item.requestId());
            if (item == throwCompletionFor) {
                throw new IllegalStateException(
                        "synthetic completion failure "
                                + item.requestId());
            }
        }

        private void failDeliveryPreparation(RequestRoute exactItem, Throwable cause) {
            failedPrepared.add(exactItem);
            preparedFailures.add(cause);
        }

        void preparationLostFor(RequestRoute item) {
            preparationLostFor = item;
        }

        void commitLostFor(RequestRoute item) {
            commitLostFor = item;
        }

        void throwCommitFor(RequestRoute item) {
            throwCommitFor = item;
        }

        void throwCompletionFor(RequestRoute item) {
            throwCompletionFor = item;
        }

        void beforeCompletion(Runnable check) {
            beforeCompletion = check;
        }

        List<RequestRoute> prepared() {
            return List.copyOf(prepared);
        }

        List<RequestRoute> committed() {
            return List.copyOf(committed);
        }

        List<ClaimIdentity> identities() {
            return List.copyOf(identities);
        }

        List<CompletionEvent> completions() {
            return List.copyOf(completions);
        }

        Map<RequestRoute, Long> unstartedWorkMs() {
            return Map.copyOf(unstartedWorkMs);
        }

        WorkSnapshot precedingWork(RequestRoute item) {
            return precedingWork.get(item);
        }

        OptionalLong remainingWorkMsAt(RequestRoute item, long nowMs) {
            OptionalLong precedingMs = precedingWork.get(item).totalRemainingWorkMsAt(nowMs);
            if (precedingMs.isEmpty()) { return OptionalLong.empty(); }
            long additionalMs = unstartedWorkMs.get(item);
            return OptionalLong.of(precedingMs.getAsLong() > Long.MAX_VALUE - additionalMs
                    ? Long.MAX_VALUE : precedingMs.getAsLong() + additionalMs);
        }

        List<RequestRoute> failedPrepared() {
            return List.copyOf(failedPrepared);
        }

        List<Throwable> preparedFailures() {
            return List.copyOf(preparedFailures);
        }

        List<String> events() {
            return List.copyOf(events);
        }

        AbstractRequestScheduler scheduler() {
            return scheduler;
        }
    }

    record ClaimIdentity(DeliveryClaimKind kind, long correlationId) {
    }

    record CompletionEvent(
            RequestRoute item,
            DeliveryResult completion) {
    }

    record SubmittedBatch(
            List<RequestRoute> exactItems,
            long batchId,
            long predictedMs,
            String decisionReason) {
    }

    static final class TestBatchSubmission {

        private CapacityBoundary prepareBoundary;
        private int closeCount;
        private int totalCloseCount;
        private SubmittedBatch command;
        private BiConsumer<RequestRoute, DeliveryResult> observer;
        private final List<CompletionEvent> synchronousCompletions =
                new ArrayList<>();
        private final List<String> events = new ArrayList<>();

        CapacityBoundary.Attempt<DefaultBatchDispatcher.PreparedSubmission>
                tryPrepareSubmission() {
            if (prepareBoundary != null) {
                return CapacityBoundary.Attempt.rejected(prepareBoundary);
            }
            return CapacityBoundary.Attempt.accepted(
                    new DefaultBatchDispatcher.PreparedSubmission() {
                        private boolean submitted;

                        @Override
                        public void submit(DefaultBatchDispatcher.Delivery delivery) {
                            submitted = true;
                            delivery.run((exactItems, batchId, predictedMs, decisionReason, exactObserver) -> {
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
                            });
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
                RequestRoute item,
                DeliveryResult completion) {
            synchronousCompletions.add(new CompletionEvent(item, completion));
        }

        void complete(
                RequestRoute item,
                DeliveryResult completion) {
            observer.accept(item, completion);
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

        BiConsumer<RequestRoute, DeliveryResult> observer() {
            return observer;
        }

        List<String> events() {
            return List.copyOf(events);
        }
    }

    static final class TestTelemetry {

        private final List<List<RequestRoute>> routes = new ArrayList<>();
        private final List<BatchTelemetry> batches = new ArrayList<>();
        private final DeliveryMetricsReporter metrics =
                Mockito.mock(DeliveryMetricsReporter.class);

        TestTelemetry() {
            Mockito.doAnswer(invocation -> {
                long batchId = invocation.getArgument(0);
                if (batchId == 0L) {
                    routesDelivered(invocation.getArgument(2), invocation.getArgument(3));
                } else {
                    batchDispatched(batchId, invocation.getArgument(1), invocation.getArgument(2),
                            invocation.getArgument(3), invocation.getArgument(4));
                }
                return null;
            }).when(metrics).reportDelivery(
                    org.mockito.ArgumentMatchers.anyLong(),
                    org.mockito.ArgumentMatchers.nullable(String.class),
                    org.mockito.ArgumentMatchers.anyInt(),
                    org.mockito.ArgumentMatchers.anyList(),
                    org.mockito.ArgumentMatchers.anyLong());
        }

        private void routesDelivered(
                int remainingQueueDepth,
                List<RequestRoute> exactItems) {
            routes.add(List.copyOf(exactItems));
        }

        private void batchDispatched(
                long batchId,
                String decisionReason,
                int remainingQueueDepth,
                List<RequestRoute> dispatched,
                long predictedMs) {
            batches.add(new BatchTelemetry(
                    batchId, decisionReason, remainingQueueDepth,
                    dispatched, predictedMs));
        }

        List<List<RequestRoute>> routes() {
            return List.copyOf(routes);
        }

        List<BatchTelemetry> batches() {
            return List.copyOf(batches);
        }

        DeliveryMetricsReporter metrics() {
            return metrics;
        }
    }

    record BatchTelemetry(
            long batchId,
            String decisionReason,
            int remainingQueueDepth,
            List<RequestRoute> dispatched,
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

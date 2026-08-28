package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.ToDoubleFunction;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Deterministic canonical ownership composition for group-policy tests. */
final class GroupPolicyTestSupport {

    static final long NOW_MS = 1_000_000L;

    private GroupPolicyTestSupport() {
    }

    static Fixture fixed(
            boolean priority,
            int maxRequests,
            long collectionWindowMs,
            long predictedExecutionBudgetMs) {
        FlexlbConfig config = new FlexlbConfig();
        if (priority) {
            SchedulingTestConfig.usePriorityQueue(config);
        } else {
            SchedulingTestConfig.useFifoQueue(config);
        }
        DecisionPolicyConfig fixed =
                SchedulingTestConfig.useFixedWindowDecision(config);
        fixed.setMaxRequests(maxRequests);
        fixed.setMaxCollectionWaitMs(collectionWindowMs);
        fixed.setMaxPredictedExecutionMs(predictedExecutionBudgetMs);
        SchedulingTestConfig.useBatchDispatcher(config);
        return new Fixture(config);
    }

    static Fixture single(boolean priority) {
        FlexlbConfig config = new FlexlbConfig();
        if (priority) {
            SchedulingTestConfig.usePriorityQueue(config);
        } else {
            SchedulingTestConfig.useFifoQueue(config);
        }
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useBatchDispatcher(config);
        return new Fixture(config);
    }

    static WorkerStatus.EngineObservation capacity(
            long maxBatchTokens,
            long maxSeqLen,
            long totalKvTokens,
            long availableKvTokens) {
        return new WorkerStatus.EngineObservation(
                RoleType.PREFILL,
                null,
                availableKvTokens,
                totalKvTokens,
                Map.of(),
                0.0,
                0L,
                0L,
                0L,
                0L,
                maxSeqLen,
                maxBatchTokens,
                0L,
                0L);
    }

    static final class Fixture {

        private final FlexlbConfig config;
        private final AtomicLong nowMs = new AtomicLong(NOW_MS);
        private final AtomicLong queueVersion = new AtomicLong();
        private final ReentrantLock queueLock = new ReentrantLock();
        private final PriorityBlockingQueue<ScheduledRequest> queue;
        private final PrefillState registry;
        private final PrefillEndpoint endpoint;
        private final WorkerStatus status;
        private final AtomicReference<List<WorkerStatus.EngineObservation>>
                statusSequence = new AtomicReference<>(List.of(capacity(
                        1_000_000L, 0L, 0L, 0L)));
        private final AtomicInteger statusReads = new AtomicInteger();
        private final AtomicReference<PrefillTimePredictor.Evaluator> evaluator =
                new AtomicReference<>(tokenEvaluator());
        private final RecordingLifecycle lifecycle = new RecordingLifecycle();
        private final RecordingDeliveryStrategy delivery;
        private final BatcherContext context;

        private Fixture(FlexlbConfig config) {
            this.config = config;
            Comparator<ScheduledRequest> itemOrder = config.isPriorityOrdering()
                    ? WorkerBatcher.PRIORITY_QUEUE_ORDER
                    : WorkerBatcher.FIFO_QUEUE_ORDER;
            Comparator<GroupPlanner.Item> projectionOrder =
                    config.isPriorityOrdering()
                            ? Comparator.comparingInt(
                                            GroupPlanner.Item::priority)
                                    .reversed()
                                    .thenComparingLong(
                                            GroupPlanner.Item::enqueueSeq)
                                    .thenComparingLong(
                                            GroupPlanner.Item::requestId)
                            : Comparator.comparingLong(
                                            GroupPlanner.Item::enqueueSeq)
                                    .thenComparingLong(
                                            GroupPlanner.Item::requestId);
            queue = new PriorityBlockingQueue<>(11, itemOrder);
            registry = new PrefillState(
                    queueLock, queue, nowMs::get, () -> { });
            status = mock(WorkerStatus.class);
            when(status.committedEngineObservation()).thenAnswer(ignored -> {
                List<WorkerStatus.EngineObservation> observations =
                        statusSequence.get();
                int index = Math.min(
                        statusReads.getAndIncrement(), observations.size() - 1);
                return observations.get(index);
            });
            PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
            when(predictor.evaluator()).thenAnswer(ignored -> evaluator.get());
            endpoint = mock(PrefillEndpoint.class);
            when(endpoint.getStatus()).thenReturn(status);
            when(endpoint.getPredictor()).thenReturn(predictor);
            delivery = new RecordingDeliveryStrategy(registry);
            context = new BatcherContext(
                    "group-policy-test",
                    endpoint,
                    config,
                    lifecycle,
                    queue,
                    queueVersion,
                    queueLock,
                    itemOrder,
                    projectionOrder,
                    true,
                    delivery,
                    registry) {
                @Override
                long now() {
                    return nowMs.get();
                }
            };
        }

        FlexlbConfig config() {
            return config;
        }

        BatcherContext context() {
            return context;
        }

        RecordingDeliveryStrategy delivery() {
            return delivery;
        }

        RecordingLifecycle lifecycle() {
            return lifecycle;
        }

        long now() {
            return nowMs.get();
        }

        void advanceTo(long newNowMs) {
            nowMs.set(newNowMs);
        }

        void status(WorkerStatus.EngineObservation observation) {
            statusSequence(observation);
        }

        void statusSequence(WorkerStatus.EngineObservation... observations) {
            if (observations.length == 0) {
                throw new IllegalArgumentException(
                        "status sequence must not be empty");
            }
            statusSequence.set(List.of(observations));
            statusReads.set(0);
        }

        int statusReads() {
            return statusReads.get();
        }

        PrefillTimePredictor.Evaluator evaluator() {
            return evaluator.get();
        }

        void evaluator(PrefillTimePredictor.Evaluator replacement) {
            evaluator.set(replacement);
        }

        ScheduledRequest item(
                long requestId,
                int priority,
                long sequenceLength,
                long enqueuedAtMs,
                long expiresAtMs) {
            Request request = new Request();
            request.setRequestId(requestId);
            request.setPriority(priority);
            request.setSeqLen(sequenceLength);
            BalanceContext balanceContext = new BalanceContext();
            balanceContext.setRequest(request);
            balanceContext.setConfig(config);
            balanceContext.setSchedulingMetadata(
                    SchedulingMetadata.explicit(priority, expiresAtMs));
            return new ScheduledRequest(
                    balanceContext,
                    new CompletableFuture<Response>(),
                    null,
                    null,
                    null,
                    endpoint,
                    null,
                    null,
                    enqueuedAtMs);
        }

        ScheduledRequest add(
                long requestId,
                int priority,
                long sequenceLength,
                long enqueuedAtMs,
                long expiresAtMs) {
            ScheduledRequest item = item(
                    requestId, priority, sequenceLength,
                    enqueuedAtMs, expiresAtMs);
            add(item);
            return item;
        }

        void add(ScheduledRequest item) {
            queueLock.lock();
            try {
                assertTrue(registry.enqueueActiveUnderLock(item),
                        "test item must acquire canonical ACTIVE ownership");
                queueVersion.incrementAndGet();
            } finally {
                queueLock.unlock();
            }
        }

        boolean remove(ScheduledRequest item) {
            queueLock.lock();
            try {
                return context.removeUnderLock(item);
            } finally {
                queueLock.unlock();
            }
        }

        void bumpSchedulingInputVersion() {
            queueLock.lock();
            try {
                context.incrementSchedulingInputVersion();
            } finally {
                queueLock.unlock();
            }
        }

        List<ScheduledRequest> activeItems() {
            return context.activeItemsInSchedulingOrder();
        }

        private static PrefillTimePredictor.Evaluator tokenEvaluator() {
            return new PrefillTimePredictor.Evaluator() {
                @Override
                public long estimateMs(long totalTokens, long hitTokens) {
                    return Math.max(0L, totalTokens - hitTokens);
                }

                @Override
                public double predictBatchMs(PrefillBatchFeatures features) {
                    return features.items().stream()
                            .mapToLong(PrefillBatchFeatures.Item::seqLen)
                            .sum();
                }
            };
        }
    }

    static final class RecordingLifecycle implements EndpointEventSink {

        private final List<ScheduledRequest> expired = new ArrayList<>();
        private final List<ScheduledRequest> offerFailureItems = new ArrayList<>();
        private final List<Throwable> offerFailures = new ArrayList<>();
        private final List<ScheduledRequest> deliveryFailureItems = new ArrayList<>();
        private final List<Throwable> deliveryFailures = new ArrayList<>();

        @Override
        public void onStatusReduced(
                org.flexlb.balance.endpoint.EndpointStatusReduction reduction) {
        }

        @Override
        public void onPrefillGenerationRetired(
                org.flexlb.balance.endpoint.PrefillEndpoint endpoint,
                List<ScheduledRequest> ownedItems) {
        }

        @Override
        public void onDecodeGenerationRetired(
                org.flexlb.balance.endpoint.DecodeEndpoint endpoint,
                List<org.flexlb.balance.endpoint.DecodeEndpoint.ReservationHandle>
                        ownedReservations) {
        }

        @Override
        public void onQueuedItemExpired(ScheduledRequest exactItem) {
            expired.add(exactItem);
        }

        @Override
        public void onQueueOfferFailure(
                ScheduledRequest exactItem,
                Throwable cause) {
            offerFailureItems.add(exactItem);
            offerFailures.add(cause);
        }

        @Override
        public void onPreparedDeliveryFailure(
                ScheduledRequest exactItem,
                Throwable cause) {
            deliveryFailureItems.add(exactItem);
            deliveryFailures.add(cause);
        }

        List<ScheduledRequest> expired() {
            return List.copyOf(expired);
        }

        List<ScheduledRequest> offerFailureItems() {
            return List.copyOf(offerFailureItems);
        }

        List<Throwable> offerFailures() {
            return List.copyOf(offerFailures);
        }

        List<ScheduledRequest> deliveryFailureItems() {
            return List.copyOf(deliveryFailureItems);
        }

        List<Throwable> deliveryFailures() {
            return List.copyOf(deliveryFailures);
        }

    }

    static final class RecordingDeliveryStrategy implements DeliveryStrategy {

        private final PrefillState registry;
        private final List<List<ScheduledRequest>> attempts = new ArrayList<>();
        private final List<PrefillTimePredictor.Evaluator> evaluators =
                new ArrayList<>();
        private final List<OptionalLong> plannedPredictions = new ArrayList<>();
        private final List<List<Long>> projectedGroups = new ArrayList<>();
        private final List<List<Long>> committedGroups = new ArrayList<>();
        private final List<DeliveryMetadata> committedMetadata =
                new ArrayList<>();
        private final AtomicBoolean blocked = new AtomicBoolean();
        private int maximumPreparedItems = Integer.MAX_VALUE;
        private ToDoubleFunction<List<ScheduledRequest>> groupProjection =
                items -> items.stream().mapToLong(ScheduledRequest::seqLen).sum();
        private final RouteProjection.AdmissionBlockSemantics blockSemantics =
                new RouteProjection.AdmissionBlockSemantics(
                        "TEST_CAPACITY",
                        RouteProjection.AfterProbeAdmission.BLOCKED,
                        "TEST_CAPACITY",
                        RoleType.PREFILL);
        private final CapacityBoundary.Availability availability =
                new CapacityBoundary.Availability() {
                    @Override
                    public boolean isAvailable() {
                        return !blocked.get();
                    }

                    @Override
                    public void addListener(Runnable listener) {
                    }

                    @Override
                    public void removeListener(Runnable listener) {
                    }
                };

        private RecordingDeliveryStrategy(PrefillState registry) {
            this.registry = registry;
        }

        @Override
        public Transaction prepare(
                List<ScheduledRequest> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            List<ScheduledRequest> exactCandidates = List.copyOf(candidates);
            attempts.add(exactCandidates);
            evaluators.add(evaluator);
            plannedPredictions.add(plannedPrediction);
            if (blocked.get()) {
                return boundaryOnly(
                        exactCandidates.getFirst(),
                        CapacityBoundary.unavailable(
                                availability, blockSemantics));
            }

            int preparedCount = Math.min(
                    maximumPreparedItems, exactCandidates.size());
            if (preparedCount == 0) {
                return boundaryOnly(
                        exactCandidates.getFirst(),
                        CapacityBoundary.unavailable(
                                availability, blockSemantics));
            }
            List<ScheduledRequest> preparedCandidates = List.copyOf(
                    exactCandidates.subList(0, preparedCount));
            ScheduledRequest blockedItem = preparedCount < exactCandidates.size()
                    ? exactCandidates.get(preparedCount) : null;
            CapacityBoundary blockedResult = blockedItem == null
                    ? null : CapacityBoundary.unavailable(
                            availability, blockSemantics);

            List<PrefillState.RouteReservation> reservations =
                    new ArrayList<>(preparedCandidates.size());
            for (ScheduledRequest item : preparedCandidates) {
                PrefillState.RouteReservationResult result =
                        registry.reserveRoute(
                                item, 0L, Integer.MAX_VALUE, () -> { });
                if (result.status()
                        != PrefillState.CapacityStatus.ACQUIRED) {
                    closeReservations(reservations);
                    return boundaryOnly(
                            exactCandidates.getFirst(),
                            CapacityBoundary.OWNERSHIP_LOST);
                }
                reservations.add(result.reservation());
            }
            return new RecordingTransaction(
                    preparedCandidates,
                    reservations,
                    registry,
                    committedGroups,
                    committedMetadata,
                    blockedItem,
                    blockedResult);
        }

        @Override
        public double projectGroupDurationMs(
                List<ScheduledRequest> items,
                PrefillTimePredictor.Evaluator evaluator) {
            List<ScheduledRequest> exact = List.copyOf(items);
            projectedGroups.add(exact.stream()
                    .map(ScheduledRequest::requestId)
                    .toList());
            return PrefillPredictionBoundary.requireValidDecisionGroupMs(
                    groupProjection.applyAsDouble(exact));
        }

        @Override
        public RouteProjection.DeliveryProjection projectionPolicy() {
            return mock(RouteProjection.DeliveryProjection.class);
        }

        void block() {
            blocked.set(true);
        }

        void limitPreparedPrefix(int maximumItems) {
            maximumPreparedItems = maximumItems;
        }

        void projection(ToDoubleFunction<List<ScheduledRequest>> replacement) {
            groupProjection = replacement;
        }

        List<List<Long>> attempts() {
            return attempts.stream()
                    .map(group -> group.stream()
                            .map(ScheduledRequest::requestId)
                            .toList())
                    .toList();
        }

        List<PrefillTimePredictor.Evaluator> evaluators() {
            return List.copyOf(evaluators);
        }

        List<OptionalLong> plannedPredictions() {
            return List.copyOf(plannedPredictions);
        }

        List<List<Long>> projectedGroups() {
            return List.copyOf(projectedGroups);
        }

        List<List<Long>> committedGroups() {
            return List.copyOf(committedGroups);
        }

        List<DeliveryMetadata> committedMetadata() {
            return List.copyOf(committedMetadata);
        }

        private static void closeReservations(
                List<PrefillState.RouteReservation> reservations) {
            for (int index = reservations.size() - 1; index >= 0; index--) {
                reservations.get(index).close();
            }
        }

    }

    private static final class RecordingTransaction
            implements DeliveryStrategy.Transaction {

        private final List<ScheduledRequest> items;
        private final List<PrefillState.RouteReservation> reservations;
        private final PrefillState registry;
        private final List<List<Long>> committedGroups;
        private final List<DeliveryMetadata> committedMetadata;
        private final ScheduledRequest blockedItem;
        private final CapacityBoundary blockedResult;
        private final AtomicBoolean resolved = new AtomicBoolean();
        private List<PrefillState.CommittedHandoff> handoffs = List.of();
        private boolean committed;

        private RecordingTransaction(
                List<ScheduledRequest> items,
                List<PrefillState.RouteReservation> reservations,
                PrefillState registry,
                List<List<Long>> committedGroups,
                List<DeliveryMetadata> committedMetadata,
                ScheduledRequest blockedItem,
                CapacityBoundary blockedResult) {
            this.items = List.copyOf(items);
            this.reservations = List.copyOf(reservations);
            this.registry = registry;
            this.committedGroups = committedGroups;
            this.committedMetadata = committedMetadata;
            this.blockedItem = blockedItem;
            this.blockedResult = blockedResult;
        }

        @Override
        public List<ScheduledRequest> items() {
            return items;
        }

        @Override
        public ScheduledRequest blockedItem() {
            return blockedItem;
        }

        @Override
        public CapacityBoundary blockedResult() {
            return blockedResult;
        }

        @Override
        public void commitUnderLock() {
            handoffs = reservations.getFirst().commitGroup(
                    items, reservations);
            committed = true;
        }

        @Override
        public void handoff(DeliveryMetadata metadata) {
            committedGroups.add(items.stream()
                    .map(ScheduledRequest::requestId)
                    .toList());
            committedMetadata.add(metadata);
            settle();
        }

        @Override
        public void abort(Throwable cause) {
            settle();
        }

        @Override
        public void close() {
            if (!committed) {
                RecordingDeliveryStrategy.closeReservations(reservations);
            }
        }

        private void settle() {
            if (!resolved.compareAndSet(false, true)) {
                return;
            }
            for (ScheduledRequest item : items) {
                if (!registry.terminalizeCommittedItem(item)) {
                    throw new IllegalStateException(
                            "test owner lost committed request_id="
                                    + item.requestId());
                }
            }
            for (PrefillState.CommittedHandoff handoff : handoffs) {
                handoff.close();
            }
        }
    }

    static DeliveryStrategy.Transaction boundaryOnly(
            ScheduledRequest item,
            CapacityBoundary boundary) {
        return new DeliveryStrategy.Transaction() {
            @Override
            public List<ScheduledRequest> items() {
                return List.of();
            }

            @Override
            public ScheduledRequest blockedItem() {
                return item;
            }

            @Override
            public CapacityBoundary blockedResult() {
                return boundary;
            }

            @Override
            public void commitUnderLock() {
                throw new IllegalStateException(
                        "boundary-only preparation cannot commit");
            }

            @Override
            public void handoff(DeliveryMetadata metadata) {
                throw new IllegalStateException(
                        "boundary-only preparation cannot hand off");
            }

            @Override
            public void abort(Throwable cause) {
            }

            @Override
            public void close() {
            }
        };
    }
}

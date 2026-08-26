package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.CommittedDelivery;
import org.flexlb.balance.delivery.DeliveryContext;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.DeliveryLifecyclePort;
import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.FixedWindowDecisionConfig;
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
        FixedWindowDecisionConfig fixed =
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
        private final PriorityBlockingQueue<BatchItem> queue;
        private final PrefillWorkRegistry registry;
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
            Comparator<BatchItem> itemOrder = config.isPriorityOrdering()
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
            registry = new PrefillWorkRegistry(
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

        BatchItem item(
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
            return new BatchItem(
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

        BatchItem add(
                long requestId,
                int priority,
                long sequenceLength,
                long enqueuedAtMs,
                long expiresAtMs) {
            BatchItem item = item(
                    requestId, priority, sequenceLength,
                    enqueuedAtMs, expiresAtMs);
            add(item);
            return item;
        }

        void add(BatchItem item) {
            queueLock.lock();
            try {
                assertTrue(registry.enqueueActiveUnderLock(item),
                        "test item must acquire canonical ACTIVE ownership");
                queueVersion.incrementAndGet();
            } finally {
                queueLock.unlock();
            }
        }

        boolean remove(BatchItem item) {
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

        List<BatchItem> activeItems() {
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

    static final class RecordingLifecycle implements DeliveryLifecyclePort {

        private final List<DeliveryItem> expired = new ArrayList<>();
        private final List<DeliveryItem> offerFailureItems = new ArrayList<>();
        private final List<Throwable> offerFailures = new ArrayList<>();
        private final List<DeliveryItem> deliveryFailureItems = new ArrayList<>();
        private final List<Throwable> deliveryFailures = new ArrayList<>();
        private final List<List<Long>> committedGroups = new ArrayList<>();
        private final List<DeliveryMetadata> committedMetadata =
                new ArrayList<>();

        @Override
        public void onExpired(DeliveryItem exactItem) {
            expired.add(exactItem);
        }

        @Override
        public void onDeliveryCommitted(
                CommittedDelivery delivery,
                DeliveryMetadata metadata) {
            committedGroups.add(delivery.items().stream()
                    .map(DeliveryItem::requestId)
                    .toList());
            committedMetadata.add(metadata);
            delivery.deliver(metadata);
        }

        @Override
        public void onOfferFailure(DeliveryItem exactItem, Throwable cause) {
            offerFailureItems.add(exactItem);
            offerFailures.add(cause);
        }

        @Override
        public void onDeliveryFailure(DeliveryItem exactItem, Throwable cause) {
            deliveryFailureItems.add(exactItem);
            deliveryFailures.add(cause);
        }

        List<DeliveryItem> expired() {
            return List.copyOf(expired);
        }

        List<DeliveryItem> offerFailureItems() {
            return List.copyOf(offerFailureItems);
        }

        List<Throwable> offerFailures() {
            return List.copyOf(offerFailures);
        }

        List<DeliveryItem> deliveryFailureItems() {
            return List.copyOf(deliveryFailureItems);
        }

        List<Throwable> deliveryFailures() {
            return List.copyOf(deliveryFailures);
        }

        List<List<Long>> committedGroups() {
            return List.copyOf(committedGroups);
        }

        List<DeliveryMetadata> committedMetadata() {
            return List.copyOf(committedMetadata);
        }
    }

    static final class RecordingDeliveryStrategy implements DeliveryStrategy {

        private final PrefillWorkRegistry registry;
        private final List<List<DeliveryItem>> attempts = new ArrayList<>();
        private final List<DeliveryMetadata> metadata = new ArrayList<>();
        private final List<PrefillTimePredictor.Evaluator> evaluators =
                new ArrayList<>();
        private final List<OptionalLong> plannedPredictions = new ArrayList<>();
        private final List<List<Long>> projectedGroups = new ArrayList<>();
        private final AtomicBoolean blocked = new AtomicBoolean();
        private ToDoubleFunction<List<DeliveryItem>> groupProjection =
                items -> items.stream().mapToLong(DeliveryItem::seqLen).sum();
        private final RouteProjection.AdmissionBlockSemantics blockSemantics =
                new RouteProjection.AdmissionBlockSemantics(
                        "TEST_CAPACITY",
                        RouteProjection.AfterProbeAdmission.BLOCKED,
                        "TEST_CAPACITY");
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

        private RecordingDeliveryStrategy(PrefillWorkRegistry registry) {
            this.registry = registry;
        }

        @Override
        public <R> R admitAndDeliver(
                List<DeliveryItem> candidates,
                DeliveryMetadata exactMetadata,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction,
                DeliveryContext<R> context) {
            List<DeliveryItem> exactCandidates = List.copyOf(candidates);
            attempts.add(exactCandidates);
            metadata.add(exactMetadata);
            evaluators.add(evaluator);
            plannedPredictions.add(plannedPrediction);
            if (blocked.get()) {
                return context.commitBoundary(
                        new DeliveryContext.SelectionBoundary(
                                exactCandidates.getFirst(),
                                new CapacityBoundary.Unavailable(
                                        availability, blockSemantics)));
            }
            if (!context.selectionStillOwned(exactCandidates)) {
                return context.noAction();
            }

            List<PrefillWorkLedger.RouteReservation> reservations =
                    new ArrayList<>(exactCandidates.size());
            for (DeliveryItem item : exactCandidates) {
                PrefillWorkLedger.RouteReservationResult result =
                        registry.reserveRoute(
                                item, 0L, Integer.MAX_VALUE, () -> { });
                if (result.status()
                        != PrefillWorkLedger.CapacityStatus.ACQUIRED) {
                    closeReservations(reservations);
                    return context.noAction();
                }
                reservations.add(result.reservation());
            }
            TestCommittedDelivery owner = new TestCommittedDelivery(
                    exactCandidates, reservations, registry);
            DeliveryContext.SelectionCommit<R> selection;
            try {
                selection = context.commitSelection(
                        new DeliveryContext.CanonicalCommit() {
                            @Override
                            public List<DeliveryItem> items() {
                                return exactCandidates;
                            }

                            @Override
                            public CommittedDelivery commitUnderLock() {
                                owner.commitUnderLock();
                                return owner;
                            }
                        },
                        null,
                        exactMetadata.reason());
            } catch (Throwable failure) {
                closeReservations(reservations);
                throw failure;
            }
            if (selection
                    instanceof DeliveryContext.SelectionCommit.Committed<R>
                    committed) {
                context.publishCommittedDelivery(
                        committed.owner(), exactMetadata);
            } else {
                closeReservations(reservations);
            }
            return selection.loopResult();
        }

        @Override
        public double projectGroupDurationMs(
                List<DeliveryItem> items,
                PrefillTimePredictor.Evaluator evaluator) {
            List<DeliveryItem> exact = List.copyOf(items);
            projectedGroups.add(exact.stream()
                    .map(DeliveryItem::requestId)
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

        void projection(ToDoubleFunction<List<DeliveryItem>> replacement) {
            groupProjection = replacement;
        }

        List<List<Long>> attempts() {
            return attempts.stream()
                    .map(group -> group.stream()
                            .map(DeliveryItem::requestId)
                            .toList())
                    .toList();
        }

        List<DeliveryMetadata> metadata() {
            return List.copyOf(metadata);
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

        private static void closeReservations(
                List<PrefillWorkLedger.RouteReservation> reservations) {
            for (int index = reservations.size() - 1; index >= 0; index--) {
                reservations.get(index).close();
            }
        }
    }

    private static final class TestCommittedDelivery
            implements CommittedDelivery {

        private final List<DeliveryItem> items;
        private final List<PrefillWorkLedger.RouteReservation> reservations;
        private final PrefillWorkRegistry registry;
        private final AtomicBoolean resolved = new AtomicBoolean();
        private List<PrefillWorkLedger.CommittedHandoff> handoffs = List.of();

        private TestCommittedDelivery(
                List<DeliveryItem> items,
                List<PrefillWorkLedger.RouteReservation> reservations,
                PrefillWorkRegistry registry) {
            this.items = List.copyOf(items);
            this.reservations = List.copyOf(reservations);
            this.registry = registry;
        }

        private void commitUnderLock() {
            handoffs = reservations.getFirst().commitGroupUnderLock(
                    items, reservations);
        }

        @Override
        public List<DeliveryItem> items() {
            return items;
        }

        @Override
        public void deliver(DeliveryMetadata metadata) {
            settle();
        }

        @Override
        public void fail(Throwable cause) {
            settle();
        }

        private void settle() {
            if (!resolved.compareAndSet(false, true)) {
                return;
            }
            for (DeliveryItem item : items) {
                if (!registry.terminalizeCommittedItem(item)) {
                    throw new IllegalStateException(
                            "test owner lost committed request_id="
                                    + item.requestId());
                }
            }
            for (PrefillWorkLedger.CommittedHandoff handoff : handoffs) {
                handoff.close();
            }
        }
    }
}

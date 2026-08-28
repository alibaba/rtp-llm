package org.flexlb.balance.delivery;

import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicBoolean;
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

    static TestItem item(long requestId) {
        return new TestItem(requestId, 50, requestId, 100L, 10L);
    }

    record TestItem(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long seqLen,
            long hitCache) implements DeliveryItem {
    }

    static final class TestContext implements DeliveryContext<String> {

        private boolean owned = true;
        private boolean commit = true;
        private boolean autoDeliver = true;
        private PreparedSelection preparedSelection;
        private SelectionBoundary committedBoundary;
        private SelectionBoundary emptyBoundary;
        private CommittedDelivery published;
        private DeliveryMetadata publishedMetadata;

        @Override
        public String noAction() {
            return "NO_ACTION";
        }

        @Override
        public boolean selectionStillOwned(List<DeliveryItem> candidates) {
            return owned;
        }

        @Override
        public CommitResult<String> commitPreparedSelection(
                PreparedSelection selection,
                String decisionReason) {
            preparedSelection = selection;
            committedBoundary = selection.boundary();
            if (!commit) {
                return new CommitResult.NotCommitted<>("NOT_COMMITTED");
            }
            return new CommitResult.Committed<>(
                    selection.commitOwnershipUnderLock(), "COMMITTED");
        }

        @Override
        public String commitBoundary(SelectionBoundary boundary) {
            emptyBoundary = boundary;
            return "BOUNDARY";
        }

        @Override
        public void handoffCommittedDelivery(
                CommittedDelivery delivery,
                DeliveryMetadata metadata) {
            published = delivery;
            publishedMetadata = metadata;
            if (autoDeliver) {
                delivery.deliver(metadata);
            }
        }

        void owned(boolean value) {
            owned = value;
        }

        void commit(boolean value) {
            commit = value;
        }

        void autoDeliver(boolean value) {
            autoDeliver = value;
        }

        PreparedSelection preparedSelection() {
            return preparedSelection;
        }

        SelectionBoundary committedBoundary() {
            return committedBoundary;
        }

        SelectionBoundary emptyBoundary() {
            return emptyBoundary;
        }

        CommittedDelivery published() {
            return published;
        }

        DeliveryMetadata publishedMetadata() {
            return publishedMetadata;
        }
    }

    static final class TestAdmissionPort implements PrefillAdmissionPort {

        private OptionalLong correlationId = OptionalLong.empty();
        private CapacityBoundary prepareBoundary;
        private int rejectAppendAt = -1;
        private CapacityBoundary appendBoundary;
        private final List<DeliveryItem> preparedItems = new ArrayList<>();
        private final List<Long> preparedPredictions = new ArrayList<>();
        private List<DeliveryItem> committedItems = List.of();
        private long committedPrediction;
        private final List<DeliveryItem> transferred = new ArrayList<>();
        private final List<String> events = new ArrayList<>();
        private int preparedCloseCount;
        private int committedCloseCount;

        @Override
        public CapacityBoundary.Attempt<PreparedAdmission> tryBegin(
                DeliveryItem firstCandidate) {
            if (prepareBoundary != null) {
                return new CapacityBoundary.Attempt.Rejected<>(prepareBoundary);
            }
            return new CapacityBoundary.Attempt.Accepted<>(new PreparedAdmission() {
                private boolean moved;

                @Override
                public OptionalLong correlationId() {
                    return correlationId;
                }

                @Override
                public CapacityBoundary.Attempt<DeliveryItem> tryAppend(
                        DeliveryItem exactNextItem,
                        long predictedMs) {
                    if (preparedItems.size() == rejectAppendAt) {
                        return new CapacityBoundary.Attempt.Rejected<>(
                                appendBoundary);
                    }
                    preparedItems.add(exactNextItem);
                    preparedPredictions.add(predictedMs);
                    return new CapacityBoundary.Attempt.Accepted<>(
                            exactNextItem);
                }

                @Override
                public CommittedAdmission commitPreparedUnderLock(
                        List<DeliveryItem> exactItems,
                        long predictedMs) {
                    moved = true;
                    committedItems = List.copyOf(exactItems);
                    committedPrediction = predictedMs;
                    events.add("admission-commit");
                    return new CommittedAdmission() {
                        private final AtomicBoolean closed =
                                new AtomicBoolean();

                        @Override
                        public boolean transferToEndpoint(
                                DeliveryItem exactItem) {
                            transferred.add(exactItem);
                            events.add("admission-transfer-"
                                    + exactItem.requestId());
                            return true;
                        }

                        @Override
                        public void close() {
                            if (closed.compareAndSet(false, true)) {
                                committedCloseCount++;
                                events.add("admission-close");
                            }
                        }
                    };
                }

                @Override
                public void close() {
                    if (!moved) {
                        preparedCloseCount++;
                        events.add("prepared-admission-close");
                    }
                }
            });
        }

        void correlationId(OptionalLong value) {
            correlationId = value;
        }

        void prepareBoundary(CapacityBoundary value) {
            prepareBoundary = value;
        }

        void rejectAppendAt(int preparedSize, CapacityBoundary boundary) {
            rejectAppendAt = preparedSize;
            appendBoundary = boundary;
        }

        List<DeliveryItem> preparedItems() {
            return List.copyOf(preparedItems);
        }

        List<Long> preparedPredictions() {
            return List.copyOf(preparedPredictions);
        }

        List<DeliveryItem> committedItems() {
            return committedItems;
        }

        long committedPrediction() {
            return committedPrediction;
        }

        List<DeliveryItem> transferred() {
            return List.copyOf(transferred);
        }

        List<String> events() {
            return List.copyOf(events);
        }

        int preparedCloseCount() {
            return preparedCloseCount;
        }

        int committedCloseCount() {
            return committedCloseCount;
        }
    }

    static final class TestSlotPort implements SlotDeliveryPort {

        private final List<DeliveryItem> prepared = new ArrayList<>();
        private final List<DeliveryItem> committed = new ArrayList<>();
        private final List<Identity> identities = new ArrayList<>();
        private final List<CompletionEvent> completions = new ArrayList<>();
        private final List<DeliveryItem> failedPrepared = new ArrayList<>();
        private final List<Throwable> preparedFailures = new ArrayList<>();
        private final List<String> events = new ArrayList<>();
        private DeliveryItem preparationLostFor;
        private DeliveryItem commitLostFor;
        private DeliveryItem throwCommitFor;
        private DeliveryItem throwCompletionFor;
        private Runnable beforeCompletion = () -> { };

        @Override
        public <T> Optional<T> prepareIfOwned(
                DeliveryItem exactItem,
                java.util.function.Supplier<T> preparation) {
            if (exactItem == preparationLostFor) {
                return Optional.empty();
            }
            prepared.add(exactItem);
            return Optional.of(preparation.get());
        }

        @Override
        public Claim tryClaimForDelivery(
                DeliveryItem exactItem,
                Identity identity,
                BooleanSupplier endpointHandoff) {
            committed.add(exactItem);
            identities.add(identity);
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
            return new TestClaim(exactItem);
        }

        @Override
        public void complete(Claim exactClaim, Completion completion) {
            beforeCompletion.run();
            completions.add(new CompletionEvent(exactClaim.item(), completion));
            events.add("complete-" + exactClaim.item().requestId());
            if (exactClaim.item() == throwCompletionFor) {
                throw new IllegalStateException(
                        "synthetic completion failure "
                                + exactClaim.item().requestId());
            }
        }

        @Override
        public void failPrepared(DeliveryItem exactItem, Throwable cause) {
            failedPrepared.add(exactItem);
            preparedFailures.add(cause);
        }

        void preparationLostFor(DeliveryItem item) {
            preparationLostFor = item;
        }

        void commitLostFor(DeliveryItem item) {
            commitLostFor = item;
        }

        void throwCommitFor(DeliveryItem item) {
            throwCommitFor = item;
        }

        void throwCompletionFor(DeliveryItem item) {
            throwCompletionFor = item;
        }

        void beforeCompletion(Runnable check) {
            beforeCompletion = check;
        }

        List<DeliveryItem> prepared() {
            return List.copyOf(prepared);
        }

        List<DeliveryItem> committed() {
            return List.copyOf(committed);
        }

        List<Identity> identities() {
            return List.copyOf(identities);
        }

        List<CompletionEvent> completions() {
            return List.copyOf(completions);
        }

        List<DeliveryItem> failedPrepared() {
            return List.copyOf(failedPrepared);
        }

        List<Throwable> preparedFailures() {
            return List.copyOf(preparedFailures);
        }

        List<String> events() {
            return List.copyOf(events);
        }
    }

    record TestClaim(DeliveryItem item) implements SlotDeliveryPort.Claim {
    }

    record CompletionEvent(
            DeliveryItem item,
            SlotDeliveryPort.Completion completion) {
    }

    static final class TestBatchSubmissionPort implements BatchSubmissionPort {

        private CapacityBoundary prepareBoundary;
        private int prepareCount;
        private int closeCount;
        private int totalCloseCount;
        private Command command;
        private BiConsumer<DeliveryItem, SlotDeliveryPort.Completion> observer;
        private final List<CompletionEvent> synchronousCompletions =
                new ArrayList<>();
        private final List<String> events = new ArrayList<>();

        @Override
        public CapacityBoundary.Attempt<PreparedSubmission>
                tryPrepareSubmission() {
            prepareCount++;
            if (prepareBoundary != null) {
                return new CapacityBoundary.Attempt.Rejected<>(prepareBoundary);
            }
            return new CapacityBoundary.Attempt.Accepted<>(
                    new PreparedSubmission() {
                        private boolean submitted;

                        @Override
                        public void submitBatch(
                                Command exactCommand,
                                BiConsumer<DeliveryItem,
                                        SlotDeliveryPort.Completion> exactObserver) {
                            submitted = true;
                            command = exactCommand;
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
                DeliveryItem item,
                SlotDeliveryPort.Completion completion) {
            synchronousCompletions.add(new CompletionEvent(item, completion));
        }

        void complete(
                DeliveryItem item,
                SlotDeliveryPort.Completion completion) {
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

        Command command() {
            return command;
        }

        BiConsumer<DeliveryItem, SlotDeliveryPort.Completion> observer() {
            return observer;
        }

        List<String> events() {
            return List.copyOf(events);
        }
    }

    static final class TestTelemetry implements DeliveryTelemetry {

        private final List<List<DeliveryItem>> routes = new ArrayList<>();
        private final List<BatchTelemetry> batches = new ArrayList<>();

        @Override
        public void routesDelivered(
                DeliveryMetadata metadata,
                List<DeliveryItem> exactItems) {
            routes.add(List.copyOf(exactItems));
        }

        @Override
        public void batchDispatched(
                long batchId,
                DeliveryMetadata metadata,
                List<DeliveryItem> dispatched,
                long predictedMs) {
            batches.add(new BatchTelemetry(
                    batchId, metadata, dispatched, predictedMs));
        }

        List<List<DeliveryItem>> routes() {
            return List.copyOf(routes);
        }

        List<BatchTelemetry> batches() {
            return List.copyOf(batches);
        }
    }

    record BatchTelemetry(
            long batchId,
            DeliveryMetadata metadata,
            List<DeliveryItem> dispatched,
            long predictedMs) {
        BatchTelemetry {
            dispatched = List.copyOf(dispatched);
        }
    }

    static CapacityBoundary.Unavailable unavailable() {
        return new CapacityBoundary.Unavailable(
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

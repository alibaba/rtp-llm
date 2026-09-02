package org.flexlb.balance.delivery;

import org.flexlb.balance.scheduler.ScheduledRequest;

import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.dao.route.RoleType;
import org.mockito.Mockito;

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

        private boolean owned = true;
        private boolean commit = true;
        private DeliveryStrategy.Transaction preparedSelection;
        private TestBoundary committedBoundary;
        private TestBoundary emptyBoundary;
        private DeliveryMetadata publishedMetadata;

        String deliver(
                DeliveryStrategy strategy,
                List<ScheduledRequest> candidates,
                DeliveryMetadata metadata,
                OptionalLong plannedPredictionMs) {
            if (candidates.isEmpty() || !owned) {
                return "NO_ACTION";
            }
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
                publishedMetadata = metadata;
                transaction.handoff(metadata);
                return "COMMITTED";
            }
        }

        void owned(boolean value) {
            owned = value;
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

        DeliveryMetadata publishedMetadata() {
            return publishedMetadata;
        }
    }

    static final class TestAdmissionPort implements PrefillAdmissionPort {

        private OptionalLong correlationId = OptionalLong.empty();
        private CapacityBoundary prepareBoundary;
        private int rejectAppendAt = -1;
        private CapacityBoundary appendBoundary;
        private final List<ScheduledRequest> preparedItems = new ArrayList<>();
        private final List<Long> preparedPredictions = new ArrayList<>();
        private List<ScheduledRequest> committedItems = List.of();
        private long committedPrediction;
        private final List<ScheduledRequest> transferred = new ArrayList<>();
        private final List<String> events = new ArrayList<>();
        private int preparedCloseCount;
        private int committedCloseCount;

        @Override
        public CapacityBoundary.Attempt<PreparedAdmission> tryBegin(
                ScheduledRequest firstCandidate) {
            if (prepareBoundary != null) {
                return CapacityBoundary.Attempt.rejected(prepareBoundary);
            }
            return CapacityBoundary.Attempt.accepted(new PreparedAdmission() {
                private boolean moved;

                @Override
                public OptionalLong correlationId() {
                    return correlationId;
                }

                @Override
                public CapacityBoundary.Attempt<ScheduledRequest> tryAppend(
                        ScheduledRequest exactNextItem,
                        long predictedMs) {
                    if (preparedItems.size() == rejectAppendAt) {
                        return CapacityBoundary.Attempt.rejected(
                                appendBoundary);
                    }
                    preparedItems.add(exactNextItem);
                    preparedPredictions.add(predictedMs);
                    return CapacityBoundary.Attempt.accepted(
                            exactNextItem);
                }

                @Override
                public CommittedAdmission commitPreparedUnderLock(
                        List<ScheduledRequest> exactItems,
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
                                ScheduledRequest exactItem) {
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

        List<ScheduledRequest> preparedItems() {
            return List.copyOf(preparedItems);
        }

        List<Long> preparedPredictions() {
            return List.copyOf(preparedPredictions);
        }

        List<ScheduledRequest> committedItems() {
            return committedItems;
        }

        long committedPrediction() {
            return committedPrediction;
        }

        List<ScheduledRequest> transferred() {
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

        private final List<ScheduledRequest> prepared = new ArrayList<>();
        private final List<ScheduledRequest> committed = new ArrayList<>();
        private final List<Identity> identities = new ArrayList<>();
        private final List<CompletionEvent> completions = new ArrayList<>();
        private final List<ScheduledRequest> failedPrepared = new ArrayList<>();
        private final List<Throwable> preparedFailures = new ArrayList<>();
        private final List<String> events = new ArrayList<>();
        private ScheduledRequest preparationLostFor;
        private ScheduledRequest commitLostFor;
        private ScheduledRequest throwCommitFor;
        private ScheduledRequest throwCompletionFor;
        private Runnable beforeCompletion = () -> { };

        @Override
        public <T> Optional<T> prepareIfOwned(
                ScheduledRequest exactItem,
                java.util.function.Supplier<T> preparation) {
            if (exactItem == preparationLostFor) {
                return Optional.empty();
            }
            prepared.add(exactItem);
            return Optional.of(preparation.get());
        }

        @Override
        public Claim tryClaimForDelivery(
                ScheduledRequest exactItem,
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
        public void failPrepared(ScheduledRequest exactItem, Throwable cause) {
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

        List<Identity> identities() {
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
    }

    record TestClaim(ScheduledRequest item) implements SlotDeliveryPort.Claim {
    }

    record CompletionEvent(
            ScheduledRequest item,
            SlotDeliveryPort.Completion completion) {
    }

    static final class TestBatchSubmissionPort implements BatchSubmissionPort {

        private CapacityBoundary prepareBoundary;
        private int prepareCount;
        private int closeCount;
        private int totalCloseCount;
        private Command command;
        private BiConsumer<ScheduledRequest, SlotDeliveryPort.Completion> observer;
        private final List<CompletionEvent> synchronousCompletions =
                new ArrayList<>();
        private final List<String> events = new ArrayList<>();

        @Override
        public CapacityBoundary.Attempt<PreparedSubmission>
                tryPrepareSubmission() {
            prepareCount++;
            if (prepareBoundary != null) {
                return CapacityBoundary.Attempt.rejected(prepareBoundary);
            }
            return CapacityBoundary.Attempt.accepted(
                    new PreparedSubmission() {
                        private boolean submitted;

                        @Override
                        public void submitBatch(
                                Command exactCommand,
                                BiConsumer<ScheduledRequest,
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
                ScheduledRequest item,
                SlotDeliveryPort.Completion completion) {
            synchronousCompletions.add(new CompletionEvent(item, completion));
        }

        void complete(
                ScheduledRequest item,
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

        BiConsumer<ScheduledRequest, SlotDeliveryPort.Completion> observer() {
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
                    org.mockito.ArgumentMatchers.any(),
                    org.mockito.ArgumentMatchers.anyList());
            org.mockito.Mockito.doAnswer(invocation -> {
                batchDispatched(
                        invocation.getArgument(0), invocation.getArgument(1),
                        invocation.getArgument(2), invocation.getArgument(3));
                return null;
            }).when(metrics).batchDispatched(
                    org.mockito.ArgumentMatchers.anyLong(),
                    org.mockito.ArgumentMatchers.any(),
                    org.mockito.ArgumentMatchers.anyList(),
                    org.mockito.ArgumentMatchers.anyLong());
        }

        private void routesDelivered(
                DeliveryMetadata metadata,
                List<ScheduledRequest> exactItems) {
            routes.add(List.copyOf(exactItems));
        }

        private void batchDispatched(
                long batchId,
                DeliveryMetadata metadata,
                List<ScheduledRequest> dispatched,
                long predictedMs) {
            batches.add(new BatchTelemetry(
                    batchId, metadata, dispatched, predictedMs));
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
            DeliveryMetadata metadata,
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

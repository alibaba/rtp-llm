package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.scheduler.QueuedRequestScheduler.PrefillQueueScheduler;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.mockito.Mockito;

import java.util.List;
import java.util.OptionalLong;

/**
 * Transport/config fixtures. Queue decisions are stubbed on PrefillQueueScheduler, never on Dispatcher.
 */
public final class DeliveryTestSupport {
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

    private DeliveryTestSupport() {}

    public static Delivery queued(
            List<RouteResult> items,
            org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput batchInput,
            long[] predictions,
            RouteResult failed,
            long now) {
        return queued(items, batchInput, predictions, failed, now, null);
    }

    public static Delivery queued(
            List<RouteResult> items,
            org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput batchInput,
            long[] predictions,
            RouteResult failed,
            long now,
            RouteResult bypassed) {
        var delivery =
                new Delivery(
                        items.isEmpty() ? null : items.getFirst().prefillEndpoint(),
                        batchInput != null,
                        items.size(),
                        null,
                        "test");
        for (var item : items) delivery.members.add(new Delivery.Member(item, null));
        if (batchInput != null) {
            delivery.batchId = batchInput.batchId();
            delivery.maxInflightBatches = batchInput.maximumInflight();
        }
        if (batchInput != null) {
            delivery.predictedMs = predictions[0];
            delivery.features =
                    PrefillBatchFeatures.from(items, RouteResult::seqLen, RouteResult::hitCache);
        } else System.arraycopy(predictions, 0, delivery.predictions, 0, predictions.length);
        delivery.blocked = failed;
        delivery.boundary =
                failed == null
                        ? null
                        : CapacityBoundary.failed(
                                new IllegalStateException("test preparation failure"));
        delivery.selectionTimeMs = now;
        delivery.bypassedHead = bypassed;
        return delivery;
    }

    public static DeliveryDispatcher route(DeliveryMetricsReporter reporter) {
        var config = SchedulingTestConfig.newConfig();
        config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
        return new DeliveryDispatcher(
                null,
                service(config),
                reporter,
                org.mockito.Mockito.mock(org.flexlb.balance.scheduler.TaskTimer.class),
                new org.flexlb.balance.scheduler.PlacementConfiguration()
                        .decodeAdmissionCapacity(service(config)),
                org.mockito.Mockito.mock(org.flexlb.balance.scheduler.DeliveryExecution.class));
    }

    public static org.flexlb.config.ConfigService service(org.flexlb.config.FlexlbConfig config) {
        var service = Mockito.mock(org.flexlb.config.ConfigService.class);
        Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        return service;
    }

    public static DeliveryDispatcher parked(CapacityBoundary boundary) {
        return new StubDispatcher() {
            @Override
            CapacityBoundary selectBoundary(
                    List<RouteResult> items,
                    PrefillTimePredictor.Evaluator evaluator,
                    OptionalLong prediction) {
                return boundary;
            }
        };
    }

    public static org.flexlb.config.DispatcherConfig.Type projection(
            DeliveryDispatcher dispatcher) {
        return dispatcher.mode();
    }

    public static void installQueueFixture(
            PrefillQueueScheduler worker, DeliveryDispatcher dispatcher) {
        if (!(dispatcher instanceof StubDispatcher stub)) {
            return;
        }
        Mockito.doAnswer(
                        call ->
                                stub.selectBoundary(
                                        call.getArgument(0),
                                        call.getArgument(2),
                                        call.getArgument(3)))
                .when(worker)
                .processSelection(
                        Mockito.any(),
                        Mockito.any(),
                        Mockito.any(),
                        Mockito.any(),
                        Mockito.any(),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.any());
        Mockito.doAnswer(call -> stub.capacityAvailable(call.getArgument(0), call.getArgument(1)))
                .when(worker)
                .capacityAvailable(Mockito.any(), Mockito.any());
        Mockito.doAnswer(
                        call ->
                                stub.watchCapacity(
                                        call.getArgument(0),
                                        call.getArgument(1),
                                        call.getArgument(2)))
                .when(worker)
                .watchCapacity(Mockito.any(), Mockito.any(), Mockito.any());
    }

    public static CapacityBoundary process(
            PrefillQueueScheduler worker,
            List<RouteResult> items,
            String reason,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong prediction) {
        return worker.processSelection(
                items,
                reason,
                evaluator,
                prediction,
                null,
                Delivery.PrefixPolicy.ORDERED_PREFIX,
                () -> true);
    }

    public static void afterSelection(
            PrefillQueueScheduler worker, java.util.function.Consumer<List<RouteResult>> selected) {
        Mockito.doAnswer(
                        call -> {
                            selected.accept(call.getArgument(0));
                            return call.callRealMethod();
                        })
                .when(worker)
                .processSelection(
                        Mockito.any(),
                        Mockito.any(),
                        Mockito.any(),
                        Mockito.any(),
                        Mockito.any(),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.any());
    }

    public static void selectWith(PrefillQueueScheduler worker, Selection selection) {
        Mockito.doAnswer(
                        call ->
                                selection.select(
                                        call.getArgument(0),
                                        call.getArgument(1),
                                        call.getArgument(2),
                                        call.getArgument(3),
                                        () -> {
                                            try {
                                                return (CapacityBoundary) call.callRealMethod();
                                            } catch (Throwable failure) {
                                                throw org.flexlb.util.Failures.propagate(
                                                        failure, "test selection failed");
                                            }
                                        }))
                .when(worker)
                .processSelection(
                        Mockito.any(),
                        Mockito.any(),
                        Mockito.any(),
                        Mockito.any(),
                        Mockito.any(),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.any());
    }

    public static void stubRouteDelivery(
            AbstractRequestScheduler requests,
            java.util.function.BiConsumer<Delivery.Member, Delivery.Result> completed) {
        Mockito.doAnswer(
                        invocation -> {
                            var claim = invocation.getArgument(0, Delivery.Member.class);
                            recordMockDelivery(claim, Delivery.SendMode.ROUTE_DECISION, 0L);
                            Mockito.doAnswer(
                                            inv -> {
                                                completed.accept(
                                                        claim, Delivery.Result.delivered());
                                                return null;
                                            })
                                    .when(requests)
                                    .onDeliveryResult(
                                            Mockito.isA(ContextResult.RouteDelivered.class));
                            return claim;
                        })
                .when(requests)
                .commitDelivery(
                        Mockito.any(Delivery.Member.class),
                        Mockito.eq(Delivery.SendMode.ROUTE_DECISION),
                        Mockito.eq(0L));
    }

    static Delivery.Member member(RouteResult route, Delivery.SendMode kind, long batchId) {
        var member = new Delivery.Member(route, null);
        member.initialize(kind, batchId);
        return member;
    }

    /**
     * Publish the accepted delivery for tests whose Scheduler collaborator is mocked.
     */
    static void recordMockDelivery(Delivery.Member member, Delivery.SendMode kind, long batchId) {
        synchronized (member.owner()) {
            member.initialize(kind, batchId);
            member.owner().recordDeliveryStarted(member, System.currentTimeMillis());
        }
    }

    static Delivery committed(
            long batchId,
            List<Delivery.Member> members,
            org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot precedingWork,
            long predictedMs,
            PrefillTimePredictor.Evaluator evaluator,
            String reason,
            int remainingQueueDepth,
            org.flexlb.balance.scheduler.Delivery handoff,
            DeliveryExecution.DeliveryPermit permit) {
        var delivery =
                new Delivery(
                        members.isEmpty() ? null : members.getFirst().route().prefillEndpoint(),
                        true,
                        members.size(),
                        evaluator,
                        reason);
        delivery.batchId = batchId;
        delivery.members = List.copyOf(members);
        delivery.precedingWork = precedingWork;
        delivery.predictedMs = predictedMs;
        delivery.remainingQueueDepth = remainingQueueDepth;
        if (handoff != null) delivery.committed(precedingWork, remainingQueueDepth, false);
        delivery.executionPermit = permit;
        return delivery;
    }

    static RouteResult item(long requestId) {
        return item(requestId, 50, requestId, 100L, 10L);
    }

    static RouteResult item(
            long requestId, int priority, long enqueuedAtMs, long seqLen, long hitCache) {
        RouteResult item = Mockito.mock(RouteResult.class);
        Mockito.when(item.requestId()).thenReturn(String.valueOf(requestId));
        Mockito.when(item.priority()).thenReturn(priority);
        Mockito.when(item.enqueuedAtMs()).thenReturn(enqueuedAtMs);
        Mockito.when(item.seqLen()).thenReturn(seqLen);
        Mockito.when(item.hitCache()).thenReturn(hitCache);
        return item;
    }

    static CapacityBoundary unavailable() {
        return CapacityBoundary.unavailable(CapacityBoundary.Resource.DISPATCH_EXECUTOR);
    }

    public record BatchInput(long batchId, int maximumInflight) {}

    /**
     * Provides transport defaults while individual tests control queue preparation and capacity boundaries.
     */
    static class StubDispatcher extends DeliveryDispatcher {
        StubDispatcher() {
            super(
                    null,
                    service(SchedulingTestConfig.newConfig()),
                    new DeliveryMetricsReporter(null),
                    org.mockito.Mockito.mock(TaskTimer.class),
                    new org.flexlb.balance.scheduler.PlacementConfiguration()
                            .decodeAdmissionCapacity(service(SchedulingTestConfig.newConfig())),
                    org.mockito.Mockito.mock(DeliveryExecution.class));
        }

        protected boolean capacityAvailable(RouteResult item, CapacityBoundary boundary) {
            return false;
        }

        protected Runnable watchCapacity(
                RouteResult item, CapacityBoundary boundary, Runnable wake) {
            return () -> {};
        }

        CapacityBoundary selectBoundary(
                List<RouteResult> items,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong prediction) {
            return CapacityBoundary.OWNERSHIP_LOST;
        }
    }

    /**
     * Lets a queue fixture inspect or replace a preparation boundary while retaining access to the real selection.
     */
    @FunctionalInterface
    public interface Selection {
        CapacityBoundary select(
                List<RouteResult> items,
                String reason,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong prediction,
                java.util.function.Supplier<CapacityBoundary> real);
    }
}

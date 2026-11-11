package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.mockito.Mockito;
import org.springframework.util.ReflectionUtils;

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

    public static void setField(Object target, String name, Object value) {
        if (target instanceof DeliveryCoordinator coordinator) {
            if (name.equals("grpcClient")) {
                target = getField(coordinator, "engine");
                if (target == null) {
                    org.springframework.test.util.ReflectionTestUtils.setField(
                            coordinator,
                            "engine",
                            new EngineDelivery((org.flexlb.engine.grpc.EngineGrpcClient) value));
                    return;
                }
            } else if (name.equals("requests")
                    || name.equals("reporter")
                    || name.equals("decodeAdmissionCapacity")) {
                target = transport(coordinator);
                if (name.equals("decodeAdmissionCapacity")) name = "capacity";
            }
        }
        org.springframework.test.util.ReflectionTestUtils.setField(target, name, value);
    }

    public static Object getField(Object target, String name) {
        if (target instanceof DeliveryCoordinator coordinator) {
            if (name.equals("grpcClient"))
                target =
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                coordinator, "engine");
            else if (name.equals("requests")
                    || name.equals("reporter")
                    || name.equals("decodeAdmissionCapacity")) {
                target =
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                coordinator, "dispatcher");
                if (name.equals("decodeAdmissionCapacity")) name = "capacity";
            }
        }
        return target == null
                ? null
                : org.springframework.test.util.ReflectionTestUtils.getField(target, name);
    }

    public static Delivery.Selection queued(
            List<RouteResult> items, BatchInput batchInput, long[] predictions) {
        return new Delivery.Selection(
                items,
                batchInput != null,
                batchInput == null ? 0L : batchInput.batchId(),
                batchInput == null ? 0 : batchInput.maximumInflight(),
                batchInput == null ? 0L : predictions[0],
                batchInput == null
                        ? null
                        : PrefillBatchFeatures.from(
                                items, RouteResult::seqLen, RouteResult::hitCache),
                batchInput == null ? predictions : null);
    }

    public static DeliveryCoordinator route(DeliveryMetricsReporter reporter) {
        var config = SchedulingTestConfig.newConfig();
        config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
        return new DeliveryCoordinator(
                new org.flexlb.balance.scheduler.RequestRepository(),
                null,
                service(config),
                reporter,
                new org.flexlb.balance.scheduler.SchedulerConfiguration()
                        .decodeAdmissionCapacity(service(config)),
                org.mockito.Mockito.mock(org.flexlb.balance.scheduler.DeliveryExecution.class));
    }

    public static org.flexlb.config.ConfigService service(org.flexlb.config.FlexlbConfig config) {
        var service = Mockito.mock(org.flexlb.config.ConfigService.class);
        Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        return service;
    }

    public static DeliveryCoordinator parked(CapacityBoundary boundary) {
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
            DeliveryCoordinator dispatcher) {
        return dispatcher.mode();
    }

    public static void installQueueFixture(
            PrefillQueueScheduler worker, DeliveryCoordinator dispatcher) {
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
                .processSelection(Mockito.any(), Mockito.any(), Mockito.any(), Mockito.any());
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
        return worker.processSelection(items, reason, evaluator, prediction);
    }

    public static void afterSelection(
            PrefillQueueScheduler worker, java.util.function.Consumer<List<RouteResult>> selected) {
        Mockito.doAnswer(
                        call -> {
                            selected.accept(call.getArgument(0));
                            return call.callRealMethod();
                        })
                .when(worker)
                .processSelection(Mockito.any(), Mockito.any(), Mockito.any(), Mockito.any());
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
                .processSelection(Mockito.any(), Mockito.any(), Mockito.any(), Mockito.any());
    }

    public static void stubRouteDelivery(
            DeliveryCoordinator dispatcher,
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
                .when(org.flexlb.balance.scheduler.DeliveryTestSupport.transport(dispatcher))
                .commit(
                        Mockito.any(Delivery.Member.class),
                        Mockito.eq(Delivery.SendMode.ROUTE_DECISION),
                        Mockito.eq(0L),
                        org.mockito.ArgumentMatchers.isNull());
    }

    static DeliveryDispatcher transport(DeliveryCoordinator coordinator) {
        var field = ReflectionUtils.findField(DeliveryCoordinator.class, "dispatcher");
        ReflectionUtils.makeAccessible(field);
        var transport = (DeliveryDispatcher) ReflectionUtils.getField(field, coordinator);
        if (transport == null || !Mockito.mockingDetails(transport).isMock()) {
            transport =
                    transport == null
                            ? Mockito.mock(DeliveryDispatcher.class)
                            : Mockito.spy(transport);
            ReflectionUtils.setField(field, coordinator, transport);
        }
        return transport;
    }

    static EngineDelivery cancellations(DeliveryCoordinator coordinator) {
        return (EngineDelivery)
                org.springframework.test.util.ReflectionTestUtils.getField(coordinator, "engine");
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
            WorkSnapshot precedingWork,
            long predictedMs,
            PrefillTimePredictor.Evaluator evaluator,
            String reason,
            DeliveryExecution.DeliveryPermit permit) {
        var delivery =
                new Delivery(
                        members.isEmpty() ? null : members.getFirst().route().prefillEndpoint(),
                        true,
                        members.size(),
                        evaluator,
                        reason);
        delivery.batchId = batchId;
        delivery.members.addAll(members);
        delivery.precedingWork = precedingWork;
        delivery.predictedMs = predictedMs;
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
    static class StubDispatcher extends DeliveryCoordinator {
        StubDispatcher() {
            super(
                    new RequestRepository(),
                    null,
                    service(SchedulingTestConfig.newConfig()),
                    new DeliveryMetricsReporter(null),
                    new org.flexlb.balance.scheduler.SchedulerConfiguration()
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

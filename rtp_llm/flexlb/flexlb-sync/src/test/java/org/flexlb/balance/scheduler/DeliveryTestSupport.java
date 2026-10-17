package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.mockito.Mockito;
import java.util.List;
import java.util.OptionalLong;

/** Transport/config fixtures. Queue decisions are stubbed on WorkerBatcher, never on Dispatcher. */
public final class DeliveryTestSupport {
    public static org.flexlb.config.ConfigService service(org.flexlb.config.FlexlbConfig config) {
        var service = Mockito.mock(org.flexlb.config.ConfigService.class);
        Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        return service;
    }

    public static DeliveryDispatcher route(DeliveryMetricsReporter reporter) {
        var config = SchedulingTestConfig.newConfig();
        config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
        return new DeliveryDispatcher(null, service(config), reporter);
    }

    public static DeliveryDispatcher parked(CapacityBoundary boundary) {
        return new StubDispatcher() {
            @Override CapacityBoundary selectBoundary(List<RequestRoute> items, PrefillTimePredictor.Evaluator evaluator,
                    OptionalLong prediction) { return boundary; }
        };
    }

    public static RouteProjection.DeliveryProjection projection(DeliveryDispatcher dispatcher) {
        return RouteProjection.forDelivery(dispatcher.mode());
    }

    static class StubDispatcher extends DeliveryDispatcher {
        StubDispatcher() { super(null, service(SchedulingTestConfig.newConfig()), new DeliveryMetricsReporter(null)); }
        CapacityBoundary selectBoundary(List<RequestRoute> items, PrefillTimePredictor.Evaluator evaluator,
                OptionalLong prediction) { return CapacityBoundary.OWNERSHIP_LOST; }
        protected boolean capacityAvailable(RequestRoute item, CapacityBoundary boundary) { return false; }
        protected Runnable watchCapacity(RequestRoute item, CapacityBoundary boundary, Runnable wake) { return () -> {}; }
    }

    public static void installQueueFixture(WorkerBatcher worker, DeliveryDispatcher dispatcher) {
        if (!(dispatcher instanceof StubDispatcher stub)) { return; }
        Mockito.doAnswer(call -> stub.selectBoundary(call.getArgument(0), call.getArgument(2), call.getArgument(3)))
                .when(worker).processSelection(Mockito.any(), Mockito.any(), Mockito.any(), Mockito.any());
        Mockito.doAnswer(call -> stub.capacityAvailable(call.getArgument(0), call.getArgument(1)))
                .when(worker).capacityAvailable(Mockito.any(), Mockito.any());
        Mockito.doAnswer(call -> stub.watchCapacity(call.getArgument(0), call.getArgument(1), call.getArgument(2)))
                .when(worker).watchCapacity(Mockito.any(), Mockito.any(), Mockito.any());
    }

    public static CapacityBoundary process(WorkerBatcher worker, List<RequestRoute> items, String reason,
            PrefillTimePredictor.Evaluator evaluator, OptionalLong prediction) {
        return worker.processSelection(items, reason, evaluator, prediction);
    }

    public static void afterSelection(WorkerBatcher worker, java.util.function.Consumer<List<RequestRoute>> selected) {
        Mockito.doAnswer(call -> {
            selected.accept(call.getArgument(0));
            return call.callRealMethod();
        }).when(worker).processSelection(Mockito.any(), Mockito.any(), Mockito.any(), Mockito.any());
    }

    @FunctionalInterface public interface Selection {
        CapacityBoundary select(List<RequestRoute> items, String reason, PrefillTimePredictor.Evaluator evaluator, OptionalLong prediction, java.util.function.Supplier<CapacityBoundary> real);
    }

    public static void selectWith(WorkerBatcher worker, Selection selection) {
        Mockito.doAnswer(call -> selection.select(call.getArgument(0), call.getArgument(1), call.getArgument(2),
                call.getArgument(3), () -> {
                    try { return (CapacityBoundary) call.callRealMethod(); }
                    catch (Throwable failure) { throw org.flexlb.util.Failures.propagate(failure, "test selection failed"); }
                })).when(worker).processSelection(Mockito.any(), Mockito.any(), Mockito.any(), Mockito.any());
    }

    public static void beforeCommit(WorkerBatcher worker, Runnable hook) {
        Mockito.doAnswer(call -> { hook.run(); return call.callRealMethod(); })
                .when(worker).commitSelection(Mockito.any());
    }

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
        Mockito.doAnswer(invocation -> {
            var permit = invocation.getArgument(3, DecodeEndpoint.EngineDispatchPermit.class);
            if (permit != null && permit.dispatch().status() != DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED) { return null; }
            var route = invocation.getArgument(0, RequestRoute.class);
            DeliveryClaim claim = new DeliveryClaim(route.ctx(), 1L, route, DeliveryClaimKind.ROUTE_DECISION, 0L);
            Mockito.doAnswer(inv -> { completed.accept(claim, DeliveryResult.delivered()); return null; })
                    .when(requests).onDeliveryEvidence(Mockito.eq(claim), Mockito.isA(ContextResult.RouteDelivered.class));
            return claim;
        }).when(requests).claimDelivery(Mockito.any(), Mockito.eq(DeliveryClaimKind.ROUTE_DECISION), Mockito.eq(0L), Mockito.any());
    }

    private DeliveryTestSupport() {
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

    static CapacityBoundary unavailable() {
        return CapacityBoundary.unavailable(CapacityBoundary.Resource.DISPATCH_EXECUTOR,
                new RouteProjection.AdmissionBlockSemantics("TEST_CAPACITY", RouteProjection.AfterProbeAdmission.BLOCKED,
                        "TEST_CAPACITY", org.flexlb.dao.route.RoleType.PREFILL));
    }
}

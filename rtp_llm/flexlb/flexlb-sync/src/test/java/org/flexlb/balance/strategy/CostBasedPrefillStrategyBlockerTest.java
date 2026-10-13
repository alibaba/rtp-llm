package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Comparator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Regression contracts for Decode capacity projected through Prefill. */
class CostBasedPrefillStrategyBlockerTest {

    @ParameterizedTest
    @CsvSource({
            "false, 50, 10, false, 0,       0,  0, false, SUCCESS, 10",
            "false, 20, 10, false, 0,       0,  0, false, SUCCESS, 10",
            "true,  50, 10, false, 0,       0,  0, false, SUCCESS, 10",
            "true,  20, 10, false, 0,       0,  0, false, BLOCKED,  0",
            "true,  80, 30, false, 0,       0,  0, false, SUCCESS, 10",
            "false, 50, 30, false, 0,       0,  0, false, SUCCESS, 20",
            "true,  50, 30, false, 0,       0,  0, false, SUCCESS, 20",
            "false, 50,  0, false, 0,       0,  0, false, SUCCESS, 20",
            "true,  50,  0, false, 0,       0,  0, false, SUCCESS, 20",
            "true,  50, 10, true,  0,       0,  0, false, BLOCKED,  0",
            "true,  50, 10, false, 1000000, 0,  0, false, SUCCESS, 10",
            "false, 60,  0, false, 0,      80, 25, false, BLOCKED,  0",
            "false, 60, 30, false, 0,      80, 25, false, BLOCKED,  0",
            "false, 60, 30, false, 0,      60, 25, false, SUCCESS, 30",
            "false, 60, 30, false, 0,      40, 25, false, SUCCESS, 30",
            "false, 60, 30, false, 0,      80, 35, false, SUCCESS, 20",
            "false, 60, 30, false, 0,      80, 25, true,  SUCCESS, 20",
            "false, 60, 10, false, 0,      80, 25, false, SUCCESS, 10",
            "true,  60, 30, false, 0,      80, 25, false, BLOCKED,  0",
            "true,  80, 30, false, 0,      60, 25, false, SUCCESS, 10"
    })
    void selectionUsesRetainedQueuePositionAndCollectionTime(
            boolean priorityOrdering, int priority, long enqueueSequence,
            boolean sharedBlock, long collectionWindowMs, int otherPriority,
            long otherSequence, boolean otherExpired,
            PlacementResult.Status expected, long expectedPrefillMs) {
        var directory = mock(WorkerDirectory.class);
        var endpoint = rejectedEndpoint(RoleType.PREFILL, "g", AdmissionRejectReason.UNSPECIFIED);
        when(endpoint.getIp()).thenReturn("worker");
        when(endpoint.getHttpPort()).thenReturn(8080);
        when(directory.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(
                List.of(new EndpointRegistry.PrefillRoutingEntry("worker:8080", endpoint)));
        var pin = mock(WorkerEndpoint.GenerationPin.class);
        when(pin.endpoint()).thenReturn(endpoint);
        long generationId = endpoint.getStatus().getGenerationId();
        when(pin.generationId()).thenReturn(generationId);
        when(directory.captureEndpoint(RoleType.PREFILL, "worker:8080")).thenReturn(pin);
        long now = System.currentTimeMillis();
        var head = new GroupPlanner.Item(2L, 50, 20L, now, Long.MAX_VALUE, 100L, 0L);
        Comparator<GroupPlanner.Item> fifo = Comparator.comparingLong(GroupPlanner.Item::enqueueSeq);
        Comparator<GroupPlanner.Item> order = priorityOrdering
                ? Comparator.comparingInt(GroupPlanner.Item::priority).reversed().thenComparing(fifo) : fifo;
        var active = new java.util.ArrayList<GroupPlanner.Item>();
        active.add(head);
        if (otherSequence != 0L) {
            active.add(new GroupPlanner.Item(3L, otherPriority, otherSequence, now,
                    otherExpired ? now : Long.MAX_VALUE, 100L, 0L));
        }
        active.sort(order);
        var observedHead = active.getFirst();
        var semantics = new RouteProjection.AdmissionBlockSemantics("BLOCKED_HEAD",
                sharedBlock ? RouteProjection.AfterProbeAdmission.BLOCKED
                        : RouteProjection.AfterProbeAdmission.INDEPENDENT, "SHARED_CAPACITY");
        when(endpoint.captureRouteProjectionInputs()).thenReturn(new RouteProjection.Inputs(
                new QueueSnapshot(now, true, order,
                        new GroupPlanner.Constraints(3, 1000000L, 1000000L, 0L, collectionWindowMs),
                        active, new QueueSnapshot.AdmissionBlock(
                                observedHead.requestId(), observedHead.enqueueSeq(), semantics)),
                new WorkSnapshot(now, List.of(), List.of(), 0L)));
        var delivery = new RouteDeliveryStrategy(
                mock(RequestRegistry.class), mock(DeliveryMetrics.class)).projectionPolicy();
        when(endpoint.deliveryProjection()).thenReturn(delivery);
        var context = new BalanceContext(new FlexlbConfig());
        var request = new Request();
        request.setRequestId(1L);
        request.setSeqLen(100L);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        context.setWorkerEnqueueSequence(enqueueSequence);
        long originalEnqueuedAt = enqueueSequence == 0L ? 0L : now - 2000000L;
        context.setFirstWorkerEnqueueTime(originalEnqueuedAt);
        var strategy = new CostBasedPrefillStrategy(directory, mock(CacheAwareService.class),
                mock(EngineHealthReporter.class));

        var result = strategy.select(context, RoleType.PREFILL, "g");
        try (SelectedRole selected = result.value()) {
            assertEquals(expected, result.status());
            if (expected == PlacementResult.Status.SUCCESS) {
                assertEquals(expectedPrefillMs,
                        selected.serverStatus().getPrefillTime(),
                        "a preceding probe must not wait for the head or restart its collection window");
            } else {
                assertEquals("worker:8080", result.blocker().endpoint());
            }
        }
        assertEquals(enqueueSequence, context.getWorkerEnqueueSequence(),
                "prediction must not assign or change the real enqueue sequence");
        assertEquals(originalEnqueuedAt, context.getFirstWorkerEnqueueTime());
        verify(pin).close();
    }

    @ParameterizedTest
    @ValueSource(strings = {"cache_hit", "selected_match", "candidate_match"})
    void failureAfterSelectionReleasesTheSelectedGeneration(String metric) {
        var directory = mock(WorkerDirectory.class);
        var endpoint = rejectedEndpoint(RoleType.PREFILL, "g", AdmissionRejectReason.UNSPECIFIED);
        when(endpoint.getIp()).thenReturn("worker");
        when(endpoint.getHttpPort()).thenReturn(8080);
        when(endpoint.canAcceptRequest()).thenReturn(true);
        when(directory.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(
                List.of(new EndpointRegistry.PrefillRoutingEntry("worker:8080", endpoint)));
        var pin = mock(WorkerEndpoint.GenerationPin.class);
        when(pin.endpoint()).thenReturn(endpoint);
        long generationId = endpoint.getStatus().getGenerationId();
        when(pin.generationId()).thenReturn(generationId);
        when(directory.captureEndpoint(RoleType.PREFILL, "worker:8080")).thenReturn(pin);
        var config = new FlexlbConfig();
        config.getScheduler().setType(SchedulerConfig.Type.DIRECT);
        config.setDispatcher(DispatcherConfig.nonBatch());
        var context = new BalanceContext(config);
        var request = new Request();
        request.setRequestId(1L);
        request.setSeqLen(100L);
        context.setRequest(request);
        var reporter = mock(EngineHealthReporter.class);
        var failure = new IllegalStateException("metric failed");
        switch (metric) {
            case "cache_hit" -> doThrow(failure).when(reporter)
                    .reportCacheHitMetrics(any(), anyLong(),
                            anyDouble());
            case "selected_match" -> doThrow(failure).when(reporter)
                    .reportRoutingSelectedCacheMatchMetrics(any(), anyLong(), anyLong());
            case "candidate_match" -> doThrow(failure).when(reporter)
                    .reportRoutingCandidateMaxCacheMatchMetrics(any(), anyLong());
            default -> throw new AssertionError(metric);
        }
        var strategy = new CostBasedPrefillStrategy(directory, mock(CacheAwareService.class), reporter);

        assertEquals(failure, assertThrows(IllegalStateException.class,
                () -> strategy.select(context, RoleType.PREFILL, "g")));
        verify(pin).close();
    }

    @Test
    void failingDirectCapacityProbeDoesNotAcquireAGenerationPin() {
        var directory = mock(WorkerDirectory.class);
        var endpoint = rejectedEndpoint(RoleType.PREFILL, "g", AdmissionRejectReason.UNSPECIFIED);
        when(directory.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(
                List.of(new EndpointRegistry.PrefillRoutingEntry("first", endpoint)));
        var pin = mock(WorkerEndpoint.GenerationPin.class);
        when(pin.endpoint()).thenReturn(endpoint);
        when(directory.captureEndpoint(RoleType.PREFILL, "first")).thenReturn(pin);
        when(endpoint.canAcceptRequest()).thenThrow(new IllegalStateException("capacity probe failed"));
        var config = new FlexlbConfig();
        config.getScheduler().setType(SchedulerConfig.Type.DIRECT);
        config.setDispatcher(DispatcherConfig.nonBatch());
        var context = new BalanceContext(config);
        var request = new Request();
        request.setRequestId(1L);
        request.setSeqLen(100L);
        context.setRequest(request);
        var strategy = new CostBasedPrefillStrategy(directory, mock(CacheAwareService.class), mock(EngineHealthReporter.class));

        assertThrows(IllegalStateException.class,
                () -> strategy.select(context, RoleType.PREFILL, "g"));
        verify(directory, never()).captureEndpoint(RoleType.PREFILL, "first");
    }

    @Test
    void directFailureDescribesTheSelectedWorkerInsteadOfCombiningOtherWorkersBlockers() {
        for (RoleType role : List.of(RoleType.PREFILL, RoleType.PDFUSION)) {
            for (AdmissionRejectReason first : AdmissionRejectReason.values()) {
                for (AdmissionRejectReason second : AdmissionRejectReason.values()) {
                    var directory = mock(WorkerDirectory.class);
                    var cache = mock(CacheAwareService.class);
                    var strategy = new CostBasedPrefillStrategy(directory, cache, mock(EngineHealthReporter.class));
                    var outside = rejectedEndpoint(role, "other", AdmissionRejectReason.UNSPECIFIED);
                    var firstEndpoint = rejectedEndpoint(role, "target", first);
                    var secondEndpoint = rejectedEndpoint(role, "target", second);
                    when(directory.prefillRoutingSnapshot(role)).thenReturn(List.of(
                            new EndpointRegistry.PrefillRoutingEntry("first", firstEndpoint),
                            new EndpointRegistry.PrefillRoutingEntry("outside", outside),
                            new EndpointRegistry.PrefillRoutingEntry("second", secondEndpoint)));
                    var pin = mock(WorkerEndpoint.GenerationPin.class);
                    when(pin.endpoint()).thenReturn(firstEndpoint);
                    when(directory.captureEndpoint(role, "first")).thenReturn(pin);
                    var config = new FlexlbConfig();
                    config.getScheduler().setType(SchedulerConfig.Type.DIRECT);
                    config.setDispatcher(DispatcherConfig.nonBatch());
                    var context = new BalanceContext(config);
                    var request = new Request();
                    request.setRequestId("1");
                    request.setSeqLen(100L);
                    context.setRequest(request);
                    context.setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));

                    var result = strategy.select(context, role, "target");
                    var failure = result.failure();

                    int expectedCode = first == AdmissionRejectReason.UNSPECIFIED ? 8432
                            : first == AdmissionRejectReason.RESOURCE_EXHAUSTED ? 8431 : 8430;
                    assertEquals(expectedCode, failure.getCode(), role + " " + first + " " + second);
                    assertEquals("first", result.diagnostics().get("selectedEndpoint"));
                    verify(directory).prefillRoutingSnapshot(role);
                    verify(outside, never()).admissionRejectReason(anyInt());
                    verify(secondEndpoint, never()).admissionRejectReason(anyInt());
                    verify(cache).findMatchingEngines(any(),
                            eq(role), eq(List.of("first", "outside", "second")));
                    verify(pin).close();
                }
            }
        }
    }

    private static PrefillEndpoint rejectedEndpoint(RoleType role, String group, AdmissionRejectReason reason) {
        var endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(
                WorkerStatus.createDiscovered(role, group, "worker", 8080, 8090, "test"));
        when(endpoint.admissionRejectReason(50)).thenReturn(reason);
        var predictor = mock(PrefillTimePredictor.class);
        var evaluator = mock(PrefillTimePredictor.Evaluator.class);
        when(evaluator.estimateMs(anyLong(), anyLong())).thenReturn(10L);
        when(predictor.evaluator()).thenReturn(evaluator);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.captureRouteProjectionInputs()).thenReturn(new RouteProjection.Inputs(
                new QueueSnapshot(0L, false, null, null, List.of(), null),
                new WorkSnapshot(0L, List.of(), List.of(), 0L)));
        return endpoint;
    }

}

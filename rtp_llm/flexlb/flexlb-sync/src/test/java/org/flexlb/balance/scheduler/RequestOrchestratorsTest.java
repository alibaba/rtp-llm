package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState.QueuedComputeSnapshot;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.InOrder;

import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Contracts for the scheduler runtime callbacks behind RequestScheduler. */
class RequestOrchestratorsTest {

    @Test
    void shutdownOwnsOneStrictCloseOrder() {
        RequestRegistry lifecycle =
                mock(RequestRegistry.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        when(lifecycle.closeAdmissionAndAwaitMutations()).thenReturn(true);

        runtime(lifecycle, registry).shutdown();

        InOrder order = inOrder(lifecycle, registry);
        order.verify(lifecycle).closeAdmissionAndAwaitMutations();
        order.verify(lifecycle).closeOutstandingAndTerminalize();
        order.verify(lifecycle).closeExpiration();
        order.verify(registry).close();
        order.verify(lifecycle).closePublisher();
    }

    @Test
    void shutdownNonOwnerDoesNotRepeatAnyCloseLeaf() {
        RequestRegistry lifecycle =
                mock(RequestRegistry.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        when(lifecycle.closeAdmissionAndAwaitMutations()).thenReturn(false);

        runtime(lifecycle, registry).shutdown();

        verify(lifecycle, never()).closeOutstandingAndTerminalize();
        verify(lifecycle, never()).closeExpiration();
        verify(registry, never()).close();
        verify(lifecycle, never()).closePublisher();
    }

    @Test
    void shutdownPreservesPrimaryFailureButStillExecutesEveryLeaf() {
        RequestRegistry lifecycle =
                mock(RequestRegistry.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        RuntimeException primary = new RuntimeException("outstanding");
        RuntimeException expiration = new RuntimeException("expiration");
        RuntimeException endpoints = new RuntimeException("endpoints");
        RuntimeException publisher = new RuntimeException("publisher");
        when(lifecycle.closeAdmissionAndAwaitMutations()).thenReturn(true);
        doThrow(primary).when(lifecycle).closeOutstandingAndTerminalize();
        doThrow(expiration).when(lifecycle).closeExpiration();
        doThrow(endpoints).when(registry).close();
        doThrow(publisher).when(lifecycle).closePublisher();

        RuntimeException thrown = assertThrows(
                RuntimeException.class,
                () -> runtime(lifecycle, registry).shutdown());

        assertSame(primary, thrown);
        assertEquals(3, thrown.getSuppressed().length);
        assertSame(expiration, thrown.getSuppressed()[0]);
        assertSame(endpoints, thrown.getSuppressed()[1]);
        assertSame(publisher, thrown.getSuppressed()[2]);
        verify(lifecycle).closePublisher();
    }

    @Test
    void expirationOrchestratorPassesOnlyTheExactRegistrySweeper() {
        RequestRegistry lifecycle =
                mock(RequestRegistry.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        AtomicBoolean exactOwnershipPredicateObserved = new AtomicBoolean();
        doAnswer(invocation -> {
            java.util.function.LongPredicate owns = invocation.getArgument(1);
            exactOwnershipPredicateObserved.set(owns.test(91L));
            return null;
        }).when(registry).evictExpiredOrphans(anyLong(), any());
        doAnswer(invocation -> {
            java.util.function.BiConsumer<Long, java.util.function.LongPredicate>
                    sweeper = invocation.getArgument(0);
            sweeper.accept(123L, requestId -> requestId == 91L);
            return null;
        }).when(lifecycle).maintainExpiration(any());

        runtime(lifecycle, registry).maintainExpiration();

        verify(registry).evictExpiredOrphans(anyLong(), any());
        org.junit.jupiter.api.Assertions.assertTrue(
                exactOwnershipPredicateObserved.get());
    }

    @Test
    void metricsDoNothingAfterLifecycleShutdownBegins() {
        RequestRegistry lifecycle =
                mock(RequestRegistry.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        RequestSchedulerReporter admissionReporter =
                mock(RequestSchedulerReporter.class);
        when(lifecycle.isShuttingDown()).thenReturn(true);

        new SchedulerRuntime(
                lifecycle, registry, reporter, admissionReporter).report();

        verify(lifecycle, never()).liveRequestCount();
        verify(registry, never()).prefillRoutingSnapshot(any());
        verify(registry, never()).snapshotDecodeEndpoints();
    }

    @Test
    void metricsIsolateEveryEndpointLeafAndContinueTraversal() {
        RequestRegistry lifecycle =
                mock(RequestRegistry.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        RequestSchedulerReporter admissionReporter =
                mock(RequestSchedulerReporter.class);
        PrefillEndpoint failingPrefill = mock(PrefillEndpoint.class);
        PrefillEndpoint healthyPrefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        when(failingPrefill.getIp()).thenReturn("host1");
        when(healthyPrefill.getIp()).thenReturn("host2");
        var enqueue = new PrefillQueueFlow.Snapshot(List.of(new PrefillQueueFlow.Entry(
                PrefillQueueFlow.Event.ENQUEUE, 2, 10)));
        when(failingPrefill.queueMetricsSnapshot()).thenThrow(new IllegalStateException("unavailable"));
        when(healthyPrefill.queueMetricsSnapshot()).thenReturn(new WorkerBatcher.QueueMetrics(
                queuedCompute(10, 2, 7), 2, enqueue));
        when(lifecycle.liveRequestCount()).thenReturn(7);
        when(registry.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(List.of(
                new EndpointRegistry.PrefillRoutingEntry("p1", failingPrefill),
                new EndpointRegistry.PrefillRoutingEntry("p2", healthyPrefill)));
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of("d1", decode));
        doThrow(new RuntimeException("metrics unavailable"))
                .when(failingPrefill).reportBatchMetrics(reporter);

        new SchedulerRuntime(
                lifecycle, registry, reporter, admissionReporter).report();

        verify(reporter).reportSchedulerInflightSize(7);
        verify(failingPrefill).reportBatchMetrics(reporter);
        verify(healthyPrefill).reportBatchMetrics(reporter);
        verify(decode).reportBatchMetrics(reporter);
        verify(reporter).reportPrefillQueuedComputeTokens(RoleType.PREFILL, "host1", Double.NaN);
        verify(reporter).reportPrefillQueuedComputeTokens(RoleType.PREFILL, "host2", 19);
        verify(reporter).reportPrefillQueuedComputeTokensByStage(RoleType.PREFILL, "host1",
                Double.NaN, Double.NaN, Double.NaN);
        verify(reporter).reportPrefillQueuedComputeTokensByStage(RoleType.PREFILL, "host2", 10, 2, 7);
        verify(reporter).reportPrefillLocalQueueRequests(RoleType.PREFILL, "host2", 2);
        verify(reporter).reportPrefillQueueFlow(RoleType.PREFILL, "host2", enqueue);
        verify(admissionReporter).reportPrefillQueueDepth("p1", 0);
        verify(admissionReporter).reportPrefillQueueDepth("p2", 0);
        verify(decode).reportAdmissionMetrics(admissionReporter);
    }

    @Test
    void reportsComputePerRoleAndIpIncludingIdleUnknownAndRetiredHosts() {
        var registry = mock(EndpointRegistry.class);
        var reporter = mock(BatchSchedulerReporter.class);
        var admission = mock(RequestSchedulerReporter.class);
        var runtime = new SchedulerRuntime(mock(RequestRegistry.class), registry, reporter, admission);
        var first = mock(PrefillEndpoint.class);
        var second = mock(PrefillEndpoint.class);
        var fusion = mock(PrefillEndpoint.class);
        when(first.getIp()).thenReturn("shared");
        when(second.getIp()).thenReturn("shared");
        when(fusion.getIp()).thenReturn("shared");
        var emptyFlow = new PrefillQueueFlow.Snapshot(List.of());
        when(first.queueMetricsSnapshot()).thenReturn(new WorkerBatcher.QueueMetrics(
                queuedCompute(10, 2, 8), 1, emptyFlow));
        when(second.queueMetricsSnapshot()).thenReturn(new WorkerBatcher.QueueMetrics(
                queuedCompute(20, 1, 9), 2, emptyFlow),
                new WorkerBatcher.QueueMetrics(new QueuedComputeSnapshot(
                        OptionalLong.of(20), OptionalLong.of(1), OptionalLong.empty(), OptionalLong.empty()), 2, emptyFlow));
        when(fusion.queueMetricsSnapshot()).thenReturn(new WorkerBatcher.QueueMetrics(
                queuedCompute(0, 0, 0), 0, emptyFlow));
        var pair = List.of(new EndpointRegistry.PrefillRoutingEntry("shared:1", first),
                new EndpointRegistry.PrefillRoutingEntry("shared:2", second));
        when(registry.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(pair)
                .thenReturn(pair).thenReturn(List.of(pair.get(0))).thenReturn(List.of());
        when(registry.prefillRoutingSnapshot(RoleType.PDFUSION)).thenReturn(
                List.of(new EndpointRegistry.PrefillRoutingEntry("shared:3", fusion)));

        for (int i = 0; i < 5; i++) {
            runtime.report();
        }

        verify(reporter).reportPrefillQueuedComputeTokens(RoleType.PREFILL, "shared", 50);
        verify(reporter).reportPrefillQueuedComputeTokens(RoleType.PREFILL, "shared", 20);
        // One unknown constituent in sample 2, then the last endpoint retired in sample 4.
        verify(reporter, times(2)).reportPrefillQueuedComputeTokens(RoleType.PREFILL, "shared", Double.NaN);
        verify(reporter, times(5)).reportPrefillQueuedComputeTokens(RoleType.PDFUSION, "shared", 0);
        verify(reporter).reportPrefillQueuedComputeTokensByStage(RoleType.PREFILL, "shared", 30, 3, 17);
        verify(reporter).reportPrefillQueuedComputeTokensByStage(RoleType.PREFILL, "shared", 30, 3, Double.NaN);
        verify(reporter).reportPrefillQueuedComputeTokensByStage(RoleType.PREFILL, "shared",
                Double.NaN, Double.NaN, Double.NaN);
        verify(reporter, times(2)).reportPrefillLocalQueueRequests(RoleType.PREFILL, "shared", 3);
        verify(reporter).reportPrefillLocalQueueRequests(RoleType.PREFILL, "shared", 1);
        verify(reporter).reportPrefillLocalQueueRequests(RoleType.PREFILL, "shared", Double.NaN);
        verify(fusion, never()).reportBatchMetrics(any());
        verify(admission, never()).reportPrefillQueueDepth("shared:3", 0);
    }

    private static QueuedComputeSnapshot queuedCompute(long local, long handoff, long waiting) {
        return new QueuedComputeSnapshot(OptionalLong.of(local), OptionalLong.of(handoff),
                OptionalLong.of(waiting), OptionalLong.of(local + handoff + waiting));
    }

    private static SchedulerRuntime runtime(
            RequestRegistry requests, EndpointRegistry endpoints) {
        return new SchedulerRuntime(
                requests,
                endpoints,
                mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
    }
}

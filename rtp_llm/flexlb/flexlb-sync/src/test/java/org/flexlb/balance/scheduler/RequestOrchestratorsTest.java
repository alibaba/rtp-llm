package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.InOrder;

import java.util.LinkedHashMap;
import java.util.Map;
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
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Contracts for the package-private lifecycle orchestrators behind RequestScheduler. */
class RequestOrchestratorsTest {

    @Test
    void shutdownOwnsOneStrictCloseOrder() {
        RequestShutdownOrchestrator.Lifecycle lifecycle =
                mock(RequestShutdownOrchestrator.Lifecycle.class);
        RequestShutdownOrchestrator.Placement placement =
                mock(RequestShutdownOrchestrator.Placement.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        when(lifecycle.closeAdmissionAndAwaitMutations()).thenReturn(true);

        new RequestShutdownOrchestrator(
                lifecycle, registry, placement).shutdown();

        InOrder order = inOrder(placement, lifecycle, registry);
        order.verify(placement).closePlacement();
        order.verify(lifecycle).closeAdmissionAndAwaitMutations();
        order.verify(lifecycle).closeOutstandingAndTerminalize();
        order.verify(lifecycle).closeExpiration();
        order.verify(registry).close();
        order.verify(lifecycle).closePublisher();
    }

    @Test
    void shutdownNonOwnerDoesNotRepeatAnyCloseLeaf() {
        RequestShutdownOrchestrator.Lifecycle lifecycle =
                mock(RequestShutdownOrchestrator.Lifecycle.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        when(lifecycle.closeAdmissionAndAwaitMutations()).thenReturn(false);

        new RequestShutdownOrchestrator(lifecycle, registry).shutdown();

        verify(lifecycle, never()).closeOutstandingAndTerminalize();
        verify(lifecycle, never()).closeExpiration();
        verify(registry, never()).close();
        verify(lifecycle, never()).closePublisher();
    }

    @Test
    void shutdownPreservesPrimaryFailureButStillExecutesEveryLeaf() {
        RequestShutdownOrchestrator.Lifecycle lifecycle =
                mock(RequestShutdownOrchestrator.Lifecycle.class);
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
                () -> new RequestShutdownOrchestrator(lifecycle, registry).shutdown());

        assertSame(primary, thrown);
        assertEquals(3, thrown.getSuppressed().length);
        assertSame(expiration, thrown.getSuppressed()[0]);
        assertSame(endpoints, thrown.getSuppressed()[1]);
        assertSame(publisher, thrown.getSuppressed()[2]);
        verify(lifecycle).closePublisher();
    }

    @Test
    void expirationOrchestratorPassesOnlyTheExactRegistrySweeper() {
        RequestExpirationOrchestrator.Lifecycle lifecycle =
                mock(RequestExpirationOrchestrator.Lifecycle.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        AtomicBoolean exactOwnershipPredicateObserved = new AtomicBoolean();
        doAnswer(invocation -> {
            java.util.function.LongPredicate owns = invocation.getArgument(1);
            exactOwnershipPredicateObserved.set(owns.test(91L));
            return null;
        }).when(registry).evictExpiredOrphans(anyLong(), any());
        doAnswer(invocation -> {
            ExpirationTimer.OrphanSweeper sweeper = invocation.getArgument(0);
            sweeper.sweep(123L, requestId -> requestId == 91L);
            return null;
        }).when(lifecycle).maintainExpiration(any());

        new RequestExpirationOrchestrator(lifecycle, registry)
                .maintainExpiration();

        verify(registry).evictExpiredOrphans(anyLong(), any());
        org.junit.jupiter.api.Assertions.assertTrue(
                exactOwnershipPredicateObserved.get());
    }

    @Test
    void metricsDoNothingAfterLifecycleShutdownBegins() {
        RequestMetricsOrchestrator.Lifecycle lifecycle =
                mock(RequestMetricsOrchestrator.Lifecycle.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        when(lifecycle.isShuttingDown()).thenReturn(true);

        new RequestMetricsOrchestrator(lifecycle, registry, reporter).report();

        verify(lifecycle, never()).liveRequestCount();
        verify(registry, never()).snapshotPrefillEndpoints();
        verify(registry, never()).snapshotDecodeEndpoints();
    }

    @Test
    void metricsIsolateEveryEndpointLeafAndContinueTraversal() {
        RequestMetricsOrchestrator.Lifecycle lifecycle =
                mock(RequestMetricsOrchestrator.Lifecycle.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        PrefillEndpoint failingPrefill = mock(PrefillEndpoint.class);
        PrefillEndpoint healthyPrefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        Map<String, PrefillEndpoint> prefill = new LinkedHashMap<>();
        prefill.put("p1", failingPrefill);
        prefill.put("p2", healthyPrefill);
        when(lifecycle.liveRequestCount()).thenReturn(7);
        when(registry.snapshotPrefillEndpoints()).thenReturn(prefill);
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of("d1", decode));
        doThrow(new RuntimeException("metrics unavailable"))
                .when(failingPrefill).reportBatchMetrics(reporter);

        new RequestMetricsOrchestrator(lifecycle, registry, reporter).report();

        verify(reporter).reportSchedulerInflightSize(7);
        verify(failingPrefill).reportBatchMetrics(reporter);
        verify(healthyPrefill).reportBatchMetrics(reporter);
        verify(decode).reportBatchMetrics(reporter);
    }
}

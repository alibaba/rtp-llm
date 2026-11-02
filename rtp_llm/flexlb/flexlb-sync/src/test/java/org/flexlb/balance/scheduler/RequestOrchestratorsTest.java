package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
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
        verify(registry, never()).snapshotPrefillEndpoints();
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
        Map<String, PrefillEndpoint> prefill = new LinkedHashMap<>();
        prefill.put("p1", failingPrefill);
        prefill.put("p2", healthyPrefill);
        when(lifecycle.liveRequestCount()).thenReturn(7);
        when(registry.snapshotPrefillEndpoints()).thenReturn(prefill);
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of("d1", decode));
        doThrow(new RuntimeException("metrics unavailable"))
                .when(failingPrefill).reportBatchMetrics(reporter);

        new SchedulerRuntime(
                lifecycle, registry, reporter, admissionReporter).report();

        verify(reporter).reportSchedulerInflightSize(7);
        verify(failingPrefill).reportBatchMetrics(reporter);
        verify(healthyPrefill).reportBatchMetrics(reporter);
        verify(decode).reportBatchMetrics(reporter);
        verify(admissionReporter).reportPrefillQueueDepth("p1", 0);
        verify(admissionReporter).reportPrefillQueueDepth("p2", 0);
        verify(decode).reportAdmissionMetrics(admissionReporter);
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

package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.InOrder;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongPredicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

/**
 * Contracts for the scheduler runtime callbacks behind RequestRepository.
 */
class RequestOrchestratorsTest {

    @Test
    void shutdownContinuesAfterFailureAndKeepsTheFirstCause() {
        var requests = mock(RequestRepository.class);
        when(requests.closeRegistration()).thenReturn(true);
        var endpoints = mock(EndpointRegistry.class);
        var dispatcher = mock(DefaultBatchDispatcher.class);
        var runtime = new SchedulerRuntime(requests, endpoints, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), dispatcher, configuration(), mock(RecentCacheKeyTraceReporter.class),
                mock(org.flexlb.balance.eviction.EngineCancelChannel.class));
        var owner = mock(AbstractRequestScheduler.class);
        runtime.initializeScheduler(owner);
        var first = new IllegalStateException("placement close failed");
        var second = new IllegalStateException("dispatcher failed");
        doThrow(first).when(owner).closePlacement();
        doThrow(second).when(dispatcher).shutdownAndAwait();
        assertSame(first, assertThrows(IllegalStateException.class, runtime::shutdown));
        assertEquals(List.of(second), List.of(first.getSuppressed()));
        verify(endpoints).close();
        verify(owner).closeOutstandingAndTerminalize();
        assertTrue(((java.util.concurrent.ExecutorService)runtime.cleanupExecutor()).isShutdown());
    }

    @Test
    void shutdownClosesRegistrationBeforeStoppingAndDrainingProducers() {
        var requests = mock(RequestRepository.class);
        when(requests.closeRegistration()).thenReturn(true, false);
        var endpoints = mock(EndpointRegistry.class);
        var dispatcher = mock(DefaultBatchDispatcher.class);
        var runtime = new SchedulerRuntime(requests, endpoints, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), dispatcher, configuration(), mock(RecentCacheKeyTraceReporter.class),
                mock(org.flexlb.balance.eviction.EngineCancelChannel.class));
        var owner = mock(AbstractRequestScheduler.class);
        runtime.initializeScheduler(owner);
        runtime.shutdown();
        InOrder order = inOrder(requests, owner, dispatcher, endpoints);
        order.verify(requests).closeRegistration();
        order.verify(owner).closePlacement();
        order.verify(owner).awaitAdmissionMutations();
        order.verify(dispatcher).shutdownAndAwait();
        order.verify(endpoints).close();
        runtime.shutdown();
        verify(owner).closePlacement();
        verify(dispatcher).shutdownAndAwait();
    }

    @Test
    void failedRegistrationGateCannotStartShutdown() {
        var requests = mock(RequestRepository.class);
        var endpoints = mock(EndpointRegistry.class);
        var failure = new IllegalStateException("gate failed");
        when(requests.closeRegistration()).thenThrow(failure);
        var runtime = runtime(requests, endpoints);
        try {
            assertSame(failure, assertThrows(IllegalStateException.class, runtime::shutdown));
        }
        finally { runtime.closeRequestExecutors(); runtime.timer().close(); }
        verifyNoInteractions(endpoints);
    }

    @Test
    void shutdownCannotReportSuccessWithUnsettledRequests() {
        var requests = mock(RequestRepository.class);
        when(requests.closeRegistration()).thenReturn(true);
        when(requests.liveRequestCount()).thenReturn(1);
        var runtime = runtime(requests, mock(EndpointRegistry.class));
        var failure = assertThrows(IllegalStateException.class, runtime::shutdown);
        assertEquals("Unsettled requests at shutdown: 1", failure.getMessage());
    }

    @Test
    void concurrentInternalFailuresAreReportedAfterResourceShutdown() throws Exception {
        var endpoints = mock(EndpointRegistry.class);
        var runtime = runtime(new RequestRepository(), endpoints);
        var first = new IllegalStateException("first internal failure");
        var second = new IllegalStateException("second internal failure");
        var start = new CountDownLatch(1);
        var one = CompletableFuture.runAsync(() -> {
            RequestProtocolTestSupport.await(start);
            runtime.recordFailure(first);
        });
        var two = CompletableFuture.runAsync(() -> {
            RequestProtocolTestSupport.await(start);
            runtime.recordFailure(second);
        });
        start.countDown();
        CompletableFuture.allOf(one, two).get(3, TimeUnit.SECONDS);
        var reported = assertThrows(IllegalStateException.class, runtime::shutdown);
        assertTrue(reported == first || reported == second);
        assertEquals(List.of(reported == first ? second : first), List.of(reported.getSuppressed()));
        verify(endpoints).close();
        assertTrue(((java.util.concurrent.ExecutorService) runtime.cleanupExecutor()).isTerminated());
    }

    @Test
    void expirationOrchestratorPassesOnlyTheExactRegistrySweeper() {
        RequestRepository requests = mock(RequestRepository.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        AtomicBoolean exactOwnershipPredicateObserved = new AtomicBoolean();
        doAnswer(invocation -> {
            java.util.function.LongPredicate owns = invocation.getArgument(1);
            exactOwnershipPredicateObserved.set(owns.test(91L));
            return null;
        }).when(registry).evictExpiredOrphans(anyLong(), any());
        when(requests.retainsIdentity(91L)).thenReturn(true);
        runtime(requests, registry).maintainExpiration();
        verify(registry).evictExpiredOrphans(anyLong(), any());
        org.junit.jupiter.api.Assertions.assertTrue(exactOwnershipPredicateObserved.get());
    }

    @Test
    void metricsDoNothingAfterLifecycleShutdownBegins() {
        RequestRepository requests = mock(RequestRepository.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
        RequestSchedulerReporter admissionReporter = mock(RequestSchedulerReporter.class);
        when(requests.isClosed()).thenReturn(true);
        new SchedulerRuntime(requests, registry, reporter, admissionReporter, org.mockito.Mockito.mock(DefaultBatchDispatcher.class), configuration(), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class), org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class)).report();
        verify(requests, never()).liveRequestCount();
        verify(registry, never()).snapshotPrefillEndpoints();
        verify(registry, never()).snapshotDecodeEndpoints();
    }

    @Test
    void metricsIsolateEveryEndpointLeafAndContinueTraversal() {
        RequestRepository requests = mock(RequestRepository.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
        RequestSchedulerReporter admissionReporter = mock(RequestSchedulerReporter.class);
        PrefillEndpoint failingPrefill = mock(PrefillEndpoint.class);
        PrefillEndpoint healthyPrefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        Map<String, PrefillEndpoint> prefill = new LinkedHashMap<>();
        prefill.put("p1", failingPrefill);
        prefill.put("p2", healthyPrefill);
        when(requests.liveRequestCount()).thenReturn(7);
        when(requests.oldestLiveRequestAgeMs()).thenReturn(19L);
        when(registry.snapshotPrefillEndpoints()).thenReturn(prefill);
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of("d1", decode));
        doThrow(new RuntimeException("metrics unavailable")).when(failingPrefill).reportBatchMetrics(reporter);
        new SchedulerRuntime(requests, registry, reporter, admissionReporter, org.mockito.Mockito.mock(DefaultBatchDispatcher.class), configuration(), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class), org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class)).report();
        verify(reporter).reportSchedulerInflight(7, 19L);
        verify(failingPrefill).reportBatchMetrics(reporter);
        verify(healthyPrefill).reportBatchMetrics(reporter);
        verify(decode).reportBatchMetrics(reporter);
        verify(admissionReporter).reportPrefillQueueDepth("p1", 0);
        verify(admissionReporter).reportPrefillQueueDepth("p2", 0);
        verify(decode).reportAdmissionMetrics(admissionReporter);
    }

    private static SchedulerRuntime runtime(RequestRepository requests, EndpointRegistry endpoints) {
        return new SchedulerRuntime(requests, endpoints, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class), org.mockito.Mockito.mock(DefaultBatchDispatcher.class), configuration(), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class), org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class));
    }

    @Test
    void failedPrefillSnapshotDoesNotSuppressDecodeMetrics() {
        RequestRepository requests = mock(RequestRepository.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        DeliveryMetricsReporter batches = mock(DeliveryMetricsReporter.class);
        RequestSchedulerReporter admission = mock(RequestSchedulerReporter.class);
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        when(registry.snapshotPrefillEndpoints()).thenThrow(new IllegalStateException("snapshot failed"));
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of("d1", decode));
        doThrow(new IllegalStateException("batch metrics failed")).when(decode).reportBatchMetrics(batches);

        new SchedulerRuntime(requests, registry, batches, admission, mock(DefaultBatchDispatcher.class), configuration(), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class), org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class)).report();

        verify(batches).reportSchedulerInflight(0, 0L);
        verify(decode).reportBatchMetrics(batches);
        verify(decode).reportAdmissionMetrics(admission);
    }
    private static ConfigService configuration() {
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        return service;
    }
    @Test
    void maintenanceStillSweepsOrphansAfterTerminalRecordFailure() {
        var requests = mock(RequestRepository.class);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        var first = new IllegalStateException("records");
        var second = new IllegalStateException("orphans");
        doThrow(first).when(requests).expireTerminalRecords(anyLong());
        var endpoints = mock(org.flexlb.balance.endpoint.EndpointRegistry.class);
        doThrow(second).when(endpoints).evictExpiredOrphans(anyLong(), org.mockito.ArgumentMatchers.any());
        var runtime = maintenanceRuntime(requests, endpoints, service);
        assertSame(first, assertThrows(IllegalStateException.class, () -> runtime.maintainExpiration()));
        assertEquals(List.of(second), List.of(first.getSuppressed()));
    }

    @Test
    void unexpectedCheckedFailureFromCallbackIsReported() {
        var requests = mock(RequestRepository.class);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        var failure = new Exception("callback failure");
        org.mockito.Mockito.doAnswer(call -> { throw failure; })
                .when(requests).expireTerminalRecords(anyLong());
        var runtime = maintenanceRuntime(requests, mock(org.flexlb.balance.endpoint.EndpointRegistry.class), service);
        var reported = assertThrows(IllegalStateException.class, () -> runtime.maintainExpiration());
        assertSame(failure, reported.getCause());
    }

    @Test
    void maintenanceUsesOnePolicyAndTimeSnapshotBeforeSweepingOrphans() {
        var requests = mock(RequestRepository.class);
        var endpoints = mock(org.flexlb.balance.endpoint.EndpointRegistry.class);
        var service = mock(ConfigService.class);
        var config = SchedulingTestConfig.newConfig();
        config.getWorkerRegistry().getHealth().setStatusStaleAfterMs(250L);
        when(service.loadBalanceConfig()).thenReturn(config);
        when(requests.retainsIdentity(91L)).thenReturn(true);
        org.mockito.Mockito.doAnswer(call -> {
            assertEquals(250L, call.getArgument(0, Long.class));
            LongPredicate retained = call.getArgument(1);
            org.junit.jupiter.api.Assertions.assertTrue(retained.test(91L));
            org.junit.jupiter.api.Assertions.assertFalse(retained.test(92L));
            return null;
        }).when(endpoints).evictExpiredOrphans(anyLong(), org.mockito.ArgumentMatchers.any());

        var runtime = maintenanceRuntime(requests, endpoints, service);
        org.mockito.Mockito.clearInvocations(service);
        runtime.maintainExpiration();

        var order = inOrder(service, requests, endpoints);
        order.verify(requests).isClosed();
        order.verify(service).loadBalanceConfig();
        order.verify(requests).expireTerminalRecords(750L);
        order.verify(endpoints).evictExpiredOrphans(org.mockito.ArgumentMatchers.eq(250L), org.mockito.ArgumentMatchers.any());
        org.mockito.Mockito.verify(service, org.mockito.Mockito.times(1)).loadBalanceConfig();
    }

    @Test
    void maintenancePreservesSaturationAndStopsAtShutdown() {
        var requests = mock(RequestRepository.class);
        var endpoints = mock(org.flexlb.balance.endpoint.EndpointRegistry.class);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        var runtime = new SchedulerRuntime(requests, endpoints, mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class), mock(org.flexlb.service.monitor.RequestSchedulerReporter.class), mock(DefaultBatchDispatcher.class), service, org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class), org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class), () -> Long.MIN_VALUE);
        runtime.maintainExpiration();
        org.mockito.Mockito.verify(requests).expireTerminalRecords(Long.MIN_VALUE);
        org.mockito.Mockito.clearInvocations(requests, service, endpoints);
        when(requests.isClosed()).thenReturn(true);
        runtime.maintainExpiration();
        org.mockito.Mockito.verifyNoInteractions(service, endpoints);
        org.mockito.Mockito.verify(requests, org.mockito.Mockito.never()).expireTerminalRecords(anyLong());
    }

    private static SchedulerRuntime maintenanceRuntime(RequestRepository requests,
            org.flexlb.balance.endpoint.EndpointRegistry endpoints, ConfigService service) {
        return new SchedulerRuntime(requests, endpoints, mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class), mock(org.flexlb.service.monitor.RequestSchedulerReporter.class), mock(DefaultBatchDispatcher.class), service, org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class), org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class), () -> 1000L);
    }
}

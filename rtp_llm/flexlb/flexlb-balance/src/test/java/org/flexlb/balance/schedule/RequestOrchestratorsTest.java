package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Contracts for the scheduler runtime callbacks behind RequestRepository.
 */
class RequestOrchestratorsTest {

    @Test
    void failedRegistrationGateCannotStartShutdown() {
        var requests = mock(RequestRepository.class);
        var endpoints = mock(EndpointRegistry.class);
        var failure = new IllegalStateException("gate failed");
        when(requests.closeRegistration()).thenThrow(failure);
        var runtime = runtime(requests, endpoints);
        try {
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () -> SchedulerTestSupport.shutdown(runtime)));
        } finally {
            SchedulerTestSupport.closeRequestExecutors(runtime);
            SchedulerTestSupport.timer(runtime).close();
        }
        verifyNoInteractions(endpoints);
    }

    @Test
    void shutdownCannotReportSuccessWithUnsettledRequests() {
        var requests = mock(RequestRepository.class);
        when(requests.closeRegistration()).thenReturn(true);
        when(requests.liveRequestCount()).thenReturn(1);
        var runtime = runtime(requests, mock(EndpointRegistry.class));
        var failure =
                assertThrows(
                        IllegalStateException.class, () -> SchedulerTestSupport.shutdown(runtime));
        assertEquals("unsettled requests at shutdown: 1", failure.getMessage());
    }

    @Test
    void concurrentInternalFailuresAreReportedAfterResourceShutdown() throws Exception {
        var endpoints = mock(EndpointRegistry.class);
        var runtime = runtime(new RequestRepository(), endpoints);
        var first = new IllegalStateException("first internal failure");
        var second = new IllegalStateException("second internal failure");
        var start = new CountDownLatch(1);
        var one =
                CompletableFuture.runAsync(
                        () -> {
                            RequestProtocolTestSupport.await(start);
                            runtime.recordFailure(first);
                        });
        var two =
                CompletableFuture.runAsync(
                        () -> {
                            RequestProtocolTestSupport.await(start);
                            runtime.recordFailure(second);
                        });
        start.countDown();
        CompletableFuture.allOf(one, two).get(3, TimeUnit.SECONDS);
        var reported =
                assertThrows(
                        IllegalStateException.class, () -> SchedulerTestSupport.shutdown(runtime));
        assertTrue(reported == first || reported == second);
        assertEquals(
                List.of(reported == first ? second : first), List.of(reported.getSuppressed()));
        verify(endpoints).close();
        assertTrue(
                ((java.util.concurrent.ExecutorService)
                                SchedulerTestSupport.cleanupExecutor(runtime))
                        .isTerminated());
    }

    @Test
    void expirationOrchestratorPassesOnlyTheExactRegistrySweeper() {
        RequestRepository requests = mock(RequestRepository.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        AtomicBoolean exactOwnershipPredicateObserved = new AtomicBoolean();
        doAnswer(
                        invocation -> {
                            java.util.function.Predicate<String> owns = invocation.getArgument(1);
                            exactOwnershipPredicateObserved.set(owns.test(String.valueOf(91L)));
                            return null;
                        })
                .when(registry)
                .evictExpiredOrphans(anyLong(), any());
        when(requests.retainsIdentity("91")).thenReturn(true);
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
        new SchedulerTestServices(
                        requests,
                        registry,
                        reporter,
                        admissionReporter,
                        org.mockito.Mockito.mock(EngineCancellation.class),
                        configuration(),
                        org.mockito.Mockito.mock(EngineGrpcClient.class))
                .report();
        verify(requests, never()).liveRequestCount();
        verify(registry, never()).endpoints(org.flexlb.config.service.RoleType.PREFILL, null);
        verify(registry, never()).endpoints(org.flexlb.config.service.RoleType.DECODE, null);
    }

    @Test
    void metricsIsolateEveryEndpointLeafAndContinueTraversal() {
        RequestRepository requests = mock(RequestRepository.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
        RequestSchedulerReporter admissionReporter = mock(RequestSchedulerReporter.class);
        PrefillEndpoint failingPrefill = mock(PrefillEndpoint.class);
        when(failingPrefill.ipPort()).thenReturn("p1");
        PrefillEndpoint healthyPrefill = mock(PrefillEndpoint.class);
        when(healthyPrefill.ipPort()).thenReturn("p2");
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        Map<String, PrefillEndpoint> prefill = new LinkedHashMap<>();
        prefill.put("p1", failingPrefill);
        prefill.put("p2", healthyPrefill);
        when(requests.liveRequestCount()).thenReturn(7);
        when(requests.oldestLiveRequestAgeMs()).thenReturn(19L);
        when(registry.endpoints(org.flexlb.config.service.RoleType.PREFILL, null))
                .thenReturn(java.util.List.copyOf(prefill.values()));
        when(registry.endpoints(org.flexlb.config.service.RoleType.DECODE, null))
                .thenReturn(java.util.List.of(decode));
        doThrow(new RuntimeException("metrics unavailable"))
                .when(reporter)
                .reportPrefill(failingPrefill);
        new SchedulerTestServices(
                        requests,
                        registry,
                        reporter,
                        admissionReporter,
                        org.mockito.Mockito.mock(EngineCancellation.class),
                        configuration(),
                        org.mockito.Mockito.mock(EngineGrpcClient.class))
                .report();
        verify(reporter).reportSchedulerInflight(7, 19L);
        verify(reporter).reportPrefill(failingPrefill);
        verify(reporter).reportPrefill(healthyPrefill);
        verify(reporter).reportDecode(decode);
        verify(admissionReporter).reportPrefillQueueDepth("p1", 0);
        verify(admissionReporter).reportPrefillQueueDepth("p2", 0);
        verify(admissionReporter)
                .reportDecodePreemption(
                        org.mockito.ArgumentMatchers.eq(decode.ipPort()),
                        org.mockito.ArgumentMatchers.any());
    }

    @Test
    void failedPrefillSnapshotDoesNotSuppressDecodeMetrics() {
        RequestRepository requests = mock(RequestRepository.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        DeliveryMetricsReporter batches = mock(DeliveryMetricsReporter.class);
        RequestSchedulerReporter admission = mock(RequestSchedulerReporter.class);
        DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        when(registry.endpoints(org.flexlb.config.service.RoleType.PREFILL, null))
                .thenThrow(new IllegalStateException("snapshot failed"));
        when(registry.endpoints(org.flexlb.config.service.RoleType.DECODE, null))
                .thenReturn(java.util.List.of(decode));
        doThrow(new IllegalStateException("batch metrics failed"))
                .when(batches)
                .reportDecode(decode);

        new SchedulerTestServices(
                        requests,
                        registry,
                        batches,
                        admission,
                        SchedulerTestSupport.dispatcher(),
                        configuration(),
                        org.mockito.Mockito.mock(EngineGrpcClient.class))
                .report();

        verify(batches).reportSchedulerInflight(0, 0L);
        verify(batches).reportDecode(decode);
        verify(admission)
                .reportDecodePreemption(
                        org.mockito.ArgumentMatchers.eq(decode.ipPort()),
                        org.mockito.ArgumentMatchers.any());
    }

    @Test
    void maintenanceStillSweepsOrphansAfterTerminalRecordFailure() {
        var requests = mock(RequestRepository.class);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        var first = new IllegalStateException("records");
        var second = new IllegalStateException("orphans");
        doThrow(first).when(requests).expireTerminalRecords(anyLong());
        var endpoints = mock(EndpointRegistry.class);
        doThrow(second)
                .when(endpoints)
                .evictExpiredOrphans(anyLong(), org.mockito.ArgumentMatchers.any());
        var runtime = maintenanceRuntime(requests, endpoints, service);
        assertSame(
                first,
                assertThrows(IllegalStateException.class, () -> runtime.maintainExpiration()));
        assertEquals(List.of(second), List.of(first.getSuppressed()));
    }

    @Test
    void unexpectedCheckedFailureFromCallbackIsReported() {
        var requests = mock(RequestRepository.class);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        var failure = new Exception("callback failure");
        org.mockito.Mockito.doAnswer(
                        call -> {
                            throw failure;
                        })
                .when(requests)
                .expireTerminalRecords(anyLong());
        var runtime = maintenanceRuntime(requests, mock(EndpointRegistry.class), service);
        var reported =
                assertThrows(IllegalStateException.class, () -> runtime.maintainExpiration());
        assertSame(failure, reported.getCause());
    }

    @Test
    void maintenanceUsesOnePolicyAndTimeSnapshotBeforeSweepingOrphans() {
        var requests = mock(RequestRepository.class);
        var endpoints = mock(EndpointRegistry.class);
        var service = mock(ConfigService.class);
        var config = SchedulingTestConfig.newConfig();
        when(service.loadBalanceConfig()).thenReturn(config);
        when(requests.retainsIdentity("91")).thenReturn(true);
        org.mockito.Mockito.doAnswer(
                        call -> {
                            assertEquals(10_000L, call.getArgument(0, Long.class));
                            java.util.function.Predicate<String> retained = call.getArgument(1);
                            org.junit.jupiter.api.Assertions.assertTrue(
                                    retained.test(String.valueOf(91L)));
                            org.junit.jupiter.api.Assertions.assertFalse(
                                    retained.test(String.valueOf(92L)));
                            return null;
                        })
                .when(endpoints)
                .evictExpiredOrphans(anyLong(), org.mockito.ArgumentMatchers.any());

        var runtime = maintenanceRuntime(requests, endpoints, service);
        org.mockito.Mockito.clearInvocations(service);
        runtime.maintainExpiration();

        var order = inOrder(service, requests, endpoints);
        order.verify(requests).isClosed();
        order.verify(requests).expireTerminalRecords(-9_000L);
        order.verify(endpoints)
                .evictExpiredOrphans(
                        org.mockito.ArgumentMatchers.eq(10_000L),
                        org.mockito.ArgumentMatchers.any());
        org.mockito.Mockito.verifyNoInteractions(service);
    }

    @Test
    void maintenancePreservesSaturationAndStopsAtShutdown() {
        var requests = mock(RequestRepository.class);
        var endpoints = mock(EndpointRegistry.class);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        var runtime =
                org.mockito.Mockito.spy(
                        new SchedulerTestServices(
                                requests,
                                endpoints,
                                mock(org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                                mock(org.flexlb.balance.metrics.RequestSchedulerReporter.class),
                                SchedulerTestSupport.dispatcher(),
                                service,
                                org.mockito.Mockito.mock(EngineGrpcClient.class),
                                () -> Long.MIN_VALUE));
        runtime.maintainExpiration();
        org.mockito.Mockito.verify(requests).expireTerminalRecords(Long.MIN_VALUE);
        org.mockito.Mockito.clearInvocations(requests, service, endpoints);
        when(requests.isClosed()).thenReturn(true);
        runtime.maintainExpiration();
        org.mockito.Mockito.verifyNoInteractions(service, endpoints);
        org.mockito.Mockito.verify(requests, org.mockito.Mockito.never())
                .expireTerminalRecords(anyLong());
    }

    private static SchedulerTestServices runtime(
            RequestRepository requests, EndpointRegistry endpoints) {
        return new SchedulerTestServices(
                requests,
                endpoints,
                mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                org.mockito.Mockito.mock(EngineCancellation.class),
                configuration(),
                org.mockito.Mockito.mock(EngineGrpcClient.class));
    }

    private static ConfigService configuration() {
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        return service;
    }

    private static SchedulerTestServices maintenanceRuntime(
            RequestRepository requests, EndpointRegistry endpoints, ConfigService service) {
        return new SchedulerTestServices(
                requests,
                endpoints,
                mock(org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                mock(org.flexlb.balance.metrics.RequestSchedulerReporter.class),
                SchedulerTestSupport.dispatcher(),
                service,
                org.mockito.Mockito.mock(EngineGrpcClient.class),
                () -> 1000L);
    }
}

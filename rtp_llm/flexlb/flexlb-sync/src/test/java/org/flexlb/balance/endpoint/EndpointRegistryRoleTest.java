package org.flexlb.balance.endpoint;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.MetricLease;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.mockito.Mockito;

import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EndpointRegistryRoleTest {

    private EndpointRegistry registry;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        registry = new EndpointRegistry(
                configService, null, Mockito.mock(BatchSchedulerReporter.class));
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void should_register_and_resolve_supported_roles_independently() {
        WorkerEndpoint prefill = registry.ensureEndpoint(
                RoleType.PREFILL, "127.0.0.1:8001", status(RoleType.PREFILL, 8001));
        WorkerEndpoint decode = registry.ensureEndpoint(
                RoleType.DECODE, "127.0.0.1:8002", status(RoleType.DECODE, 8002));
        WorkerEndpoint pdFusion = registry.ensureEndpoint(
                RoleType.PDFUSION, "127.0.0.1:8003", status(RoleType.PDFUSION, 8003));
        WorkerEndpoint vit = registry.ensureEndpoint(
                RoleType.VIT, "127.0.0.1:8004", status(RoleType.VIT, 8004));

        assertInstanceOf(PrefillEndpoint.class, prefill);
        assertInstanceOf(DecodeEndpoint.class, decode);
        assertInstanceOf(PrefillEndpoint.class, pdFusion);
        assertInstanceOf(SimpleWorkerEndpoint.class, vit);
        assertSame(prefill, registry.get(RoleType.PREFILL, "127.0.0.1:8001"));
        assertSame(decode, registry.get(RoleType.DECODE, "127.0.0.1:8002"));
        assertSame(pdFusion, registry.get(RoleType.PDFUSION, "127.0.0.1:8003"));
        assertSame(vit, registry.get(RoleType.VIT, "127.0.0.1:8004"));

        for (RoleType roleType : new RoleType[]{
                RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION, RoleType.VIT}) {
            assertEquals(1, registry.getEndpointCount(roleType));
        }
    }

    @Test
    void should_not_mix_roles_when_ip_port_is_shared() {
        String ipPort = "127.0.0.1:8080";
        WorkerEndpoint prefill = registry.ensureEndpoint(
                RoleType.PREFILL, ipPort, status(RoleType.PREFILL, 8080));
        WorkerEndpoint decode = registry.ensureEndpoint(
                RoleType.DECODE, ipPort, status(RoleType.DECODE, 8080));

        assertNotSame(prefill, decode);
        assertSame(prefill, registry.get(RoleType.PREFILL, ipPort));
        assertSame(decode, registry.get(RoleType.DECODE, ipPort));
    }

    @Test
    void simple_endpoint_should_report_status_task_count_as_load() {
        WorkerStatus status = status(RoleType.VIT, 8080);
        status.setRunningTaskList(Map.of("1", new TaskInfo(), "2", new TaskInfo()));
        SimpleWorkerEndpoint endpoint = registry.ensureVitEndpoint("127.0.0.1:8080", status);

        assertEquals(2, endpoint.getLocalTaskCount());
        assertEquals(2, endpoint.getLoadMetric());
    }

    @Test
    void should_not_remove_new_generation_with_expired_status() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus expired = status(RoleType.VIT, 8080);
        WorkerEndpoint oldEndpoint = registry.ensureEndpoint(RoleType.VIT, ipPort, expired);

        WorkerStatus replacement = status(RoleType.VIT, 8080);
        WorkerEndpoint newEndpoint = registry.ensureEndpoint(RoleType.VIT, ipPort, replacement);

        assertNotSame(oldEndpoint, newEndpoint);
        assertFalse(registry.remove(RoleType.VIT, ipPort, expired));
        assertFalse(expired.isAlive());
        assertSame(newEndpoint, registry.get(RoleType.VIT, ipPort));

        assertTrue(registry.remove(RoleType.VIT, ipPort, replacement));
        assertNull(registry.get(RoleType.VIT, ipPort));
    }

    @Test
    void should_close_endpoint_after_conditional_removal() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = status(RoleType.PREFILL, 8080);
        PrefillEndpoint endpoint = Mockito.mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        registry.putPrefill(ipPort, endpoint);

        assertTrue(registry.remove(RoleType.PREFILL, ipPort, status));
        verify(endpoint).close();
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void concurrent_candidates_release_only_losing_metric_leases() throws Exception {
        int threadCount = 16;
        AtomicInteger acquired = new AtomicInteger();
        AtomicInteger released = new AtomicInteger();
        CountDownLatch allCandidatesCreated = new CountDownLatch(threadCount);
        BatchSchedulerReporter trackedReporter = Mockito.mock(BatchSchedulerReporter.class);
        when(trackedReporter.acquireEndpointMetrics(
                Mockito.eq("PREFILL"), Mockito.eq("127.0.0.1"),
                Mockito.eq("127.0.0.1:8080"), Mockito.anyLong()))
                .thenAnswer(ignored -> {
                    acquired.incrementAndGet();
                    allCandidatesCreated.countDown();
                    assertTrue(allCandidatesCreated.await(10, TimeUnit.SECONDS));
                    AtomicInteger closed = new AtomicInteger();
                    return (MetricLease) () -> {
                        if (closed.compareAndSet(0, 1)) {
                            released.incrementAndGet();
                        }
                    };
                });
        ConfigService configService = Mockito.mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        EndpointRegistry concurrentRegistry = new EndpointRegistry(configService, null, trackedReporter);
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        WorkerStatus sharedStatus = status(RoleType.PREFILL, 8080);
        try {
            var futures = java.util.stream.IntStream.range(0, threadCount)
                    .mapToObj(ignored -> executor.submit(() -> concurrentRegistry.ensurePrefillEndpoint(
                            "127.0.0.1:8080", sharedStatus)))
                    .toList();
            for (var future : futures) {
                assertSame(futures.get(0).get(), future.get());
            }

            assertEquals(threadCount, acquired.get());
            assertEquals(threadCount - 1, released.get());
            assertTrue(concurrentRegistry.remove(
                    RoleType.PREFILL, "127.0.0.1:8080", sharedStatus));
            assertEquals(threadCount, released.get());
        } finally {
            concurrentRegistry.close();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void endpoint_construction_failure_releases_metric_lease() {
        AtomicInteger released = new AtomicInteger();
        BatchSchedulerReporter trackedReporter = Mockito.mock(BatchSchedulerReporter.class);
        when(trackedReporter.acquireEndpointMetrics(
                Mockito.anyString(), Mockito.anyString(), Mockito.anyString(), Mockito.anyLong()))
                .thenReturn((MetricLease) released::incrementAndGet);
        FlexlbConfig invalidConfig = new FlexlbConfig();
        invalidConfig.setCostFormula("(");
        ConfigService configService = Mockito.mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(invalidConfig);
        EndpointRegistry failingRegistry = new EndpointRegistry(configService, null, trackedReporter);

        assertThrows(IllegalArgumentException.class, () -> failingRegistry.ensurePrefillEndpoint(
                "127.0.0.1:8080", status(RoleType.PREFILL, 8080)));
        assertEquals(1, released.get());
    }

    private static WorkerStatus status(RoleType roleType, int port) {
        WorkerStatus status = new WorkerStatus();
        status.setRole(roleType);
        status.setIp("127.0.0.1");
        status.setPort(port);
        status.setGrpcPort(port + 1);
        status.setAlive(true);
        return status;
    }
}

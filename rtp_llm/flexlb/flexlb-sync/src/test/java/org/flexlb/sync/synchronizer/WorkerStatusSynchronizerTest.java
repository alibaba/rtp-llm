package org.flexlb.sync.synchronizer;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class WorkerStatusSynchronizerTest {
    @Test
    void aSlowDiscoveryCannotBeOvertakenByNewerCyclesOfTheSameRole() throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.getWorkerRegistry().getHealth().setStatusPollIntervalMs(60_000L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.modelName()).thenReturn("model");
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL));
        WorkerAddressService discovery = mock(WorkerAddressService.class);
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicInteger calls = new AtomicInteger();
        when(discovery.getEngineWorkerList("model", RoleType.PREFILL)).thenAnswer(invocation -> {
            calls.incrementAndGet();
            entered.countDown();
            assertTrue(release.await(3L, TimeUnit.SECONDS));
            return List.of();
        });
        var sync = new WorkerStatusSynchronizer(discovery, mock(EngineHealthReporter.class),
                mock(EndpointRegistry.class), mock(WorkerStatusRpcClient.class), model,
                mock(CacheAwareService.class), mock(DynamicCacheIntervalService.class), service);
        try {
            assertTrue(entered.await(2L, TimeUnit.SECONDS));
            for (int i = 0; i < 20; i++) { sync.syncEngineStatus(); }
            Thread.sleep(100L);
            assertEquals(1, calls.get(), "new discovery cycles must coalesce behind the accepted cycle");
        } finally {
            sync.destroy();
            release.countDown();
        }
        sync.syncEngineStatus();
        assertEquals(1, calls.get(), "shutdown must stop accepting discovery cycles");
        var statusExecutor = (ThreadPoolExecutor) org.springframework.test.util.ReflectionTestUtils
                .getField(sync, "statusCheckExecutor");
        assertThrows(RejectedExecutionException.class, () -> statusExecutor.submit(() -> { }),
                "shutdown must reject polls so their callers can release the acquired lease");
    }
}

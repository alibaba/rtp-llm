package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.match.localsync.DynamicCacheIntervalService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineSyncRunnerTraceTest {
    @Test
    void slowRetirementDoesNotBlockDispatchingHealthyWorkers() throws Exception {
        var directory = new WorkerDirectory(mock(EndpointRegistry.class));
        var addresses = mock(WorkerAddressService.class);
        var statusExecutor = mock(ExecutorService.class);
        var cacheExecutor = mock(ExecutorService.class);
        var cache = mock(CacheAwareService.class);
        WorkerStatus retiring = RunnerTestSupport.discovered(
                RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null);
        WorkerStatus healthy = RunnerTestSupport.alive(
                RoleType.PREFILL, "", "127.0.0.2", 8080, 8081, null);
        directory.currentOrDiscover(RoleType.PREFILL, retiring.getIpPort(), () -> retiring);
        directory.currentOrDiscover(RoleType.PREFILL, healthy.getIpPort(), () -> healthy);
        when(addresses.getEngineWorkerList("model", RoleType.PREFILL))
                .thenReturn(List.of(WorkerHost.of(healthy.getIp(), healthy.getPort())));
        var cleanupEntered = new CountDownLatch(1);
        var finishCleanup = new CountDownLatch(1);
        var retirementTrace = retiring.trace();
        doAnswer(invocation -> {
            assertFalse(retiring.lock.isHeldByCurrentThread());
            assertTrue(retiring.cacheMutationLock.isHeldByCurrentThread());
            cleanupEntered.countDown();
            assertTrue(finishCleanup.await(10, TimeUnit.SECONDS));
            return null;
        }).when(cache).removeEngineBlockCache(retiring, retirementTrace);
        EngineSyncRunner first = runner(directory, addresses, statusExecutor, cacheExecutor, cache, 11L);
        EngineSyncRunner second = runner(directory, addresses, statusExecutor, cacheExecutor, cache, 12L);
        ExecutorService runners = Executors.newFixedThreadPool(2);
        try {
            var firstRun = runners.submit(first);
            assertTrue(cleanupEntered.await(5, TimeUnit.SECONDS));
            firstRun.get(5, TimeUnit.SECONDS);
            var healthyPoll = healthy.trace().snapshot().statusPoll();
            assertNotNull(healthyPoll);
            assertEquals(11L, healthyPoll.syncCycleId(), "the first cycle must dispatch before cleanup finishes");
            verify(statusExecutor).submit(any(GrpcWorkerStatusRunner.class));
            var secondRun = runners.submit(second);
            secondRun.get(5, TimeUnit.SECONDS);
            assertEquals(1L, finishCleanup.getCount(), "cleanup is still blocked after both sync cycles return");
            assertTrue(healthy.pollHealth().reportedAlive());
            assertTrue(healthy.isActiveGeneration());
            finishCleanup.countDown();
        } finally {
            finishCleanup.countDown();
            runners.shutdownNow();
            directory.close();
            assertTrue(runners.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    private static EngineSyncRunner runner(WorkerDirectory directory, WorkerAddressService addresses,
                                           ExecutorService statusExecutor, ExecutorService cacheExecutor,
                                           CacheAwareService cache, long cycle) {
        return new EngineSyncRunner("model", directory, addresses, statusExecutor, cacheExecutor, statusExecutor,
                mock(EngineHealthReporter.class), mock(EngineGrpcService.class), RoleType.PREFILL,
                cache, mock(DynamicCacheIntervalService.class), 5_000L, cycle, 20L, false);
    }
}

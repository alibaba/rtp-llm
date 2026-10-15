package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.master.EndpointTrace.Stage;
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
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineSyncRunnerTraceTest {
    @Test
    void slowRetirementTraceRemainsReadableWhileAnotherCycleDispatchesHealthyWorkers() throws Exception {
        var directory = new WorkerDirectory(mock(EndpointRegistry.class));
        var addresses = mock(WorkerAddressService.class);
        var statusExecutor = mock(ExecutorService.class);
        var cacheExecutor = mock(ExecutorService.class);
        var cache = mock(CacheAwareService.class);
        WorkerStatus retiring = spy(RunnerTestSupport.discovered(
                RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null));
        when(retiring.pollHealth()).thenReturn(new WorkerStatus.PollHealth(
                System.nanoTime() / 1000L - 20_000_000L, 20_000L, 0L, true));
        WorkerStatus healthy = RunnerTestSupport.alive(
                RoleType.PREFILL, "", "127.0.0.2", 8080, 8081, null);
        directory.currentOrDiscover(RoleType.PREFILL, retiring.getIpPort(), () -> retiring);
        directory.currentOrDiscover(RoleType.PREFILL, healthy.getIpPort(), () -> healthy);
        when(addresses.getEngineWorkerList("model", RoleType.PREFILL))
                .thenReturn(List.of(WorkerHost.of(healthy.getIp(), healthy.getPort())));
        var cleanupEntered = new CountDownLatch(1);
        var finishCleanup = new CountDownLatch(1);
        String retiringAddress = retiring.getIpPort();
        var retirementTrace = retiring.trace();
        doAnswer(invocation -> {
            assertFalse(retiring.lock.isHeldByCurrentThread());
            assertTrue(retiring.cacheMutationLock.isHeldByCurrentThread());
            cleanupEntered.countDown();
            assertTrue(finishCleanup.await(10, TimeUnit.SECONDS));
            return null;
        }).when(cache).removeEngineBlockCache(retiringAddress, retirementTrace);
        EngineSyncRunner first = runner(directory, addresses, statusExecutor, cacheExecutor, cache, 11L);
        EngineSyncRunner second = runner(directory, addresses, statusExecutor, cacheExecutor, cache, 12L);
        ExecutorService runners = Executors.newFixedThreadPool(2);
        ExecutorService reader = Executors.newSingleThreadExecutor();
        try {
            var firstRun = runners.submit(first);
            assertTrue(cleanupEntered.await(5, TimeUnit.SECONDS));
            assertNull(healthy.trace().snapshot().statusPoll());
            verify(statusExecutor, never()).submit(any(Runnable.class));
            var secondRun = runners.submit(second);
            secondRun.get(5, TimeUnit.SECONDS);

            // Reading traces must finish before the first cycle releases its cache cleanup lock.
            var snapshots = reader.submit(() -> List.of(first.trace().snapshot(), second.trace().snapshot()))
                    .get(1, TimeUnit.SECONDS);
            assertEquals(11L, snapshots.getFirst().syncCycleId());
            assertEquals(12L, snapshots.getLast().syncCycleId());
            assertNotEquals(snapshots.getFirst().thread(), snapshots.getLast().thread());
            assertEquals(Stage.SYNC_RETIREMENT.name(), snapshots.getFirst().stage());
            assertEquals(Stage.RUNNER_ENDED.name(), snapshots.getLast().stage());
            for (var snapshot : snapshots) {
                assertNotNull(snapshot.thread());
                assertTrue(snapshot.runUs() > 0L);
            }
            TimeUnit.MILLISECONDS.sleep(2L);
            assertTrue(first.trace().snapshot().runUs() > snapshots.getFirst().runUs());
            assertEquals(snapshots.getLast().runUs(), second.trace().snapshot().runUs());
            assertTrue(healthy.pollHealth().reportedAlive());
            assertTrue(healthy.isActiveGeneration());
            var healthyPoll = healthy.trace().snapshot().statusPoll();
            assertNotNull(healthyPoll);
            assertEquals(12L, healthyPoll.syncCycleId(), "the second cycle dispatches before cleanup finishes");
            verify(statusExecutor).submit(any(GrpcWorkerStatusRunner.class));
            assertFalse(firstRun.isDone());

            finishCleanup.countDown();
            firstRun.get(5, TimeUnit.SECONDS);
            assertEquals(Stage.RUNNER_ENDED.name(), first.trace().snapshot().stage());
        } finally {
            finishCleanup.countDown();
            runners.shutdownNow();
            reader.shutdownNow();
            assertTrue(runners.awaitTermination(5, TimeUnit.SECONDS));
            assertTrue(reader.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    private static EngineSyncRunner runner(WorkerDirectory directory, WorkerAddressService addresses,
                                           ExecutorService statusExecutor, ExecutorService cacheExecutor,
                                           CacheAwareService cache, long cycle) {
        return new EngineSyncRunner("model", directory, addresses, statusExecutor, cacheExecutor,
                mock(EngineHealthReporter.class), mock(EngineGrpcService.class), RoleType.PREFILL,
                cache, mock(DynamicCacheIntervalService.class), 5_000L, cycle, 20L, false,
                10_000_000L, 60_000_000L);
    }
}

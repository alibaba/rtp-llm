package org.flexlb.sync.runner;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerGenerationFence;
import org.flexlb.sync.status.WorkerGenerationManager;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anySet;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GrpcCacheStatusCheckRunnerTest {

    private static final String MODEL = "test-model";
    private static final String IP_PORT = "127.0.0.1:8080";

    private EngineGrpcService engineGrpcService;
    private EngineHealthReporter engineHealthReporter;
    private CacheAwareService cacheAwareService;
    private ConcurrentMap<String, WorkerStatus> statuses;
    private WorkerStatus workerStatus;

    @BeforeEach
    void setUp() {
        engineGrpcService = Mockito.mock(EngineGrpcService.class);
        engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        cacheAwareService = Mockito.mock(CacheAwareService.class);
        statuses = new ConcurrentHashMap<>();
        workerStatus = status();
        statuses.put(IP_PORT, workerStatus);
        when(cacheAwareService.publishEngineCacheSnapshot(anyString(), any(), anySet()))
                .thenReturn(successfulPublication());
    }

    @Test
    void callsGrpcServiceWithCurrentCacheVersion() {
        when(engineGrpcService.getCacheStatusAsync(
                anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(),
                eq(RoleType.PREFILL)))
                .thenReturn(CompletableFuture.completedFuture(cacheResponse(1, 101L)));

        runner(workerStatus).run();

        verify(engineGrpcService).getCacheStatusAsync(
                eq("127.0.0.1"), eq(8081), eq(workerStatus), eq(-1L), eq(20L),
                eq(RoleType.PREFILL));
        verify(cacheAwareService).publishEngineCacheSnapshot(
                IP_PORT, RoleType.PREFILL, Set.of(101L));
        assertEquals(1L, workerStatus.getCacheStatus().getVersion());
    }

    @Test
    void ignoresCallbackFromStaleGenerationBeforePublishingCache() {
        WorkerStatus stale = status();
        when(engineGrpcService.getCacheStatusAsync(
                anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(),
                eq(RoleType.PREFILL)))
                .thenReturn(CompletableFuture.completedFuture(cacheResponse(5, 501L)));

        runner(stale).run();

        verify(cacheAwareService, never()).publishEngineCacheSnapshot(
                anyString(), any(), anySet());
        assertNull(stale.getCacheStatus());
        assertNull(workerStatus.getCacheStatus());
    }

    @Test
    void lowerCacheVersionIsANewCacheEpochAndRepublishesFullSnapshot() {
        workerStatus.setCacheStatus(CacheStatus.builder().version(10L).build());
        when(engineGrpcService.getCacheStatusAsync(
                anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(),
                eq(RoleType.PREFILL)))
                .thenReturn(CompletableFuture.completedFuture(
                        cacheResponse(2L, 201L, 202L)));

        runner(workerStatus).run();

        verify(cacheAwareService).publishEngineCacheSnapshot(
                IP_PORT, RoleType.PREFILL, Set.of(201L, 202L));
        assertEquals(2L, workerStatus.getCacheStatus().getVersion());
        assertEquals(2L, workerStatus.getCacheStatus().getCacheKeySize());
    }

    @Test
    void publicationFailureDoesNotCommitVersionAndSameResponseCanRetry() {
        when(cacheAwareService.publishEngineCacheSnapshot(
                anyString(), any(), anySet()))
                .thenReturn(failedPublication())
                .thenReturn(successfulPublication());
        when(engineGrpcService.getCacheStatusAsync(
                anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(),
                eq(RoleType.PREFILL)))
                .thenReturn(CompletableFuture.completedFuture(cacheResponse(5L, 501L)));
        GrpcCacheStatusCheckRunner runner = runner(workerStatus);

        runner.run();
        assertNull(workerStatus.getCacheStatus(),
                "failed publication must not advance the cache commit marker");
        runner.run();

        verify(cacheAwareService, times(2)).publishEngineCacheSnapshot(
                IP_PORT, RoleType.PREFILL, Set.of(501L));
        verify(engineGrpcService, times(2)).getCacheStatusAsync(
                eq("127.0.0.1"), eq(8081), eq(workerStatus), eq(-1L), eq(20L),
                eq(RoleType.PREFILL));
        assertEquals(5L, workerStatus.getCacheStatus().getVersion());
    }

    @Test
    void retirementWaitsForOldPublicationThenClearsBeforeNewGenerationPublishes()
            throws Exception {
        WorkerStatus old = workerStatus;
        WorkerGenerationFence fence = new WorkerGenerationFence();
        EndpointRegistry registry = Mockito.mock(EndpointRegistry.class);
        WorkerGenerationManager generationManager =
                new WorkerGenerationManager(registry, cacheAwareService, fence);
        CountDownLatch oldPublicationEntered = new CountDownLatch(1);
        CountDownLatch releaseOldPublication = new CountDownLatch(1);
        CountDownLatch retirementStarted = new CountDownLatch(1);
        List<String> order = Collections.synchronizedList(new ArrayList<>());
        AtomicReference<Set<Long>> visibleKeys = new AtomicReference<>(Set.of());

        when(cacheAwareService.publishEngineCacheSnapshot(
                eq(IP_PORT), eq(RoleType.PREFILL), anySet()))
                .thenAnswer(invocation -> {
                    Set<?> publishedKeys = invocation.getArgument(2);
                    Set<Long> keys = publishedKeys.stream()
                            .map(Long.class::cast)
                            .collect(Collectors.toUnmodifiableSet());
                    if (keys.contains(101L)) {
                        order.add("publish-A");
                        oldPublicationEntered.countDown();
                        if (!releaseOldPublication.await(2, TimeUnit.SECONDS)) {
                            throw new AssertionError("timed out releasing old publication");
                        }
                    } else {
                        order.add("publish-B");
                    }
                    visibleKeys.set(keys);
                    return successfulPublication();
                });
        doAnswer(invocation -> {
            order.add("clear");
            visibleKeys.set(Set.of());
            return null;
        }).when(cacheAwareService).clearEngineCache(IP_PORT);
        when(engineGrpcService.getCacheStatusAsync(
                anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(),
                eq(RoleType.PREFILL)))
                .thenReturn(CompletableFuture.completedFuture(cacheResponse(1L, 101L)))
                .thenReturn(CompletableFuture.completedFuture(cacheResponse(2L, 202L)));

        ExecutorService callbackExecutor = Executors.newSingleThreadExecutor();
        ExecutorService retirementExecutor = Executors.newSingleThreadExecutor();
        try {
            runner(old, fence, callbackExecutor).run();
            assertTrue(oldPublicationEntered.await(1, TimeUnit.SECONDS));

            Future<Boolean> retirement = retirementExecutor.submit(() -> {
                retirementStarted.countDown();
                return generationManager.retireIf(
                        statuses, RoleType.PREFILL, IP_PORT, old, ignored -> true);
            });
            assertTrue(retirementStarted.await(1, TimeUnit.SECONDS));
            assertThrows(TimeoutException.class,
                    () -> retirement.get(100, TimeUnit.MILLISECONDS),
                    "the generation writer must wait for the old cache reader");
            assertFalse(order.contains("clear"));

            releaseOldPublication.countDown();
            assertTrue(retirement.get(2, TimeUnit.SECONDS));
            assertEquals(List.of("publish-A", "clear"), List.copyOf(order));

            WorkerStatus fresh = generationManager.getOrCreate(
                    statuses, RoleType.PREFILL, IP_PORT);
            assertSame(fresh, statuses.get(IP_PORT));
            runner(fresh, fence, Runnable::run).run();

            assertEquals(List.of("publish-A", "clear", "publish-B"),
                    List.copyOf(order));
            assertEquals(Set.of(202L), visibleKeys.get());
            assertEquals(2L, fresh.getCacheStatus().getVersion());
            assertSame(fresh, statuses.get(IP_PORT));
        } finally {
            releaseOldPublication.countDown();
            callbackExecutor.shutdownNow();
            retirementExecutor.shutdownNow();
        }
    }

    private GrpcCacheStatusCheckRunner runner(WorkerStatus generation) {
        return runner(generation, new WorkerGenerationFence(), Runnable::run);
    }

    private GrpcCacheStatusCheckRunner runner(
            WorkerStatus generation, WorkerGenerationFence fence,
            Executor callbackExecutor) {
        return new GrpcCacheStatusCheckRunner(
                MODEL, IP_PORT, RoleType.PREFILL, generation,
                engineHealthReporter, engineGrpcService, cacheAwareService,
                statuses, fence, 20L, new LongAdder(), 50L, callbackExecutor);
    }

    private static WorkerStatus status() {
        WorkerStatus status = new WorkerStatus();
        status.setRole(RoleType.PREFILL);
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        return status;
    }

    private static EngineRpcService.CacheStatusPB cacheResponse(
            long version, Long... keys) {
        EngineRpcService.CacheStatusPB.Builder response =
                EngineRpcService.CacheStatusPB.newBuilder()
                        .setVersion(version)
                        .setAvailableKvCache(1_000)
                        .setTotalKvCache(2_000)
                        .setBlockSize(128);
        for (Long key : keys) {
            response.putCacheKeys(key, true);
        }
        return response.build();
    }

    private static WorkerCacheUpdateResult successfulPublication() {
        return WorkerCacheUpdateResult.builder()
                .success(true)
                .engineIpPort(IP_PORT)
                .build();
    }

    private static WorkerCacheUpdateResult failedPublication() {
        return WorkerCacheUpdateResult.builder()
                .success(false)
                .engineIpPort(IP_PORT)
                .errorMessage("test failure")
                .build();
    }
}

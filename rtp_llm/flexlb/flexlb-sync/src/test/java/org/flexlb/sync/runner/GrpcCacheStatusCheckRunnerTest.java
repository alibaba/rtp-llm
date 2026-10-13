package org.flexlb.sync.runner;

import org.flexlb.testfixture.WorkerStatusTestFixtures;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.match.localsync.DynamicCacheIntervalService;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class GrpcCacheStatusCheckRunnerTest {

    private final EngineGrpcService engineGrpcService = Mockito.mock(EngineGrpcService.class);

    private final EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);

    private final CacheAwareService localKvCacheAwareManager = Mockito.mock(CacheAwareService.class);

    private final DynamicCacheIntervalService cacheIntervalService =
            Mockito.mock(DynamicCacheIntervalService.class);

    @Test
    void cachePollAndPublicationProceedWhileStatusLockIsHeld() throws Exception {
        var status = workerStatus();
        var directory = directory(status);
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), eq(status), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(
                        EngineRpcService.CacheStatusPB.newBuilder().setVersion(1L).build()));
        when(localKvCacheAwareManager.updateEngineBlockCache(eq(status), any()))
                .thenReturn(WorkerCacheUpdateResult.builder().success(true).build());
        var executor = java.util.concurrent.Executors.newSingleThreadExecutor();
        status.lock.lock();
        try {
            executor.submit(() -> {
                var lease = status.tryBeginCachePoll(1L);
                org.junit.jupiter.api.Assertions.assertNotNull(lease);
                new GrpcCacheStatusCheckRunner("model", status.getIpPort(), "test", RoleType.PREFILL,
                        status, lease, directory, engineHealthReporter, engineGrpcService,
                        localKvCacheAwareManager, cacheIntervalService, 1000L, true, Runnable::run).run();
            }).get(2, java.util.concurrent.TimeUnit.SECONDS);
            assertEquals(1L, status.cacheIndexSnapshot().indexedVersion());
        } finally {
            status.lock.unlock();
            executor.shutdownNow();
            directory.close();
        }
    }

    @Test
    void delayedOldCacheResponseCannotWriteAfterCleanupAndReplacement() {
        var old = workerStatus();
        var directory = directory(old);
        var rpc = new CompletableFuture<EngineRpcService.CacheStatusPB>();
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), eq(old), anyLong(), anyLong(), any()))
                .thenReturn(rpc);
        new GrpcCacheStatusCheckRunner("model", old.getIpPort(), "test", RoleType.PREFILL,
                old, old.tryBeginCachePoll(1L), directory, engineHealthReporter, engineGrpcService,
                localKvCacheAwareManager, cacheIntervalService, 1000L, true, Runnable::run).run();
        old.lock.lock();
        EndpointRegistry.DetachedGeneration detached;
        try {
            detached = directory.beginRetirement(RoleType.PREFILL, old.getIpPort(), old, "TEST");
        } finally {
            old.lock.unlock();
        }
        directory.completeRetirement(RoleType.PREFILL, old.getIpPort(), old, detached,
                localKvCacheAwareManager, org.slf4j.LoggerFactory.getLogger(getClass()));
        var replacement = workerStatus();
        org.junit.jupiter.api.Assertions.assertSame(replacement,
                directory.currentOrDiscover(RoleType.PREFILL, old.getIpPort(), () -> replacement));
        rpc.complete(EngineRpcService.CacheStatusPB.newBuilder().setVersion(2L).putCacheKeys(123L, true).build());
        verify(localKvCacheAwareManager, never()).updateEngineBlockCache(any(), any());
        org.junit.jupiter.api.Assertions.assertNull(replacement.getCacheStatus());
        assertEquals(-1L, replacement.cacheIndexSnapshot().indexedVersion());
        var discarded = old.trace().snapshot().cachePoll();
        assertEquals(-1L, discarded.times().convertStartMs(), "retired responses must not copy cache keys");
        org.junit.jupiter.api.Assertions.assertTrue(discarded.leaseReleased());
        directory.close();
    }

    @Test
    void cacheConversionWaitsForCacheExecutorInsteadOfGrpcCompletionThread() {
        var status = workerStatus();
        var lease = status.tryBeginCachePoll(1L);
        var rpc = new CompletableFuture<EngineRpcService.CacheStatusPB>();
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), any(), anyLong(), anyLong(), any()))
                .thenReturn(rpc);
        var queue = new java.util.ArrayList<Runnable>();
        new GrpcCacheStatusCheckRunner("model", status.getIpPort(), "test", RoleType.PREFILL,
                status, lease, directory(status), engineHealthReporter, engineGrpcService,
                localKvCacheAwareManager, cacheIntervalService, 1000L, true, queue::add).run();
        rpc.complete(EngineRpcService.CacheStatusPB.newBuilder().setVersion(1L).build());
        assertEquals(1, queue.size());
        assertEquals(-1, status.trace().snapshot().cachePoll().times().convertStartMs());
        verifyNoInteractions(localKvCacheAwareManager);
        queue.removeFirst().run();
        org.junit.jupiter.api.Assertions.assertTrue(status.trace().snapshot().cachePoll().times().convertStartMs() >= 0);
        org.junit.jupiter.api.Assertions.assertTrue(status.trace().snapshot().cachePoll().leaseReleased());
    }

    @Test
    void cacheUpdateDoesNotHoldHealthLockAndRetirementWaitsForCacheWriter() throws Exception {
        WorkerStatus status = workerStatus();
        WorkerDirectory directory = directory(status);
        java.util.concurrent.CountDownLatch updating = new java.util.concurrent.CountDownLatch(1);
        java.util.concurrent.CountDownLatch release = new java.util.concurrent.CountDownLatch(1);
        java.util.concurrent.atomic.AtomicBoolean cleanupHeldCacheLock =
                new java.util.concurrent.atomic.AtomicBoolean();
        java.util.concurrent.atomic.AtomicBoolean cleanupHeldStatusLock =
                new java.util.concurrent.atomic.AtomicBoolean(true);
        Mockito.doAnswer(call -> {
            cleanupHeldCacheLock.set(status.cacheMutationLock.isHeldByCurrentThread());
            cleanupHeldStatusLock.set(status.lock.isHeldByCurrentThread());
            return null;
        }).when(localKvCacheAwareManager).removeEngineBlockCache(status, status.trace());
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), eq(status),
                anyLong(), anyLong(), eq(RoleType.PREFILL)))
                .thenReturn(CompletableFuture.completedFuture(
                        EngineRpcService.CacheStatusPB.newBuilder().setVersion(1).build()));
        when(localKvCacheAwareManager.updateEngineBlockCache(eq(status), any())).thenAnswer(call -> {
            org.junit.jupiter.api.Assertions.assertFalse(status.lock.isHeldByCurrentThread());
            org.junit.jupiter.api.Assertions.assertTrue(status.cacheMutationLock.isHeldByCurrentThread());
            updating.countDown();
            org.junit.jupiter.api.Assertions.assertTrue(release.await(5, java.util.concurrent.TimeUnit.SECONDS));
            return WorkerCacheUpdateResult.builder().success(true).cacheVersion(1).build();
        });
        java.util.concurrent.ExecutorService threads = java.util.concurrent.Executors.newFixedThreadPool(2);
        try {
            java.util.concurrent.Future<?> update = threads.submit(() ->
                    new GrpcCacheStatusCheckRunner("test-model", status.getIpPort(), "test-site",
                            RoleType.PREFILL, status, status.tryBeginCachePoll(1L), directory,
                            engineHealthReporter, engineGrpcService, localKvCacheAwareManager, cacheIntervalService,
                            5000, true, Runnable::run).run());
            org.junit.jupiter.api.Assertions.assertTrue(updating.await(2, java.util.concurrent.TimeUnit.SECONDS));
            org.junit.jupiter.api.Assertions.assertTrue(status.lock.tryLock(1, java.util.concurrent.TimeUnit.SECONDS));
            org.flexlb.balance.endpoint.EndpointRegistry.DetachedGeneration detached;
            try {
                WorkerStatusTestFixtures.readyForScheduling(status);
                detached = directory.beginRetirement(RoleType.PREFILL, status.getIpPort(), status, "TEST_RETIREMENT");
            } finally {
                status.lock.unlock();
            }
            java.util.concurrent.Future<?> retiring = threads.submit(() -> directory.completeRetirement(
                    RoleType.PREFILL, status.getIpPort(), status, detached, localKvCacheAwareManager,
                    org.slf4j.LoggerFactory.getLogger(getClass())));
            org.junit.jupiter.api.Assertions.assertSame(status,
                    directory.currentOrDiscover(RoleType.PREFILL, status.getIpPort(), () -> {
                        throw new AssertionError("replacement published before old cache writer drained");
                    }));
            // Even final cleanup/removal must not wait on the status lock.
            status.lock.lock();
            try {
                release.countDown();
                update.get(2, java.util.concurrent.TimeUnit.SECONDS);
                retiring.get(2, java.util.concurrent.TimeUnit.SECONDS);
            } finally {
                status.lock.unlock();
            }
            verify(localKvCacheAwareManager).removeEngineBlockCache(status, status.trace());
            org.junit.jupiter.api.Assertions.assertTrue(cleanupHeldCacheLock.get(),
                    "detached-key cleanup must serialize with the old cache writer");
            org.junit.jupiter.api.Assertions.assertFalse(cleanupHeldStatusLock.get(),
                    "index cleanup must not hold the health/lifecycle lock");
            org.junit.jupiter.api.Assertions.assertTrue(directory.statusSnapshot(RoleType.PREFILL).isEmpty());
            org.junit.jupiter.api.Assertions.assertEquals(-1L, status.cacheIndexSnapshot().indexedVersion(),
                    "retiring callback cannot commit its index version");
        } finally {
            release.countDown();
            threads.shutdownNow();
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {
            "RPC_THROW", "RPC_FAILED", "CONVERSION_FAILED", "CALLBACK_REJECTED", "CALLBACK_FAILED"})
    void failurePathsReleaseCachePollAndPreserveBoundary(String failurePoint) {
        var status = workerStatus();
        var lease = status.tryBeginCachePoll(1L);
        var failure = new IllegalStateException("injected failure");
        var stub = when(engineGrpcService.getCacheStatusAsync(
                anyString(), anyInt(), any(), anyLong(), anyLong(), any()));
        if (failurePoint.equals("RPC_THROW")) {
            stub.thenThrow(failure);
        } else if (failurePoint.equals("CALLBACK_REJECTED")) {
            stub.thenReturn(CompletableFuture.completedFuture(EngineRpcService.CacheStatusPB.getDefaultInstance()));
        } else if (failurePoint.equals("CONVERSION_FAILED")) {
            stub.thenReturn(CompletableFuture.completedFuture(null));
        } else {
            stub.thenReturn(CompletableFuture.failedFuture(new java.util.concurrent.CompletionException(failure)));
        }
        if (failurePoint.equals("CALLBACK_FAILED")) {
            Mockito.doThrow(failure).when(engineHealthReporter).reportCacheStatusCheckerFail(anyString(), any(), any());
        }
        java.util.concurrent.Executor executor = failurePoint.equals("CALLBACK_REJECTED")
                ? task -> { throw new java.util.concurrent.RejectedExecutionException("pool full"); }
                : Runnable::run;
        var runner = new GrpcCacheStatusCheckRunner("test-model", status.getIpPort(), "test", RoleType.PREFILL,
                status, lease, directory(status), engineHealthReporter, engineGrpcService,
                localKvCacheAwareManager, cacheIntervalService, 1000L, true, executor);
        if (failurePoint.equals("RPC_THROW")) {
            org.junit.jupiter.api.Assertions.assertSame(failure,
                    org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, runner::run));
        } else {
            runner.run();
        }
        var snapshot = status.trace().snapshot().cachePoll();
        String expectedFailure = switch (failurePoint) {
            case "RPC_THROW" -> "RPC_INITIATION_FAILED";
            case "CONVERSION_FAILED" -> "CONVERSION_FAILED";
            case "CALLBACK_REJECTED" -> "CALLBACK_FAILED";
            default -> "RPC_COMPLETED";
        };
        assertEquals(expectedFailure, snapshot.failureStage());
        org.junit.jupiter.api.Assertions.assertTrue(snapshot.times().rpcResultMs() >= 0);
        org.junit.jupiter.api.Assertions.assertTrue(snapshot.leaseReleased());
        if (failurePoint.equals("CONVERSION_FAILED")) {
            assertEquals("OK", snapshot.rpcResult());
            org.junit.jupiter.api.Assertions.assertTrue(snapshot.times().convertStartMs() >= 0);
        }
        if (failurePoint.equals("CALLBACK_REJECTED")) {
            assertEquals(-1, snapshot.times().callbackStartMs());
        }
        verifyNoInteractions(localKvCacheAwareManager);
        try (var next = status.tryBeginCachePoll(1L)) {
            org.junit.jupiter.api.Assertions.assertNotNull(next, "a failed poll must not prevent future polling");
        }
    }

    @Test
    void decodeCacheResponseUpdatesCapacityWithoutBuildingDetailedIndex() {
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        EngineRpcService.CacheStatusPB response = EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(1)
                .setAvailableKvCache(1000)
                .setTotalKvCache(2000)
                .setBlockSize(128)
                .build();
        when(engineGrpcService.getCacheStatusAsync(
                anyString(), anyInt(), eq(status), anyLong(), anyLong(), eq(RoleType.DECODE)))
                .thenReturn(CompletableFuture.completedFuture(response));

        new GrpcCacheStatusCheckRunner(
                "test-model", status.getIpPort(), "test-site", RoleType.DECODE,
                status, status.tryBeginCachePoll(1L), directory(status),
                engineHealthReporter, engineGrpcService, localKvCacheAwareManager, cacheIntervalService,
                20, true, Runnable::run).run();

        assertEquals(1000, status.getCacheStatus().getAvailableKvCache());
        assertEquals(2000, status.getCacheStatus().getTotalKvCache());
        verifyNoInteractions(localKvCacheAwareManager);
    }

    @Test
    void testGrpcCacheStatusCheckRunner() {
        // Arrange
        String modelName = "test-model";
        String site = "test-site";

        WorkerStatus workerStatus = workerStatus();
        String ipPort = workerStatus.getLogicalIpPort();
        WorkerDirectory directory = directory(workerStatus);

        EngineRpcService.CacheStatusPB cacheStatusPB = EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(1)
                .setAvailableKvCache(1000)
                .setTotalKvCache(2000)
                .setBlockSize(128)
                .build();
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(), eq(RoleType.PREFILL))).thenReturn(CompletableFuture.completedFuture(cacheStatusPB));
        when(localKvCacheAwareManager.updateEngineBlockCache(eq(workerStatus), any(EndpointTrace.Poll.class)))
                .thenReturn(WorkerCacheUpdateResult.builder()
                        .success(true)
                        .cacheVersion(1)
                        .build());

        // Act
        GrpcCacheStatusCheckRunner runner = new GrpcCacheStatusCheckRunner(
                modelName, ipPort, site, RoleType.PREFILL, workerStatus,
                workerStatus.tryBeginCachePoll(1L),
                directory,
                engineHealthReporter, engineGrpcService,
                localKvCacheAwareManager, cacheIntervalService,
                20, true, Runnable::run);
        runner.run();

        // Give some time for async execution
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Assert
        verify(engineGrpcService).getCacheStatusAsync(eq("127.0.0.1"), eq(8081), any(WorkerStatus.class), eq(-1L), eq(20L), eq(RoleType.PREFILL));
        assertEquals(1L, workerStatus.cacheIndexSnapshot().indexedVersion());
    }

    @Test
    void failedCacheIndexUpdateRetriesTheSameWorkerVersion() {
        WorkerStatus workerStatus = workerStatus();
        String ipPort = workerStatus.getLogicalIpPort();
        WorkerDirectory directory = directory(workerStatus);
        EngineRpcService.CacheStatusPB firstResponse =
                EngineRpcService.CacheStatusPB.newBuilder()
                        .setVersion(1)
                        .setBlockSize(128)
                        .build();
        EngineRpcService.CacheStatusPB secondResponse =
                EngineRpcService.CacheStatusPB.newBuilder()
                        .setVersion(2)
                        .setBlockSize(128)
                        .build();
        when(engineGrpcService.getCacheStatusAsync(
                anyString(), anyInt(), any(WorkerStatus.class), anyLong(),
                anyLong(), eq(RoleType.PREFILL)))
                .thenReturn(
                        CompletableFuture.completedFuture(firstResponse),
                        CompletableFuture.completedFuture(secondResponse),
                        CompletableFuture.completedFuture(secondResponse));
        when(localKvCacheAwareManager.updateEngineBlockCache(eq(workerStatus), any(EndpointTrace.Poll.class)))
                .thenReturn(
                        WorkerCacheUpdateResult.builder()
                                .success(true)
                                .cacheVersion(1)
                                .build(),
                        WorkerCacheUpdateResult.builder()
                                .success(false)
                                .errorMessage("index failed")
                                .build(),
                        WorkerCacheUpdateResult.builder()
                                .success(true)
                                .cacheVersion(2)
                                .build());

        runCachePoll(ipPort, workerStatus, directory, false);
        assertEquals(1L, workerStatus.cacheIndexSnapshot().indexedVersion());

        runCachePoll(ipPort, workerStatus, directory, false);
        assertEquals(1L, workerStatus.cacheIndexSnapshot().indexedVersion());
        assertEquals(2L, workerStatus.getCacheStatus().getVersion());

        runCachePoll(ipPort, workerStatus, directory, false);
        assertEquals(2L, workerStatus.cacheIndexSnapshot().indexedVersion());
        verify(engineGrpcService).getCacheStatusAsync(
                eq("127.0.0.1"), eq(8081), any(WorkerStatus.class),
                eq(-1L), eq(20L), eq(RoleType.PREFILL));
        verify(engineGrpcService, times(2)).getCacheStatusAsync(
                eq("127.0.0.1"), eq(8081), any(WorkerStatus.class),
                eq(1L), eq(20L), eq(RoleType.PREFILL));
    }

    @Test
    void staleGenerationCallbackCannotPublishAddressCache() {
        WorkerStatus oldStatus = workerStatus();
        String ipPort = oldStatus.getLogicalIpPort();
        WorkerDirectory directory = Mockito.mock(WorkerDirectory.class);
        when(directory.isCurrentStatus(
                RoleType.PREFILL, ipPort, oldStatus)).thenReturn(false);
        CompletableFuture<EngineRpcService.CacheStatusPB> response =
                new CompletableFuture<>();
        when(engineGrpcService.getCacheStatusAsync(
                anyString(), anyInt(), any(WorkerStatus.class), anyLong(),
                anyLong(), eq(RoleType.PREFILL))).thenReturn(response);

        GrpcCacheStatusCheckRunner runner = new GrpcCacheStatusCheckRunner(
                "test-model", ipPort, "test-site", RoleType.PREFILL,
                oldStatus, oldStatus.tryBeginCachePoll(1L), directory,
                engineHealthReporter, engineGrpcService,
                localKvCacheAwareManager, cacheIntervalService,
                20, true, Runnable::run);
        runner.run();

        response.complete(EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(7)
                .setAvailableKvCache(1000)
                .setTotalKvCache(2000)
                .setBlockSize(128)
                .build());

        verify(localKvCacheAwareManager, never())
                .updateEngineBlockCache(eq(oldStatus), any(EndpointTrace.Poll.class));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void unchangedVersionKeepsOwnedKeysAndRefreshesCapacityWithoutFullConversion(boolean omitKeys) {
        WorkerStatus status = workerStatus();
        WorkerDirectory directory = directory(status);
        var full = EngineRpcService.CacheStatusPB.newBuilder().setVersion(7)
                .putCacheKeys(11L, true).putCacheKeys(22L, true)
                .setAvailableKvCache(100).setTotalKvCache(200).setBlockSize(16).build();
        var unchangedBuilder = full.toBuilder().setAvailableKvCache(80);
        if (omitKeys) unchangedBuilder.clearCacheKeys();
        var unchanged = unchangedBuilder.build();
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), eq(status),
                anyLong(), anyLong(), eq(RoleType.PREFILL)))
                .thenReturn(CompletableFuture.completedFuture(full),
                        CompletableFuture.completedFuture(unchanged));
        when(localKvCacheAwareManager.updateEngineBlockCache(eq(status), any(EndpointTrace.Poll.class)))
                .thenReturn(WorkerCacheUpdateResult.builder().success(true).cacheVersion(7).build());
        runCachePoll(status.getIpPort(), status, directory, false);
        var ownedKeys = status.getCacheStatus().getCachedKeys();
        try (var converter = Mockito.mockStatic(org.flexlb.service.grpc.EngineStatusConverter.class,
                Mockito.CALLS_REAL_METHODS)) {
            runCachePoll(status.getIpPort(), status, directory, false);
            converter.verify(() -> org.flexlb.service.grpc.EngineStatusConverter
                    .convertToCacheStatus(unchanged), never());
        }
        org.junit.jupiter.api.Assertions.assertSame(ownedKeys, status.getCacheStatus().getCachedKeys());
        assertEquals(java.util.Set.of(11L, 22L), status.getCacheStatus().getCachedKeys());
        assertEquals(80, status.getCacheStatus().getAvailableKvCache());
        assertEquals(7, status.cacheIndexSnapshot().indexedVersion());
        verify(localKvCacheAwareManager, times(1)).updateEngineBlockCache(eq(status), any(EndpointTrace.Poll.class));
    }

    @Test
    void changedVersionWithEmptyKeysIsARealEmptySnapshot() {
        WorkerStatus status = workerStatus();
        WorkerDirectory directory = directory(status);
        var first = EngineRpcService.CacheStatusPB.newBuilder().setVersion(7).putCacheKeys(11L, true).build();
        var emptied = EngineRpcService.CacheStatusPB.newBuilder().setVersion(8).build();
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), eq(status),
                anyLong(), anyLong(), eq(RoleType.PREFILL)))
                .thenReturn(CompletableFuture.completedFuture(first), CompletableFuture.completedFuture(emptied));
        when(localKvCacheAwareManager.updateEngineBlockCache(eq(status), any(EndpointTrace.Poll.class)))
                .thenReturn(WorkerCacheUpdateResult.builder().success(true).cacheVersion(7).build(),
                        WorkerCacheUpdateResult.builder().success(true).cacheVersion(8).build());
        runCachePoll(status.getIpPort(), status, directory, false);
        runCachePoll(status.getIpPort(), status, directory, false);
        org.junit.jupiter.api.Assertions.assertTrue(status.getCacheStatus().getCachedKeys().isEmpty());
        assertEquals(8, status.cacheIndexSnapshot().indexedVersion());
        verify(localKvCacheAwareManager, times(2)).updateEngineBlockCache(eq(status), any(EndpointTrace.Poll.class));
    }

    private void runCachePoll(
            String ipPort,
            WorkerStatus workerStatus,
            WorkerDirectory directory,
            boolean debug) {
        new GrpcCacheStatusCheckRunner(
                "test-model", ipPort, "test-site", RoleType.PREFILL,
                workerStatus, workerStatus.tryBeginCachePoll(1L), directory,
                engineHealthReporter, engineGrpcService,
                localKvCacheAwareManager, cacheIntervalService,
                20, debug, Runnable::run).run();
    }

    private static WorkerStatus workerStatus() {
        return RunnerTestSupport.discovered(
                RoleType.PREFILL, null, "127.0.0.1",
                8080, 8081, "test-site");
    }

    private static WorkerDirectory directory(WorkerStatus status) {
        WorkerDirectory directory = new WorkerDirectory(
                Mockito.mock(EndpointRegistry.class));
        directory.currentOrDiscover(
                status.getRole(), status.getLogicalIpPort(), () -> status);
        return directory;
    }

}

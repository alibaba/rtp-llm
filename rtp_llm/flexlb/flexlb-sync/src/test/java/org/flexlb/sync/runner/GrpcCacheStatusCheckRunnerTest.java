package org.flexlb.sync.runner;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.when;

class GrpcCacheStatusCheckRunnerTest {

    private final EngineGrpcService engineGrpcService = Mockito.mock(EngineGrpcService.class);

    private final EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);

    private final CacheAwareService localKvCacheAwareManager = Mockito.mock(CacheAwareService.class);

    @Test
    void testGrpcCacheStatusCheckRunner() {
        // Arrange
        String modelName = "test-model";
        String ipPort = "127.0.0.1:8080";
        String site = "test-site";

        WorkerStatus workerStatus = workerStatus();

        EngineRpcService.CacheStatusPB cacheStatusPB = EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(1)
                .setAvailableKvCache(1000)
                .setTotalKvCache(2000)
                .setBlockSize(128)
                .build();
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(), eq(RoleType.PREFILL))).thenReturn(CompletableFuture.completedFuture(cacheStatusPB));

        // Act
        GrpcCacheStatusCheckRunner runner = new GrpcCacheStatusCheckRunner(
                modelName, ipPort, site, RoleType.PREFILL, workerStatus, engineHealthReporter, engineGrpcService, localKvCacheAwareManager,
                20, new LongAdder(), 50L, true, Runnable::run);
        runner.run();

        // Give some time for async execution
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Assert
        verify(engineGrpcService).getCacheStatusAsync(eq("127.0.0.1"), eq(8081), any(WorkerStatus.class), eq(-1L), eq(20L), eq(RoleType.PREFILL));
    }

    @Test
    void delayedOldCallbackIsRejectedAfterReplacementGenerationActivates() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus oldStatus = workerStatus();
        WorkerStatus replacementStatus = workerStatus();
        CompletableFuture<EngineRpcService.CacheStatusPB> oldResponse =
                new CompletableFuture<>();

        when(engineGrpcService.getCacheStatusAsync(
                eq("127.0.0.1"),
                eq(8081),
                eq(oldStatus),
                eq(-1L),
                eq(20L),
                eq(RoleType.PREFILL))).thenReturn(oldResponse);
        when(engineGrpcService.getCacheStatusAsync(
                eq("127.0.0.1"),
                eq(8081),
                eq(replacementStatus),
                eq(-1L),
                eq(20L),
                eq(RoleType.PREFILL))).thenReturn(
                        CompletableFuture.completedFuture(cacheStatus(1L, 21L)));

        AtomicLong activeGeneration =
                new AtomicLong(oldStatus.getGenerationId());
        when(localKvCacheAwareManager.updateEngineBlockCache(
                eq(ipPort),
                eq(RoleType.PREFILL),
                anyLong(),
                any())).thenAnswer(invocation -> {
                    long callbackGeneration = invocation.getArgument(2);
                    WorkerCacheUpdateResult.Outcome outcome =
                            callbackGeneration == activeGeneration.get()
                                    ? WorkerCacheUpdateResult.Outcome.APPLIED
                                    : WorkerCacheUpdateResult.Outcome.STALE_GENERATION;
                    return WorkerCacheUpdateResult.builder()
                            .outcome(outcome)
                            .engineIpPort(ipPort)
                            .build();
                });

        GrpcCacheStatusCheckRunner oldRunner = runner(ipPort, oldStatus);
        oldRunner.run();

        // Service discovery publishes and activates the replacement before its
        // cache callback can run.
        activeGeneration.set(replacementStatus.getGenerationId());
        GrpcCacheStatusCheckRunner replacementRunner =
                runner(ipPort, replacementStatus);
        replacementRunner.run();
        oldResponse.complete(cacheStatus(2L, 11L));

        assertNotNull(replacementStatus.getCacheStatus());
        assertTrue(replacementStatus.getCacheStatus()
                .getCachedKeys().contains(21L));
        assertNull(oldStatus.getCacheStatus());
        verify(engineHealthReporter, times(1))
                .reportCacheStatusCheckerSuccess(anyString(), any());
        verify(localKvCacheAwareManager).updateEngineBlockCache(
                eq(ipPort),
                eq(RoleType.PREFILL),
                eq(oldStatus.getGenerationId()),
                any());
    }

    private GrpcCacheStatusCheckRunner runner(
            String ipPort, WorkerStatus workerStatus) {
        return new GrpcCacheStatusCheckRunner(
                "test-model",
                ipPort,
                "test-site",
                RoleType.PREFILL,
                workerStatus,
                engineHealthReporter,
                engineGrpcService,
                localKvCacheAwareManager,
                20,
                new LongAdder(),
                50L,
                true,
                Runnable::run);
    }

    private static WorkerStatus workerStatus() {
        return RunnerTestSupport.discovered(
                RoleType.PREFILL, null, "127.0.0.1",
                8080, 8081, "test-site");
    }

    private static EngineRpcService.CacheStatusPB cacheStatus(
            long version, long key) {
        return EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(version)
                .setAvailableKvCache(1000)
                .setTotalKvCache(2000)
                .setBlockSize(128)
                .putCacheKeys(key, true)
                .build();
    }
}

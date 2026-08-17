package org.flexlb.sync.runner;

import io.grpc.Status;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.LongAdder;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
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

        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);

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
                20, new LongAdder(), 50L, Runnable::run);
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
    void shouldReportGrpcDeadlineByStatusCode() {
        WorkerStatus workerStatus = workerStatus();
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(), eq(RoleType.PREFILL)))
                .thenReturn(CompletableFuture.failedFuture(Status.DEADLINE_EXCEEDED.asRuntimeException()));

        createRunner(workerStatus).run();

        verify(engineHealthReporter).reportCacheStatusCheckerFail(
                "test-model", BalanceStatusEnum.CACHE_GRPC_TIMEOUT, RoleType.PREFILL);
    }

    @Test
    void shouldNotTreatDeadlineTextAsGrpcDeadline() {
        WorkerStatus workerStatus = workerStatus();
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(), eq(RoleType.PREFILL)))
                .thenReturn(CompletableFuture.failedFuture(
                        Status.INTERNAL.withDescription("contains DEADLINE_EXCEEDED text").asRuntimeException()));

        createRunner(workerStatus).run();

        verify(engineHealthReporter).reportCacheStatusCheckerFail(
                "test-model", BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE, RoleType.PREFILL);
        verify(engineHealthReporter, never()).reportCacheStatusCheckerFail(
                "test-model", BalanceStatusEnum.CACHE_GRPC_TIMEOUT, RoleType.PREFILL);
    }

    @Test
    void shouldSkipCacheStatusRpcForVitWorkers() {
        WorkerStatus workerStatus = workerStatus();
        workerStatus.getCacheCheckInProgress().set(true);
        GrpcCacheStatusCheckRunner runner = new GrpcCacheStatusCheckRunner(
                "test-model", "127.0.0.1:8080", "test-site", RoleType.VIT,
                workerStatus, engineHealthReporter, engineGrpcService, localKvCacheAwareManager,
                20, new LongAdder(), 50L, Runnable::run);

        runner.run();

        verify(engineGrpcService, never()).getCacheStatusAsync(
                anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(), eq(RoleType.VIT));
        org.junit.jupiter.api.Assertions.assertFalse(workerStatus.getCacheCheckInProgress().get());
    }

    private WorkerStatus workerStatus() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);
        return workerStatus;
    }

    private GrpcCacheStatusCheckRunner createRunner(WorkerStatus workerStatus) {
        return new GrpcCacheStatusCheckRunner(
                "test-model", "127.0.0.1:8080", "test-site", RoleType.PREFILL,
                workerStatus, engineHealthReporter, engineGrpcService, localKvCacheAwareManager,
                20, new LongAdder(), 50L, Runnable::run);
    }
}

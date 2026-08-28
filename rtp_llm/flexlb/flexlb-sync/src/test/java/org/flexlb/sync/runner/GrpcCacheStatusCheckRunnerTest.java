package org.flexlb.sync.runner;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.LongAdder;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.when;

class GrpcCacheStatusCheckRunnerTest {

    private final EngineGrpcService engineGrpcService = Mockito.mock(EngineGrpcService.class);

    private final EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);

    private final CacheAwareService localKvCacheAwareManager = Mockito.mock(CacheAwareService.class);

    private final DynamicCacheIntervalService cacheIntervalService =
            Mockito.mock(DynamicCacheIntervalService.class);

    @Test
    void testGrpcCacheStatusCheckRunner() {
        // Arrange
        String modelName = "test-model";
        String ipPort = "127.0.0.1:8080";
        String site = "test-site";

        WorkerStatus workerStatus = workerStatus();
        WorkerDirectory directory = directory(workerStatus);

        EngineRpcService.CacheStatusPB cacheStatusPB = EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(1)
                .setAvailableKvCache(1000)
                .setTotalKvCache(2000)
                .setBlockSize(128)
                .build();
        when(engineGrpcService.getCacheStatusAsync(anyString(), anyInt(), any(WorkerStatus.class), anyLong(), anyLong(), eq(RoleType.PREFILL))).thenReturn(CompletableFuture.completedFuture(cacheStatusPB));

        // Act
        GrpcCacheStatusCheckRunner runner = new GrpcCacheStatusCheckRunner(
                modelName, ipPort, site, RoleType.PREFILL, workerStatus,
                workerStatus.tryBeginCachePoll(),
                directory,
                engineHealthReporter, engineGrpcService,
                localKvCacheAwareManager, cacheIntervalService,
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
    void staleGenerationCallbackCannotPublishAddressCache() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus oldStatus = workerStatus();
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
                oldStatus, oldStatus.tryBeginCachePoll(), directory,
                engineHealthReporter, engineGrpcService,
                localKvCacheAwareManager, cacheIntervalService,
                20, new LongAdder(), 50L, true, Runnable::run);
        runner.run();

        response.complete(EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(7)
                .setAvailableKvCache(1000)
                .setTotalKvCache(2000)
                .setBlockSize(128)
                .build());

        verify(localKvCacheAwareManager, never())
                .updateEngineBlockCache(oldStatus);
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
                status.getRole(), status.getIpPort(), () -> status);
        return directory;
    }

}

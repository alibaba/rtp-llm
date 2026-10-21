package org.flexlb.sync.synchronizer;

import io.grpc.Status;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.match.localsync.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.RunnerTestSupport;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class MasterEngineSynchronizerTest {
    @Test
    @Timeout(20)
    void fullApplicationPoolAndRpcTimeoutsPreserveWorkerHealth() throws Exception {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.modelName()).thenReturn("model");
        when(model.requiredRoles()).thenReturn(List.of(RoleType.DECODE));
        WorkerAddressService addresses = mock(WorkerAddressService.class);
        AtomicBoolean discovered = new AtomicBoolean();
        when(addresses.getEngineWorkerList("model", RoleType.DECODE)).thenAnswer(call ->
                discovered.get() ? List.of(WorkerHost.of("127.0.0.1", 8080)) : List.of());
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        AtomicBoolean transportFailed = new AtomicBoolean();
        CountDownLatch successfulPolls = new CountDownLatch(3);
        CountDownLatch failedPolls = new CountDownLatch(3);
        var reporter = mock(EngineHealthReporter.class);
        doAnswer(call -> {
            failedPolls.countDown();
            return null;
        }).when(reporter).reportStatusCheckerFail(
                eq("model"), eq(BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT), eq(RoleType.DECODE),
                any(), eq("DEADLINE_EXCEEDED"), any());
        var response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("DECODE").setStatusVersion(1L).setAlive(true).build();
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenAnswer(call -> {
                    if (transportFailed.get()) {
                        return CompletableFuture.failedFuture(Status.DEADLINE_EXCEEDED.asRuntimeException());
                    }
                    successfulPolls.countDown();
                    return CompletableFuture.completedFuture(response);
                });
        when(grpc.getCacheStatusAsync(anyString(), anyInt(), any(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.CacheStatusPB.newBuilder()
                        .setVersion(1L).build()));
        CountDownLatch releaseApplications = new CountDownLatch(1);
        var registry = RunnerTestSupport.endpointRegistry(config);
        try {
            var directory = new WorkerDirectory(registry);
            var synchronizer = new MasterEngineSynchronizer(addresses, reporter,
                    directory, grpc, model, mock(CacheAwareService.class),
                    mock(DynamicCacheIntervalService.class), config);
            var field = MasterEngineSynchronizer.class.getDeclaredField("statusApplicationExecutor");
            field.setAccessible(true);
            var applications = (ThreadPoolExecutor) field.get(synchronizer);
            try {
                CountDownLatch occupied = new CountDownLatch(applications.getCorePoolSize());
                for (int i = 0; i < applications.getCorePoolSize(); i++) {
                    applications.execute(() -> {
                        occupied.countDown();
                        try {
                            releaseApplications.await();
                        } catch (InterruptedException interrupted) {
                            Thread.currentThread().interrupt();
                        }
                    });
                }
                assertTrue(occupied.await(5, TimeUnit.SECONDS));
                discovered.set(true);
                assertTrue(successfulPolls.await(5, TimeUnit.SECONDS), "RPCs continue with every application thread occupied");
                WorkerStatus status = directory.statusSnapshot(RoleType.DECODE).get("127.0.0.1:8080");
                assertNotNull(status);
                assertEquals(1, applications.getQueue().size(), "only one response per generation may wait");
                assertEquals(-1L, status.appliedStatusCursor().statusVersion());
                transportFailed.set(true);
                assertTrue(failedPolls.await(5, TimeUnit.SECONDS), "three failed polls finish despite the application backlog");
                assertNull(status.retirementReason(), "timeouts cannot prove worker failure");
                assertTrue(status.isActiveGeneration());
                assertTrue(status.isAvailableForScheduling());
                assertEquals(0L, status.pollHealth().consecutiveWorkerFailures());
                assertEquals(1, applications.getQueue().size());
                releaseApplications.countDown();
                applications.shutdown();
                assertTrue(applications.awaitTermination(5, TimeUnit.SECONDS));
                assertEquals(1L, status.appliedStatusCursor().statusVersion(), "known resource state may still finish applying");
                assertNotNull(registry.get(RoleType.DECODE, status.getIpPort()));
                assertTrue(status.isAvailableForScheduling(), "neither local backlog nor a timeout proves worker failure");
            } finally {
                releaseApplications.countDown();
                synchronizer.destroy();
            }
        } finally {
            registry.close();
        }
    }
}

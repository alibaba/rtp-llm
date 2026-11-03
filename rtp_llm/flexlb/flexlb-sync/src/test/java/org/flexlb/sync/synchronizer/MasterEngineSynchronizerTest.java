package org.flexlb.sync.synchronizer;

import io.grpc.Status;
import org.flexlb.balance.endpoint.WorkerEndpoint;
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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE", "PDFUSION", "VIT"})
    void readinessRequiresFreshHealthyPollAndMatchingPublishedEndpoint(RoleType role) throws Exception {
        WorkerDirectory directory = mock(WorkerDirectory.class);
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.topologySnapshot()).thenReturn(
                new WorkerStatus.TopologySnapshot(null, "127.0.0.1", 8080, 8081, null));
        String address = "127.0.0.1:8080";
        when(directory.statusSnapshot(role)).thenReturn(Map.of(address, status));
        MasterEngineSynchronizer synchronizer = synchronizer(directory, List.of(role));
        try {
            when(status.pollHealth()).thenReturn(new WorkerStatus.PollHealth(0L, 0L, 0L, false, false));
            assertFalse(synchronizer.isReady(), "discovery alone cannot establish readiness");

            when(status.isAvailableForScheduling()).thenReturn(true);
            when(status.pollHealth()).thenReturn(new WorkerStatus.PollHealth(
                    System.nanoTime() / 1000, 0L, 0L, true, true));
            assertFalse(synchronizer.isReady(), "the successful response must also be applied and published");

            WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
            when(directory.exactEndpoint(role, address, status)).thenReturn(endpoint);
            assertTrue(synchronizer.isReady());

            when(status.pollHealth()).thenReturn(new WorkerStatus.PollHealth(
                    Math.max(1L, System.nanoTime() / 1000 - TimeUnit.MINUTES.toMicros(1)),
                    0L, 0L, true, true));
            assertFalse(synchronizer.isReady(), "a published endpoint cannot keep stale status ready forever");

            when(status.pollHealth()).thenReturn(new WorkerStatus.PollHealth(
                    System.nanoTime() / 1000, 0L, 1L, false, true));
            assertFalse(synchronizer.isReady(), "an explicit unhealthy report withdraws readiness immediately");

            when(status.pollHealth()).thenReturn(new WorkerStatus.PollHealth(
                    System.nanoTime() / 1000, 0L, 0L, true, true));
            when(status.isAvailableForScheduling()).thenReturn(false);
            assertFalse(synchronizer.isReady(), "retiring generations cannot establish readiness");
            when(status.isAvailableForScheduling()).thenReturn(true);
            assertTrue(synchronizer.isReady(), "fresh synchronized capacity restores readiness");
        } finally {
            synchronizer.destroy();
        }
    }

    @Test
    void readinessRequiresEveryConfiguredRoleAndRejectsEmptyTopology() throws Exception {
        WorkerDirectory directory = mock(WorkerDirectory.class);
        MasterEngineSynchronizer empty = synchronizer(directory, List.of());
        try {
            assertFalse(empty.isReady());
        } finally {
            empty.destroy();
        }
        WorkerStatus prefill = mock(WorkerStatus.class);
        when(prefill.topologySnapshot()).thenReturn(
                new WorkerStatus.TopologySnapshot(null, "127.0.0.1", 8080, 8081, null));
        when(prefill.isAvailableForScheduling()).thenReturn(true);
        when(prefill.pollHealth()).thenReturn(new WorkerStatus.PollHealth(
                System.nanoTime() / 1000, 0L, 0L, true, true));
        when(directory.statusSnapshot(RoleType.PREFILL)).thenReturn(Map.of("p:8080", prefill));
        when(directory.exactEndpoint(RoleType.PREFILL, "p:8080", prefill))
                .thenReturn(mock(WorkerEndpoint.class));
        MasterEngineSynchronizer separated = synchronizer(directory, List.of(RoleType.PREFILL, RoleType.DECODE));
        try {
            assertFalse(separated.isReady(), "Prefill alone cannot serve a PD separated topology");
        } finally {
            separated.destroy();
        }
    }

    @Test
    void readinessRequiresACompleteRouteWithinTheSameGroup() throws Exception {
        WorkerDirectory directory = mock(WorkerDirectory.class);
        WorkerStatus decode = readyWorker(directory, RoleType.DECODE, "d:8080", "a");
        WorkerStatus prefill = readyWorker(directory, RoleType.PREFILL, "p:8080", "b");
        MasterEngineSynchronizer synchronizer = synchronizer(directory, List.of(RoleType.DECODE, RoleType.PREFILL));
        try {
            assertFalse(synchronizer.isReady(), "disjoint role groups cannot form a complete route");
            when(prefill.topologySnapshot()).thenReturn(
                    new WorkerStatus.TopologySnapshot("a", "p", 8080, 8081, null));
            assertTrue(synchronizer.isReady());
            when(prefill.topologySnapshot()).thenReturn(
                    new WorkerStatus.TopologySnapshot(null, "p", 8080, 8081, null));
            assertFalse(synchronizer.isReady(), "a selected non-null group must match the next role");
            when(decode.topologySnapshot()).thenReturn(
                    new WorkerStatus.TopologySnapshot(null, "d", 8080, 8081, null));
            assertTrue(synchronizer.isReady(), "a null selected group leaves the next role unrestricted");
        } finally {
            synchronizer.destroy();
        }
    }

    private WorkerStatus readyWorker(WorkerDirectory directory, RoleType role, String address, String group) {
        WorkerStatus status = mock(WorkerStatus.class);
        when(status.isAvailableForScheduling()).thenReturn(true);
        when(status.pollHealth()).thenReturn(new WorkerStatus.PollHealth(
                System.nanoTime() / 1000, 0L, 0L, true, true));
        when(status.topologySnapshot()).thenReturn(
                new WorkerStatus.TopologySnapshot(group, address, 8080, 8081, null));
        when(directory.statusSnapshot(role)).thenReturn(Map.of(address, status));
        when(directory.exactEndpoint(role, address, status)).thenReturn(mock(WorkerEndpoint.class));
        return status;
    }

    private MasterEngineSynchronizer synchronizer(WorkerDirectory directory, List<RoleType> roles) throws Exception {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        ModelMetaConfig model = mock(ModelMetaConfig.class);
        when(model.modelName()).thenReturn("model");
        when(model.requiredRoles()).thenReturn(roles);
        MasterEngineSynchronizer synchronizer = new MasterEngineSynchronizer(mock(WorkerAddressService.class),
                mock(EngineHealthReporter.class), directory, mock(EngineGrpcService.class),
                model, mock(CacheAwareService.class), mock(DynamicCacheIntervalService.class), config);
        synchronizer.destroy();
        for (String name : List.of("scheduler", "engineSyncExecutor", "statusCheckExecutor",
                "statusApplicationExecutor", "cacheStatusExecutor")) {
            var field = MasterEngineSynchronizer.class.getDeclaredField(name);
            field.setAccessible(true);
            assertTrue(((ThreadPoolExecutor) field.get(synchronizer)).awaitTermination(5, TimeUnit.SECONDS));
        }
        return synchronizer;
    }

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
                eq("127.0.0.1:8080"), eq(BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT), eq(RoleType.DECODE),
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
                WorkerStatus status = directory.statusSnapshot(RoleType.DECODE).get("127.0.0.1:8080@0");
                assertNotNull(status);
                assertEquals(1, applications.getQueue().size(), "only one response per generation may wait");
                assertEquals(-1L, status.appliedStatusCursor().statusVersion());
                assertFalse(synchronizer.isReady(), "pending status application has no routing capacity yet");
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
                assertNotNull(registry.get(RoleType.DECODE, status.getLogicalIpPort()));
                assertTrue(synchronizer.isReady(), "published fresh capacity is ready despite application backlog");
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

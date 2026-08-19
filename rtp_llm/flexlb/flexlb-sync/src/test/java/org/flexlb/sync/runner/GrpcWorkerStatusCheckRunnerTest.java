package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerGenerationFence;
import org.flexlb.sync.status.WorkerGenerationManager;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Generation, cursor, and failure semantics for {@link GrpcWorkerStatusRunner}. */
class GrpcWorkerStatusCheckRunnerTest {

    private static final String MODEL = "test-model";
    private static final String IP_PORT = "127.0.0.1:8080";

    private final EngineGrpcService engineGrpcService =
            Mockito.mock(EngineGrpcService.class);
    private final EngineHealthReporter healthReporter =
            Mockito.mock(EngineHealthReporter.class);

    @Test
    void callsGrpcServiceAndAppliesCurrentGeneration() {
        WorkerStatus status = status(RoleType.PREFILL, 8080);
        ConcurrentMap<String, WorkerStatus> statuses = mapWith(status);
        EndpointRegistry registry = Mockito.mock(EndpointRegistry.class);
        CacheAwareService cache = Mockito.mock(CacheAwareService.class);
        whenStatusRpc(workerResponse(RoleType.PREFILL, 100L, true, -1L));

        runner(RoleType.PREFILL, status, statuses, null, registry, cache).run();

        verify(engineGrpcService).getWorkerStatusAsync(
                "127.0.0.1", 8081, -1L, 20L, RoleType.PREFILL);
        verify(registry).updateEndpointFromWorkerStatus(
                Mockito.eq(status), any());
        assertEquals(100L, status.getStatusVersion().get());
        assertTrue(status.isAlive());
    }

    @Test
    void equalStatusVersionDoesNotReplaceRunningSnapshotButRefreshesHeartbeat() {
        WorkerStatus status = status(RoleType.PREFILL, 8080);
        status.getStatusVersion().set(100L);
        status.getLastAppliedStatusVersion().set(100L);
        status.getLatestFinishedTaskVersion().set(0L);
        ConcurrentMap<String, WorkerStatus> statuses = mapWith(status);
        EndpointRegistry registry = Mockito.mock(EndpointRegistry.class);
        CacheAwareService cache = Mockito.mock(CacheAwareService.class);
        FlexlbBatchScheduler scheduler = Mockito.mock(FlexlbBatchScheduler.class);
        EngineRpcService.TaskInfoPB task = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(123L)
                .setBatchId(7L)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED)
                .setIsWaiting(true)
                .build();
        EngineRpcService.WorkerStatusPB response = baseResponse(
                RoleType.PREFILL, 100L, true)
                .addRunningTaskInfo(task)
                .build();
        whenStatusRpc(response);
        long before = status.getStatusLastUpdateTime().get();

        runner(RoleType.PREFILL, status, statuses, scheduler, registry, cache).run();

        assertNull(status.getRunningTaskList(),
                "equal version must not churn the status snapshot");
        assertTrue(status.getStatusLastUpdateTime().get() >= before,
                "a successful equal-version probe is still a liveness heartbeat");
        verify(registry).refreshEndpointActivity(Mockito.eq(status), any());
        verify(registry, never()).updateEndpointFromWorkerStatus(
                Mockito.eq(status), any());
        verify(scheduler).recordRequestActivity(Mockito.eq(status), any());
        verify(scheduler, never()).updateRequestLifecycleFromWorkerStatus(
                Mockito.eq(status), any());
    }

    @Test
    void ignoresStatusCallbackFromStaleGeneration() {
        WorkerStatus stale = status(RoleType.VIT, 8080);
        WorkerStatus current = status(RoleType.VIT, 8080);
        ConcurrentMap<String, WorkerStatus> statuses = mapWith(current);
        EndpointRegistry registry = registry();
        CacheAwareService cache = Mockito.mock(CacheAwareService.class);
        WorkerEndpoint currentEndpoint = registry.ensureEndpoint(
                RoleType.VIT, IP_PORT, current);
        whenStatusRpc(workerResponse(RoleType.VIT, 100L, true, -1L));

        runner(RoleType.VIT, stale, statuses, null, registry, cache).run();

        assertSame(currentEndpoint, registry.get(RoleType.VIT, IP_PORT));
        assertSame(current, currentEndpoint.getStatus());
        assertEquals(-1L, stale.getStatusVersion().get());
        verify(cache, never()).clearEngineCache(anyString());
        registry.close();
    }

    @Test
    void removesEndpointAfterThreeConsecutiveGrpcFailures() {
        WorkerStatus status = status(RoleType.VIT, 8080);
        ConcurrentMap<String, WorkerStatus> statuses = mapWith(status);
        EndpointRegistry registry = registry();
        CacheAwareService cache = Mockito.mock(CacheAwareService.class);
        registry.ensureEndpoint(RoleType.VIT, IP_PORT, status);
        when(engineGrpcService.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any(RoleType.class)))
                .thenReturn(CompletableFuture.failedFuture(
                        new RuntimeException("unavailable")));
        GrpcWorkerStatusRunner runner = runner(
                RoleType.VIT, status, statuses, null, registry, cache);
        long lastSuccessfulHeartbeat = status.getStatusLastUpdateTime().get();

        runner.run();
        runner.run();
        runner.run();

        assertFalse(status.isAlive());
        assertEquals(3L, status.getConsecutiveFailures().get());
        assertNull(registry.get(RoleType.VIT, IP_PORT));
        assertEquals(lastSuccessfulHeartbeat, status.getStatusLastUpdateTime().get(),
                "failed RPCs must not extend the last-success TTL");
        assertSame(status, statuses.get(IP_PORT),
                "probe failures make a generation unroutable but do not retire it");
        registry.close();
    }

    @Test
    void sameVersionRecoveryRecreatesEndpointWithoutDiscoveryRevival() {
        WorkerStatus status = status(RoleType.VIT, 8080);
        status.getStatusVersion().set(100L);
        ConcurrentMap<String, WorkerStatus> statuses = mapWith(status);
        EndpointRegistry registry = registry();
        CacheAwareService cache = Mockito.mock(CacheAwareService.class);
        registry.ensureEndpoint(RoleType.VIT, IP_PORT, status);
        when(engineGrpcService.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any(RoleType.class)))
                .thenReturn(CompletableFuture.failedFuture(
                        new RuntimeException("unavailable")));
        GrpcWorkerStatusRunner runner = runner(
                RoleType.VIT, status, statuses, null, registry, cache);
        runner.run();
        runner.run();
        runner.run();
        whenStatusRpc(workerResponse(RoleType.VIT, 100L, true, -1L));

        runner.run();

        assertTrue(status.isAlive());
        assertEquals(0L, status.getConsecutiveFailures().get());
        assertSame(status, registry.get(RoleType.VIT, IP_PORT).getStatus());
        registry.close();
    }

    @Test
    void lowerStatusVersionRotatesGenerationAndRetiresOldSideEffects() {
        WorkerStatus old = status(RoleType.VIT, 8080);
        old.getStatusVersion().set(100L);
        ConcurrentMap<String, WorkerStatus> statuses = mapWith(old);
        EndpointRegistry registry = registry();
        CacheAwareService cache = Mockito.mock(CacheAwareService.class);
        registry.ensureEndpoint(RoleType.VIT, IP_PORT, old);
        whenStatusRpc(workerResponse(RoleType.VIT, 2L, true, -1L));

        runner(RoleType.VIT, old, statuses, null, registry, cache).run();

        WorkerStatus replacement = statuses.get(IP_PORT);
        assertNotSame(old, replacement);
        assertEquals(RoleType.VIT, replacement.getRole());
        assertEquals(-1L, replacement.getStatusVersion().get(),
                "the rollback response is discarded and the new epoch starts clean");
        assertFalse(old.isAlive());
        assertNull(registry.get(RoleType.VIT, IP_PORT));
        verify(cache, never()).clearEngineCache(anyString());
        registry.close();
    }

    @Test
    void rollbackToVersionZeroRotatesThenFreshGenerationAcceptsVersionZero() {
        WorkerStatus old = status(RoleType.VIT, 8080);
        old.getStatusVersion().set(100L);
        ConcurrentMap<String, WorkerStatus> statuses = mapWith(old);
        EndpointRegistry registry = registry();
        CacheAwareService cache = Mockito.mock(CacheAwareService.class);
        registry.ensureEndpoint(RoleType.VIT, IP_PORT, old);
        whenStatusRpc(workerResponse(RoleType.VIT, 0L, true, -1L));

        runner(RoleType.VIT, old, statuses, null, registry, cache).run();
        WorkerStatus fresh = statuses.get(IP_PORT);

        assertNotSame(old, fresh);
        assertEquals(-1L, fresh.getStatusVersion().get());
        runner(RoleType.VIT, fresh, statuses, null, registry, cache).run();

        assertEquals(0L, fresh.getStatusVersion().get());
        assertEquals(0L, fresh.getLastAppliedStatusVersion().get());
        assertTrue(fresh.isAlive());
        assertSame(fresh, registry.get(RoleType.VIT, IP_PORT).getStatus());
        registry.close();
    }

    @Test
    void pipelineFailureDoesNotAdvanceMarkersAndEqualVersionRetries() {
        WorkerStatus status = status(RoleType.PREFILL, 8080);
        status.getStatusVersion().set(100L);
        status.getLatestFinishedTaskVersion().set(7L);
        ConcurrentMap<String, WorkerStatus> statuses = mapWith(status);
        EndpointRegistry registry = Mockito.mock(EndpointRegistry.class);
        CacheAwareService cache = Mockito.mock(CacheAwareService.class);
        FlexlbBatchScheduler scheduler = Mockito.mock(FlexlbBatchScheduler.class);
        doThrow(new IllegalStateException("first endpoint apply fails"))
                .doNothing()
                .when(registry).updateEndpointFromWorkerStatus(Mockito.eq(status), any());
        whenStatusRpc(workerResponse(RoleType.PREFILL, 100L, true, 8L));
        GrpcWorkerStatusRunner runner = runner(
                RoleType.PREFILL, status, statuses, scheduler, registry, cache);

        runner.run();

        assertEquals(-1L, status.getLastAppliedStatusVersion().get());
        assertEquals(7L, status.getLatestFinishedTaskVersion().get());

        runner.run();

        assertEquals(100L, status.getLastAppliedStatusVersion().get());
        assertEquals(8L, status.getLatestFinishedTaskVersion().get());
        verify(registry, times(2)).updateEndpointFromWorkerStatus(
                Mockito.eq(status), any());
        verify(scheduler, times(2)).updateRequestLifecycleFromWorkerStatus(
                Mockito.eq(status), any());
    }

    @Test
    void equalStatusVersionStillAdvancesIndependentFinishedCursor() {
        WorkerStatus status = status(RoleType.PREFILL, 8080);
        status.getStatusVersion().set(100L);
        status.getLastAppliedStatusVersion().set(100L);
        status.getLatestFinishedTaskVersion().set(7L);
        ConcurrentMap<String, WorkerStatus> statuses = mapWith(status);
        EndpointRegistry registry = Mockito.mock(EndpointRegistry.class);
        CacheAwareService cache = Mockito.mock(CacheAwareService.class);
        FlexlbBatchScheduler scheduler = Mockito.mock(FlexlbBatchScheduler.class);
        whenStatusRpc(workerResponse(RoleType.PREFILL, 100L, true, 8L));

        runner(RoleType.PREFILL, status, statuses, scheduler, registry, cache).run();

        verify(scheduler).recordRequestActivity(Mockito.eq(status), any());
        verify(scheduler).updateRequestLifecycleFromWorkerStatus(
                Mockito.eq(status), any());
        verify(registry).updateEndpointFromWorkerStatus(
                Mockito.eq(status), any());
        assertEquals(8L, status.getLatestFinishedTaskVersion().get());
        assertEquals(100L, status.getLastAppliedStatusVersion().get());
    }

    private GrpcWorkerStatusRunner runner(
            RoleType role, WorkerStatus status,
            ConcurrentMap<String, WorkerStatus> statuses,
            FlexlbBatchScheduler scheduler, EndpointRegistry registry,
            CacheAwareService cache) {
        WorkerGenerationFence fence = new WorkerGenerationFence();
        WorkerGenerationManager manager =
                new WorkerGenerationManager(registry, cache, fence);
        return new GrpcWorkerStatusRunner(
                MODEL, IP_PORT, "test-site", role, "test-group",
                status, statuses, healthReporter, engineGrpcService, 20L,
                scheduler, registry, manager, fence, Runnable::run);
    }

    private void whenStatusRpc(EngineRpcService.WorkerStatusPB response) {
        when(engineGrpcService.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any(RoleType.class)))
                .thenReturn(CompletableFuture.completedFuture(response));
    }

    private static EngineRpcService.WorkerStatusPB workerResponse(
            RoleType role, long statusVersion, boolean alive,
            long latestFinishedVersion) {
        return baseResponse(role, statusVersion, alive)
                .setLatestFinishedVersion(latestFinishedVersion)
                .build();
    }

    private static EngineRpcService.WorkerStatusPB.Builder baseResponse(
            RoleType role, long statusVersion, boolean alive) {
        EngineRpcService.RoleTypePB protoRole = switch (role) {
            case PREFILL -> EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL;
            case DECODE -> EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE;
            case PDFUSION -> EngineRpcService.RoleTypePB.ROLE_TYPE_PDFUSION;
            case VIT -> EngineRpcService.RoleTypePB.ROLE_TYPE_VIT;
            case FRONTEND -> EngineRpcService.RoleTypePB.ROLE_TYPE_FRONTEND;
        };
        return EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(role.getCode())
                .setRoleType(protoRole)
                .setStatusVersion(statusVersion)
                .setAlive(alive)
                .setDpSize(1)
                .setTpSize(1);
    }

    private static ConcurrentMap<String, WorkerStatus> mapWith(WorkerStatus status) {
        ConcurrentMap<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(IP_PORT, status);
        return statuses;
    }

    private static EndpointRegistry registry() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return new EndpointRegistry(
                configService, () -> null,
                Mockito.mock(BatchSchedulerReporter.class));
    }

    private static WorkerStatus status(RoleType role, int port) {
        WorkerStatus status = new WorkerStatus();
        status.setRole(role);
        status.setIp("127.0.0.1");
        status.setPort(port);
        status.setGrpcPort(port + 1);
        status.setAlive(true);
        return status;
    }
}

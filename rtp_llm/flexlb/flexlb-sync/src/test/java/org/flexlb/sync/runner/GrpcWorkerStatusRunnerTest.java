package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class GrpcWorkerStatusRunnerTest {

    @ParameterizedTest
    @CsvSource({"DECODE,false,false", "DECODE,false,true", "PREFILL,true,false",
            "PREFILL,true,true", "PDFUSION,true,false", "PDFUSION,true,true"})
    void repeatedRpcFailuresRetireWorkerAndOnlyClearDetailedCacheIndexes(
            RoleType role, boolean needsCacheKeys, boolean telemetryFails) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                role, null, "127.0.0.1", 8080, 8081, "test-site");
        EndpointRegistry directory = directory(registry, status);
        WorkerEndpoint endpoint = RunnerTestSupport.publishEndpoint(
                registry, role, status.getIpPort(), status);
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        var failuresAtReport = new ArrayList<Long>();
        var lockHeldAtReport = new ArrayList<Boolean>();
        org.mockito.Mockito.doAnswer(invocation -> {
            failuresAtReport.add(status.pollHealth().consecutiveTransportFailures());
            lockHeldAtReport.add(status.lock.isHeldByCurrentThread());
            if (telemetryFails) { throw new IllegalStateException("monitor unavailable"); }
            return null;
        }).when(reporter).reportStatusCheckerFail(anyString(), any(), any());
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.failedFuture(
                        io.grpc.Status.UNAVAILABLE.asRuntimeException()));

        for (int failure = 1; failure <= 3; failure++) {
            WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
            assertNotNull(lease);
            new GrpcWorkerStatusRunner(
                    "test-model",
                    status, lease, directory, reporter,
                    grpc, 5_000L, cache, Runnable::run).run();
            assertEquals(failure, status.pollHealth().consecutiveTransportFailures(),
                    "failed telemetry must not suppress the transport failure fact");
            if (failure < 3) {
                assertSame(status, directory.statusSnapshot(role).get(status.getIpPort()));
                assertSame(endpoint, registry.get(role, status.getIpPort()));
                verifyNoInteractions(cache);
            }
        }

        assertNull(registry.get(role, status.getIpPort()));
        assertTrue(directory.statusSnapshot(role).isEmpty());
        assertNull(status.tryBeginStatusPoll(), "retired generation cannot poll again");
        assertEquals(List.of(1L, 2L, 3L), failuresAtReport, "business facts precede telemetry");
        assertEquals(List.of(false, false, false), lockHeldAtReport, "telemetry must run outside the generation lock");
        verify(reporter, org.mockito.Mockito.times(3)).reportStatusCheckerFail(
                "test-model", org.flexlb.enums.BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, role);
        if (needsCacheKeys) {
            verify(cache).removeEngineBlockCache(status.getIpPort());
        } else {
            verifyNoInteractions(cache);
        }
    }

    @Test
    void newGenerationPublishesUnderLockAndLaterStatusProjectsOutsideLock() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1",
                8080, 8081, "test-site");
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll();
        assertNotNull(pollLease);

        AtomicBoolean publicationHeldLock = new AtomicBoolean();
        AtomicBoolean projectionReleasedLock = new AtomicBoolean();
        WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
        EndpointRegistry registry = org.mockito.Mockito.spy(RunnerTestSupport.endpointRegistry(mock(ConfigService.class)));
        EndpointRegistry directory = directory(registry, status);
        org.mockito.Mockito.doAnswer(invocation -> {
                    publicationHeldLock.set(
                            status.lock.isHeldByCurrentThread());
                    status.publishPreparedStatus(invocation.getArgument(2));
                    return endpoint;
                }).when(registry).publishPreparedEndpoint(
                        org.mockito.Mockito.eq(ipPort), org.mockito.Mockito.eq(status),
                        any(WorkerStatus.PreparedStatus.class));
        when(registry.get(RoleType.DECODE, ipPort, status)).thenReturn(null, endpoint);
        when(endpoint.applyPreparedStatus(any(), any())).thenAnswer(invocation -> {
            assertTrue(status.lock.isHeldByCurrentThread(),
                    "existing endpoint reduction must hold the generation lock");
            status.publishPreparedStatus(invocation.getArgument(1));
            return (Runnable) () -> projectionReleasedLock.set(!status.lock.isHeldByCurrentThread());
        });

        EngineRpcService.WorkerStatusPB response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.DECODE.getCode())
                        .setRoleType(
                                EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(1L)
                        .setAlive(true)
                        .build();
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response),
                        CompletableFuture.completedFuture(response.toBuilder().setStatusVersion(2L).build()));

        new GrpcWorkerStatusRunner(
                "test-model",
                status, pollLease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), Runnable::run)
                .run();

        assertTrue(publicationHeldLock.get(),
                "publication must commit under the generation lock");
        assertFalse(projectionReleasedLock.get(),
                "a new generation has no local request facts to project");
        WorkerStatus.PollLease nextPoll = status.tryBeginStatusPoll();
        assertNotNull(nextPoll);
        new GrpcWorkerStatusRunner(
                "test-model", status, nextPoll, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), Runnable::run).run();
        assertTrue(projectionReleasedLock.get(),
                "endpoint facts must project after releasing that lock");
    }

    @Test
    void sameVersionResponseProjectsExactEndpointActivity() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = RunnerTestSupport.alive(
                RoleType.DECODE, null, "127.0.0.1",
                8080, 8081, "test-site");
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll();
        assertNotNull(pollLease);

        WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
        EndpointRegistry registry = org.mockito.Mockito.spy(RunnerTestSupport.endpointRegistry(mock(ConfigService.class)));
        EndpointRegistry directory = directory(registry, status);
        when(registry.get(RoleType.DECODE, ipPort, status))
                .thenReturn(endpoint);
        java.util.concurrent.atomic.AtomicBoolean projected =
                new java.util.concurrent.atomic.AtomicBoolean();
        Runnable activity = () -> projected.set(true);
        when(endpoint.applyStatusHeartbeat(any(), any()))
                .thenReturn(activity);

        EngineRpcService.TaskInfoPB task = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(123L)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                .build();
        EngineRpcService.WorkerStatusPB response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.DECODE.getCode())
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(
                                status.appliedStatusCursor().statusVersion())
                        .setAlive(true)
                        .addRunningTaskInfo(task)
                        .build();
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        new GrpcWorkerStatusRunner(
                "test-model",
                status, pollLease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), Runnable::run)
                .run();

        ArgumentCaptor<WorkerStatus.StatusObservation> observation =
                ArgumentCaptor.forClass(WorkerStatus.StatusObservation.class);
        verify(endpoint).applyStatusHeartbeat(
                org.mockito.Mockito.eq(status), observation.capture());
        assertTrue(observation.getValue().runningTasks().values().stream()
                .anyMatch(active -> active.requestId() == 123L));
        assertTrue(projected.get());
        WorkerStatus.PollLease nextPoll = status.tryBeginStatusPoll();
        assertNotNull(nextPoll, "the asynchronous owner must close the exact poll lease");
        nextPoll.close();
    }

    @ParameterizedTest
    @ValueSource(longs = {1L, 2L})
    void committedStatusWithoutItsExactEndpointRetiresOnHeartbeatOrNewVersion(long responseVersion) {
        RoleType role = RoleType.DECODE;
        WorkerStatus status = RunnerTestSupport.alive(
                role, null, "127.0.0.1", 8080, 8081, "test-site");
        EndpointRegistry registry = org.mockito.Mockito.spy(
                RunnerTestSupport.endpointRegistry(mock(ConfigService.class)));
        directory(registry, status);
        assertNull(registry.get(role, status.getIpPort()), "the committed cursor has no endpoint");
        WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
        assertNotNull(lease);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(role.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .setStatusVersion(responseVersion)
                .setAlive(true)
                .build();
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));

        new GrpcWorkerStatusRunner("test-model", status, lease, registry,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), Runnable::run).run();

        assertTrue(registry.statusSnapshot(role).isEmpty());
        assertNull(status.tryBeginStatusPoll(), "a missing committed endpoint retires its status generation");
        assertNull(registry.get(role, status.getIpPort()));
        verify(registry, org.mockito.Mockito.never()).publishPreparedEndpoint(
                anyString(), org.mockito.Mockito.same(status), any(WorkerStatus.PreparedStatus.class));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void changedWorkerRoleRetiresGenerationWithoutPublishingStatus(boolean published) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, "test-site");
        directory(registry, status);
        if (published) {
            RunnerTestSupport.publishEndpoint(registry, RoleType.PREFILL, status.getIpPort(), status);
        }
        var committed = status.committedWorkerStatus();
        WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
        assertNotNull(lease);
        // Both protocol fields agree: rejection must enforce discovery identity, not converter validation.
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.DECODE.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .setStatusVersion(2L)
                .setAlive(true)
                .build();
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));

        new GrpcWorkerStatusRunner("test-model", status, lease, registry,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), Runnable::run).run();

        assertSame(committed, status.committedWorkerStatus(), "rejected role must not advance fields or cursors");
        assertEquals(RoleType.PREFILL, status.getRole());
        assertFalse(status.isActiveGeneration());
        assertNull(status.tryBeginStatusPoll(), "retired generation cannot poll again");
        assertTrue(registry.statusSnapshot(RoleType.PREFILL).isEmpty());
        assertTrue(registry.statusSnapshot(RoleType.DECODE).isEmpty());
        assertNull(registry.get(RoleType.PREFILL, status.getIpPort()));
        assertNull(registry.get(RoleType.DECODE, status.getIpPort()));
    }

    private static EndpointRegistry directory(
            EndpointRegistry registry, WorkerStatus status) {
        EndpointRegistry directory = registry;
        directory.currentOrDiscover(
                status.getRole(), status.getIpPort(), () -> status);
        return directory;
    }
}

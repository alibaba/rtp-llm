package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GrpcWorkerStatusRunnerTest {

    @Test
    void newGenerationProjectionRunsOutsideWorkerStatusLock() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1",
                8080, 8081, "test-site");
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll();
        assertNotNull(pollLease);

        AtomicBoolean publicationHeldLock = new AtomicBoolean();
        AtomicBoolean projectionReleasedLock = new AtomicBoolean();
        WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        WorkerDirectory directory = directory(registry, status);
        when(registry.publishPreparedEndpoint(
                org.mockito.Mockito.eq(ipPort),
                org.mockito.Mockito.eq(status),
                any(WorkerStatus.PreparedStatus.class)))
                .thenAnswer(invocation -> {
                    publicationHeldLock.set(
                            status.lock.isHeldByCurrentThread());
                    status.publishPreparedStatus(invocation.getArgument(2));
                    return new EndpointRegistry.EndpointPublication(
                            endpoint,
                            () -> projectionReleasedLock.set(
                                    !status.lock.isHeldByCurrentThread()));
                });

        EngineRpcService.WorkerStatusPB response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.DECODE.getCode())
                        .setRoleType(
                                EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(1L)
                        .setAlive(true)
                        .build();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));

        new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.DECODE, null,
                status, pollLease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), Runnable::run)
                .run();

        assertTrue(publicationHeldLock.get(),
                "publication must commit under the generation lock");
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
        EndpointRegistry registry = mock(EndpointRegistry.class);
        WorkerDirectory directory = directory(registry, status);
        when(registry.get(RoleType.DECODE, ipPort, status))
                .thenReturn(endpoint);
        java.util.concurrent.atomic.AtomicBoolean projected =
                new java.util.concurrent.atomic.AtomicBoolean();
        Runnable activity = () -> projected.set(true);
        when(endpoint.observeStatusHeartbeat(any(), any()))
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
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.DECODE, null,
                status, pollLease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), Runnable::run)
                .run();

        ArgumentCaptor<WorkerStatus.StatusObservation> observation =
                ArgumentCaptor.forClass(WorkerStatus.StatusObservation.class);
        verify(endpoint).observeStatusHeartbeat(
                org.mockito.Mockito.eq(status), observation.capture());
        assertTrue(observation.getValue().runningTasks().values().stream()
                .anyMatch(active -> active.requestId() == 123L));
        assertTrue(projected.get());
        WorkerStatus.PollLease nextPoll = status.tryBeginStatusPoll();
        assertNotNull(nextPoll, "the asynchronous owner must close the exact poll lease");
        nextPoll.close();
    }

    private static WorkerDirectory directory(
            EndpointRegistry registry, WorkerStatus status) {
        WorkerDirectory directory = new WorkerDirectory(registry);
        directory.currentOrDiscover(
                status.getRole(), status.getIpPort(), () -> status);
        return directory;
    }
}

package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;

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
    @ValueSource(booleans = {false, true})
    void transportFailureNeverRetiresWorkerEvenWhenTelemetryFails(boolean telemetryFails) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status =
                RunnerTestSupport.discovered(
                        RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "site");
        directory(registry, status);
        WorkerEndpoint endpoint =
                RunnerTestSupport.publishEndpoint(
                        registry, status.getRole(), status.getLogicalIpPort(), status);
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        if (telemetryFails) {
            org.mockito.Mockito.doThrow(new IllegalStateException("monitor unavailable"))
                    .when(reporter)
                    .reportStatusCheckerFail(any(), anyString(), any());
        }
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(
                        CompletableFuture.failedFuture(
                                io.grpc.Status.UNAVAILABLE.asRuntimeException()));
        try {
            for (int i = 0; i < 5; i++) {
                new GrpcWorkerStatusRunner(
                                "model",
                                status,
                                status.tryBeginStatusPoll(),
                                registry,
                                reporter,
                                grpc,
                                500L,
                                cache,
                                Runnable::run)
                        .run();
            }
            assertSame(endpoint, registry.get(status.getRole(), status.getLogicalIpPort()));
            assertEquals(0L, status.pollHealth().consecutiveWorkerFailures());
            assertNull(status.retirementReason());
            verifyNoInteractions(cache);
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @CsvSource({
        "DEADLINE_EXCEEDED,WORKER_STATUS_GRPC_TIMEOUT",
        "UNAVAILABLE,WORKER_SERVICE_UNAVAILABLE"
    })
    void reportsFailedRpcLatencyWithSameWorkerTags(
            String grpcCode, BalanceStatusEnum expectedError) {
        WorkerStatus status =
                RunnerTestSupport.discovered(
                        RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, "test-site");
        EndpointRegistry directory = directory(mock(EndpointRegistry.class), status);
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        io.grpc.Status failure =
                "DEADLINE_EXCEEDED".equals(grpcCode)
                        ? io.grpc.Status.DEADLINE_EXCEEDED
                        : io.grpc.Status.UNAVAILABLE;
        when(grpc.getWorkerStatusAsync(
                        org.mockito.ArgumentMatchers.anyString(),
                        anyInt(),
                        anyLong(),
                        anyLong(),
                        any()))
                .thenReturn(CompletableFuture.failedFuture(failure.asRuntimeException()));
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);

        new GrpcWorkerStatusRunner(
                        "test-model",
                        status,
                        status.tryBeginStatusPoll(),
                        directory,
                        reporter,
                        grpc,
                        5000L,
                        mock(CacheAwareService.class),
                        Runnable::run)
                .run();

        ArgumentCaptor<Long> latency = ArgumentCaptor.forClass(Long.class);
        verify(reporter)
                .reportStatusCheckerFail(expectedError, status.getMetricIpPort(), RoleType.PREFILL);
        verify(reporter)
                .reportStatusCheckFailureLatency(
                        org.mockito.ArgumentMatchers.eq(expectedError),
                        org.mockito.ArgumentMatchers.eq(status.getMetricIpPort()),
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        latency.capture());
        assertTrue(latency.getValue() >= 0L);
    }

    @Test
    void reportsEachObservedStepOnceAcrossPolls() {
        WorkerStatus status =
                RunnerTestSupport.discovered(
                        RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        String ipPort = status.getLogicalIpPort();
        WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        EndpointRegistry directory = directory(registry, status);
        when(registry.publishPreparedEndpoint(
                        org.mockito.ArgumentMatchers.anyString(), any(), any()))
                .thenAnswer(
                        invocation -> {
                            status.publishPreparedStatus(invocation.getArgument(2));
                            return endpoint;
                        });
        when(registry.get(RoleType.DECODE, ipPort, status)).thenReturn(endpoint);
        when(endpoint.applyPreparedStatus(any(), any()))
                .thenAnswer(
                        invocation -> {
                            status.publishPreparedStatus(invocation.getArgument(1));
                            return new WorkerEndpoint.StatusChange.None();
                        });
        when(endpoint.applyStatusHeartbeat(any(), any()))
                .thenReturn(new WorkerEndpoint.StatusChange.None());
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        long[] versions = {1, 2, 2, 3, 4};
        long[] steps = {0, 42, 42, 42, 43};
        for (int i = 0; i < versions.length; i++) {
            var response =
                    EngineRpcService.WorkerStatusPB.newBuilder()
                            .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                            .setStatusVersion(versions[i])
                            .setAlive(true);
            if (steps[i] > 0) {
                response.setLastStepMetrics(
                        EngineRpcService.WorkerStepMetricsPB.newBuilder()
                                .setStepId(steps[i])
                                .setTotalScheduledTokens(64)
                                .setTokenBudget(32000)
                                .setBudgetFillRatio(0.002));
            }
            when(grpc.getWorkerStatusAsync(
                            org.mockito.ArgumentMatchers.anyString(),
                            anyInt(),
                            anyLong(),
                            anyLong(),
                            any()))
                    .thenReturn(CompletableFuture.completedFuture(response.build()));
            new GrpcWorkerStatusRunner(
                            "test-model",
                            status,
                            status.tryBeginStatusPoll(),
                            directory,
                            reporter,
                            grpc,
                            5000L,
                            mock(CacheAwareService.class),
                            Runnable::run)
                    .run();
        }
        var observed = ArgumentCaptor.forClass(WorkerStatus.StepMetrics.class);
        verify(reporter, org.mockito.Mockito.times(2))
                .reportWorkerStepMetrics(org.mockito.Mockito.eq(status), observed.capture());
        org.junit.jupiter.api.Assertions.assertEquals(
                java.util.List.of(42L, 43L),
                observed.getAllValues().stream().map(WorkerStatus.StepMetrics::stepId).toList());
    }

    @Test
    void newGenerationPublishesUnderLockAndLaterStatusProjectsOutsideLock() {
        WorkerStatus status =
                RunnerTestSupport.discovered(
                        RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        String ipPort = status.getLogicalIpPort();
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll();
        assertNotNull(pollLease);

        AtomicBoolean publicationHeldLock = new AtomicBoolean();
        AtomicBoolean projectionReleasedLock = new AtomicBoolean();
        WorkerEndpoint.StatusChange expectedChange =
                new WorkerEndpoint.StatusChange.Decode(List.of(), false, false);
        WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
        EndpointRegistry registry =
                org.mockito.Mockito.spy(
                        RunnerTestSupport.endpointRegistry(mock(ConfigService.class)));
        EndpointRegistry directory = directory(registry, status);
        org.mockito.Mockito.doAnswer(
                        invocation -> {
                            publicationHeldLock.set(status.lock.isHeldByCurrentThread());
                            status.publishPreparedStatus(invocation.getArgument(2));
                            return endpoint;
                        })
                .when(registry)
                .publishPreparedEndpoint(
                        org.mockito.Mockito.eq(ipPort),
                        org.mockito.Mockito.eq(status),
                        any(WorkerStatus.PreparedStatus.class));
        when(registry.get(RoleType.DECODE, ipPort, status)).thenReturn(null, endpoint);
        when(endpoint.applyPreparedStatus(any(), any()))
                .thenAnswer(
                        invocation -> {
                            assertTrue(
                                    status.lock.isHeldByCurrentThread(),
                                    "existing endpoint reduction must hold the generation lock");
                            status.publishPreparedStatus(invocation.getArgument(1));
                            return expectedChange;
                        });
        org.mockito.Mockito.doAnswer(
                        invocation -> {
                            assertFalse(
                                    status.lock.isHeldByCurrentThread(),
                                    "status publication must release the generation lock");
                            if (invocation.getArgument(0) == expectedChange) {
                                projectionReleasedLock.set(true);
                            } else {
                                assertSame(
                                        WorkerEndpoint.NO_STATUS_CHANGE, invocation.getArgument(0));
                            }
                            return null;
                        })
                .when(endpoint)
                .publishStatus(any());

        EngineRpcService.WorkerStatusPB response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.DECODE.getCode())
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(1L)
                        .setAlive(true)
                        .build();
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        when(grpc.getWorkerStatusAsync(
                        org.mockito.ArgumentMatchers.anyString(),
                        anyInt(),
                        anyLong(),
                        anyLong(),
                        any()))
                .thenReturn(
                        CompletableFuture.completedFuture(response),
                        CompletableFuture.completedFuture(
                                response.toBuilder().setStatusVersion(2L).build()));

        new GrpcWorkerStatusRunner(
                        "test-model",
                        status,
                        pollLease,
                        directory,
                        mock(EngineHealthReporter.class),
                        grpc,
                        5_000L,
                        mock(CacheAwareService.class),
                        Runnable::run)
                .run();

        assertTrue(publicationHeldLock.get(), "publication must commit under the generation lock");
        assertFalse(
                projectionReleasedLock.get(),
                "a new generation has no local request facts to project");
        WorkerStatus.PollLease nextPoll = status.tryBeginStatusPoll();
        assertNotNull(nextPoll);
        new GrpcWorkerStatusRunner(
                        "test-model",
                        status,
                        nextPoll,
                        directory,
                        mock(EngineHealthReporter.class),
                        grpc,
                        5_000L,
                        mock(CacheAwareService.class),
                        Runnable::run)
                .run();
        assertTrue(
                projectionReleasedLock.get(),
                "endpoint facts must project after releasing that lock");
    }

    @Test
    void sameVersionResponseProjectsExactEndpointActivity() {
        WorkerStatus status =
                RunnerTestSupport.alive(
                        RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        String ipPort = status.getLogicalIpPort();
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll();
        assertNotNull(pollLease);

        WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
        EndpointRegistry registry =
                org.mockito.Mockito.spy(
                        RunnerTestSupport.endpointRegistry(mock(ConfigService.class)));
        EndpointRegistry directory = directory(registry, status);
        when(registry.get(RoleType.DECODE, ipPort, status)).thenReturn(endpoint);
        java.util.concurrent.atomic.AtomicBoolean projected =
                new java.util.concurrent.atomic.AtomicBoolean();
        WorkerEndpoint.StatusChange activity =
                new WorkerEndpoint.StatusChange.Decode(List.of(), false, false);
        when(endpoint.applyStatusHeartbeat(any(), any()))
                .thenAnswer(
                        invocation -> {
                            assertTrue(
                                    status.lock.isHeldByCurrentThread(),
                                    "heartbeat reduction must hold its generation lock");
                            return activity;
                        });
        org.mockito.Mockito.doAnswer(
                        invocation -> {
                            assertSame(activity, invocation.getArgument(0));
                            assertFalse(
                                    status.lock.isHeldByCurrentThread(),
                                    "heartbeat facts must publish outside the generation lock");
                            projected.set(true);
                            return null;
                        })
                .when(endpoint)
                .publishStatus(any());

        EngineRpcService.TaskInfoPB task =
                EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId(123)
                        .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                        .build();
        EngineRpcService.WorkerStatusPB response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.DECODE.getCode())
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(status.appliedStatusCursor().statusVersion())
                        .setAlive(true)
                        .addRunningTaskInfo(task)
                        .build();
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        when(grpc.getWorkerStatusAsync(
                        org.mockito.ArgumentMatchers.anyString(),
                        anyInt(),
                        anyLong(),
                        anyLong(),
                        any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        new GrpcWorkerStatusRunner(
                        "test-model",
                        status,
                        pollLease,
                        directory,
                        mock(EngineHealthReporter.class),
                        grpc,
                        5_000L,
                        mock(CacheAwareService.class),
                        Runnable::run)
                .run();

        ArgumentCaptor<WorkerStatus.StatusObservation> observation =
                ArgumentCaptor.forClass(WorkerStatus.StatusObservation.class);
        verify(endpoint)
                .applyStatusHeartbeat(org.mockito.Mockito.eq(status), observation.capture());
        assertTrue(
                observation.getValue().activeTasks().values().stream()
                        .anyMatch(active -> active.requestId().equals("123")));
        assertTrue(projected.get());
        verify(endpoint, org.mockito.Mockito.never()).applyPreparedStatus(any(), any());
        WorkerStatus.PollLease nextPoll = status.tryBeginStatusPoll();
        assertNotNull(nextPoll, "the asynchronous owner must close the exact poll lease");
        nextPoll.close();
    }

    @ParameterizedTest
    @ValueSource(longs = {1L, 2L})
    void committedStatusWithoutItsExactEndpointRetiresOnHeartbeatOrNewVersion(
            long responseVersion) {
        RoleType role = RoleType.DECODE;
        WorkerStatus status =
                RunnerTestSupport.alive(role, null, "127.0.0.1", 8080, 8081, "test-site");
        EndpointRegistry registry =
                org.mockito.Mockito.spy(
                        RunnerTestSupport.endpointRegistry(mock(ConfigService.class)));
        directory(registry, status);
        assertNull(registry.get(role, status.getIpPort()), "the committed cursor has no endpoint");
        WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
        assertNotNull(lease);
        EngineRpcService.WorkerStatusPB response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(role.getCode())
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(responseVersion)
                        .setAlive(true)
                        .build();
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        when(grpc.getWorkerStatusAsync(
                        org.mockito.ArgumentMatchers.anyString(),
                        anyInt(),
                        anyLong(),
                        anyLong(),
                        any()))
                .thenReturn(CompletableFuture.completedFuture(response));

        new GrpcWorkerStatusRunner(
                        "test-model",
                        status,
                        lease,
                        registry,
                        mock(EngineHealthReporter.class),
                        grpc,
                        5_000L,
                        mock(CacheAwareService.class),
                        Runnable::run)
                .run();

        org.flexlb.sync.runner.RunnerTestSupport.awaitCondition(
                () -> registry.statusSnapshot(role).isEmpty());
        assertNull(
                status.tryBeginStatusPoll(),
                "a missing committed endpoint retires its status generation");
        assertNull(registry.get(role, status.getIpPort()));
        verify(registry, org.mockito.Mockito.never())
                .publishPreparedEndpoint(
                        org.mockito.ArgumentMatchers.anyString(),
                        org.mockito.Mockito.same(status),
                        any(WorkerStatus.PreparedStatus.class));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void changedWorkerRoleRetiresGenerationWithoutPublishingStatus(boolean published) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status =
                RunnerTestSupport.discovered(
                        RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, "test-site");
        directory(registry, status);
        if (published) {
            RunnerTestSupport.publishEndpoint(
                    registry, RoleType.PREFILL, status.getIpPort(), status);
        }
        var committed = status.committedWorkerStatus();
        WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
        assertNotNull(lease);
        // Both protocol fields agree: rejection must enforce discovery identity, not converter
        // validation.
        EngineRpcService.WorkerStatusPB response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.DECODE.getCode())
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(2L)
                        .setAlive(true)
                        .build();
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        when(grpc.getWorkerStatusAsync(
                        org.mockito.ArgumentMatchers.anyString(),
                        anyInt(),
                        anyLong(),
                        anyLong(),
                        any()))
                .thenReturn(CompletableFuture.completedFuture(response));

        new GrpcWorkerStatusRunner(
                        "test-model",
                        status,
                        lease,
                        registry,
                        mock(EngineHealthReporter.class),
                        grpc,
                        5_000L,
                        mock(CacheAwareService.class),
                        Runnable::run)
                .run();

        assertSame(
                committed,
                status.committedWorkerStatus(),
                "rejected role must not advance fields or cursors");
        assertEquals(RoleType.PREFILL, status.getRole());
        assertFalse(status.isActiveGeneration());
        assertNull(status.tryBeginStatusPoll(), "retired generation cannot poll again");
        org.flexlb.sync.runner.RunnerTestSupport.awaitCondition(
                () -> registry.statusSnapshot(RoleType.PREFILL).isEmpty());
        assertTrue(registry.statusSnapshot(RoleType.DECODE).isEmpty());
        assertNull(registry.get(RoleType.PREFILL, status.getIpPort()));
        assertNull(registry.get(RoleType.DECODE, status.getIpPort()));
    }

    private static EndpointRegistry directory(EndpointRegistry registry, WorkerStatus status) {
        EndpointRegistry directory = registry;
        if (org.mockito.Mockito.mockingDetails(registry).isMock()
                && !org.mockito.Mockito.mockingDetails(registry).isSpy()) {
            org.mockito.Mockito.when(
                            registry.isCurrentStatus(
                                    status.getRole(), status.getLogicalIpPort(), status))
                    .thenReturn(true);
        }
        directory.currentOrDiscover(status.getRole(), status.getLogicalIpPort(), () -> status);
        return directory;
    }

    @ParameterizedTest
    @CsvSource({
        "DECODE,false,false",
        "PREFILL,true,false",
        "PDFUSION,true,false",
        "PREFILL,true,true"
    })
    void workerFailuresRetireButInvalidFramesKeepTheGeneration(
            RoleType role, boolean needsCacheKeys, boolean invalidRole) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status =
                RunnerTestSupport.discovered(role, null, "127.0.0.1", 8080, 8081, "test-site");
        EndpointRegistry directory = directory(registry, status);
        WorkerEndpoint endpoint =
                RunnerTestSupport.publishEndpoint(
                        registry, role, status.getLogicalIpPort(), status);
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        when(grpc.getWorkerStatusAsync(
                        org.mockito.ArgumentMatchers.anyString(),
                        anyInt(),
                        anyLong(),
                        anyLong(),
                        any()))
                .thenReturn(
                        invalidRole
                                ? CompletableFuture.completedFuture(
                                        EngineRpcService.WorkerStatusPB.newBuilder()
                                                .setRoleType(
                                                        EngineRpcService.RoleTypePB
                                                                .ROLE_TYPE_PREFILL)
                                                .setRole("DECODE")
                                                .setStatusVersion(1L)
                                                .setAlive(true)
                                                .build())
                                : CompletableFuture.completedFuture(
                                        EngineRpcService.WorkerStatusPB.newBuilder()
                                                .setRoleType(
                                                        EngineRpcService.RoleTypePB.valueOf(
                                                                "ROLE_TYPE_" + role.name()))
                                                .setRole(role.name())
                                                .setStatusVersion(1L)
                                                .setAlive(false)
                                                .build()));

        for (int failure = 1; failure <= 3; failure++) {
            WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
            assertNotNull(lease);
            new GrpcWorkerStatusRunner(
                            "test-model",
                            status,
                            lease,
                            directory,
                            mock(EngineHealthReporter.class),
                            grpc,
                            5_000L,
                            cache,
                            Runnable::run)
                    .run();
            if (failure < 3) {
                assertSame(status, directory.statusSnapshot(role).get(status.getLogicalIpPort()));
                assertSame(endpoint, registry.get(role, status.getLogicalIpPort()));
                verifyNoInteractions(cache);
            }
        }

        if (invalidRole) {
            assertSame(status, directory.statusSnapshot(role).get(status.getLogicalIpPort()));
            assertSame(endpoint, registry.get(role, status.getLogicalIpPort()));
            verifyNoInteractions(cache);
            registry.close();
            return;
        }
        RunnerTestSupport.awaitCondition(() -> directory.statusSnapshot(role).isEmpty());
        assertNull(registry.get(role, status.getLogicalIpPort()));
        assertTrue(directory.statusSnapshot(role).isEmpty());
        assertNull(status.tryBeginStatusPoll(), "retired generation cannot poll again");
        if (needsCacheKeys) {
            verify(cache).removeEngineBlockCache(status.getLogicalIpPort());
        } else {
            verifyNoInteractions(cache);
        }
    }
}

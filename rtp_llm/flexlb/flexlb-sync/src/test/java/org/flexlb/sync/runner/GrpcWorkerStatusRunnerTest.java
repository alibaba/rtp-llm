package org.flexlb.sync.runner;

import com.google.protobuf.Int64Value;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;

import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

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

/**
 * Checks worker status projection, distinct step observations and generation-safe publication.
 */
class GrpcWorkerStatusRunnerTest {

    @ParameterizedTest
    @CsvSource({
        "DECODE,false,false",
        "PREFILL,true,false",
        "PDFUSION,true,false",
        "DECODE,false,true",
        "PREFILL,true,true"
    })
    void transportFailuresPreserveRoutingAndExplicitDeadStatusRetires(
            RoleType role, boolean needsCacheKeys, boolean reportedDead) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status =
                RunnerTestSupport.discovered(role, null, "127.0.0.1", 8080, 8081, "test-site");
        WorkerDirectory directory = directory(registry, status);
        WorkerEndpoint endpoint =
                RunnerTestSupport.publishEndpoint(
                        registry, role, status.getLogicalIpPort(), status);
        CacheAwareService cache = mock(CacheAwareService.class);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        long lastSuccessfulPoll = status.pollHealth().lastSuccessfulPollUs();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(
                        reportedDead
                                ? CompletableFuture.completedFuture(
                                        EngineRpcService.WorkerStatusPB.newBuilder()
                                                .setRole(role.getCode())
                                                .setStatusVersion(1L)
                                                .setAlive(false)
                                                .build())
                                : CompletableFuture.failedFuture(
                                        io.grpc.Status.UNAVAILABLE.asRuntimeException()));

        for (int failure = 1; failure <= (reportedDead ? 1 : 3); failure++) {
            WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
            assertNotNull(lease);
            new GrpcWorkerStatusRunner(
                            "test-model",
                            status.getLogicalIpPort(),
                            "test-site",
                            role,
                            null,
                            status,
                            lease,
                            directory,
                            reporter,
                            grpc,
                            5_000L,
                            cache,
                            Runnable::run)
                    .run();
            if (!reportedDead) {
                assertSame(status, directory.statusSnapshot(role).get(status.getLogicalIpPort()));
                assertSame(endpoint, registry.get(role, status.getLogicalIpPort()));
                verifyNoInteractions(cache);
            }
        }
        if (!reportedDead) {
            assertEquals(3, status.pollHealth().consecutiveTransportFailures());
            assertEquals(lastSuccessfulPoll, status.pollHealth().lastSuccessfulPollUs());
            assertTrue(status.pollHealth().reportedAlive());
            return;
        }

        verify(reporter)
                .reportStatusCheckerFail(
                        BalanceStatusEnum.WORKER_STATUS_SERVER_FAILED,
                        status.getMetricIpPort(),
                        role);
        assertNull(registry.get(role, status.getLogicalIpPort()));
        assertTrue(directory.statusSnapshot(role).isEmpty());
        assertNull(status.tryBeginStatusPoll(), "retired generation cannot poll again");
        if (needsCacheKeys) {
            verify(cache).removeEngineBlockCache(status.getLogicalIpPort());
        } else {
            verifyNoInteractions(cache);
        }
    }

    @ParameterizedTest
    @CsvSource({
        "DEADLINE_EXCEEDED,WORKER_STATUS_GRPC_TIMEOUT",
        "UNAVAILABLE,WORKER_SERVICE_UNAVAILABLE",
        "LOCAL_FAILURE,MASTER_STATUS_PROCESSING_FAILED"
    })
    void reportsFailedRpcLatencyWithSameWorkerTags(
            String grpcCode, BalanceStatusEnum expectedError) {
        WorkerStatus status =
                RunnerTestSupport.discovered(
                        RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, "test-site");
        WorkerDirectory directory = directory(mock(EndpointRegistry.class), status);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        Throwable failure =
                "LOCAL_FAILURE".equals(grpcCode)
                        ? new IllegalStateException("Local completion failed")
                        : io.grpc.Status.fromCode(io.grpc.Status.Code.valueOf(grpcCode))
                                .asRuntimeException();
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.failedFuture(failure));
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);

        new GrpcWorkerStatusRunner(
                        "test-model",
                        status.getLogicalIpPort(),
                        "test-site",
                        RoleType.PREFILL,
                        null,
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
        WorkerDirectory directory = directory(registry, status);
        when(registry.publishPreparedEndpoint(anyString(), any(), any()))
                .thenAnswer(
                        invocation -> {
                            status.publishPreparedStatus(invocation.getArgument(2));
                            return new EndpointRegistry.EndpointPublication(endpoint, () -> {});
                        });
        when(registry.get(RoleType.DECODE, ipPort, status)).thenReturn(endpoint);
        when(endpoint.applyPreparedStatus(any(), any()))
                .thenAnswer(
                        invocation -> {
                            status.publishPreparedStatus(invocation.getArgument(1));
                            return (Runnable) () -> {};
                        });
        when(endpoint.observeStatusHeartbeat(any(), any())).thenReturn(() -> {});
        EngineGrpcService grpc = mock(EngineGrpcService.class);
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
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.completedFuture(response.build()));
            new GrpcWorkerStatusRunner(
                            "test-model",
                            ipPort,
                            "test-site",
                            RoleType.DECODE,
                            null,
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
    void newGenerationProjectionRunsOutsideWorkerStatusLock() {
        WorkerStatus status =
                RunnerTestSupport.discovered(
                        RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        String ipPort = status.getLogicalIpPort();
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
                .thenAnswer(
                        invocation -> {
                            publicationHeldLock.set(status.lock.isHeldByCurrentThread());
                            status.publishPreparedStatus(invocation.getArgument(2));
                            return new EndpointRegistry.EndpointPublication(
                                    endpoint,
                                    () ->
                                            projectionReleasedLock.set(
                                                    !status.lock.isHeldByCurrentThread()));
                        });

        EngineRpcService.WorkerStatusPB response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.DECODE.getCode())
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(1L)
                        .setAlive(true)
                        .build();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));

        new GrpcWorkerStatusRunner(
                        "test-model",
                        ipPort,
                        "test-site",
                        RoleType.DECODE,
                        null,
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
        EndpointRegistry registry = mock(EndpointRegistry.class);
        WorkerDirectory directory = directory(registry, status);
        when(registry.get(RoleType.DECODE, ipPort, status)).thenReturn(endpoint);
        java.util.concurrent.atomic.AtomicBoolean projected =
                new java.util.concurrent.atomic.AtomicBoolean();
        Runnable activity = () -> projected.set(true);
        when(endpoint.observeStatusHeartbeat(any(), any())).thenReturn(activity);

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
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        new GrpcWorkerStatusRunner(
                        "test-model",
                        ipPort,
                        "test-site",
                        RoleType.DECODE,
                        null,
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
                .observeStatusHeartbeat(org.mockito.Mockito.eq(status), observation.capture());
        assertTrue(
                observation.getValue().activeTasks().values().stream()
                        .anyMatch(active -> active.requestId().equals("123")));
        assertTrue(projected.get());
        WorkerStatus.PollLease nextPoll = status.tryBeginStatusPoll();
        assertNotNull(nextPoll, "the asynchronous owner must close the exact poll lease");
        nextPoll.close();
    }

    @ParameterizedTest
    @CsvSource({"8,8", "0,0", ",9223372036854775807"})
    void acceptedSameVersionVitStatusReplacesLoadAtApplication(Long reported, long expected) {
        WorkerStatus status =
                RunnerTestSupport.alive(RoleType.VIT, null, "127.0.0.1", 8080, 8081, "test-site");
        String address = status.getLogicalIpPort();
        EndpointRegistry registry = mock(EndpointRegistry.class);
        WorkerDirectory directory = directory(registry, status);
        when(registry.get(RoleType.VIT, address, status)).thenReturn(new WorkerEndpoint(status));
        status.recordVitLoad(OptionalLong.of(17), status.vitRoutedInputs(), System.nanoTime());
        var response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_VIT)
                        .setStatusVersion(status.appliedStatusCursor().statusVersion())
                        .setAlive(true);
        if (reported != null) {
            response.setMmInflightInputs(Int64Value.of(reported));
        }

        var pendingResponse = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        var queuedApplication = new AtomicReference<Runnable>();
        runVitPoll(status, directory, pendingResponse, queuedApplication::set);
        pendingResponse.complete(response.build());
        long afterReceive = System.nanoTime();
        assertEquals(17, status.vitLoadEstimate(afterReceive, Long.MAX_VALUE));

        queuedApplication.get().run();

        assertEquals(expected, status.vitLoadEstimate(System.nanoTime(), Long.MAX_VALUE));
        assertEquals(Long.MAX_VALUE, status.vitLoadEstimate(afterReceive + 1, 0));
    }

    @Test
    void heartbeatsAdvanceDuringBlockedStatusApplication() throws Exception {
        WorkerStatus status =
                RunnerTestSupport.discovered(
                        RoleType.VIT, null, "127.0.0.1", 8080, 8081, "test-site");
        status.configureStatusFreshness(10_000_000L);
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        try {
            WorkerDirectory directory = directory(registry, status);
            WorkerEndpoint endpoint =
                    RunnerTestSupport.publishEndpoint(
                            registry, RoleType.VIT, status.getLogicalIpPort(), status);
            var response =
                    EngineRpcService.WorkerStatusPB.newBuilder()
                            .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_VIT)
                            .setStatusVersion(status.appliedStatusCursor().statusVersion())
                            .setAlive(true)
                            .build();
            var pendingApplication = new AtomicReference<Runnable>();
            var busyApplicationQueued = new AtomicBoolean();

            status.lock.lock();
            try {
                status.recordStatusApplied(System.nanoTime() / 1000 - 11_000_000L);
                CompletableFuture.runAsync(
                                () ->
                                        runVitPoll(
                                                status,
                                                directory,
                                                CompletableFuture.completedFuture(response),
                                                pendingApplication::set))
                        .get(5, TimeUnit.SECONDS);
                long firstReceipt = status.pollHealth().lastSuccessfulPollUs();
                CompletableFuture.runAsync(
                                () ->
                                        runVitPoll(
                                                status,
                                                directory,
                                                CompletableFuture.completedFuture(response),
                                                ignored -> busyApplicationQueued.set(true)))
                        .get(5, TimeUnit.SECONDS);
                assertTrue(status.pollHealth().lastSuccessfulPollUs() > firstReceipt);
                assertFalse(status.tryClaimHealthRetirement(System.nanoTime() / 1000, 10_000_000L));
                assertFalse(busyApplicationQueued.get());
                try (var captured =
                        directory.captureEndpoint(RoleType.VIT, status.getLogicalIpPort())) {
                    assertNull(captured);
                }
                try (var existingWork = endpoint.tryPinGeneration()) {
                    assertNotNull(existingWork);
                }
            } finally {
                status.lock.unlock();
            }

            assertNotNull(pendingApplication.get());
            pendingApplication.get().run();
            try (var captured =
                    directory.captureEndpoint(RoleType.VIT, status.getLogicalIpPort())) {
                assertNotNull(captured);
            }
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @CsvSource({"ROLE_TYPE_VIT,1,true", "ROLE_TYPE_DECODE,2,true", "ROLE_TYPE_VIT,2,false"})
    void rejectedVitStatusKeepsThePreviousLoad(
            EngineRpcService.RoleTypePB reportedRole, long version, boolean hasEndpoint) {
        WorkerStatus status =
                RunnerTestSupport.discovered(
                        RoleType.VIT, null, "127.0.0.1", 8080, 8081, "test-site");
        RunnerTestSupport.publish(status, RunnerTestSupport.response(status, true, 2));
        EndpointRegistry registry = mock(EndpointRegistry.class);
        WorkerDirectory directory = directory(registry, status);
        if (hasEndpoint) {
            when(registry.get(RoleType.VIT, status.getLogicalIpPort(), status))
                    .thenReturn(new WorkerEndpoint(status));
        }
        status.recordVitLoad(OptionalLong.of(17), status.vitRoutedInputs(), System.nanoTime());
        var response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRoleType(reportedRole)
                        .setStatusVersion(version)
                        .setAlive(true)
                        .setMmInflightInputs(Int64Value.of(0))
                        .build();

        runVitPoll(status, directory, CompletableFuture.completedFuture(response), Runnable::run);

        assertEquals(17, status.vitLoadEstimate(System.nanoTime(), Long.MAX_VALUE));
    }

    private static void runVitPoll(
            WorkerStatus status,
            WorkerDirectory directory,
            CompletableFuture<EngineRpcService.WorkerStatusPB> response,
            Executor callbackExecutor) {
        WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
        assertNotNull(lease);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(response);
        new GrpcWorkerStatusRunner(
                        "test-model",
                        status.getLogicalIpPort(),
                        "test-site",
                        RoleType.VIT,
                        null,
                        status,
                        lease,
                        directory,
                        mock(EngineHealthReporter.class),
                        grpc,
                        5_000L,
                        mock(CacheAwareService.class),
                        callbackExecutor)
                .run();
    }

    private static WorkerDirectory directory(EndpointRegistry registry, WorkerStatus status) {
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        directory.currentOrDiscover(status.getRole(), status.getLogicalIpPort(), () -> status);
        return directory;
    }
}

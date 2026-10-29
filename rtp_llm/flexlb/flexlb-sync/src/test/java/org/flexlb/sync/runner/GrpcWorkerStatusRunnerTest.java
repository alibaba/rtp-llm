package org.flexlb.sync.runner;

import io.grpc.Status;
import io.netty.channel.ConnectTimeoutException;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.schedule.ExpirationCleaner;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.ArgumentCaptor;

import java.io.EOFException;
import java.net.BindException;
import java.net.ConnectException;
import java.net.NoRouteToHostException;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.nio.channels.AsynchronousCloseException;
import java.nio.channels.ClosedByInterruptException;
import java.nio.channels.ClosedChannelException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

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
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class GrpcWorkerStatusRunnerTest {

    @Test
    void vitLoadWireContractMatchesPythonFixture() throws Exception {
        // Shared with vit_rpc_control_test: field 28, Int64Value(7).
        byte[] wire = java.util.HexFormat.of().parseHex("e201020807");
        var response = EngineRpcService.WorkerStatusPB.parseFrom(wire);
        assertTrue(response.hasMmInflightInputs());
        assertEquals(7L, response.getMmInflightInputs().getValue());
        org.junit.jupiter.api.Assertions.assertArrayEquals(wire, response.toByteArray());
    }

    @Test
    void vitLoadUpdatesAtTheSameStatusVersionAndPreservesUnknownVersusZero() {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(RoleType.VIT, "g", "127.0.0.1", 8080, 8081, "test");
        WorkerDirectory directory = directory(registry, status);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        var response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("VIT").setStatusVersion(1L).setAlive(true);
        try {
            for (int index = 0; index < 2; index++) {
                long count = index == 0 ? 18L : 3L;
                when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                        .thenReturn(CompletableFuture.completedFuture(response
                                .setMmInflightInputs(com.google.protobuf.Int64Value.of(count)).build()));
                poll(status, directory, grpc, Runnable::run, index + 1L);
                assertNotNull(status.vitLoad());
                assertEquals(count, status.vitLoad().inflightInputs());
                assertEquals(1L, status.appliedStatusCursor().statusVersion());
            }
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.completedFuture(response.clearMmInflightInputs().build()));
            poll(status, directory, grpc, Runnable::run, 3L);
            assertNull(status.vitLoad(), "legacy or unknown load is not idle");
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.completedFuture(response
                            .setMmInflightInputs(com.google.protobuf.Int64Value.of(0L)).build()));
            poll(status, directory, grpc, Runnable::run, 4L);
            assertNotNull(status.vitLoad());
            assertEquals(0L, status.vitLoad().inflightInputs());
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.completedFuture(response
                            .setMmInflightInputs(com.google.protobuf.Int64Value.of(-1L)).build()));
            poll(status, directory, grpc, Runnable::run, 5L);
            assertNull(status.vitLoad(), "invalid load must not win least-load routing");
        } finally {
            registry.close();
        }
    }

    @Test
    void vitRpcFailureDoesNotRenewLoadAndExplicitNotAliveClearsIt() {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(RoleType.VIT, "g", "127.0.0.1", 8080, 8081, "test");
        WorkerDirectory directory = directory(registry, status);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        var response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("VIT").setStatusVersion(1L).setAlive(true)
                .setMmInflightInputs(com.google.protobuf.Int64Value.of(9L)).build();
        try {
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.completedFuture(response));
            poll(status, directory, grpc, Runnable::run, 1L);
            var sample = status.vitLoad();
            assertNotNull(sample);
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.failedFuture(Status.DEADLINE_EXCEEDED.asRuntimeException()));
            poll(status, directory, grpc, Runnable::run, 2L);
            assertSame(sample, status.vitLoad(), "a timeout must not refresh the sample timestamp");
            assertTrue(status.isAvailableForScheduling());
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.completedFuture(response.toBuilder().setAlive(false).build()));
            poll(status, directory, grpc, Runnable::run, 3L);
            assertNull(status.vitLoad());
        } finally {
            registry.close();
        }
    }

    static Stream<Arguments> rpcFailures() {
        return Stream.of(
                Arguments.of(
                        Status.UNKNOWN.withCause(new OutOfMemoryError("opaque")).asRuntimeException(),
                        BalanceStatusEnum.MASTER_STATUS_MEMORY_EXHAUSTED,
                        GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED, true),
                Arguments.of(
                        Status.UNAVAILABLE.withCause(new RejectedExecutionException()).asRuntimeException(),
                        BalanceStatusEnum.MASTER_STATUS_EXECUTOR_REJECTED,
                        GrpcFailure.Kind.LOCAL_EXECUTOR_REJECTED, true),
                Arguments.of(Status.UNAVAILABLE.withCause(new BindException()).asRuntimeException(),
                        BalanceStatusEnum.MASTER_STATUS_BIND_FAILED, GrpcFailure.Kind.LOCAL_BIND_FAILED, true),
                Arguments.of(
                        Status.DEADLINE_EXCEEDED.asRuntimeException(),
                        BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, GrpcFailure.Kind.DEADLINE_EXCEEDED, false),
                Arguments.of(
                        new IllegalStateException("DEADLINE_EXCEEDED OutOfMemoryError Connection reset"),
                        BalanceStatusEnum.WORKER_STATUS_GRPC_FAILED, GrpcFailure.Kind.RPC_FAILURE, false),
                Arguments.of(
                        Status.UNAVAILABLE.withCause(new ConnectException()).asRuntimeException(),
                        BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED, GrpcFailure.Kind.CONNECT_FAILED, false),
                Arguments.of(new ConnectTimeoutException(), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.CONNECT_TIMEOUT, false),
                Arguments.of(new ConnectException("Connection refused"), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.CONNECT_REFUSED, false),
                Arguments.of(new NoRouteToHostException(), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.NO_ROUTE, false),
                Arguments.of(new SocketException("Connection reset"), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.CONNECTION_RESET, false),
                Arguments.of(new SocketException("Connection reset by peer"), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.CONNECTION_RESET, false),
                Arguments.of(new SocketException("Broken pipe"), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.BROKEN_PIPE, false),
                Arguments.of(new ClosedChannelException(), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.CHANNEL_CLOSED, false),
                Arguments.of(new EOFException(), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.UNEXPECTED_EOF, false),
                Arguments.of(new ClosedByInterruptException(), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.IO_INTERRUPTED, false),
                Arguments.of(new AsynchronousCloseException(), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.ASYNC_CHANNEL_CLOSED, false),
                Arguments.of(new SocketTimeoutException(), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.IO_TIMEOUT, false),
                Arguments.of(new SocketException(), BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED,
                        GrpcFailure.Kind.SOCKET_FAILURE, false),
                Arguments.of(Status.INTERNAL.asRuntimeException(), BalanceStatusEnum.WORKER_STATUS_GRPC_FAILED,
                        GrpcFailure.Kind.RPC_FAILURE, false),
                Arguments.of(Status.RESOURCE_EXHAUSTED.asRuntimeException(), BalanceStatusEnum.WORKER_STATUS_GRPC_FAILED,
                        GrpcFailure.Kind.RPC_FAILURE, false),
                Arguments.of(Status.UNAVAILABLE.asRuntimeException(), BalanceStatusEnum.WORKER_STATUS_GRPC_FAILED,
                        GrpcFailure.Kind.RPC_FAILURE, false));
    }

    @ParameterizedTest
    @MethodSource("rpcFailures")
    void rpcFailureMetricsUseExceptionTypesAndStatusCodes(
            Throwable failure, BalanceStatusEnum expected, GrpcFailure.Kind kind, boolean confirmedLocal) {
        var status = RunnerTestSupport.alive(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        var health = status.pollHealth();
        var lease = status.tryBeginStatusPoll(1L);
        var grpc = mock(EngineGrpcService.class);
        var reporter = mock(EngineHealthReporter.class);
        var response = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any())).thenReturn(response);
        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test", RoleType.DECODE, null, status, lease,
                directory(mock(EndpointRegistry.class), status), reporter, grpc, 1000L,
                mock(CacheAwareService.class), Runnable::run).run();
        response.completeExceptionally(new java.util.concurrent.CompletionException(failure));
        verify(reporter).reportStatusCheckerFail(eq(status.getMetricIpPort()), eq(expected), eq(RoleType.DECODE),
                any(Stage.class), anyString(), any(GrpcFailure.Analysis.class));
        var analysis = verifyFailureDetail(reporter, Stage.RPC_COMPLETED, kind.name());
        assertEquals(GrpcFailure.analyze(failure), analysis);
        assertEquals(kind, analysis.kind());
        assertEquals(confirmedLocal, analysis.confirmedLocalFailure());
        verify(reporter).reportStatusCheckFailureLatency(eq(expected), eq(status.getMetricIpPort()),
                eq(RoleType.DECODE), anyLong());
        org.mockito.Mockito.verifyNoMoreInteractions(reporter);
        assertEquals("ERROR", status.trace().snapshot().statusPoll().rpcResult());
        assertTrue(status.trace().snapshot().statusPoll().leaseReleased());
        assertTrue(status.isAvailableForScheduling());
        assertSame(health, status.pollHealth());
        assertEquals(0L, status.pollHealth().consecutiveWorkerFailures());
        try (var next = status.tryBeginStatusPoll(2L)) {
            assertNotNull(next);
        }
    }

    @ParameterizedTest
    @MethodSource("rpcFailures")
    void rpcFailuresPreserveEarlierNotAliveEvidenceAndAdmission(
            Throwable failure, BalanceStatusEnum expected, GrpcFailure.Kind kind, boolean confirmedLocal) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = WorkerStatus.createDiscovered(RoleType.DECODE, "g", "127.0.0.1", 8080, 8081, "test");
        WorkerDirectory directory = directory(registry, status);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.DECODE.getCode()).setStatusVersion(1L).setAlive(true)
                .setAvailableKvCache(1000L).setTotalKvCache(1000L).build();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        try {
            poll(status, directory, grpc, Runnable::run, 1L);
            WorkerEndpoint endpoint = directory.exactEndpoint(status.getRole(), status.getLogicalIpPort(), status);
            assertNotNull(endpoint);
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.completedFuture(response.toBuilder().setAlive(false).build()));
            poll(status, directory, grpc, Runnable::run, 2L);
            var previousHealth = status.pollHealth();
            assertEquals(1L, previousHealth.consecutiveWorkerFailures());
            assertFalse(previousHealth.reportedAlive());

            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.failedFuture(failure));
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            for (long cycle = 3L; cycle <= 5L; cycle++) {
                var lease = status.tryBeginStatusPoll(cycle);
                assertNotNull(lease);
                new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test", status.getRole(), null, status,
                        lease, directory, reporter, grpc, 1000L, mock(CacheAwareService.class), Runnable::run).run();
                assertSame(previousHealth, status.pollHealth(), "RPC failures neither count nor clear not-alive evidence");
                assertTrue(status.isAvailableForScheduling());
                assertNull(status.retirementReason());
                assertSame(endpoint, directory.exactEndpoint(status.getRole(), status.getLogicalIpPort(), status));
                try (var pin = endpoint.tryPinSchedulingGeneration()) {
                    assertNotNull(pin, "unattributed failures must preserve scheduling admission");
                }
                assertTrue(status.trace().snapshot().statusPoll().leaseReleased());
            }
            var details = ArgumentCaptor.forClass(GrpcFailure.Analysis.class);
            verify(reporter, times(3)).reportStatusCheckerFail(eq(status.getMetricIpPort()), eq(expected), eq(status.getRole()),
                    eq(Stage.RPC_COMPLETED), eq(kind.name()), details.capture());
            for (var detail : details.getAllValues()) {
                assertEquals(kind, detail.kind());
                assertEquals(confirmedLocal, detail.confirmedLocalFailure());
            }
        } finally {
            registry.close();
        }
    }

    @Test
    void rejectedApplicationIsReportedAsLocalAfterRpcSucceeded() {
        var status = RunnerTestSupport.discovered(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        var grpc = mock(EngineGrpcService.class);
        var reporter = mock(EngineHealthReporter.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(1).setAlive(true).build()));
        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test", RoleType.DECODE, null, status,
                status.tryBeginStatusPoll(1L), directory(mock(EndpointRegistry.class), status), reporter, grpc, 1000L,
                mock(CacheAwareService.class), task -> { throw new java.util.concurrent.RejectedExecutionException(); }).run();
        assertEquals("OK", status.trace().snapshot().statusPoll().rpcResult());
        assertEquals("CALLBACK_SUBMISSION_FAILED", status.trace().snapshot().statusPoll().failureStage());
        assertEquals(0, status.pollHealth().consecutiveWorkerFailures());
        verify(reporter).reportStatusCheckerFail(eq(status.getMetricIpPort()), eq(BalanceStatusEnum.MASTER_STATUS_EXECUTOR_REJECTED),
                eq(RoleType.DECODE), any(Stage.class), anyString(), any(GrpcFailure.Analysis.class));
        assertTrue(verifyFailureDetail(reporter, Stage.CALLBACK_SUBMISSION_FAILED,
                "LOCAL_EXECUTOR_REJECTED").confirmedLocalFailure());
        verify(reporter).reportStatusCheckFailureLatency(eq(BalanceStatusEnum.MASTER_STATUS_EXECUTOR_REJECTED),
                eq(status.getMetricIpPort()), eq(RoleType.DECODE), anyLong());
        org.mockito.Mockito.verifyNoMoreInteractions(reporter);
    }

    @Test
    void responseCompletionFailureIsObservedAndReleasesPoll() {
        var status = RunnerTestSupport.alive(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        var grpc = mock(EngineGrpcService.class);
        var reporter = mock(EngineHealthReporter.class);
        var response = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any())).thenReturn(response);
        // Deliberately fail response validation after recording a successful RPC.
        var invalid = mock(EngineRpcService.WorkerStatusPB.class);
        when(invalid.getStatusVersion()).thenThrow(new IllegalStateException("validation failed"));
        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test", RoleType.DECODE, null, status,
                status.tryBeginStatusPoll(1L), directory(mock(EndpointRegistry.class), status),
                reporter, grpc, 1000L, mock(CacheAwareService.class), Runnable::run).run();
        response.complete(invalid);
        assertEquals("OK", status.trace().snapshot().statusPoll().rpcResult());
        assertEquals("CALLBACK_FAILED", status.trace().snapshot().statusPoll().failureStage());
        assertTrue(status.trace().snapshot().statusPoll().leaseReleased());
        assertTrue(status.isAvailableForScheduling(), "local validation failure must preserve worker health");
        assertNull(status.retirementReason());
        verify(reporter).reportStatusCheckerFail(eq(status.getMetricIpPort()), eq(BalanceStatusEnum.MASTER_STATUS_PROCESSING_FAILED),
                eq(RoleType.DECODE), any(Stage.class), anyString(), any(GrpcFailure.Analysis.class));
        try (var next = status.tryBeginStatusPoll(2L)) { assertNotNull(next); }
    }

    @Test
    void rpcFailureDiagnosticsCannotChangeWorkerHealth() {
        var status = RunnerTestSupport.alive(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        var grpc = mock(EngineGrpcService.class);
        Throwable failure = new IllegalStateException() {
            @Override
            public String getMessage() { throw new OutOfMemoryError("diagnostics failed"); }
        };
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.failedFuture(
                        new java.util.concurrent.CompletionException("RPC failed", failure)));
        poll(status, directory(mock(EndpointRegistry.class), status), grpc, Runnable::run, 2L);
        assertTrue(status.isAvailableForScheduling());
        assertEquals(0L, status.pollHealth().consecutiveWorkerFailures());
        assertTrue(status.pollHealth().reportedAlive());
        assertNull(status.retirementReason());
        try (var next = status.tryBeginStatusPoll(3L)) { assertNotNull(next); }
    }

    @Test
    void synchronousInitiationOomRetainsOriginalErrorAndReportsLocalFailure() {
        var status = RunnerTestSupport.discovered(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        var grpc = mock(EngineGrpcService.class);
        var reporter = mock(EngineHealthReporter.class);
        var failure = new OutOfMemoryError("opaque");
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any())).thenThrow(failure);
        var runner = new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test", RoleType.DECODE, null, status,
                status.tryBeginStatusPoll(1L), directory(mock(EndpointRegistry.class), status), reporter,
                grpc, 1000L, mock(CacheAwareService.class), Runnable::run);
        assertSame(failure, org.junit.jupiter.api.Assertions.assertThrows(OutOfMemoryError.class, runner::run));
        verify(reporter).reportStatusCheckerFail(eq(status.getMetricIpPort()), eq(BalanceStatusEnum.MASTER_STATUS_MEMORY_EXHAUSTED),
                eq(RoleType.DECODE), any(Stage.class), anyString(), any(GrpcFailure.Analysis.class));
        assertEquals("RPC_INITIATION_FAILED", status.trace().snapshot().statusPoll().failureStage());
        assertTrue(status.trace().snapshot().statusPoll().leaseReleased());
    }

    static Stream<Arguments> localProcessingFailures() {
        return Stream.of(
                Arguments.of(new OutOfMemoryError("direct buffer"),
                        BalanceStatusEnum.MASTER_STATUS_MEMORY_EXHAUSTED),
                Arguments.of(new RejectedExecutionException("executor stopped"),
                        BalanceStatusEnum.MASTER_STATUS_EXECUTOR_REJECTED));
    }

    @ParameterizedTest
    @MethodSource("localProcessingFailures")
    void conversionResourceFailuresPreserveCommittedStateAndAllowNextPoll(
            Throwable failure, BalanceStatusEnum expected) {
        var status = RunnerTestSupport.alive(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        var grpc = mock(EngineGrpcService.class);
        var reporter = mock(EngineHealthReporter.class);
        var response = org.mockito.Mockito.spy(EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.DECODE.getCode()).setStatusVersion(2L).setAlive(true).build());
        org.mockito.Mockito.doThrow(failure).when(response).getRunningTaskInfoList();
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test", RoleType.DECODE, null, status,
                status.tryBeginStatusPoll(1L), directory(mock(EndpointRegistry.class), status), reporter,
                grpc, 1000L, mock(CacheAwareService.class), Runnable::run).run();

        verify(reporter).reportStatusCheckerFail(eq(status.getMetricIpPort()), eq(expected), eq(RoleType.DECODE),
                any(Stage.class), anyString(), any(GrpcFailure.Analysis.class));
        var analysis = verifyFailureDetail(reporter, Stage.CONVERSION_FAILED, GrpcFailure.classify(failure).name());
        assertSame(failure, analysis.rootCause());
        assertEquals(1L, status.appliedStatusCursor().statusVersion());
        assertEquals(0L, status.pollHealth().consecutiveWorkerFailures());
        assertTrue(status.isAvailableForScheduling());
        assertNull(status.retirementReason());
        try (var next = status.tryBeginStatusPoll(2L)) {
            assertNotNull(next);
            assertTrue(status.tryBeginStatusApplication(next));
            status.finishStatusApplication(next);
        }
    }

    @ParameterizedTest
    @MethodSource("localProcessingFailures")
    void publicationFailureAfterCommitReportsLocalCauseAndCleansUpExactGeneration(
            Throwable failure, BalanceStatusEnum expected) {
        var status = RunnerTestSupport.discovered(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        var registry = mock(EndpointRegistry.class);
        var directory = directory(registry, status);
        var reporter = mock(EngineHealthReporter.class);
        when(registry.publishPreparedEndpoint(eq(status.getLogicalIpPort()), eq(status), any()))
                .thenAnswer(invocation -> {
                    status.publishPreparedStatus(invocation.getArgument(2));
                    throw failure;
                });
        var grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.DECODE.getCode()).setStatusVersion(1L).setAlive(true).build()));
        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test", RoleType.DECODE, null, status,
                status.tryBeginStatusPoll(1L), directory, reporter, grpc, 1000L,
                mock(CacheAwareService.class), Runnable::run).run();

        verify(reporter).reportStatusCheckerFail(eq(status.getMetricIpPort()), eq(expected), eq(RoleType.DECODE),
                any(Stage.class), anyString(), any(GrpcFailure.Analysis.class));
        var analysis = verifyFailureDetail(reporter, Stage.STATUS_RECONCILIATION_FAILED,
                GrpcFailure.classify(failure).name());
        assertSame(failure, analysis.rootCause());
        assertEquals(1L, status.appliedStatusCursor().statusVersion(), "failure happened after state commit");
        assertEquals(0L, status.pollHealth().consecutiveWorkerFailures());
        assertFalse(status.isActiveGeneration());
        assertNull(directory.statusSnapshot(RoleType.DECODE).get(status.getLogicalIpPort()));
        assertEquals("STATUS_RECONCILIATION_FAILED", status.trace().snapshot().unavailableReason());
        assertTrue(status.trace().snapshot().statusPoll().leaseReleased());
        var replacement = RunnerTestSupport.discovered(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        assertSame(replacement, directory.currentOrDiscover(RoleType.DECODE, status.getLogicalIpPort(), () -> replacement));
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {
            "ROLE_MISMATCH", "INVALID_STATUS_VERSION", "STATUS_VERSION_REGRESSED", "ENDPOINT_MISSING"})
    void invalidStatusRetiresLocalGenerationWithoutBlamingWorker(String reason) {
        var status = RunnerTestSupport.alive(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        RunnerTestSupport.publish(status, RunnerTestSupport.response(status, true, 2L));
        var registry = mock(EndpointRegistry.class);
        var directory = directory(registry, status);
        var reporter = mock(EngineHealthReporter.class);
        var response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(reason.equals("ROLE_MISMATCH") ? RoleType.PREFILL.getCode() : RoleType.DECODE.getCode())
                .setStatusVersion(reason.equals("INVALID_STATUS_VERSION") ? 0L
                        : reason.equals("STATUS_VERSION_REGRESSED") ? 1L : 3L)
                .setAlive(true).build();
        var grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test", RoleType.DECODE, null, status,
                status.tryBeginStatusPoll(1L), directory, reporter, grpc, 1000L,
                mock(CacheAwareService.class), Runnable::run).run();

        verify(reporter).reportStatusCheckerFail(eq(status.getMetricIpPort()), eq(BalanceStatusEnum.MASTER_STATUS_PROCESSING_FAILED),
                eq(RoleType.DECODE), any(Stage.class), anyString(), any(GrpcFailure.Analysis.class));
        assertFalse(verifyFailureDetail(reporter, Stage.STATUS_RESPONSE_FAILED, reason).confirmedLocalFailure());
        assertEquals(0L, status.pollHealth().consecutiveWorkerFailures());
        assertFalse(status.isActiveGeneration());
        assertEquals(reason, status.trace().snapshot().unavailableReason());
        assertNull(directory.statusSnapshot(RoleType.DECODE).get(status.getLogicalIpPort()));
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "PDFUSION", "DECODE", "VIT"})
    void normalRpcSuccessKeepsTheExistingEndpointWithoutExtraCapacityNotifications(RoleType role) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        PlacementAvailability availability = mock(PlacementAvailability.class);
        EndpointRegistry registry = new EndpointRegistry(config, mock(EndpointEventProjector.class),
                mock(BatchSchedulerReporter.class), mock(DeliveryStrategy.class), availability);
        WorkerStatus status = WorkerStatus.createDiscovered(role, "g", "127.0.0.1", 8080, 8081, "test");
        WorkerDirectory directory = directory(registry, status);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(role.getCode()).setStatusVersion(1L).setAlive(true)
                .setAvailableKvCache(1000L).setTotalKvCache(1000L).build();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        try {
            poll(status, directory, grpc, Runnable::run, 1L);
            WorkerEndpoint endpoint = directory.exactEndpoint(role, status.getLogicalIpPort(), status);
            assertNotNull(endpoint);
            clearInvocations(availability);
            poll(status, directory, grpc, Runnable::run, 2L);
            poll(status, directory, grpc, Runnable::run, 3L);
            try (var pin = endpoint.tryPinSchedulingGeneration()) {
                assertNotNull(pin);
            }
            assertSame(endpoint, directory.exactEndpoint(role, status.getLogicalIpPort(), status));
            verifyNoInteractions(availability);
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @CsvSource({"PREFILL,true", "DECODE,true", "PREFILL,false", "DECODE,false"})
    void onlyValidNotAliveReportsDisableAdmission(RoleType role, boolean reportsNotAlive) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = WorkerStatus.createDiscovered(role, "g", "127.0.0.1", 8080, 8081, "test");
        WorkerDirectory directory = directory(registry, status);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(role.getCode()).setStatusVersion(1L).setAlive(true)
                .setAvailableKvCache(1000L).setTotalKvCache(1000L).build();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        try {
            poll(status, directory, grpc, Runnable::run, 1L);
            WorkerEndpoint endpoint = directory.exactEndpoint(role, status.getLogicalIpPort(), status);
            DecodeEndpoint.ReservationHandle reservation = null;
            if (endpoint instanceof DecodeEndpoint decode) {
                try (var pin = decode.tryPinSchedulingGeneration()) {
                    reservation = decode.reserve(pin, "42", 1L, 1L, 40);
                }
            }
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(reportsNotAlive
                            ? CompletableFuture.completedFuture(response.toBuilder().setAlive(false).build())
                            : CompletableFuture.failedFuture(
                                    Status.UNKNOWN.withCause(new OutOfMemoryError("opaque")).asRuntimeException()));
            for (int index = 3; index <= 5; index++) {
                poll(status, directory, grpc, Runnable::run, index);
                assertEquals(reportsNotAlive ? index - 2 : 0L, status.pollHealth().consecutiveWorkerFailures());
                assertEquals(!reportsNotAlive || index < 5, status.isAvailableForScheduling());
            }
            if (!reportsNotAlive) {
                assertNull(status.retirementReason());
                try (var pin = endpoint.tryPinSchedulingGeneration()) { assertNotNull(pin); }
                if (endpoint instanceof PrefillEndpoint prefill) {
                    assertTrue(prefill.batchAdmissionAvailability(2).isAvailable());
                } else {
                    var decode = (DecodeEndpoint) endpoint;
                    var capacity = new DecodeEndpoint.AdmissionCapacity(0L, 100L);
                    assertTrue(decode.shouldRetryDispatch("42", capacity));
                    decode.release(reservation, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
                }
                return;
            }
            assertEquals(WorkerStatus.RetirementReason.CONSECUTIVE_WORKER_FAILURES, status.retirementReason());
            if (endpoint instanceof PrefillEndpoint prefill) {
                assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, prefill.reserveBatch(null, 1L, 2).status());
                assertFalse(prefill.batchAdmissionAvailability(2).isAvailable(),
                        "free batch slots cannot trigger retry for an unavailable worker");
            } else {
                DecodeEndpoint decode = (DecodeEndpoint) endpoint;
                DecodeEndpoint.AdmissionCapacity capacity = new DecodeEndpoint.AdmissionCapacity(0L, 100L);
                assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                        decode.acquireDispatchPermit(reservation, capacity).status());
                assertFalse(decode.shouldRetryDispatch("42", capacity),
                        "free Decode capacity cannot trigger retry for an unavailable worker");
                decode.release(reservation, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
                assertTrue(decode.shouldRetryDispatch("42", capacity),
                        "lost ownership must still wake a waiter for an unavailable worker");
            }
        } finally {
            registry.close();
        }
    }

    @Test
    void queuedApplicationCannotReviveAWorkerRemovedFromDiscovery() {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = WorkerStatus.createDiscovered(RoleType.DECODE, "g", "127.0.0.1", 8080, 8081, "test");
        WorkerDirectory directory = directory(registry, status);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("DECODE").setStatusVersion(1L).setAlive(true).build();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        java.util.List<Runnable> applications = new java.util.ArrayList<>();
        try {
            poll(status, directory, grpc, Runnable::run, 1L);
            WorkerEndpoint endpoint = directory.exactEndpoint(RoleType.DECODE, status.getLogicalIpPort(), status);
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.completedFuture(response.toBuilder().setStatusVersion(2L).build()));
            poll(status, directory, grpc, applications::add, 2L);
            status.markRemovedFromDiscovery();
            assertEquals(WorkerStatus.RetirementReason.DISCOVERY_REMOVED, status.retirementReason());
            assertFalse(status.isAvailableForScheduling());
            assertNull(endpoint.tryPinSchedulingGeneration());
            assertNull(status.tryBeginStatusPoll(6L), "UNAVAILABLE is terminal for this generation");
            applications.removeFirst().run();
            assertFalse(status.isAvailableForScheduling(), "the queued response cannot undo confirmed discovery removal");
            assertNull(endpoint.tryPinSchedulingGeneration());
            assertEquals(1L, status.appliedStatusCursor().statusVersion(), "unavailable generations cannot commit the queued response");
        } finally {
            applications.forEach(Runnable::run);
            registry.close();
        }
    }

    @Test
    void queuedApplicationAllowsHeartbeatsWithoutAccumulatingResponses() {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        try {
            WorkerStatus status = WorkerStatus.createDiscovered(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
            WorkerDirectory directory = directory(registry, status);
            EngineGrpcService grpc = mock(EngineGrpcService.class);
            java.util.concurrent.atomic.AtomicLong version = new java.util.concurrent.atomic.AtomicLong();
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenAnswer(call -> CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.newBuilder()
                            .setRole("DECODE").setStatusVersion(version.incrementAndGet()).setAlive(true)
                            .setLatestFinishedVersion(10L * version.get())
                            .setAvailableKvCache(1000L - 100L * version.get()).setTotalKvCache(1000L).build()));
            java.util.List<Runnable> applications = new java.util.ArrayList<>();
            poll(status, directory, grpc, Runnable::run, 1L);
            WorkerEndpoint endpoint = directory.exactEndpoint(RoleType.DECODE, status.getLogicalIpPort(), status);
            WorkerStatus.CommittedWorkerStatus committed = status.committedWorkerStatus();
            poll(status, directory, grpc, applications::add, 2L);
            for (int index = 3; index <= 7; index++) {
                poll(status, directory, grpc, applications::add, index);
                assertEquals(1, applications.size(), "busy application must not queue another response");
            }
            assertEquals(1, applications.size(), "at most one response is retained for local application");
            assertSame(committed, status.committedWorkerStatus(), "RPC completion cannot overwrite committed resource state");
            try (var pin = endpoint.tryPinSchedulingGeneration()) {
                assertNotNull(pin, "slow local application keeps using the previous successful snapshot");
            }
            applications.removeFirst().run();
            assertEquals(2L, status.appliedStatusCursor().statusVersion());
            assertEquals(20L, status.appliedStatusCursor().latestFinishedTaskVersion());
            assertEquals(800L, status.committedEngineObservation().availableKvCacheTokens());
            assertTrue(status.isAvailableForScheduling());
            try (var pin = endpoint.tryPinSchedulingGeneration()) { assertNotNull(pin); }

            poll(status, directory, grpc, Runnable::run, 8L);
            assertEquals(8L, status.appliedStatusCursor().statusVersion());
            assertEquals(80L, status.appliedStatusCursor().latestFinishedTaskVersion());
            assertEquals(200L, status.committedEngineObservation().availableKvCacheTokens());
            ArgumentCaptor<Long> requestedFinishedVersions = ArgumentCaptor.forClass(Long.class);
            verify(grpc, org.mockito.Mockito.times(8)).getWorkerStatusAsync(anyString(), anyInt(),
                    requestedFinishedVersions.capture(), anyLong(), any());
            assertEquals(java.util.List.of(-1L, 10L, 10L, 10L, 10L, 10L, 10L, 20L),
                    requestedFinishedVersions.getAllValues(), "each RPC starts from the previous committed cursor");
        } finally {
            registry.close();
        }
    }

    private static void poll(WorkerStatus status, WorkerDirectory directory, EngineGrpcService grpc,
                             java.util.concurrent.Executor applicationExecutor, long cycle) {
        new GrpcWorkerStatusRunner("model", status.getLogicalIpPort(), "test", status.getRole(), null, status,
                status.tryBeginStatusPoll(cycle), directory, mock(EngineHealthReporter.class), grpc,
                1000L, mock(CacheAwareService.class), applicationExecutor).run();
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void repeatedLocalConversionFailuresDoNotRetireReachableWorker(boolean malformedRole) {
        var status = RunnerTestSupport.discovered(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        var directory = directory(mock(EndpointRegistry.class), status);
        var grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(malformedRole
                        ? EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).setStatusVersion(1L)
                                .setRole("invalid-role").build() : null));
        for (int index = 0; index < 4; index++) {
            new GrpcWorkerStatusRunner("model", status.getLogicalIpPort(), "test", RoleType.DECODE, null,
                    status, status.tryBeginStatusPoll(index), directory, mock(EngineHealthReporter.class), grpc,
                    1000L, mock(CacheAwareService.class), Runnable::run).run();
        }
        assertTrue(status.isActiveGeneration());
        org.junit.jupiter.api.Assertions.assertEquals(0, status.pollHealth().consecutiveWorkerFailures());
        org.junit.jupiter.api.Assertions.assertEquals("OK", status.trace().snapshot().statusPoll().rpcResult());
        assertSame(status, directory.statusSnapshot(RoleType.DECODE).get(status.getLogicalIpPort()));
    }

    @ParameterizedTest
    @CsvSource({"true,1,DECODE,true", "false,1,DECODE,false",
            "true,0,DECODE,false", "true,1,PREFILL,false"})
    void validAliveRpcUpdatesHealthBeforeLocalApplication(
            boolean alive, long version, RoleType responseRole, boolean reportedAlive) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        WorkerDirectory directory = directory(registry, status);
        var lease = status.tryBeginStatusPoll(1L);
        String ipPort = status.getLogicalIpPort();
        var rpc = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(rpc);
        java.util.List<Runnable> queue = new java.util.ArrayList<>();
        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test-site",
                RoleType.DECODE, null, status, lease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), queue::add).run();
        assertNull(status.tryBeginStatusPoll(2L), "only one wire poll may be in flight");
        rpc.complete(EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(responseRole.getCode())
                .setRoleType(responseRole == RoleType.DECODE
                        ? EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE
                        : EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setStatusVersion(version).setAlive(alive).build());
        org.junit.jupiter.api.Assertions.assertEquals(reportedAlive, status.pollHealth().reportedAlive());
        assertTrue(status.trace().snapshot().statusPoll().leaseReleased(), "RPC ownership ends before local application");
        try (var next = status.tryBeginStatusPoll(2L)) {
            assertNotNull(next, "one unhealthy report must not retire the generation");
        }
        // A successful contact does not commit local resource state.
        org.junit.jupiter.api.Assertions.assertEquals(-1L, status.appliedStatusCursor().statusVersion());
        queue.forEach(Runnable::run);
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {
            "RPC_THROW", "RPC_FAILED", "CONVERSION_FAILED", "CALLBACK_REJECTED", "FAILURE_REPORTING_FAILED"})
    void failurePathsReleaseThePollAndPreserveBoundary(String failurePoint) {
        var status = RunnerTestSupport.discovered(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test");
        var lease = status.tryBeginStatusPoll(1L);
        String ipPort = status.getLogicalIpPort();
        var grpc = mock(EngineGrpcService.class);
        var reporter = mock(EngineHealthReporter.class);
        var failure = new IllegalStateException("injected failure");
        var stub = when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()));
        if (failurePoint.equals("RPC_THROW")) {
            stub.thenThrow(failure);
        } else if (failurePoint.equals("CALLBACK_REJECTED")) {
            stub.thenReturn(CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.getDefaultInstance()));
        } else if (failurePoint.equals("CONVERSION_FAILED")) {
            stub.thenReturn(CompletableFuture.completedFuture(null));
        } else {
            stub.thenReturn(CompletableFuture.failedFuture(new java.util.concurrent.CompletionException(failure)));
        }
        if (failurePoint.equals("FAILURE_REPORTING_FAILED")) {
            org.mockito.Mockito.doThrow(failure).when(reporter).reportStatusCheckerFail(
                    anyString(), any(), any(), any(Stage.class), anyString(), any(GrpcFailure.Analysis.class));
        }
        java.util.concurrent.Executor executor = failurePoint.equals("CALLBACK_REJECTED")
                ? task -> { throw new java.util.concurrent.RejectedExecutionException("pool full"); }
                : Runnable::run;
        var runner = new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test", RoleType.DECODE, null, status, lease, directory(mock(EndpointRegistry.class), status), reporter, grpc,
                1000L, mock(CacheAwareService.class), executor);
        if (failurePoint.equals("RPC_THROW")) {
            assertSame(failure, org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, runner::run));
        } else {
            runner.run();
        }
        var snapshot = status.trace().snapshot().statusPoll();
        String expectedFailure = switch (failurePoint) {
            case "RPC_THROW" -> "RPC_INITIATION_FAILED";
            case "CONVERSION_FAILED" -> "CONVERSION_FAILED";
            case "CALLBACK_REJECTED" -> "CALLBACK_SUBMISSION_FAILED";
            default -> "RPC_COMPLETED";
        };
        org.junit.jupiter.api.Assertions.assertEquals(expectedFailure, snapshot.failureStage());
        assertTrue(snapshot.times().rpcResultMs() >= 0);
        assertTrue(snapshot.leaseReleased());
        if (failurePoint.equals("CONVERSION_FAILED")) {
            org.junit.jupiter.api.Assertions.assertEquals("OK", snapshot.rpcResult());
            assertTrue(snapshot.times().convertStartMs() >= 0);
        }
        if (failurePoint.equals("CALLBACK_REJECTED")) {
            org.junit.jupiter.api.Assertions.assertEquals(-1, snapshot.times().callbackStartMs());
        }
        try (var next = status.tryBeginStatusPoll(1L)) {
            assertNotNull(next, "a failed poll must not prevent future polling");
            assertTrue(status.tryBeginStatusApplication(next), "failures must release the application slot too");
            status.finishStatusApplication(next);
        }
    }

    @Test
    void invalidVersionRecordsOnlyOneResponseFailure() {
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        EndpointRegistry registry = mock(EndpointRegistry.class);
        WorkerDirectory directory = directory(registry, status);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(
                        EngineRpcService.WorkerStatusPB.newBuilder()
                                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                                .setStatusVersion(0).setAlive(true).build()));

        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test-site",
                RoleType.DECODE, null, status, status.tryBeginStatusPoll(1L), directory,
                mock(EngineHealthReporter.class), grpc, 5000L,
                mock(CacheAwareService.class), Runnable::run).run();

        org.junit.jupiter.api.Assertions.assertEquals("STATUS_RESPONSE_FAILED",
                status.trace().snapshot().statusPoll().failureStage());
    }

    /** Capture the worker diagnostic line emitted through syncLogger. */
    private static final class RetirementLogCapture implements AutoCloseable {
        private final ch.qos.logback.classic.Logger logger = (ch.qos.logback.classic.Logger)
                org.slf4j.LoggerFactory.getLogger("syncLogger");
        private final java.util.concurrent.ConcurrentLinkedQueue<String> messages =
                new java.util.concurrent.ConcurrentLinkedQueue<>();
        private final java.util.concurrent.CountDownLatch first = new java.util.concurrent.CountDownLatch(1);
        private final ch.qos.logback.core.AppenderBase<ch.qos.logback.classic.spi.ILoggingEvent> appender;

        RetirementLogCapture(WorkerStatus status) {
            appender = new ch.qos.logback.core.AppenderBase<>() {
                @Override
                protected void append(ch.qos.logback.classic.spi.ILoggingEvent event) {
                    String message = event.getFormattedMessage();
                    if (!message.startsWith("event=worker_trace ")
                            || !message.contains(" generation=" + status.getGenerationId() + " ")
                            || !message.contains(" endpoint=" + status.getLogicalIpPort() + " ")) {
                        return;
                    }
                    messages.add(message);
                    first.countDown();
                }
            };
            appender.start();
            logger.addAppender(appender);
        }

        String awaitFirst() throws InterruptedException {
            assertTrue(first.await(5, java.util.concurrent.TimeUnit.SECONDS));
            String message = messages.element();
            return message;
        }

        @Override
        public void close() {
            logger.detachAppender(appender);
            appender.stop();
        }
    }

    @Test
    void timeoutsOnlyEnterRetirementHistoryAfterConfirmedDiscoveryRemoval() throws Exception {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.PREFILL, "example-group", "127.0.0.1", 8080, 8081, "example-site");
        RetirementLogCapture logs = new RetirementLogCapture(status);
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        directory.currentOrDiscover(status.getRole(), status.getLogicalIpPort(), () -> status);
        CacheAwareService cache = mock(CacheAwareService.class);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.PREFILL.getCode())
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                        .setStatusVersion(1).setAlive(true).build()));
        new GrpcWorkerStatusRunner("example-model", status.getLogicalIpPort(), "example-site",
                RoleType.PREFILL, null, status, status.tryBeginStatusPoll(1L), directory,
                mock(EngineHealthReporter.class), grpc, 30L, cache, Runnable::run).run();

        verify(cache).observeCacheHitFeedback(eq(status), any());
        clearInvocations(cache);
        var timer = java.util.concurrent.Executors.newSingleThreadScheduledExecutor();
        ExpirationCleaner cleaner = new ExpirationCleaner(cache, directory);
        try {
            for (int attempt = 1; attempt <= 3; attempt++) {
                var rpc = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
                when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                        .thenReturn(rpc);
                var finished = new CompletableFuture<Void>();
                var lease = status.tryBeginStatusPoll(1L);
        String ipPort = status.getLogicalIpPort();
                assertNotNull(lease);
                lease.pollTrace().record(Stage.RUNNER_SUBMITTED);
                new GrpcWorkerStatusRunner("example-model", status.getLogicalIpPort(), "example-site",
                        RoleType.PREFILL, null, status, lease, directory,
                        mock(EngineHealthReporter.class), grpc, 30L, cache,
                        Runnable::run).run();
                timer.schedule(() -> {
                    rpc.completeExceptionally(io.grpc.Status.DEADLINE_EXCEEDED
                            .withDescription("simulated 30ms RPC deadline").asRuntimeException());
                    finished.complete(null);
                }, 30L, java.util.concurrent.TimeUnit.MILLISECONDS);
                finished.get(5, java.util.concurrent.TimeUnit.SECONDS);
                cleaner.retireUnavailableWorkers();
                assertSame(status, directory.statusSnapshot(RoleType.PREFILL).get(status.getLogicalIpPort()));
                assertNull(status.retirementReason(), "deadlines do not prove worker failure");
                verifyNoInteractions(cache);
            }
            status.markRemovedFromDiscovery();
            cleaner.retireUnavailableWorkers();
            assertTrue(directory.statusSnapshot(RoleType.PREFILL).isEmpty());
            verify(cache).removeEngineBlockCache(
                    org.mockito.Mockito.same(status),
                    org.mockito.Mockito.same(status.trace()));
            String json = logs.awaitFirst();
            assertTrue(json.contains("DEADLINE_EXCEEDED"));
            assertTrue(json.contains("DISCOVERY_REMOVED"));
            assertTrue(json.contains("failures=0"));
            var snapshot = status.trace().snapshot();
            org.junit.jupiter.api.Assertions.assertEquals(0, snapshot.consecutiveFailures());
            assertTrue(snapshot.statusPoll().leaseReleased());
            assertEquals(-1L, snapshot.statusPoll().times().callbackEndMs(), "failed RPC health is recorded without local application");
        } finally {
            timer.shutdownNow();
            logs.close();
            registry.close();
        }
    }

    @Test
    void successfulRpcQueuedBehindCallbacksIsVisibleInRetirementHistory() throws Exception {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        RetirementLogCapture logs = new RetirementLogCapture(status);
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        directory.currentOrDiscover(status.getRole(), status.getLogicalIpPort(), () -> status);
        var lease = status.tryBeginStatusPoll(1L);
        String ipPort = status.getLogicalIpPort();
        var rpc = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(rpc);
        java.util.List<Runnable> queue = new java.util.ArrayList<>();
        CacheAwareService cache = mock(CacheAwareService.class);
        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test-site",
                RoleType.DECODE, null, status, lease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L, cache, queue::add).run();
        rpc.complete(EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.DECODE.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .setStatusVersion(1).setAlive(true).build());
        var queuedPoll = status.trace().snapshot().statusPoll();
        assertTrue(queuedPoll.times().rpcResultMs() >= 0);
        assertEquals("LEASE_RELEASED", queuedPoll.stage());
        assertTrue(queuedPoll.leaseReleased(), "RPC is complete while application waits");
        assertTrue(queuedPoll.times().leaseEndMs() >= 0L);
        assertEquals(-1L, queuedPoll.times().callbackStartMs(), "the queued callback has not started");
        assertEquals(-1L, queuedPoll.times().workerAttemptMs(), "the queued callback has not attempted the Worker lock");
        assertEquals(1, queue.size());
        var logger = mock(org.slf4j.Logger.class);
        status.lock.lock();
        EndpointRegistry.DetachedGeneration detached;
        try {
            detached = directory.beginRetirement(RoleType.DECODE, status.getLogicalIpPort(), status,
                    "DISCOVERY_TOPOLOGY_CHANGED");
        } finally {
            status.lock.unlock();
        }
        try (logs) {
            directory.completeRetirement(RoleType.DECODE, status.getLogicalIpPort(), status, detached, cache, logger);
            String json = logs.awaitFirst();
            assertTrue(json.contains("rpc:OK"));
            assertTrue(json.contains("stage:LEASE_RELEASED"));
            assertTrue(json.contains("released:true"));
            assertTrue(json.contains("DISCOVERY_TOPOLOGY_CHANGED"));
            assertTrue(!json.contains("RPC_FAILED"), "a queued successful RPC must not be reported as a transport failure");
            queue.removeFirst().run();
            assertTrue(status.trace().snapshot().statusPoll().leaseReleased());
            assertNull(registry.get(RoleType.DECODE, status.getLogicalIpPort()), "late callback must not resurrect a retired worker");
        }
    }

    @Test
    @org.junit.jupiter.api.Timeout(10)
    void callbackWaitingForWorkerLockHasStartedButHasNotCommitted() throws Exception {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        WorkerDirectory directory = directory(registry, status);
        var lease = status.tryBeginStatusPoll(1L);
        String ipPort = status.getLogicalIpPort();
        var rpc = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any())).thenReturn(rpc);
        var callback = new java.util.concurrent.atomic.AtomicReference<Thread>();
        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test-site", RoleType.DECODE, null, status, lease, directory, mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), task -> {
                    Thread thread = Thread.ofPlatform().unstarted(task);
                    callback.set(thread);
                    thread.start();
                }).run();
        status.lock.lock();
        try {
            rpc.complete(EngineRpcService.WorkerStatusPB.newBuilder()
                    .setRole(RoleType.DECODE.getCode())
                    .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                    .setStatusVersion(1).setAlive(true).build());
            long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(3);
            while (!status.lock.hasQueuedThread(callback.get()) && System.nanoTime() < deadline) {
                Thread.sleep(1);
            }
            assertTrue(status.lock.hasQueuedThread(callback.get()));
            var snapshot = status.trace().snapshot().statusPoll();
            org.junit.jupiter.api.Assertions.assertEquals("STATUS_LOCK_ATTEMPT", snapshot.stage());
            org.junit.jupiter.api.Assertions.assertEquals("OK", snapshot.rpcResult());
            assertTrue(snapshot.times().callbackStartMs() >= 0);
            assertTrue(snapshot.times().workerAttemptMs() >= 0);
            org.junit.jupiter.api.Assertions.assertEquals(-1, snapshot.times().workerAcquiredMs());
            org.junit.jupiter.api.Assertions.assertEquals(-1, snapshot.times().commitMs());
            var before = status.pollHealth();
            // Real Worker lock contention cannot prevent a later RPC or create another application.
            poll(status, directory, grpc, task -> { throw new AssertionError("application already occupied"); }, 2L);
            assertTrue(status.pollHealth() != before, "later RPC updated transport health while the lock was held");
            assertEquals(-1L, status.appliedStatusCursor().statusVersion());
        } finally {
            status.lock.unlock();
            if (callback.get() != null) {callback.get().join(3000);}
        }
        assertTrue(status.trace().snapshot().statusPoll().leaseReleased());
        assertNotNull(registry.get(RoleType.DECODE, status.getLogicalIpPort()));
    }

    @ParameterizedTest
    @CsvSource({"DECODE,false", "PREFILL,true", "PDFUSION,true"})
    void repeatedRpcFailuresPreserveWorkerUntilItReportsNotAlive(
            RoleType role, boolean needsCacheKeys) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                role, null, "127.0.0.1", 8080, 8081, "test-site");
        WorkerDirectory directory = directory(registry, status);
        WorkerEndpoint endpoint = RunnerTestSupport.publishEndpoint(
                registry, role, status.getLogicalIpPort(), status);
        CacheAwareService cache = mock(CacheAwareService.class);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.failedFuture(io.grpc.Status.UNAVAILABLE.asRuntimeException()));

        ExpirationCleaner cleaner = new ExpirationCleaner(cache, directory);
        for (int failure = 1; failure <= 3; failure++) {
            WorkerStatus.PollLease lease = status.tryBeginStatusPoll(1L);
        String ipPort = status.getLogicalIpPort();
            assertNotNull(lease);
            new GrpcWorkerStatusRunner(
                    "test-model", status.getLogicalIpPort(), "test-site", role, null,
                    status, lease, directory, mock(EngineHealthReporter.class),
                    grpc, 5_000L, cache, Runnable::run).run();
            cleaner.retireUnavailableWorkers();
            assertSame(status, directory.statusSnapshot(role).get(status.getLogicalIpPort()));
            assertSame(endpoint, registry.get(role, status.getLogicalIpPort()));
            assertNull(status.retirementReason());
            assertTrue(status.isAvailableForScheduling());
            assertEquals(0L, status.pollHealth().consecutiveWorkerFailures());
            verifyNoInteractions(cache);
        }

        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(role.getCode()).setStatusVersion(1L).setAlive(false).build()));
        for (int index = 1; index <= 3; index++) {
            poll(status, directory, grpc, Runnable::run, 3L + index);
            assertEquals(index, status.pollHealth().consecutiveWorkerFailures());
            assertEquals(index < 3, status.isAvailableForScheduling());
        }
        cleaner.retireUnavailableWorkers();
        assertNull(registry.get(role, status.getLogicalIpPort()));
        assertTrue(directory.statusSnapshot(role).isEmpty());
        var snapshot = status.trace().snapshot();
        assertEquals("CONSECUTIVE_WORKER_FAILURES", snapshot.unavailableReason());
        assertEquals(3L, snapshot.consecutiveFailures());
        assertNull(status.tryBeginStatusPoll(1L), "retired generation cannot poll again");
        if (needsCacheKeys) {
            verify(cache).removeEngineBlockCache(
                    org.mockito.Mockito.same(status),
                    org.mockito.Mockito.same(status.trace()));
        } else {
            verifyNoInteractions(cache);
        }
        registry.close();
    }

    @ParameterizedTest
    @CsvSource({"DEADLINE_EXCEEDED,WORKER_STATUS_GRPC_TIMEOUT", "UNAVAILABLE,WORKER_STATUS_GRPC_FAILED"})
    void reportsFailedRpcLatencyWithSameWorkerTags(String grpcCode, BalanceStatusEnum expectedError) {
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, "test-site");
        WorkerDirectory directory = directory(mock(EndpointRegistry.class), status);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        io.grpc.Status failure = "DEADLINE_EXCEEDED".equals(grpcCode)
                ? io.grpc.Status.DEADLINE_EXCEEDED : io.grpc.Status.UNAVAILABLE;
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.failedFuture(failure.asRuntimeException()));
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);

        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test-site", RoleType.PREFILL,
                null, status, status.tryBeginStatusPoll(1L), directory, reporter, grpc, 5000L,
                mock(CacheAwareService.class), Runnable::run).run();

        ArgumentCaptor<Long> latency = ArgumentCaptor.forClass(Long.class);
        verify(reporter).reportStatusCheckerFail(eq(status.getMetricIpPort()), eq(expectedError),
                eq(RoleType.PREFILL), eq(Stage.RPC_COMPLETED), anyString(), any(GrpcFailure.Analysis.class));
        verify(reporter).reportStatusCheckFailureLatency(org.mockito.ArgumentMatchers.eq(expectedError),
                org.mockito.ArgumentMatchers.eq(status.getMetricIpPort()),
                org.mockito.ArgumentMatchers.eq(RoleType.PREFILL), latency.capture());
        assertTrue(latency.getValue() >= 0L);
    }

    @Test
    void reportsEachObservedStepOnceAcrossPolls() {
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        String ipPort = status.getLogicalIpPort();
        WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        WorkerDirectory directory = directory(registry, status);
        when(registry.publishPreparedEndpoint(anyString(), any(), any())).thenAnswer(invocation -> {
            status.publishPreparedStatus(invocation.getArgument(2));
            return new EndpointRegistry.EndpointPublication(endpoint, () -> { });
        });
        when(registry.get(RoleType.DECODE, ipPort, status)).thenReturn(endpoint);
        when(endpoint.applyPreparedStatus(any(), any())).thenAnswer(invocation -> {
            status.publishPreparedStatus(invocation.getArgument(1));
            return (Runnable) () -> { };
        });
        when(endpoint.observeStatusHeartbeat(any(), any())).thenReturn(() -> { });
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        long[] versions = {1, 2, 2, 3, 4};
        long[] steps = {0, 42, 42, 42, 43};
        for (int i = 0; i < versions.length; i++) {
            var response = EngineRpcService.WorkerStatusPB.newBuilder()
                    .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                    .setStatusVersion(versions[i]).setAlive(true);
            if (steps[i] > 0) {
                response.setLastStepMetrics(EngineRpcService.WorkerStepMetricsPB.newBuilder()
                        .setStepId(steps[i]).setTotalScheduledTokens(64).setTokenBudget(32000)
                        .setBudgetFillRatio(0.002));
            }
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.completedFuture(response.build()));
            new GrpcWorkerStatusRunner("test-model", ipPort, "test-site", RoleType.DECODE, null,
                    status, status.tryBeginStatusPoll(1L), directory, reporter, grpc, 5000L,
                    mock(CacheAwareService.class), Runnable::run).run();
        }
        var observed = ArgumentCaptor.forClass(WorkerStatus.StepMetrics.class);
        verify(reporter, org.mockito.Mockito.times(2)).reportWorkerStepMetrics(
                org.mockito.Mockito.eq(status), observed.capture());
        org.junit.jupiter.api.Assertions.assertEquals(java.util.List.of(42L, 43L),
                observed.getAllValues().stream().map(WorkerStatus.StepMetrics::stepId).toList());
    }

    @Test
    void newGenerationProjectionRunsOutsideWorkerStatusLock() {
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1",
                8080, 8081, "test-site");
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll(19L);
        String ipPort = status.getLogicalIpPort();
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
        var poll = status.trace().snapshot().statusPoll();
        org.junit.jupiter.api.Assertions.assertEquals(19L, poll.syncCycleId());
        assertTrue(poll.times().healthMs() >= 0);
        assertTrue(poll.times().commitMs() >= poll.times().healthMs());
    }

    @Test
    void sameVersionResponseProjectsExactEndpointActivity() {
        WorkerStatus status = RunnerTestSupport.alive(
                RoleType.DECODE, null, "127.0.0.1",
                8080, 8081, "test-site");
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll(1L);
        String ipPort = status.getLogicalIpPort();
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
                .setRequestId(123)
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
        assertTrue(observation.getValue().activeTasks().values().stream()
                .anyMatch(active -> active.requestId().equals("123")));
        assertTrue(projected.get());
        WorkerStatus.PollLease nextPoll = status.tryBeginStatusPoll(1L);
        assertNotNull(nextPoll, "the asynchronous owner must close the exact poll lease");
        nextPoll.close();
    }

    private static GrpcFailure.Analysis verifyFailureDetail(
            EngineHealthReporter reporter, Stage stage, String reason) {
        var analysis = ArgumentCaptor.forClass(GrpcFailure.Analysis.class);
        verify(reporter).reportStatusCheckerFail(anyString(), any(BalanceStatusEnum.class), eq(RoleType.DECODE),
                eq(stage), eq(reason), analysis.capture());
        return analysis.getValue();
    }

    private static WorkerDirectory directory(
            EndpointRegistry registry, WorkerStatus status) {
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        directory.currentOrDiscover(
                status.getRole(), status.getLogicalIpPort(), () -> status);
        return directory;
    }
}

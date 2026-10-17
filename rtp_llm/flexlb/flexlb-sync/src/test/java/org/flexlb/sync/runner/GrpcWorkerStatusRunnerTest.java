package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.sync.schedule.WorkerRetirementCleaner;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.mockito.ArgumentCaptor;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class GrpcWorkerStatusRunnerTest {

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
            WorkerEndpoint endpoint = directory.exactEndpoint(role, status.getIpPort(), status);
            assertNotNull(endpoint);
            clearInvocations(availability);
            poll(status, directory, grpc, Runnable::run, 2L);
            poll(status, directory, grpc, Runnable::run, 3L);
            try (var pin = endpoint.tryPinSchedulingGeneration()) {
                assertNotNull(pin);
            }
            assertSame(endpoint, directory.exactEndpoint(role, status.getIpPort(), status));
            verifyNoInteractions(availability);
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void unavailableWorkerRejectionUsesTheSameCapacityWaitPredicate(RoleType role) {
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
            WorkerEndpoint endpoint = directory.exactEndpoint(role, status.getIpPort(), status);
            DecodeEndpoint.ReservationHandle reservation = null;
            if (endpoint instanceof DecodeEndpoint decode) {
                try (var pin = decode.tryPinSchedulingGeneration()) {
                    reservation = decode.reserve(pin, 42L, 1L, 1L, 40);
                }
            }
            when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                    .thenReturn(CompletableFuture.failedFuture(new IllegalStateException("connection unavailable")));
            for (int index = 3; index <= 5; index++) {
                poll(status, directory, grpc, Runnable::run, index);
            }
            assertFalse(status.isAvailableForScheduling());
            if (endpoint instanceof PrefillEndpoint prefill) {
                assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, prefill.reserveBatch(null, 1L, 2).status());
                assertFalse(prefill.batchAdmissionAvailability(2).isAvailable(),
                        "free batch slots cannot trigger retry for an unavailable worker");
            } else {
                DecodeEndpoint decode = (DecodeEndpoint) endpoint;
                DecodeEndpoint.AdmissionCapacity capacity = new DecodeEndpoint.AdmissionCapacity(0L, 100L);
                assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                        decode.acquireDispatchPermit(reservation, capacity).status());
                assertFalse(decode.shouldRetryDispatch(42L, capacity),
                        "free Decode capacity cannot trigger retry for an unavailable worker");
                decode.release(reservation, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
                assertTrue(decode.shouldRetryDispatch(42L, capacity),
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
            WorkerEndpoint endpoint = directory.exactEndpoint(RoleType.DECODE, status.getIpPort(), status);
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
    void queuedApplicationBlocksNewRpcsAndKeepsCommittedStateSchedulable() {
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
            WorkerEndpoint endpoint = directory.exactEndpoint(RoleType.DECODE, status.getIpPort(), status);
            WorkerStatus.CommittedWorkerStatus committed = status.committedWorkerStatus();
            poll(status, directory, grpc, applications::add, 2L);
            for (int index = 3; index <= 7; index++) {
                assertNull(status.tryBeginStatusPoll(index), "do not fetch another response while the callback waits");
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
            assertEquals(3L, status.appliedStatusCursor().statusVersion());
            assertEquals(30L, status.appliedStatusCursor().latestFinishedTaskVersion());
            assertEquals(700L, status.committedEngineObservation().availableKvCacheTokens());
            ArgumentCaptor<Long> requestedFinishedVersions = ArgumentCaptor.forClass(Long.class);
            verify(grpc, org.mockito.Mockito.times(3)).getWorkerStatusAsync(anyString(), anyInt(),
                    requestedFinishedVersions.capture(), anyLong(), any());
            assertEquals(java.util.List.of(-1L, 10L, 20L),
                    requestedFinishedVersions.getAllValues(), "each RPC starts from the previous committed cursor");
        } finally {
            registry.close();
        }
    }

    private static void poll(WorkerStatus status, WorkerDirectory directory, EngineGrpcService grpc,
                             java.util.concurrent.Executor applicationExecutor, long cycle) {
        new GrpcWorkerStatusRunner("model", status.getIpPort(), "test", status.getRole(), status,
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
            new GrpcWorkerStatusRunner("model", status.getIpPort(), "test", RoleType.DECODE,
                    status, status.tryBeginStatusPoll(index), directory, mock(EngineHealthReporter.class), grpc,
                    1000L, mock(CacheAwareService.class), Runnable::run).run();
        }
        assertTrue(status.isActiveGeneration());
        org.junit.jupiter.api.Assertions.assertEquals(0, status.pollHealth().consecutiveTransportFailures());
        org.junit.jupiter.api.Assertions.assertEquals("OK", status.trace().snapshot().statusPoll().rpcResult());
        assertSame(status, directory.statusSnapshot(RoleType.DECODE).get(status.getIpPort()));
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
        var rpc = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(rpc);
        java.util.List<Runnable> queue = new java.util.ArrayList<>();
        new GrpcWorkerStatusRunner("test-model", status.getIpPort(), "test-site",
                RoleType.DECODE, status, lease, directory,
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
        assertFalse(status.trace().snapshot().statusPoll().leaseReleased(), "the poll owns its slot until local application finishes");
        assertNull(status.tryBeginStatusPoll(2L));
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
            org.mockito.Mockito.doThrow(failure).when(reporter).reportStatusCheckerFail(anyString(), any(), any());
        }
        java.util.concurrent.Executor executor = failurePoint.equals("CALLBACK_REJECTED")
                ? task -> { throw new java.util.concurrent.RejectedExecutionException("pool full"); }
                : Runnable::run;
        var runner = new GrpcWorkerStatusRunner("test-model", status.getIpPort(), "test", RoleType.DECODE, status, lease, directory(mock(EndpointRegistry.class), status), reporter, grpc,
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
            case "CALLBACK_REJECTED" -> "CALLBACK_FAILED";
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

        new GrpcWorkerStatusRunner("test-model", status.getIpPort(), "test-site",
                RoleType.DECODE, status, status.tryBeginStatusPoll(1L), directory,
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
                            || !message.contains(" endpoint=" + status.getIpPort() + " ")) {
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
    void consecutiveTimeoutsProduceCorrelatedRetirementHistory() throws Exception {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.PREFILL, "example-group", "127.0.0.1", 8080, 8081, "example-site");
        RetirementLogCapture logs = new RetirementLogCapture(status);
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        directory.currentOrDiscover(status.getRole(), status.getIpPort(), () -> status);
        CacheAwareService cache = mock(CacheAwareService.class);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.PREFILL.getCode())
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                        .setStatusVersion(1).setAlive(true).build()));
        new GrpcWorkerStatusRunner("example-model", status.getIpPort(), "example-site",
                RoleType.PREFILL, status, status.tryBeginStatusPoll(1L), directory,
                mock(EngineHealthReporter.class), grpc, 30L, cache, Runnable::run).run();

        var timer = java.util.concurrent.Executors.newSingleThreadScheduledExecutor();
        WorkerRetirementCleaner cleaner = new WorkerRetirementCleaner(cache, directory);
        try {
            for (int attempt = 1; attempt <= 3; attempt++) {
                var rpc = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
                when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                        .thenReturn(rpc);
                var finished = new CompletableFuture<Void>();
                var lease = status.tryBeginStatusPoll(1L);
                assertNotNull(lease);
                lease.pollTrace().record(Stage.RUNNER_SUBMITTED);
                new GrpcWorkerStatusRunner("example-model", status.getIpPort(), "example-site",
                        RoleType.PREFILL, status, lease, directory,
                        mock(EngineHealthReporter.class), grpc, 30L, cache,
                        Runnable::run).run();
                timer.schedule(() -> {
                    rpc.completeExceptionally(io.grpc.Status.DEADLINE_EXCEEDED
                            .withDescription("simulated 30ms RPC deadline").asRuntimeException());
                    finished.complete(null);
                }, 30L, java.util.concurrent.TimeUnit.MILLISECONDS);
                finished.get(5, java.util.concurrent.TimeUnit.SECONDS);
                cleaner.retireUnavailableWorkers();
            }
            assertTrue(directory.statusSnapshot(RoleType.PREFILL).isEmpty());
            verify(cache).removeEngineBlockCache(
                    org.mockito.Mockito.same(status),
                    org.mockito.Mockito.same(status.trace()));
            String json = logs.awaitFirst();
            assertTrue(json.contains("DEADLINE_EXCEEDED"));
            assertTrue(json.contains("CONSECUTIVE_STATUS_FAILURES"));
            assertTrue(json.contains("failures=3"));
            var snapshot = status.trace().snapshot();
            org.junit.jupiter.api.Assertions.assertEquals(3, snapshot.consecutiveFailures());
            assertTrue(snapshot.statusPoll().leaseReleased());
            assertEquals(-1L, snapshot.statusPoll().times().callbackEndMs(), "failed RPC health is recorded without local application");
        } finally {
            timer.shutdownNow();
            logs.close();
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
        directory.currentOrDiscover(status.getRole(), status.getIpPort(), () -> status);
        var lease = status.tryBeginStatusPoll(1L);
        var rpc = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(rpc);
        java.util.List<Runnable> queue = new java.util.ArrayList<>();
        CacheAwareService cache = mock(CacheAwareService.class);
        new GrpcWorkerStatusRunner("test-model", status.getIpPort(), "test-site",
                RoleType.DECODE, status, lease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L, cache, queue::add).run();
        rpc.complete(EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.DECODE.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .setStatusVersion(1).setAlive(true).build());
        var queuedPoll = status.trace().snapshot().statusPoll();
        assertTrue(queuedPoll.times().rpcResultMs() >= 0);
        assertEquals("CALLBACK_READY", queuedPoll.stage());
        assertFalse(queuedPoll.leaseReleased(), "backpressure includes callback queueing");
        assertNull(status.tryBeginStatusPoll(2L));
        assertEquals(-1L, queuedPoll.times().leaseEndMs());
        assertEquals(-1L, queuedPoll.times().callbackStartMs(), "the queued callback has not started");
        assertEquals(-1L, queuedPoll.times().workerAttemptMs(), "the queued callback has not attempted the Worker lock");
        assertEquals(1, queue.size());
        var logger = mock(org.slf4j.Logger.class);
        status.lock.lock();
        EndpointRegistry.DetachedGeneration detached;
        try {
            detached = directory.beginRetirement(RoleType.DECODE, status.getIpPort(), status,
                    "DISCOVERY_TOPOLOGY_CHANGED");
        } finally {
            status.lock.unlock();
        }
        try (logs) {
            directory.completeRetirement(RoleType.DECODE, status.getIpPort(), status, detached, cache, logger);
            String json = logs.awaitFirst();
            assertTrue(json.contains("rpc:OK"));
            assertTrue(json.contains("stage:CALLBACK_READY"));
            assertTrue(json.contains("released:false"));
            assertTrue(json.contains("DISCOVERY_TOPOLOGY_CHANGED"));
            assertTrue(!json.contains("RPC_FAILED"), "a queued successful RPC must not be reported as a transport failure");
            queue.removeFirst().run();
            assertTrue(status.trace().snapshot().statusPoll().leaseReleased());
            assertNull(registry.get(RoleType.DECODE, status.getIpPort()), "late callback must not resurrect a retired worker");
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
        var rpc = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any())).thenReturn(rpc);
        var callback = new java.util.concurrent.atomic.AtomicReference<Thread>();
        new GrpcWorkerStatusRunner("test-model", status.getIpPort(), "test-site", RoleType.DECODE, status, lease, directory, mock(EngineHealthReporter.class), grpc, 5_000L,
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
            assertNull(status.tryBeginStatusPoll(2L), "waiting for the Worker lock still owns this poll");
            var snapshot = status.trace().snapshot().statusPoll();
            org.junit.jupiter.api.Assertions.assertEquals("STATUS_LOCK_ATTEMPT", snapshot.stage());
            org.junit.jupiter.api.Assertions.assertEquals("OK", snapshot.rpcResult());
            assertTrue(snapshot.times().callbackStartMs() >= 0);
            assertTrue(snapshot.times().workerAttemptMs() >= 0);
            org.junit.jupiter.api.Assertions.assertEquals(-1, snapshot.times().workerAcquiredMs());
            org.junit.jupiter.api.Assertions.assertEquals(-1, snapshot.times().commitMs());
        } finally {
            status.lock.unlock();
            if (callback.get() != null) {callback.get().join(3000);}
        }
        assertTrue(status.trace().snapshot().statusPoll().leaseReleased());
        assertNotNull(registry.get(RoleType.DECODE, status.getIpPort()));
    }

    @ParameterizedTest
    @CsvSource({"DECODE,false", "PREFILL,true", "PDFUSION,true"})
    void repeatedRpcFailuresRetireWorkerAndOnlyClearDetailedCacheIndexes(
            RoleType role, boolean needsCacheKeys) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                role, null, "127.0.0.1", 8080, 8081, "test-site");
        WorkerDirectory directory = directory(registry, status);
        WorkerEndpoint endpoint = RunnerTestSupport.publishEndpoint(
                registry, role, status.getIpPort(), status);
        CacheAwareService cache = mock(CacheAwareService.class);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.failedFuture(
                        io.grpc.Status.UNAVAILABLE.asRuntimeException()));

        WorkerRetirementCleaner cleaner = new WorkerRetirementCleaner(cache, directory);
        for (int failure = 1; failure <= 3; failure++) {
            WorkerStatus.PollLease lease = status.tryBeginStatusPoll(1L);
            assertNotNull(lease);
            new GrpcWorkerStatusRunner(
                    "test-model", status.getIpPort(), "test-site", role,
                    status, lease, directory, mock(EngineHealthReporter.class),
                    grpc, 5_000L, cache, Runnable::run).run();
            cleaner.retireUnavailableWorkers();
            if (failure < 3) {
                assertSame(status, directory.statusSnapshot(role).get(status.getIpPort()));
                assertSame(endpoint, registry.get(role, status.getIpPort()));
                verifyNoInteractions(cache);
            }
        }

        assertNull(registry.get(role, status.getIpPort()));
        assertTrue(directory.statusSnapshot(role).isEmpty());
        var snapshot = status.trace().snapshot();
        org.junit.jupiter.api.Assertions.assertEquals(3, snapshot.consecutiveFailures());
        org.junit.jupiter.api.Assertions.assertEquals("CONSECUTIVE_STATUS_FAILURES", snapshot.unavailableReason());
        assertNull(status.tryBeginStatusPoll(1L), "retired generation cannot poll again");
        if (needsCacheKeys) {
            verify(cache).removeEngineBlockCache(
                    org.mockito.Mockito.same(status),
                    org.mockito.Mockito.same(status.trace()));
        } else {
            verifyNoInteractions(cache);
        }
    }

    @Test
    void newGenerationProjectionRunsOutsideWorkerStatusLock() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1",
                8080, 8081, "test-site");
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll(19L);
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
                "test-model", ipPort, "test-site", RoleType.DECODE,
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
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = RunnerTestSupport.alive(
                RoleType.DECODE, null, "127.0.0.1",
                8080, 8081, "test-site");
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll(1L);
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
                "test-model", ipPort, "test-site", RoleType.DECODE,
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
        WorkerStatus.PollLease nextPoll = status.tryBeginStatusPoll(1L);
        assertNotNull(nextPoll, "the asynchronous owner must close the exact poll lease");
        nextPoll.close();
    }

    private static WorkerDirectory directory(
            EndpointRegistry registry, WorkerStatus status) {
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        directory.currentOrDiscover(
                status.getRole(), status.getIpPort(), () -> status);
        return directory;
    }
}

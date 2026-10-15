package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

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
    @org.junit.jupiter.params.provider.ValueSource(strings = {
            "RPC_THROW", "RPC_FAILED", "CONVERSION_FAILED", "CALLBACK_REJECTED", "CALLBACK_FAILED"})
    void failurePathsReleaseStatusPollAndPreserveBoundary(String failurePoint) {
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
        if (failurePoint.equals("CALLBACK_FAILED")) {
            org.mockito.Mockito.doThrow(failure).when(reporter).reportStatusCheckerFail(anyString(), any(), any());
        }
        java.util.concurrent.Executor executor = failurePoint.equals("CALLBACK_REJECTED")
                ? task -> { throw new java.util.concurrent.RejectedExecutionException("pool full"); }
                : Runnable::run;
        var runner = new GrpcWorkerStatusRunner("test-model", status.getIpPort(), "test", RoleType.DECODE,
                null, status, lease, directory(mock(EndpointRegistry.class), status), reporter, grpc,
                1000L, mock(CacheAwareService.class), executor);
        if (failurePoint.equals("RPC_THROW")) {
            assertSame(failure, org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, runner::run));
        } else {
            runner.run();
        }
        var snapshot = status.trace().snapshot().poll();
        String expectedFailure = switch (failurePoint) {
            case "RPC_THROW" -> "RPC_INITIATION_FAILED";
            case "CONVERSION_FAILED" -> "CALLBACK_READY";
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
                RoleType.DECODE, null, status, status.tryBeginStatusPoll(1L), directory,
                mock(EngineHealthReporter.class), grpc, 5000L,
                mock(CacheAwareService.class), Runnable::run).run();

        org.junit.jupiter.api.Assertions.assertEquals("STATUS_RESPONSE_FAILED",
                status.trace().snapshot().poll().failureStage());
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
        WorkerDirectory directory = new WorkerDirectory(registry);
        directory.currentOrDiscover(status.getRole(), status.getIpPort(), () -> status);
        CacheAwareService cache = mock(CacheAwareService.class);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.PREFILL.getCode())
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                        .setStatusVersion(1).setAlive(true).build()));
        new GrpcWorkerStatusRunner("example-model", status.getIpPort(), "example-site",
                RoleType.PREFILL, "example-group", status, status.tryBeginStatusPoll(1L), directory,
                mock(EngineHealthReporter.class), grpc, 30L, cache, Runnable::run).run();

        var timer = java.util.concurrent.Executors.newSingleThreadScheduledExecutor();
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
                        RoleType.PREFILL, "example-group", status, lease, directory,
                        mock(EngineHealthReporter.class), grpc, 30L, cache,
                        task -> timer.schedule(() -> {
                            try {
                                task.run();
                                finished.complete(null);
                            } catch (Throwable failure) {
                                finished.completeExceptionally(failure);
                            }
                        }, 20L, java.util.concurrent.TimeUnit.MILLISECONDS)).run();
                timer.schedule(() -> rpc.completeExceptionally(io.grpc.Status.DEADLINE_EXCEEDED
                        .withDescription("simulated 30ms RPC deadline").asRuntimeException()),
                        30L, java.util.concurrent.TimeUnit.MILLISECONDS);
                finished.get(5, java.util.concurrent.TimeUnit.SECONDS);
            }
            assertTrue(directory.statusSnapshot(RoleType.PREFILL).isEmpty());
            verify(cache).removeEngineBlockCache(
                    org.mockito.Mockito.eq(status.getIpPort()),
                    org.mockito.Mockito.same(status.trace()));
            String json = logs.awaitFirst();
            assertTrue(json.contains("DEADLINE_EXCEEDED"));
            assertTrue(json.contains("CONSECUTIVE_STATUS_FAILURES"));
            assertTrue(json.contains("failures=3"));
            var snapshot = status.trace().snapshot();
            org.junit.jupiter.api.Assertions.assertEquals(3, snapshot.consecutiveFailures());
            assertTrue(snapshot.poll().leaseReleased());
            assertTrue(snapshot.poll().times().callbackEndMs() >= 0);
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
        WorkerDirectory directory = new WorkerDirectory(registry);
        directory.currentOrDiscover(status.getRole(), status.getIpPort(), () -> status);
        var lease = status.tryBeginStatusPoll(1L);
        var rpc = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(rpc);
        java.util.List<Runnable> queue = new java.util.ArrayList<>();
        CacheAwareService cache = mock(CacheAwareService.class);
        new GrpcWorkerStatusRunner("test-model", status.getIpPort(), "test-site",
                RoleType.DECODE, null, status, lease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L, cache, queue::add).run();
        rpc.complete(EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.DECODE.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .setStatusVersion(1).setAlive(true).build());
        assertTrue(status.trace().snapshot().poll().times().rpcResultMs() >= 0);
        org.junit.jupiter.api.Assertions.assertEquals("CALLBACK_READY",
                status.trace().snapshot().poll().stage());
        var logger = mock(org.slf4j.Logger.class);
        status.lock.lock();
        EndpointRegistry.DetachedGeneration detached;
        try {
            detached = directory.beginRetirement(RoleType.DECODE, status.getIpPort(), status,
                    "STATUS_EXPIRED", 10_000_000L);
        } finally {
            status.lock.unlock();
        }
        try (logs) {
            directory.completeRetirement(RoleType.DECODE, status.getIpPort(), status, detached, cache, logger);
            String json = logs.awaitFirst();
            assertTrue(json.contains("rpc:OK"));
            assertTrue(json.contains("CALLBACK_READY"));
            assertTrue(json.contains("STATUS_EXPIRED"));
            assertTrue(!json.contains("RPC_FAILED"), "a queued successful RPC must not be reported as a transport failure");
            queue.removeFirst().run();
            assertTrue(status.trace().snapshot().poll().leaseReleased());
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
        new GrpcWorkerStatusRunner("test-model", status.getIpPort(), "test-site", RoleType.DECODE,
                null, status, lease, directory, mock(EngineHealthReporter.class), grpc, 5_000L,
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
            var snapshot = status.trace().snapshot().poll();
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
        assertTrue(status.trace().snapshot().poll().leaseReleased());
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

        for (int failure = 1; failure <= 3; failure++) {
            WorkerStatus.PollLease lease = status.tryBeginStatusPoll(1L);
            assertNotNull(lease);
            new GrpcWorkerStatusRunner(
                    "test-model", status.getIpPort(), "test-site", role, null,
                    status, lease, directory, mock(EngineHealthReporter.class),
                    grpc, 5_000L, cache, Runnable::run).run();
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
                    org.mockito.Mockito.eq(status.getIpPort()),
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
                "test-model", ipPort, "test-site", RoleType.DECODE, null,
                status, pollLease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), Runnable::run)
                .run();

        assertTrue(publicationHeldLock.get(),
                "publication must commit under the generation lock");
        assertTrue(projectionReleasedLock.get(),
                "endpoint facts must project after releasing that lock");
        var poll = status.trace().snapshot().poll();
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
        WorkerStatus.PollLease nextPoll = status.tryBeginStatusPoll(1L);
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

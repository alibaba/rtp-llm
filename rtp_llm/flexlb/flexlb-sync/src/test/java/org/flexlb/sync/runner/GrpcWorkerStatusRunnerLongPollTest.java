package org.flexlb.sync.runner;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/**
 * Long-poll re-arm chain of {@link GrpcWorkerStatusRunner}: with long-poll
 * enabled the next status poll is launched as soon as the previous response
 * arrives (response-driven resident loop) instead of waiting for the fixed
 * SYNC_STATUS_INTERVAL tick, and statusCheckInProgress stays true across the
 * hand-off so the periodic loop cannot double-poll. Failure / stale / disabled
 * paths break the chain and return ownership to the periodic loop.
 */
class GrpcWorkerStatusRunnerLongPollTest {

    private final EngineGrpcService engineGrpcService = Mockito.mock(EngineGrpcService.class);
    private final EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
    private final ScheduledExecutorService rearmScheduler = Executors.newScheduledThreadPool(2);

    @AfterEach
    void tearDown() throws InterruptedException {
        rearmScheduler.shutdownNow();
        rearmScheduler.awaitTermination(2, TimeUnit.SECONDS);
    }

    @Test
    void should_rearm_next_poll_immediately_after_response() throws Exception {
        WorkerStatus status = status();
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put("127.0.0.1:8080", status);
        AtomicInteger calls = new AtomicInteger();
        AtomicLong seenWaitTimeout = new AtomicLong(-1);
        AtomicLong seenRequestTimeout = new AtomicLong(-1);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                any(RoleType.class), anyLong()))
                .thenAnswer(invocation -> {
                    calls.incrementAndGet();
                    seenRequestTimeout.set(invocation.getArgument(3));
                    seenWaitTimeout.set(invocation.getArgument(5));
                    return CompletableFuture.completedFuture(alivePb(100L));
                });

        GrpcWorkerStatusRunner runner = runner(status, statuses,
                new StatusLongPollConfig(true, 1000L, rearmScheduler));
        status.getStatusCheckInProgress().set(true);
        runner.run();

        // Response-driven chain: with a 1ms re-arm delay and instant stub
        // responses, far more polls than any 20ms fixed tick could produce.
        Thread.sleep(300);
        assertTrue(calls.get() >= 5,
                "chain must re-issue polls on response arrival, got " + calls.get());
        assertTrue(status.getStatusCheckInProgress().get(),
                "in-progress flag must stay true across the re-arm hand-off");
        assertEquals(1000L, seenWaitTimeout.get(), "request must carry wait_timeout_ms");
        assertEquals(20L + 1000L, seenRequestTimeout.get(),
                "gRPC deadline must cover the parked long-poll time");
    }

    @Test
    void should_wait_for_delayed_response_then_rearm() throws Exception {
        WorkerStatus status = status();
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put("127.0.0.1:8080", status);
        AtomicInteger calls = new AtomicInteger();
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                any(RoleType.class), anyLong()))
                .thenAnswer(invocation -> {
                    calls.incrementAndGet();
                    // Simulate an engine that parks the long-poll for 120ms.
                    CompletableFuture<EngineRpcService.WorkerStatusPB> future = new CompletableFuture<>();
                    rearmScheduler.schedule(() -> future.complete(alivePb(100L)),
                            120, TimeUnit.MILLISECONDS);
                    return future;
                });

        GrpcWorkerStatusRunner runner = runner(status, statuses,
                new StatusLongPollConfig(true, 1000L, rearmScheduler));
        status.getStatusCheckInProgress().set(true);
        runner.run();

        // While the poll is parked there must be exactly one in-flight request
        // (no fixed-interval churn) and the flag guards against re-entry.
        Thread.sleep(60);
        assertEquals(1, calls.get(), "no extra polls while the response is parked");
        assertTrue(status.getStatusCheckInProgress().get());

        // Once the response lands, the next poll follows within milliseconds.
        Thread.sleep(120);
        assertTrue(calls.get() >= 2, "next poll must launch right after the response");
    }

    @Test
    void should_not_rearm_when_long_poll_disabled() throws Exception {
        WorkerStatus status = status();
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put("127.0.0.1:8080", status);
        AtomicInteger calls = new AtomicInteger();
        AtomicLong seenWaitTimeout = new AtomicLong(-1);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                any(RoleType.class), anyLong()))
                .thenAnswer(invocation -> {
                    calls.incrementAndGet();
                    seenWaitTimeout.set(invocation.getArgument(5));
                    return CompletableFuture.completedFuture(alivePb(100L));
                });

        GrpcWorkerStatusRunner runner = runner(status, statuses,
                new StatusLongPollConfig(false, 1000L, rearmScheduler));
        status.getStatusCheckInProgress().set(true);
        runner.run();

        Thread.sleep(100);
        assertEquals(1, calls.get(), "disabled long-poll must fall back to one-shot polling");
        assertFalse(status.getStatusCheckInProgress().get(),
                "flag must return to the periodic loop when the chain is off");
        assertEquals(0L, seenWaitTimeout.get(), "request must not ask the engine to park");
    }

    @Test
    void should_break_chain_on_failure_and_return_to_periodic_loop() throws Exception {
        WorkerStatus status = status();
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put("127.0.0.1:8080", status);
        AtomicInteger calls = new AtomicInteger();
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                any(RoleType.class), anyLong()))
                .thenAnswer(invocation -> {
                    calls.incrementAndGet();
                    return CompletableFuture.failedFuture(new RuntimeException("unavailable"));
                });

        GrpcWorkerStatusRunner runner = runner(status, statuses,
                new StatusLongPollConfig(true, 1000L, rearmScheduler));
        status.getStatusCheckInProgress().set(true);
        runner.run();

        Thread.sleep(100);
        assertEquals(1, calls.get(), "failures must not re-arm (periodic loop is the backoff)");
        assertFalse(status.getStatusCheckInProgress().get());
    }

    @Test
    void should_break_chain_when_worker_generation_is_stale() throws Exception {
        WorkerStatus expired = status();
        WorkerStatus current = status();
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put("127.0.0.1:8080", current);
        AtomicInteger calls = new AtomicInteger();
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                any(RoleType.class), anyLong()))
                .thenAnswer(invocation -> {
                    calls.incrementAndGet();
                    return CompletableFuture.completedFuture(alivePb(100L));
                });

        GrpcWorkerStatusRunner runner = runner(expired, statuses,
                new StatusLongPollConfig(true, 1000L, rearmScheduler));
        expired.getStatusCheckInProgress().set(true);
        runner.run();

        Thread.sleep(100);
        assertEquals(1, calls.get(), "stale generation must not keep its own chain alive");
        assertFalse(expired.getStatusCheckInProgress().get());
    }

    private GrpcWorkerStatusRunner runner(WorkerStatus status,
                                          Map<String, WorkerStatus> statuses,
                                          StatusLongPollConfig longPollConfig) {
        return new GrpcWorkerStatusRunner(
                "test-model", "127.0.0.1:8080", "test-site", RoleType.PREFILL, "test-group",
                status, statuses, engineHealthReporter, engineGrpcService,
                20L, null, null, Runnable::run, longPollConfig);
    }

    private static WorkerStatus status() {
        WorkerStatus status = new WorkerStatus();
        status.setRole(RoleType.PREFILL);
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setAlive(true);
        return status;
    }

    private static EngineRpcService.WorkerStatusPB alivePb(long version) {
        return EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setStatusVersion(version)
                .setAlive(true)
                .build();
    }
}

package org.flexlb.sync.runner;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.LongAdder;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class PollRunnerLifecycleTest {
    @ParameterizedTest
    @CsvSource({"false,SEND", "true,SEND", "false,REPLY", "true,REPLY",
            "false,EXECUTOR", "true,EXECUTOR", "false,CALLBACK", "true,CALLBACK"})
    void failedPollReleasesExactLeaseAndUsesGenerationAddress(boolean cachePoll, String failureAt) {
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 19000, "site");
        var registry = RunnerTestSupport.endpointRegistry(mock(ConfigService.class));
        registry.currentOrDiscover(RoleType.DECODE, status.getIpPort(), () -> status);
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        CacheAwareService cache = mock(CacheAwareService.class);
        DynamicCacheIntervalService interval = mock(DynamicCacheIntervalService.class);
        IllegalStateException failure = new IllegalStateException("poll unavailable");
        EngineHealthReporter reporter = failureAt.equals("CALLBACK")
                ? mock(EngineHealthReporter.class, invocation -> { throw failure; })
                : mock(EngineHealthReporter.class);
        CompletableFuture<EngineRpcService.WorkerStatusPB> statusReply = new CompletableFuture<>();
        CompletableFuture<EngineRpcService.CacheStatusPB> cacheReply = new CompletableFuture<>();
        if (cachePoll) {
            var call = when(grpc.getCacheStatusAsync("127.0.0.1", 19000, -1L, 500L, RoleType.DECODE));
            if (failureAt.equals("SEND")) { call.thenThrow(failure); }
            else { call.thenReturn(cacheReply); }
        } else {
            var call = when(grpc.getWorkerStatusAsync("127.0.0.1", 19000,
                    status.appliedStatusCursor().latestFinishedTaskVersion(), 500L, RoleType.DECODE));
            if (failureAt.equals("SEND")) { call.thenThrow(failure); }
            else { call.thenReturn(statusReply); }
        }
        Executor callbacks = failureAt.equals("EXECUTOR")
                ? task -> { throw new RejectedExecutionException("closed"); } : Runnable::run;
        WorkerStatus.PollLease lease = acquire(status, cachePoll);
        assertNotNull(lease);
        Runnable runner = cachePoll
                ? new GrpcCacheStatusCheckRunner("model", status, lease, registry, reporter, grpc,
                        cache, interval, 500L, new LongAdder(), 50L, false, callbacks)
                : new GrpcWorkerStatusRunner("model", status, lease, registry, reporter, grpc,
                        500L, cache, callbacks);
        if (failureAt.equals("SEND")) {
            assertSame(failure, assertThrows(IllegalStateException.class, runner::run));
        } else {
            runner.run();
            assertNull(acquire(status, cachePoll), "an outstanding RPC retains the poll lease");
            if (cachePoll) { cacheReply.completeExceptionally(failure); }
            else { statusReply.completeExceptionally(failure); }
        }
        try (WorkerStatus.PollLease next = acquire(status, cachePoll)) {
            assertNotNull(next, "every failed poll must permit the next poll");
        }
        if (failureAt.equals("CALLBACK")) {
            if (cachePoll) {
                verify(reporter).reportCacheStatusCheckerFail(eq("model"), any(), eq(RoleType.DECODE));
            } else {
                verify(reporter).reportStatusCheckerFail(eq("model"), any(), eq(RoleType.DECODE));
            }
        }
        if (!cachePoll && (failureAt.equals("REPLY") || failureAt.equals("CALLBACK"))) {
            assertEquals(1L, status.pollHealth().consecutiveTransportFailures(),
                    "the original transport failure must be recorded even when its report fails");
        }
        if (cachePoll) {
            verify(grpc).getCacheStatusAsync("127.0.0.1", 19000, -1L, 500L, RoleType.DECODE);
            verifyNoInteractions(cache);
        } else {
            verify(grpc).getWorkerStatusAsync("127.0.0.1", 19000,
                    status.appliedStatusCursor().latestFinishedTaskVersion(), 500L, RoleType.DECODE);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cachePreparationReleasesLeaseWithoutStartingRpc(boolean skip) {
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.PREFILL, null, "127.0.0.1", 8080, 19000, "site");
        var registry = RunnerTestSupport.endpointRegistry(mock(ConfigService.class));
        WorkerStatusRpcClient grpc = mock(WorkerStatusRpcClient.class);
        DynamicCacheIntervalService interval = mock(DynamicCacheIntervalService.class);
        IllegalStateException failure = new IllegalStateException("interval unavailable");
        if (skip) { when(interval.getCurrentIntervalMs()).thenReturn(100L); }
        else { when(interval.getCurrentIntervalMs()).thenThrow(failure); }
        LongAdder ticks = new LongAdder();
        ticks.increment();
        Runnable runner = new GrpcCacheStatusCheckRunner("model", status, acquire(status, true),
                registry, mock(EngineHealthReporter.class), grpc, mock(CacheAwareService.class),
                interval, 500L, ticks, 50L, false, Runnable::run);

        if (skip) { runner.run(); }
        else { assertSame(failure, assertThrows(IllegalStateException.class, runner::run)); }

        verifyNoInteractions(grpc);
        try (WorkerStatus.PollLease next = acquire(status, true)) {
            assertNotNull(next);
        }
    }

    private static WorkerStatus.PollLease acquire(WorkerStatus status, boolean cachePoll) {
        return cachePoll ? status.tryBeginCachePoll() : status.tryBeginStatusPoll();
    }
}

package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.awaitCondition;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.*;

class ColdTokenDispatchTest {
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(longs = {-1L, 0L})
    void asynchronousNonBatchReportsCannotBiasColdRouting(long wireBatchId) throws Exception {
        try (var f = new Fixture(false, false, false, true, config -> {
            config.getDispatcher().setMaxInflightPerPrefillWorker(256);
            config.getRouter().getRoles().getPrefill().getTokenBalance().setEngineQueueAdmissionThreshold(256L);
            var balance = config.getRouter().getRoles().getPrefill().getTokenBalance();
            balance.setEnableWindow(false);
            balance.setMaxPendingComputeTokens(2_000_000L);
        })) {
            var workers = List.of(f.a, f.b, (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.3"),
                    (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.4"));
            var tasks = new java.util.ArrayList<java.util.List<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>>();
            for (int w = 0; w < 4; w++) { tasks.add(new java.util.ArrayList<>()); }
            int[] deliveries = new int[4];
            for (int i = 0; i < 256; i++) {
                var ctx = f.context(5_000_000L + i);
                ctx.data().setRequest(ctx.data().getRequest().toBuilder().seqLen(50_000L).build());
                assertTrue(f.scheduler.submit(ctx).get(2, TimeUnit.SECONDS).isSuccess());
                int w = workers.indexOf(ctx.snapshot().route().prefillEp());
                deliveries[w]++;
                tasks.get(w).add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId(ctx.data().getRequestId()).setBatchId(wireBatchId).setInputLength(50_000L)
                        .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build());
            }
            assertArrayEquals(new int[]{64, 64, 64, 64}, deliveries);
            // C++ emits -1 for ordinary requests; older payloads may omit the field (0).
            // Only two polls have arrived when the next burst starts. Service/load is identical.
            for (int round = 0; round < 6; round++) {
                int firstPoll = (round % 2) * 2;
                f.observeTaskSnapshot(workers.get(firstPoll), tasks.get(firstPoll), List.of());
                f.observeTaskSnapshot(workers.get(firstPoll + 1), tasks.get(firstPoll + 1), List.of());
                for (int i = 0; i < 128; i++) {
                    var ctx = f.context(5_100_000L + round * 128L + i);
                    ctx.data().setRequest(ctx.data().getRequest().toBuilder().seqLen(50_000L).build());
                    assertTrue(f.scheduler.submit(ctx).get(2, TimeUnit.SECONDS).isSuccess());
                    int w = workers.indexOf(ctx.snapshot().route().prefillEp());
                    deliveries[w]++;
                    tasks.get(w).add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                            .setRequestId(ctx.data().getRequestId()).setBatchId(wireBatchId).setInputLength(50_000L)
                            .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build());
                }
                int expected = 96 + round * 32;
                assertArrayEquals(new int[]{expected, expected, expected, expected}, deliveries,
                        "poll order must not manufacture load: " + java.util.Arrays.toString(deliveries));
                for (var worker : workers) {
                    assertEquals(expected, worker.ownership.progress().outstandingRequests());
                    assertEquals(expected * 50_000L, worker.ownership.tokenWorkload().pendingTokens());
                }
            }
            var extra = f.context(5_200_000L);
            var response = f.scheduler.submit(extra);
            awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
            assertFalse(response.isDone(), "all four configured 256-request limits are full");
            var workerTasks = tasks.getFirst();
            f.observeTaskSnapshot(workers.getFirst(), workerTasks.subList(1, workerTasks.size()),
                    List.of(workerTasks.getFirst()));
            assertTrue(response.get(2, TimeUnit.SECONDS).isSuccess());
            assertSame(workers.getFirst(), extra.snapshot().route().prefillEp());
            for (var worker : workers) { assertEquals(256, worker.ownership.progress().outstandingRequests()); }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints = {0, 6, 10})
    void lowActualWaitingCanFeedRepeatedSixtyFourRequestPrefillBatches(int waiting) throws Exception {
        try (var f = new Fixture(false, false, false, true, config -> {
            config.getDispatcher().setMaxInflightPerPrefillWorker(256);
            config.getRouter().getRoles().getPrefill().getTokenBalance().setMaxPendingComputeTokens(2_000_000L);
        })) {
            when(f.cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(f.a.ipPort(), 9));
            var background = new java.util.ArrayList<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>();
            for (int i = 0; i < waiting; i++) {
                background.add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                        .setBatchId(-1L).setRequestId(4_000_000L + i).setInputLength(50_000L).setIsWaiting(true)
                        .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED).build());
            }
            f.observeTaskSnapshot(f.a, background, List.of());
            for (int round = 0; round < 4; round++) {
                var responses = new java.util.ArrayList<java.util.concurrent.CompletableFuture<org.flexlb.dao.loadbalance.Response>>();
                var running = new java.util.ArrayList<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>();
                for (int i = 0; i < 64; i++) {
                    long id = 4_100_000L + round * 64 + i;
                    var ctx = f.context(id);
                    ctx.data().setRequest(ctx.data().getRequest().toBuilder().seqLen(50_000L).cacheKeyBlockSize(5000L).build());
                    responses.add(f.scheduler.submit(ctx));
                    running.add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                            .setBatchId(-1L).setRequestId(id).setInputLength(50_000L)
                            .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build());
                }
                // Known running requests release speculative waiting credit on each status round.
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                while (responses.stream().anyMatch(response -> !response.isDone()) && System.nanoTime() < deadline) {
                    Thread.sleep(20L);
                    var feedback = new java.util.ArrayList<>(background);
                    for (int i = 0; i < responses.size(); i++) {
                        if (responses.get(i).isDone()) { feedback.add(running.get(i)); }
                    }
                    f.observeTaskSnapshot(f.a, feedback, List.of());
                }
                java.util.concurrent.CompletableFuture.allOf(responses.toArray(java.util.concurrent.CompletableFuture[]::new))
                        .get(1, TimeUnit.SECONDS);
                assertTrue(responses.stream().allMatch(r -> r.join().isSuccess()));
                assertEquals(64 + waiting, f.a.ownership.progress().outstandingRequests());
                assertEquals(0, f.b.ownership.progress().outstandingRequests(), "hot work keeps its cache leader");
                var observed = new java.util.ArrayList<>(running);
                observed.addAll(background);
                f.observeTaskSnapshot(f.a, observed, List.of());
                f.observeTaskSnapshot(f.a, background, running);
                assertEquals(waiting, f.a.ownership.progress().outstandingRequests());
            }
        }
    }

    @Test
    void decodeAllocationIsNotProofThatAnUnobservedPrefillRequestHasFinished() throws Exception {
        try (var f = new Fixture(false, true, false, true)) {
            var proof = new java.util.concurrent.CompletableFuture<org.flexlb.engine.grpc.EngineGrpcClient.CancelReply>();
            when(f.grpc.cancelAsync(anyString(), anyInt(), any(), anyLong())).thenReturn(proof);
            var ctx = f.context(870_000L);
            assertTrue(f.scheduler.submit(ctx).get(2, TimeUnit.SECONDS).isSuccess());
            var route = ctx.snapshot().route();
            ctx.recordEngineEvidence(new ContextResult.EngineEvidence(route, route.decodeReservation(), RoleType.DECODE,
                    ContextResult.EngineKind.ALLOCATED, 0L, null, System.currentTimeMillis()));
            assertTrue(ctx.snapshot().engine().decodeAccepted());
            f.scheduler.cancel(ctx, 0L, CancelReason.CLIENT_CANCELLED);
            awaitCondition(() -> route.prefillEp().ownership.stats().locallyOwnedRequests() == 0);
            assertEquals(1, route.prefillEp().ownership.progress().outstandingRequests());
            proof.complete(org.flexlb.engine.grpc.EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            awaitCondition(() -> route.prefillEp().ownership.progress().outstandingRequests() == 0);
        }
    }

    @Test
    void localCancellationBeforeFirstEngineReportCannotRecycleInflightSlots() throws Exception {
        try (var f = new Fixture(false, false, false, true,
                config -> config.getDispatcher().setMaxInflightPerPrefillWorker(2))) {
            when(f.cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(f.a.ipPort(), 9));
            var proof = new java.util.concurrent.CompletableFuture<org.flexlb.engine.grpc.EngineGrpcClient.CancelReply>();
            when(f.grpc.cancelAsync(anyString(), anyInt(), any(), anyLong())).thenReturn(proof);
            for (long id = 880_000L; id < 880_002L; id++) {
                var ctx = f.context(id);
                assertTrue(f.scheduler.submit(ctx).get(2, TimeUnit.SECONDS).isSuccess());
                f.scheduler.cancel(ctx, 0L, CancelReason.CLIENT_CANCELLED);
                awaitCondition(() -> f.a.ownership.stats().locallyOwnedRequests() == 0);
            }
            assertEquals(2, f.a.ownership.progress().outstandingRequests());
            var next = f.context(880_002L);
            var response = f.scheduler.submit(next);
            awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
            assertFalse(response.isDone());
            assertEquals(0, f.b.ownership.progress().outstandingRequests(), "cache first must not fall back to an idle peer");
            proof.complete(org.flexlb.engine.grpc.EngineGrpcClient.CancelReply.REQUEST_FENCED);
            assertTrue(response.get(2, TimeUnit.SECONDS).isSuccess());
            assertSame(f.a, next.snapshot().route().prefillEp());
            assertEquals(1, f.a.ownership.progress().outstandingRequests());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"5000,1,10", "5000,4,10", "0,4,10"})
    @org.junit.jupiter.api.Timeout(60)
    void sustainedColdOverloadKeepsEveryWorkerServingWithinHardLimit(long duration, int pollTicks, long threshold) throws Exception {
        try (var f = new Fixture(false, false, false, true, config -> {
            config.getDispatcher().setMaxInflightPerPrefillWorker(256);
            var balance = config.getRouter().getRoles().getPrefill().getTokenBalance();
            balance.setMinPrefixHitPercent(5);
            balance.setEnableWindow(false);
            balance.setMaxPendingComputeTokens(2_000_000L);
            balance.setEngineQueueOverloadDurationMs(duration);
            balance.setEngineQueueAdmissionThreshold(threshold);
        })) {
            var workers = List.of(f.a, f.b, (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.3"),
                    (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.4"));
            var admitted = new java.util.concurrent.ConcurrentLinkedQueue<RequestContext>();
            var responses = new java.util.ArrayList<java.util.concurrent.CompletableFuture<org.flexlb.dao.loadbalance.Response>>();
            var active = new java.util.ArrayList<java.util.ArrayDeque<RequestContext>>();
            var finished = new java.util.ArrayList<java.util.List<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>>();
            for (int w = 0; w < 4; w++) { active.add(new java.util.ArrayDeque<>()); finished.add(new java.util.ArrayList<>()); }
            int[] deliveries = new int[4], completions = new int[4], peaks = new int[4];
            long id = 1_000_000L;
            int tick = 0;
            for (; tick < 1200; tick++) {
                // Eight arrivals versus four completions per tick: twice fleet service capacity.
                if (tick < 300) {
                    for (int n = 0; n < 8; n++) {
                        var ctx = f.context(id++);
                        ctx.data().setRequest(ctx.data().getRequest().toBuilder().seqLen(50_000L).build());
                        var response = f.scheduler.submit(ctx);
                        response.thenAccept(value -> { if (value.isSuccess()) { admitted.add(ctx); } });
                        responses.add(response);
                    }
                }
                RequestContext next;
                while ((next = admitted.poll()) != null) {
                    int w = workers.indexOf(next.snapshot().route().prefillEp());
                    assertTrue(next.snapshot().route().coldTokenBalance());
                    active.get(w).add(next); deliveries[w]++;
                }
                for (int w = 0; w < 4; w++) {
                    peaks[w] = Math.max(peaks[w], active.get(w).size());
                    var done = active.get(w).poll();
                    if (done != null) {
                        completions[w]++;
                        finished.get(w).add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                                .setBatchId(-1L).setRequestId(done.data().getRequestId()).setInputLength(50_000L)
                                .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build());
                    }
                    // Two fast polls and two slower, offset polls; completions remain deltas until observed.
                    if (w < 2 || tick % pollTicks == ((w - 2) * 2) % pollTicks) {
                        var tasks = new java.util.ArrayList<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>();
                        int position = 0;
                        for (var ctx : active.get(w)) {
                            boolean waiting = position++ > 0;
                            tasks.add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                                    .setBatchId(-1L).setRequestId(ctx.data().getRequestId()).setInputLength(50_000L)
                                    .setIsWaiting(waiting).setPhase(waiting
                                            ? org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED
                                            : org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build());
                        }
                        f.observeTaskSnapshot(workers.get(w), tasks, finished.get(w));
                        finished.get(w).clear();
                    }
                    assertTrue(workers.get(w).ownership.progress().outstandingRequests() <= 256);
                }
                if (tick >= 25 && tick < 300 && tick % 25 == 24) {
                    // Compare actual service while traffic is above capacity. Final assignment
                    // totals include unequal tails already sent before traffic stops.
                    for (int w = 0; w < 4; w++) {
                        assertTrue(completions[w] >= tick - 4,
                                "idle worker during sustained pressure at tick " + tick + ": " + java.util.Arrays.toString(completions));
                    }
                    assertTrue(java.util.Arrays.stream(completions).max().orElseThrow()
                            - java.util.Arrays.stream(completions).min().orElseThrow() <= 4,
                            "unbalanced service during pressure: " + java.util.Arrays.toString(completions));
                }
                if (tick >= 300 && java.util.Arrays.stream(completions).sum() == 2400) { break; }
                Thread.sleep(25L);
            }
            assertEquals(2400, java.util.Arrays.stream(completions).sum());
            assertTrue(responses.stream().allMatch(r -> r.isDone() && r.join().isSuccess()));
            assertTrue(java.util.Arrays.stream(peaks).allMatch(peak -> peak <= 256),
                    "reported and unobserved work must share the configured hard limit: " + java.util.Arrays.toString(peaks));
            assertTrue(tick <= 600 + 256 + pollTicks,
                    "drain must stay within fleet service time plus one bounded in-flight tail: " + tick);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void coldRequestDoesNotWaitForWorkerCollectionWindow(boolean batch) throws Exception {
        try (var f = new Fixture(batch)) {
            var request = f.context(1);
            var response = f.scheduler.submit(request).get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(), response.getErrorMessage());
            assertEquals(batch, response.isEnqueuedByMaster());
            assertEquals(0, f.a.ownership.progress().queueDepth());
            assertEquals(0, f.b.ownership.progress().queueDepth());
            assertEquals(1000, f.a.ownership.tokenWorkload().pendingTokens() + f.b.ownership.tokenWorkload().pendingTokens());
            assertEquals(0, f.a.ownership.tokenWorkload().plannedTokens() + f.b.ownership.tokenWorkload().plannedTokens());
        }
    }

    @Test
    void capacityLostAfterSelectionRetriesAnotherWorkerWithoutBindingToTheOldQueue() throws Exception {
        try (var f = new Fixture()) {
            var selected = new CountDownLatch(1);
            var proceed = new CountDownLatch(1);
            var first = new AtomicBoolean(true);
            var selectedWorker = new java.util.concurrent.atomic.AtomicReference<PrefillEndpoint>();
            doAnswer(call -> {
                var result = (org.flexlb.balance.PlacementResult<RequestRoute, PlacementKey>) call.callRealMethod();
                if (result.value() != null && first.getAndSet(false)) {
                    selectedWorker.set(result.value().prefillEp());
                    selected.countDown();
                    assertTrue(proceed.await(3, TimeUnit.SECONDS));
                }
                return result;
            }).when(f.router).select(any(), nullable(String.class));
            var context = f.context(2);
            var response = f.scheduler.submit(context);
            assertTrue(selected.await(2, TimeUnit.SECONDS));
            var oldWorker = selectedWorker.get();
            var reservation = f.occupy(oldWorker, 99);
            proceed.countDown();
            try {
                assertTrue(response.get(2, TimeUnit.SECONDS).isSuccess());
                assertNotSame(oldWorker, context.snapshot().route().prefillEp());
                assertEquals(0, oldWorker.ownership.progress().queueDepth());
                assertEquals(0, oldWorker.ownership.tokenWorkload().plannedTokens());
            } finally { oldWorker.release(reservation, null); }
        }
    }

    @Test
    void waitingColdRequestHasNoWorkerAssignmentAndUsesTheWorkerThatBecomesFree() throws Exception {
        try (var f = new Fixture()) {
            var a = f.occupy(f.a, 90);
            var b = f.occupy(f.b, 91);
            var context = f.context(3);
            var response = f.scheduler.submit(context);
            try {
                awaitCondition(() -> !f.scheduler.getLatestQueueWaitSnapshot().isEmpty()
                        && f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
                assertFalse(response.isDone());
                assertNull(context.snapshot().route());
                f.b.release(b, null);
                assertTrue(response.get(2, TimeUnit.SECONDS).isSuccess());
                assertSame(f.b, context.snapshot().route().prefillEp());
                assertEquals(0, f.a.ownership.progress().queueDepth());
                assertEquals(0, f.b.ownership.progress().queueDepth());
            } finally {
                f.a.release(a, null);
                f.b.release(b, null);
            }
        }
    }

    @Test
    void cancellationWhileWaitingLeavesNoTokenOrWorkerOwnership() throws Exception {
        try (var f = new Fixture()) {
            var a = f.occupy(f.a, 90);
            var b = f.occupy(f.b, 91);
            var context = f.context(4);
            var response = f.scheduler.submit(context);
            try {
                awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
                assertTrue(response.cancel(false));
                f.a.release(a, null);
                f.b.release(b, null);
                awaitCondition(() -> f.scheduler.requests.findActive(4) == null);
                assertEquals(0, f.a.ownership.tokenWorkload().score());
                assertEquals(0, f.b.ownership.tokenWorkload().score());
            } finally {
                f.a.release(a, null);
                f.b.release(b, null);
            }
        }
    }

    @Test
    void batchCapacityReleaseWakesGlobalWaiterWithoutAWorkerQueue() throws Exception {
        try (var f = new Fixture(true)) {
            var one = f.context(10);
            var two = f.context(11);
            assertTrue(f.scheduler.submit(one).get(2, TimeUnit.SECONDS).isSuccess());
            assertTrue(f.scheduler.submit(two).get(2, TimeUnit.SECONDS).isSuccess());
            assertNotSame(one.snapshot().route().prefillEp(), two.snapshot().route().prefillEp());
            var three = f.context(12);
            var future = f.scheduler.submit(three);
            awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
            assertFalse(future.isDone());
            assertNull(three.snapshot().route());
            assertEquals(0, f.a.ownership.progress().queueDepth() + f.b.ownership.progress().queueDepth());
            var released = one.snapshot().route().prefillEp();
            f.scheduler.cancel(10, one.snapshot().delivery().identity().batchId(), CancelReason.CLIENT_CANCELLED);
            assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
            assertSame(released, three.snapshot().route().prefillEp());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void decodeWaitAndAsyncPreemptionRebalancePrefillAtDispatch(boolean preempt) throws Exception {
        try (var f = new Fixture(false, true, preempt)) {
            org.flexlb.balance.endpoint.DecodeResources.ReservationHandle occupied;
            try (var pin = f.decode.tryPinGeneration()) {
                occupied = f.decode.tryReserveQueuedRequest(pin, 99, 1000, 1001, 10, null).handle();
            }
            var firstWorker = new java.util.concurrent.atomic.AtomicReference<PrefillEndpoint>();
            doAnswer(call -> {
                var result = (org.flexlb.balance.PlacementResult<RequestRoute, PlacementKey>) call.callRealMethod();
                if (result.value() != null) { firstWorker.compareAndSet(null, result.value().prefillEp()); }
                return result;
            }).when(f.router).select(any(), nullable(String.class));
            var reclaim = new java.util.concurrent.CompletableFuture<DecodeCapacityAcquirer.PreemptionResult>();
            var reclaimStarted = new CountDownLatch(1);
            when(f.eviction.tryReclaim(any(), any())).thenAnswer(call -> { reclaimStarted.countDown(); return reclaim; });
            var context = f.context(20);
            var future = f.scheduler.submit(context);
            if (preempt) { assertTrue(reclaimStarted.await(2, TimeUnit.SECONDS)); }
            else { awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role")); }
            assertFalse(future.isDone());
            assertNull(context.snapshot().route());
            assertEquals(0, f.a.ownership.tokenWorkload().pendingTokens() + f.b.ownership.tokenWorkload().pendingTokens());
            var old = firstWorker.get();
            var load = f.occupy(old, 98);
            try {
                f.decode.release(occupied, org.flexlb.balance.endpoint.DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
                if (preempt) {
                    org.flexlb.balance.endpoint.DecodeResources.ReservationHandle incoming;
                    try (var pin = f.decode.tryPinGeneration()) {
                        incoming = f.decode.tryReserveQueuedRequest(pin, 20, 1000, 1001, 50, null).handle();
                    }
                    reclaim.complete(new DecodeCapacityAcquirer.PreemptionResult(incoming, false, "committed"));
                }
                assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
                assertNotSame(old, context.snapshot().route().prefillEp());
                assertEquals(0, f.a.ownership.progress().queueDepth() + f.b.ownership.progress().queueDepth());
                assertEquals(0, f.a.ownership.tokenWorkload().plannedTokens() + f.b.ownership.tokenWorkload().plannedTokens());
                assertEquals(1, f.decode.routingView().engineCapacityUsed());
            } finally { old.release(load, null); }
        }
    }

    @Test
    void batchExecutorCapacityWaitDoesNotBindWorkerAndReleaseWakesIt() throws Exception {
        try (var f = new Fixture(true)) {
            var permits = new java.util.ArrayList<DeliveryExecution.DeliveryPermit>();
            try {
                while (true) {
                    var attempt = f.runtime.deliveries().tryReserveDelivery();
                    if (!attempt.accepted()) { break; }
                    permits.add(attempt.value());
                    assertTrue(permits.size() < 1024);
                }
                var request = f.context(30);
                var future = f.scheduler.submit(request);
                awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
                assertFalse(future.isDone());
                assertNull(request.snapshot().route());
                assertEquals(0, f.a.ownership.tokenWorkload().pendingTokens() + f.b.ownership.tokenWorkload().pendingTokens());
                permits.removeLast().close();
                assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
            } finally { permits.forEach(DeliveryExecution.DeliveryPermit::close); }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void cancellationAfterSelectionRollsBackImmediateAdmission(boolean batch) throws Exception {
        try (var f = new Fixture(batch)) {
            var selected = new CountDownLatch(1);
            var proceed = new CountDownLatch(1);
            doAnswer(call -> {
                var result = call.callRealMethod();
                selected.countDown();
                assertTrue(proceed.await(3, TimeUnit.SECONDS));
                return result;
            }).when(f.router).select(any(), nullable(String.class));
            var context = f.context(31);
            var future = f.scheduler.submit(context);
            assertTrue(selected.await(2, TimeUnit.SECONDS));
            try { assertTrue(future.cancel(false)); }
            finally { proceed.countDown(); }
            awaitCondition(() -> f.scheduler.requests.findActive(31) == null);
            awaitCondition(() -> f.a.ownership.tokenWorkload().score() + f.b.ownership.tokenWorkload().score() == 0);
            assertEquals(0, f.a.ownership.stats().batchCount() + f.b.ownership.stats().batchCount());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void lowEngineQueueDeliversColdAndCacheFirstHotRequestsDespiteTokenWaterline(boolean batch, boolean hot) throws Exception {
        try (var f = new Fixture(batch, false, false, true)) {
            var a = f.occupy(f.a, 90);
            var b = f.occupy(f.b, 91);
            f.updateWaiting(f.a, 5L, 0L);
            f.updateWaiting(f.b, 5L, 0L);
            if (hot) { when(f.cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(f.a.ipPort(), 9)); }
            try {
                // Both waiting credit and the hard inflight limit have room; tokens and KV cannot block these requests.
                for (long id = 92; id <= 94; id++) {
                    var context = f.context(id);
                    var response = f.scheduler.submit(context).get(2, TimeUnit.SECONDS);
                    assertTrue(response.isSuccess(), response.getErrorMessage());
                    assertEquals(batch, response.isEnqueuedByMaster());
                    var route = context.snapshot().route();
                    assertNotNull(route);
                    if (hot) { assertSame(f.a, route.prefillEp()); assertEquals(900L, route.hitCache()); }
                    assertEquals(0, f.a.ownership.progress().queueDepth() + f.b.ownership.progress().queueDepth());
                    assertTrue(route.prefillEp().ownership.tokenWorkload().pendingTokens() > 1L);
                }
                if (batch) { assertEquals(3, f.a.ownership.stats().batchCount() + f.b.ownership.stats().batchCount()); }
            } finally { f.a.release(a, null); f.b.release(b, null); }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,false,false", "false,true,false", "true,false,false", "true,true,false",
            "false,false,true", "false,true,true", "true,false,true", "true,true,true"})
    void liveEngineWaitingTasksStopDeliveryAndRecoveryWakesIt(boolean batch, boolean hot, boolean heartbeat) throws Exception {
        try (var f = new Fixture(batch, false, false, true)) {
            // Observe the default five-second interval without sleeping on wall time.
            f.updateWaitingTasks(f.a, 32, heartbeat);
            var policy = ReflectionTestUtils.getField(f.a, "engineQueueAdmission");
            ReflectionTestUtils.setField(policy, "aboveSinceMs", System.nanoTime() / 1_000_000L - 5001L);
            f.updateWaitingTasks(f.a, 32, heartbeat);
            assertTrue(f.a.engineQueueCongested());
            if (hot) { when(f.cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(f.a.ipPort(), 9)); }
            var request = f.context(100);
            var response = f.scheduler.submit(request);
            if (hot) {
                awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
                assertFalse(response.isDone());
                assertNull(request.snapshot().route());
                assertEquals(32_000L, f.a.ownership.tokenWorkload().pendingTokens());
                assertEquals(0, f.a.ownership.stats().locallyOwnedRequests());
                f.updateWaitingTasks(f.a, 10, heartbeat);
                assertFalse(f.a.engineQueueCongested());
            }
            assertTrue(response.get(2, TimeUnit.SECONDS).isSuccess());
            assertSame(hot ? f.a : f.b, request.snapshot().route().prefillEp());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,0", "false,1", "false,2", "false,3", "false,4",
            "true,0", "true,1", "true,2", "true,3", "true,4"})
    void fourWorkerConcurrentBurstPreservesColdBalanceAndCacheFirst(boolean batch, int cacheMode) throws Exception {
        try (var f = new Fixture(batch, false, false, true, config -> {
            // This case isolates cache eligibility and token-score balancing, not feedback headroom.
            config.getRouter().getRoles().getPrefill().getTokenBalance().setEngineQueueAdmissionThreshold(1000L);
            var balance = config.getRouter().getRoles().getPrefill().getTokenBalance();
            balance.setMinPrefixHitPercent(5);
            balance.setEnableWindow(false);
            balance.setMaxPendingComputeTokens(2_000_000L);
            config.getDispatcher().setMaxInflightPerPrefillWorker(512);
        })) {
            var c = (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.3");
            var d = (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.4");
            var workers = List.of(f.a, f.b, c, d);
            Map<String, Integer> hits = switch (cacheMode) {
                case 1 -> Map.of(f.a.ipPort(), 1);
                case 2 -> Map.of(f.a.ipPort(), 45, f.b.ipPort(), 45, c.ipPort(), 45, d.ipPort(), 45);
                case 3 -> Map.of(f.a.ipPort(), 45);
                case 4 -> Map.of(f.a.ipPort(), 3);
                default -> Map.of();
            };
            when(f.cache.findMatchingEngines(any(), any(), any())).thenReturn(hits);
            var counts = new java.util.concurrent.ConcurrentHashMap<PrefillEndpoint, java.util.concurrent.atomic.AtomicInteger>();
            var start = new CountDownLatch(1);
            try (var clients = java.util.concurrent.Executors.newFixedThreadPool(8)) {
                var runs = new java.util.ArrayList<java.util.concurrent.Future<?>>();
                for (int client = 0; client < 8; client++) {
                    final int clientId = client;
                    runs.add(clients.submit(() -> {
                        assertTrue(start.await(3, TimeUnit.SECONDS));
                        var requests = new java.util.ArrayList<RequestContext>();
                        var responses = new java.util.ArrayList<java.util.concurrent.CompletableFuture<org.flexlb.dao.loadbalance.Response>>();
                        for (int i = 0; i < 50; i++) {
                            var request = f.context(10_000 + clientId * 50 + i);
                            request.data().setRequest(request.data().getRequest().toBuilder().seqLen(50_000L)
                                    .blockCacheKeys(java.util.stream.LongStream.rangeClosed(1, 50).boxed().toList())
                                    .cacheKeyBlockSize(1000L).build());
                            requests.add(request);
                            responses.add(f.scheduler.submit(request));
                        }
                        for (int i = 0; i < responses.size(); i++) {
                            var response = responses.get(i).get(10, TimeUnit.SECONDS);
                            assertTrue(response.isSuccess(), response.getErrorMessage());
                            var route = requests.get(i).snapshot().route();
                            assertEquals(cacheMode < 2, route.coldTokenBalance());
                            counts.computeIfAbsent(route.prefillEp(), ignored -> new java.util.concurrent.atomic.AtomicInteger()).incrementAndGet();
                        }
                        return null;
                    }));
                }
                start.countDown();
                for (var run : runs) { run.get(15, TimeUnit.SECONDS); }
            }
            var distribution = workers.stream().map(worker -> counts.getOrDefault(worker,
                    new java.util.concurrent.atomic.AtomicInteger()).get()).toList();
            assertEquals(cacheMode >= 3 ? List.of(400, 0, 0, 0) : List.of(100, 100, 100, 100), distribution);
            for (var worker : workers) {
                assertEquals(0L, worker.ownership.tokenWorkload().plannedTokens());
                assertEquals((cacheMode < 2 ? 50_000L : cacheMode == 4 ? 47_000L : 5_000L) * counts.getOrDefault(worker,
                        new java.util.concurrent.atomic.AtomicInteger()).get(), worker.ownership.tokenWorkload().pendingTokens());
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void burstHonorsHardInflightLimitAndCompletionWakesExactlyOneRequest(boolean batch, boolean hot) throws Exception {
        try (var f = new Fixture(batch, false, false, true,
                config -> {
                    config.getDispatcher().setMaxInflightPerPrefillWorker(256);
                    config.getRouter().getRoles().getPrefill().getTokenBalance().setEngineQueueAdmissionThreshold(256L);
                })) {
            var workers = List.of(f.a, f.b, (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.3"),
                    (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.4"));
            if (hot) { when(f.cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(f.a.ipPort(), 9)); }
            var requests = new java.util.ArrayList<RequestContext>();
            var responses = new java.util.ArrayList<java.util.concurrent.CompletableFuture<org.flexlb.dao.loadbalance.Response>>();
            for (int i = 0; i < 1280; i++) {
                var request = f.context(900_000L + i);
                requests.add(request);
                responses.add(f.scheduler.submit(request));
            }
            int admitted = hot ? 256 : 1024;
            awaitCondition(() -> responses.stream().filter(java.util.concurrent.CompletableFuture::isDone).count() >= admitted);
            awaitCondition(() -> f.scheduler.getLatestQueueWaitSnapshot().containsKey("role"));
            assertEquals(admitted, responses.stream().filter(java.util.concurrent.CompletableFuture::isDone).count());
            for (var worker : workers) {
                long expected = hot && worker != f.a ? 0 : 256;
                assertEquals(expected, batch ? worker.ownership.progress().batchSlots() : worker.ownership.progress().outstandingRequests());
            }
            RequestContext completed = null;
            for (int i = 0; i < requests.size(); i++) {
                if (!responses.get(i).isDone()) {
                    assertNull(requests.get(i).snapshot().delivery(), "waiting work must not own a delivery slot");
                    continue;
                }
                assertTrue(responses.get(i).join().isSuccess());
                if (hot) { assertSame(f.a, requests.get(i).snapshot().route().prefillEp()); }
                completed = requests.get(i);
            }
            assertNotNull(completed);
            var route = completed.snapshot().route();
            var terminal = org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                    .setRequestId(route.requestId()).setInputLength(1000L)
                    .setBatchId(completed.snapshot().delivery().identity().batchId())
                    .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build();
            f.observeTaskSnapshot(route.prefillEp(), List.of(), List.of(terminal));
            awaitCondition(() -> responses.stream().filter(java.util.concurrent.CompletableFuture::isDone).count() >= admitted + 1);
            assertEquals(admitted + 1, responses.stream().filter(java.util.concurrent.CompletableFuture::isDone).count());
            assertEquals(256, batch ? route.prefillEp().ownership.progress().batchSlots()
                    : route.prefillEp().ownership.progress().outstandingRequests());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,false,false", "true,false,false", "false,true,false", "true,true,false",
            "false,false,true", "true,false,true"})
    void engineBacklogWithoutLocalOwnersMustNotAttractTheColdBurst(boolean batch, boolean fullyOwned, boolean migrating) throws Exception {
        try (var f = new Fixture(batch, false, false, true, config -> {
            var balance = config.getRouter().getRoles().getPrefill().getTokenBalance();
            balance.setMinPrefixHitPercent(5);
            balance.setEngineQueueAdmissionThreshold(32L); // Equal token scores with room for another burst.
            balance.setMaxPendingComputeTokens(2_000_000L);
            var rescue = new RoutingConfig.HotspotRescueConfig();
            rescue.setMaxIdlePendingTokens(64768);
            balance.setHotspotRescue(rescue);
            config.getDispatcher().setMaxInflightPerPrefillWorker(256);
        })) {
            var workers = List.of(f.a, f.b, (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.3"),
                    (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.4"));
            var leases = new java.util.ArrayList<Map.Entry<PrefillEndpoint, PrefillState.RouteReservation>>();
            try {
                for (int w = 0; w < workers.size(); w++) {
                    var tasks = new java.util.ArrayList<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>();
                    for (int i = 0; i < 10; i++) {
                        long id = 100_000L + w * 100 + i;
                        tasks.add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder().setRequestId(id)
                                .setInputLength(50_000L).setIsWaiting(true)
                                .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED).build());
                        if (w > 0 || fullyOwned) {
                            var ctx = f.context(id);
                            ctx.data().setRequest(ctx.data().getRequest().toBuilder().seqLen(50_000L).build());
                            var route = SchedulingTestConfig.createRoute(ctx, null, null, null, workers.get(w), null, null,
                                    System.currentTimeMillis());
                            var acquired = workers.get(w).ownership.reserveUnqueuedRoute(route, 1000, 0);
                            assertEquals(PrefillState.CapacityStatus.ACQUIRED, acquired.status());
                            leases.add(Map.entry(workers.get(w), acquired.reservation()));
                        }
                    }
                    f.observeTaskSnapshot(workers.get(w), tasks, List.of());
                    assertEquals(10, workers.get(w).getStatus().committedEngineObservation().waitingQueryLen());
                }
                assertEquals(List.of(500_000.0, 500_000.0, 500_000.0, 500_000.0),
                        workers.stream().map(w -> w.ownership.tokenWorkload().score()).toList());
                if (migrating) {
                    var request = f.context(300_000L);
                    request.markPrefillMigrationSource(workers.getLast());
                    var result = f.router.select(request, null);
                    try (var selected = result.value()) {

                        assertEquals(org.flexlb.balance.PlacementResult.Status.BLOCKED, result.status(),
                                "no worker is below the 64768-token migration target limit");
                    }
                    return;
                }
                var counts = new int[4];
                for (int i = 0; i < 10; i++) {
                    var request = f.context(200_000L + i);
                    request.data().setRequest(request.data().getRequest().toBuilder().seqLen(50_000L).build());
                    assertTrue(f.scheduler.submit(request).get(2, TimeUnit.SECONDS).isSuccess());
                    var route = request.snapshot().route();
                    assertTrue(route.coldTokenBalance());
                    counts[workers.indexOf(route.prefillEp())]++;
                }
                assertTrue(java.util.Arrays.stream(counts).max().orElseThrow() - java.util.Arrays.stream(counts).min().orElseThrow() <= 1,
                        "equal Engine token backlog must not look empty solely because local request owners are missing");
            } finally {
                for (var lease : leases) { lease.getKey().release(lease.getValue(), null); }
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void repeatedColdBurstsAndEngineCompletionsDoNotBiasOneWorker(boolean batch) throws Exception {
        try (var f = new Fixture(batch, false, false, true)) {
            var workers = List.of(f.a, f.b, (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.3"),
                    (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.4"));
            int[] totals = new int[4];
            for (int wave = 0; wave < 30; wave++) {
                var finished = new java.util.ArrayList<java.util.List<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>>();
                for (int w = 0; w < 4; w++) { finished.add(new java.util.ArrayList<>()); }
                var requests = new java.util.ArrayList<RequestContext>();
                var responses = new java.util.ArrayList<java.util.concurrent.CompletableFuture<org.flexlb.dao.loadbalance.Response>>();
                for (int i = 0; i < 40; i++) {
                    var ctx = f.context(400_000L + wave * 40 + i);
                    ctx.data().setRequest(ctx.data().getRequest().toBuilder().seqLen(50_000L).build());
                    requests.add(ctx);
                    responses.add(f.scheduler.submit(ctx));
                }
                for (int i = 0; i < 40; i++) {
                    assertTrue(responses.get(i).get(5, TimeUnit.SECONDS).isSuccess());
                    var ctx = requests.get(i);
                    var route = ctx.snapshot().route();
                    int w = workers.indexOf(route.prefillEp());
                    totals[w]++;
                    assertTrue(route.coldTokenBalance());
                    finished.get(w).add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                            .setRequestId(route.requestId()).setInputLength(50_000L)
                            .setBatchId(ctx.snapshot().delivery().identity().batchId())
                            .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build());
                }
                for (int w = 0; w < 4; w++) {
                    assertEquals(10, finished.get(w).size());
                    f.observeTaskSnapshot(workers.get(w), List.of(), finished.get(w));
                    assertEquals(0L, workers.get(w).ownership.tokenWorkload().pendingTokens());
                    assertEquals(0L, workers.get(w).ownership.tokenWorkload().plannedTokens());
                }
            }
            assertArrayEquals(new int[]{300, 300, 300, 300}, totals);
        }
    }

    @Test
    void cancelledNonBatchEngineBacklogMustRemainInColdScore() throws Exception {
        var reason = CancelReason.CLIENT_CANCELLED;
        try (var f = new Fixture(false, false, false, true)) {
            var c = (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.3");
            var d = (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.4");
            var bLease = f.occupy(f.b, 600_001);
            var cLease = f.occupy(c, 600_002);
            var dLease = f.occupy(d, 600_003);
            try {
                var workers = List.of(f.a, f.b, c, d);
                var engineWaiting = new java.util.HashMap<PrefillEndpoint,
                        java.util.List<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>>();
                int[] counts = new int[4];
                for (int i = 0; i < 40; i++) {
                    var request = f.context(700_000L + i);
                    request.data().setRequest(request.data().getRequest().toBuilder().seqLen(50_000L).build());
                    assertTrue(f.scheduler.submit(request).get(2, TimeUnit.SECONDS).isSuccess());
                    var route = request.snapshot().route();
                    var worker = route.prefillEp();
                    assertTrue(route.coldTokenBalance());
                    counts[workers.indexOf(worker)]++;
                    var tasks = engineWaiting.computeIfAbsent(worker, ignored -> new java.util.ArrayList<>());
                    tasks.add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                            .setRequestId(route.requestId()).setInputLength(50_000L).setIsWaiting(true)
                            .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED).build());
                    f.observeTaskSnapshot(worker, tasks, List.of());
                    f.scheduler.cancel(request, 0L, reason);
                    awaitCondition(() -> worker.ownership.stats().locallyOwnedRequests() == 0);
                    assertTrue(worker.ownership.tokenWorkload().pendingTokens() >= tasks.size() * 50_000L,
                            "Engine work must remain visible even before the next heartbeat");
                    f.observeTaskSnapshot(worker, tasks, List.of());
                }
                assertArrayEquals(new int[]{10, 10, 10, 10}, counts);
                assertEquals(2_003_000L, workers.stream().mapToLong(w -> w.ownership.tokenWorkload().pendingTokens()).sum());
                for (var worker : workers) { f.observeTaskSnapshot(worker, List.of(), List.of()); }
                assertEquals(3000L, workers.stream().mapToLong(w -> w.ownership.tokenWorkload().pendingTokens()).sum());
            } finally {
                f.b.release(bLease, null); c.release(cLease, null); d.release(dLease, null);
            }
        }
    }

    @Test
    void sustainedThousandWaitingWorkerStopsTakingColdTraffic() throws Exception {
        try (var f = new Fixture(false, false, false, true)) {
            var workers = List.of(f.a, f.b, (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.3"),
                    (PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0.4"));
            f.updateWaitingTasks(f.a, 1000, false);
            var policy = ReflectionTestUtils.getField(f.a, "engineQueueAdmission");
            ReflectionTestUtils.setField(policy, "aboveSinceMs", System.nanoTime() / 1_000_000L - 5001L);
            f.updateWaitingTasks(f.a, 1000, false);
            for (var worker : workers.subList(1, 4)) { f.updateWaitingTasks(worker, 0, false); }
            assertTrue(f.a.engineQueueCongested());
            var running = new java.util.HashMap<PrefillEndpoint, java.util.List<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>>();
            int[] counts = new int[4];
            for (int i = 0; i < 100; i++) {
                var ctx = f.context(800_000L + i);
                assertTrue(f.scheduler.submit(ctx).get(2, TimeUnit.SECONDS).isSuccess());
                var worker = ctx.snapshot().route().prefillEp();
                counts[workers.indexOf(worker)]++;
                var tasks = running.computeIfAbsent(worker, ignored -> new java.util.ArrayList<>());
                tasks.add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder().setRequestId(ctx.data().getRequestId())
                        .setInputLength(1000L).setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build());
                f.observeTaskSnapshot(worker, tasks, List.of());
            }
            assertEquals(0, counts[0]);
            assertTrue(java.util.Arrays.stream(counts, 1, 4).max().orElseThrow()
                    - java.util.Arrays.stream(counts, 1, 4).min().orElseThrow() <= 1);
        }
    }

    @Test
    void coldCapacityHeadDoesNotStrandAdmissibleParkedHotRequests() throws Exception {
        try (var f = new Fixture(false, false, false, true, config -> {
            config.getDispatcher().setMaxInflightPerPrefillWorker(256);
            config.queueScheduler().setQueueTimeoutMs(120_000L);
            config.getRequestLifecycle().getRequest().setTimeoutMs(100_000L);
            var balance = config.getRouter().getRoles().getPrefill().getTokenBalance();
            balance.setMinPrefixHitPercent(5);
            balance.setMaxPendingComputeTokens(200_000L);
            balance.setEngineQueueAdmissionThreshold(30L);
            var rescue = new RoutingConfig.HotspotRescueConfig();
            rescue.setMaxInputTokens(32768L);
            rescue.setMaxIdlePendingTokens(64768L);
            rescue.setMigrationIntervalMs(1000L);
            balance.setHotspotRescue(rescue);
            var shedding = new org.flexlb.config.PrefillQueueSheddingConfig();
            var budget = new org.flexlb.config.PrefillQueueSheddingConfig.DeadlineBudgetConfig();
            budget.setTokensPerSecond(70_000L);
            shedding.setDeadlineBudget(budget);
            config.queueScheduler().setPrefillQueueShedding(shedding);
        })) {
            when(f.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                List<Long> keys = call.getArgument(0);
                return keys.getFirst() == 111L ? Map.of(f.a.ipPort(), 19, f.b.ipPort(), 19) : Map.of();
            });
            var running = new java.util.ArrayList<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>();
            for (int i = 0; i < 3; i++) {
                running.add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                        .setBatchId(-1).setRequestId(7_000_000 + i).setInputLength(100_000)
                        .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build());
            }
            var congested = new java.util.ArrayList<>(running);
            for (int i = 0; i < 31; i++) {
                congested.add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                        .setBatchId(-1).setRequestId(7_010_000 + i).setInputLength(1000).setIsWaiting(true)
                        .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED).build());
            }
            f.observeTaskSnapshot(f.a, congested, List.of());
            f.observeTaskSnapshot(f.b, congested, List.of());
            var cold = f.context(7_100_000);
            cold.data().setRequest(cold.data().getRequest().toBuilder().seqLen(100_000L).build());
            var coldResponse = f.scheduler.submit(cold);
            verify(f.router, timeout(1000)).select(same(cold), nullable(String.class));
            var hot = f.context(7_100_001);
            hot.data().setRequest(hot.data().getRequest().toBuilder()
                    .seqLen(100_000L).cacheKeyBlockSize(5000L)
                    .blockCacheKeys(java.util.stream.LongStream.range(111, 131).boxed().toList()).build());
            var hotResponse = f.scheduler.submit(hot);
            verify(f.router, timeout(1000)).select(same(hot), nullable(String.class));
            var lock = (java.util.concurrent.locks.ReentrantLock) ReflectionTestUtils.getField(f.scheduler, "lock");
            awaitCondition(() -> {
                lock.lock();
                try {
                    var entries = (Map<RequestContext, GlobalQueueEntry>) ReflectionTestUtils.getField(f.scheduler, "queuedEntries");
                    var waiters = (PlacementCapacityWaitQueue) ReflectionTestUtils.getField(f.scheduler, "waitingRequests");
                    return entries.get(cold) != null && entries.get(hot) != null
                            && waiters.isWaiting(entries.get(cold)) && waiters.isWaiting(entries.get(hot));
                } finally { lock.unlock(); }
            });
            f.observeTaskSnapshot(f.a, running, List.of());
            f.observeTaskSnapshot(f.b, running, List.of());
            assertTrue(f.router.hotPrefillQueueAllowsAdmission(hot, null));
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.INSUFFICIENT_CAPACITY,
                    f.router.coldPrefillAdmission(cold, null));
            assertTrue(hotResponse.get(2, TimeUnit.SECONDS).isSuccess(),
                    "hot work must retry while the cold token gate remains closed");
            assertFalse(coldResponse.isDone());
            var fresh = f.context(7_100_002);
            fresh.data().setRequest(fresh.data().getRequest().toBuilder().seqLen(100_000L).cacheKeyBlockSize(5000L)
                    .blockCacheKeys(java.util.stream.LongStream.range(111, 131).boxed().toList()).build());
            assertTrue(f.scheduler.submit(fresh).get(2, TimeUnit.SECONDS).isSuccess(),
                    "both parked and new hot requests retain admission while cold work waits");
            // Cold work still waits for a real reduction below its own token gate.
            f.observeTaskSnapshot(f.a, running.subList(2, 3), running.subList(0, 2));
            f.observeTaskSnapshot(f.b, running.subList(2, 3), running.subList(0, 2));
            assertTrue(coldResponse.get(2, TimeUnit.SECONDS).isSuccess());
            assertTrue(hotResponse.get(2, TimeUnit.SECONDS).isSuccess());
        }
    }

    @Test
    void hotCapacityHeadDoesNotStrandAnotherWorkersCachedRequests() throws Exception {
        try (var f = new Fixture(false, false, false, true, config -> {
            config.getDispatcher().setMaxInflightPerPrefillWorker(256);
            config.queueScheduler().setQueueTimeoutMs(120_000L);
            config.getRequestLifecycle().getRequest().setTimeoutMs(100_000L);
            var balance = config.getRouter().getRoles().getPrefill().getTokenBalance();
            balance.setMinPrefixHitPercent(5);
            balance.setMaxPendingComputeTokens(200_000L);
            balance.setEngineQueueAdmissionThreshold(30L);
            var shedding = new org.flexlb.config.PrefillQueueSheddingConfig();
            shedding.setDeadlineBudget(new org.flexlb.config.PrefillQueueSheddingConfig.DeadlineBudgetConfig());
            config.queueScheduler().setPrefillQueueShedding(shedding);
        })) {
            when(f.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                List<Long> keys = call.getArgument(0);
                return Map.of(keys.getFirst() == 111L ? f.a.ipPort() : f.b.ipPort(), 19);
            });
            f.updateWaitingTasks(f.a, 31, false);
            f.updateWaitingTasks(f.b, 31, false);
            var head = f.context(7_200_000);
            head.data().setRequest(head.data().getRequest().toBuilder().seqLen(100_000L).cacheKeyBlockSize(5000L)
                    .blockCacheKeys(java.util.stream.LongStream.range(111, 131).boxed().toList()).build());
            var headResponse = f.scheduler.submit(head);
            verify(f.router, timeout(1000)).select(same(head), nullable(String.class));
            var hot = f.context(7_200_001);
            hot.data().setRequest(hot.data().getRequest().toBuilder().seqLen(100_000L).cacheKeyBlockSize(5000L)
                    .blockCacheKeys(java.util.stream.LongStream.range(211, 231).boxed().toList()).build());
            var hotResponse = f.scheduler.submit(hot);
            var lock = (java.util.concurrent.locks.ReentrantLock) ReflectionTestUtils.getField(f.scheduler, "lock");
            awaitCondition(() -> {
                lock.lock();
                try {
                    var entries = (Map<RequestContext, GlobalQueueEntry>) ReflectionTestUtils.getField(f.scheduler, "queuedEntries");
                    var waiters = (PlacementCapacityWaitQueue) ReflectionTestUtils.getField(f.scheduler, "waitingRequests");
                    return entries.get(head) != null && entries.get(hot) != null
                            && waiters.isWaiting(entries.get(head)) && waiters.isWaiting(entries.get(hot));
                } finally { lock.unlock(); }
            });
            f.updateWaitingTasks(f.b, 0, false);
            assertFalse(f.router.hotPrefillQueueAllowsAdmission(head, null));
            assertTrue(f.router.hotPrefillQueueAllowsAdmission(hot, null));
            // Further healthy B status reports must not repeatedly retry only A's head.
            for (int i = 0; i < 5; i++) {
                f.updateWaitingTasks(f.b, 0, false);
                Thread.sleep(20L);
            }
            var fresh = f.context(7_200_002);
            fresh.data().setRequest(fresh.data().getRequest().toBuilder().seqLen(100_000L).cacheKeyBlockSize(5000L)
                    .blockCacheKeys(java.util.stream.LongStream.range(211, 231).boxed().toList()).build());
            assertTrue(f.scheduler.submit(fresh).get(2, TimeUnit.SECONDS).isSuccess());
            assertSame(f.b, fresh.snapshot().route().prefillEp());
            assertTrue(hotResponse.get(2, TimeUnit.SECONDS).isSuccess(),
                    "B's parked hot request must progress even while A remains full");
            assertSame(f.b, hot.snapshot().route().prefillEp());
            assertFalse(headResponse.isDone());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,100", "false,80", "true,100", "true,80"})
    void shrinkAndRandomTrafficReductionDoNotStrandSurvivingCacheCohorts(boolean shrink, int retainedPercent) throws Exception {
        try (var f = new Fixture(false, false, false, true, config -> {
            config.getDispatcher().setMaxInflightPerPrefillWorker(256);
            config.queueScheduler().setQueueTimeoutMs(120_000L);
            config.getRequestLifecycle().getRequest().setTimeoutMs(100_000L);
            var balance = config.getRouter().getRoles().getPrefill().getTokenBalance();
            balance.setMinPrefixHitPercent(5);
            balance.setMaxPendingComputeTokens(200_000L);
            balance.setEngineQueueAdmissionThreshold(30L);
            var shedding = new org.flexlb.config.PrefillQueueSheddingConfig();
            shedding.setDeadlineBudget(new org.flexlb.config.PrefillQueueSheddingConfig.DeadlineBudgetConfig());
            config.queueScheduler().setPrefillQueueShedding(shedding);
        })) {
            var workers = new java.util.ArrayList<PrefillEndpoint>(List.of(f.a, f.b));
            for (int i = 3; i <= 13; i++) { workers.add((PrefillEndpoint) f.publish(RoleType.PREFILL, "127.0.0." + i)); }
            when(f.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                List<Long> keys = call.getArgument(0);
                List<String> candidates = call.getArgument(2);
                int owner = Math.toIntExact(keys.getFirst() / 100L);
                var hits = new java.util.HashMap<String, Integer>();
                if (owner == 0) {
                    hits.put(workers.getLast().ipPort(), 19);
                    hits.put(f.a.ipPort(), 18);
                } else { hits.put(workers.get(owner).ipPort(), 19); }
                hits.keySet().retainAll(candidates);
                return hits;
            });
            for (var worker : workers) { f.updateWaitingTasks(worker, 31, false); }
            var head = f.context(7_300_000);
            setCachedRequest(head, 0L);
            var headResponse = f.scheduler.submit(head);
            awaitParked(f, List.of(head));
            var requests = new java.util.ArrayList<RequestContext>();
            var responses = new java.util.ArrayList<java.util.concurrent.CompletableFuture<org.flexlb.dao.loadbalance.Response>>();
            var expectedWorkers = new java.util.ArrayList<PrefillEndpoint>();
            var sampling = new java.util.Random(17);
            for (int i = 1; i < 12; i++) {
                for (int request = 0; request < 10; request++) {
                    if (sampling.nextInt(100) >= retainedPercent) { continue; }
                    var ctx = f.context(7_300_000L + i * 100 + request);
                    setCachedRequest(ctx, i * 100L);
                    requests.add(ctx);
                    expectedWorkers.add(workers.get(i));
                    responses.add(f.scheduler.submit(ctx));
                }
            }
            awaitParked(f, requests);
            if (shrink) {
                var removed = workers.getLast();
                EndpointRegistry.Retirement retirement;
                removed.getStatus().lock.lock();
                try { retirement = f.endpoints.beginRetirement(RoleType.PREFILL, removed.ipPort(), removed.getStatus()); }
                finally { removed.getStatus().lock.unlock(); }
                retirement.complete(f.cache, org.slf4j.LoggerFactory.getLogger(getClass()));
                assertEquals(12, f.endpoints.prefillRoutingSnapshot(RoleType.PREFILL).size());
                verify(f.cache).removeEngineBlockCache(removed.ipPort());
            } else { f.updateWaitingTasks(workers.getLast(), 0, false); }
            for (int i = 1; i < 12; i++) { f.updateWaitingTasks(workers.get(i), 0, false); }
            for (int i = 0; i < responses.size(); i++) {
                assertTrue(responses.get(i).get(3, TimeUnit.SECONDS).isSuccess());
                assertSame(expectedWorkers.get(i), requests.get(i).snapshot().route().prefillEp());
            }
            assertEquals(!shrink, headResponse.isDone());
            if (shrink) {
                // The removed worker's head now waits only for its surviving cache candidate A.
                f.updateWaitingTasks(f.a, 0, false);
                assertTrue(headResponse.get(2, TimeUnit.SECONDS).isSuccess());
                assertSame(f.a, head.snapshot().route().prefillEp());
            }
        }
    }

    @Test
    void newlyVisibleCacheWakesAHotRequestWhoseOldCacheWorkerIsStillFull() throws Exception {
        try (var f = new Fixture(false, false, false, true, config -> {
            config.getRouter().getRoles().getPrefill().getTokenBalance().setMaxPendingComputeTokens(200_000L);
        })) {
            var warmed = new AtomicBoolean();
            when(f.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> warmed.get()
                    ? Map.of(f.a.ipPort(), 19, f.b.ipPort(), 19) : Map.of(f.a.ipPort(), 19));
            f.updateWaitingTasks(f.a, 31, false);
            f.updateWaitingTasks(f.b, 0, false);
            var request = f.context(7_400_000);
            setCachedRequest(request, 100L);
            var response = f.scheduler.submit(request);
            awaitParked(f, List.of(request));
            warmed.set(true);
            f.endpoints.publishCacheAvailability(f.b.getStatus());
            assertTrue(response.get(2, TimeUnit.SECONDS).isSuccess());
            assertSame(f.b, request.snapshot().route().prefillEp());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(longs = {0L, 95_000L})
    void delayedEngineFeedbackCannotAdmitAnEntireHotBacklogAgainstTheSameQueueSnapshot(long reportedPrefix) throws Exception {
        try (var f = new Fixture(false, false, false, true, config -> {
            config.getDispatcher().setMaxInflightPerPrefillWorker(256);
            config.getRouter().getRoles().getPrefill().getTokenBalance().setEngineQueueAdmissionThreshold(30L);
            config.queueScheduler().setQueueTimeoutMs(120_000L);
            config.getRequestLifecycle().getRequest().setTimeoutMs(100_000L);
            config.getRouter().getRoles().getPrefill().getTokenBalance().setMaxPendingComputeTokens(200_000L);
        })) {
            when(f.cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(f.a.ipPort(), 19));
            f.updateWaitingTasks(f.a, 0, false);
            var contexts = new java.util.ArrayList<RequestContext>();
            var responses = new java.util.ArrayList<java.util.concurrent.CompletableFuture<org.flexlb.dao.loadbalance.Response>>();
            for (int i = 0; i < 80; i++) {
                var ctx = f.context(7_500_000L + i);
                setCachedRequest(ctx, 100L);
                contexts.add(ctx);
                responses.add(f.scheduler.submit(ctx));
            }
            verify(f.router, timeout(1000).atLeastOnce()).select(same(contexts.getLast()), nullable(String.class));
            awaitCondition(() -> responses.stream().filter(java.util.concurrent.CompletableFuture::isDone).count() >= 31);
            Thread.sleep(20L); // One transport-delay window with no new Engine evidence.
            assertEquals(31L, responses.stream().filter(java.util.concurrent.CompletableFuture::isDone).count(),
                    "waiting=0 must not be reused by all 80 hot requests before the next status");
            var observed = new java.util.ArrayList<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>();
            while (responses.stream().anyMatch(response -> !response.isDone())) {
                int previouslyObserved = observed.size();
                observed.clear();
                for (int i = 0; i < responses.size(); i++) {
                    if (!responses.get(i).isDone()) { continue; }
                    assertTrue(responses.get(i).get().isSuccess());
                    observed.add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                            .setRequestId(contexts.get(i).data().getRequestId()).setBatchId(-1L)
                            .setInputLength(100_000L).setPrefixLength(reportedPrefix)
                            .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build());
                }
                assertTrue(observed.size() > previouslyObserved);
                int count = observed.size();
                f.observeTaskSnapshot(f.a, observed, List.of());
                // Completion can race the snapshot scan and finish the entire backlog.
                if (count < responses.size()) {
                    awaitCondition(() -> responses.stream().filter(java.util.concurrent.CompletableFuture::isDone).count() > count);
                }
            }
            assertEquals(80, f.a.ownership.progress().outstandingRequests(),
                    "observed running work must not become a 31-request concurrency cap");
        }
    }

    @Test
    void engineWaitingRecoveryWakesReservedWorkEvenWhenNewAdmissionCreditIsFull() throws Exception {
        try (var f = new Fixture(false, false, false, true, config -> {
            config.getDispatcher().setMaxInflightPerPrefillWorker(256);
            var balance = config.getRouter().getRoles().getPrefill().getTokenBalance();
            balance.setMaxPendingComputeTokens(200_000L);
            balance.setEngineQueueAdmissionThreshold(30L);
        })) {
            var reservations = new java.util.ArrayList<PrefillState.RouteReservation>();
            try {
                for (int i = 0; i < 31; i++) {
                    var ctx = f.context(7_600_000L + i);
                    var route = SchedulingTestConfig.createRoute(ctx, null, null, null, f.a, null, null,
                            System.currentTimeMillis());
                    var result = f.a.ownership.reserveUnqueuedRoute(route, 1L, 256L);
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                    reservations.add(result.reservation());
                }
                f.updateWaitingTasks(f.a, 31, false);
                assertTrue(f.a.engineQueueDispatchBlocked());
                var running = new java.util.ArrayList<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB>();
                for (int i = 0; i < 31; i++) {
                    running.add(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                            .setRequestId(1000L + i).setInputLength(1000L)
                            .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build());
                }
                var change = (org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange.Prefill)
                        f.observeTaskSnapshot(f.a, running, List.of());
                assertTrue(f.a.engineQueueAdmissionBlocked(), "new requests still have no unobserved credit");
                assertFalse(f.a.engineQueueDispatchBlocked(), "already reserved work may dispatch");
                assertTrue(change.reconciliation().capacityReleased(), "dispatch recovery must still notify the worker");
            } finally { for (var reservation : reservations) { f.a.release(reservation, null); } }
        }
    }

    private static void setCachedRequest(RequestContext ctx, long keyBase) {
        ctx.data().setRequest(ctx.data().getRequest().toBuilder().seqLen(100_000L).cacheKeyBlockSize(5000L)
                .blockCacheKeys(java.util.stream.LongStream.range(keyBase, keyBase + 20).boxed().toList()).build());
    }

    private static void awaitParked(Fixture f, List<RequestContext> contexts) throws Exception {
        var lock = (java.util.concurrent.locks.ReentrantLock) ReflectionTestUtils.getField(f.scheduler, "lock");
        var entries = (Map<RequestContext, GlobalQueueEntry>) ReflectionTestUtils.getField(f.scheduler, "queuedEntries");
        var waiters = (PlacementCapacityWaitQueue) ReflectionTestUtils.getField(f.scheduler, "waitingRequests");
        awaitCondition(() -> {
            lock.lock();
            try { return contexts.stream().allMatch(ctx -> entries.get(ctx) != null && waiters.isWaiting(entries.get(ctx))); }
            finally { lock.unlock(); }
        });
    }

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final EndpointRegistry endpoints;
        final PrefillEndpoint a;
        final PrefillEndpoint b;
        final org.flexlb.balance.endpoint.DecodeEndpoint decode;
        final DecodeCapacityAcquirer eviction = mock(DecodeCapacityAcquirer.class);
        final RequestWorkerSelector router;
        final QueuedRequestScheduler scheduler;
        final SchedulerRuntime runtime;
        final CacheAwareService cache = mock(CacheAwareService.class);
        final org.flexlb.engine.grpc.EngineGrpcClient grpc = mock(org.flexlb.engine.grpc.EngineGrpcClient.class);

        Fixture() { this(false); }

        Fixture(boolean batch) { this(batch, false, false); }

        Fixture(boolean batch, boolean separateDecode, boolean priority) { this(batch, separateDecode, priority, false); }

        Fixture(boolean batch, boolean separateDecode, boolean priority, boolean engineAdmissionTest) {
            this(batch, separateDecode, priority, engineAdmissionTest, ignored -> {});
        }
        Fixture(boolean batch, boolean separateDecode, boolean priority, boolean engineAdmissionTest,
                java.util.function.Consumer<FlexlbConfig> customize) {
            SchedulingTestConfig.useFifoQueue(config);
            if (priority) { SchedulingTestConfig.usePriorityQueue(config); }
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
            SchedulingTestConfig.useNonBatchDispatcher(config);
            config.getDispatcher().setMaxInflightPerPrefillWorker(1);
            if (batch) { config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.BATCH); }
            config.queueScheduler().getDecision().setMaxRequests(8);
            config.queueScheduler().getDecision().setMaxCollectionWaitMs(60_000L);
            config.getRouter().getRoles().getPrefill().setTokenBalance(new RoutingConfig.TokenBalanceConfig());
            if (engineAdmissionTest) {
                config.getDispatcher().setMaxInflightPerPrefillWorker(128);
                config.queueScheduler().getDecision().setType(org.flexlb.config.DecisionPolicyConfig.Type.SINGLE);
                config.getRouter().getRoles().getPrefill().getTokenBalance().setMaxPendingComputeTokens(1L);
            } else {
                // These cases exercise the explicitly retained legacy capacity policy.
                config.getRouter().getRoles().getPrefill().getTokenBalance().setEngineQueueAdmissionThreshold(null);
            }
            customize.accept(config);
            var service = SchedulingTestConfig.configService(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            when(grpc.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call ->
                    java.util.concurrent.CompletableFuture.completedFuture(LinearDeliveryFixture.ack(call.getArgument(2))));
            when(grpc.cancelAsync(anyString(), anyInt(), any(), anyLong())).thenReturn(
                    java.util.concurrent.CompletableFuture.completedFuture(org.flexlb.engine.grpc.EngineGrpcClient.CancelReply.REQUEST_CLEANED));
            var dispatcher = new DeliveryDispatcher(grpc, service, reporter);
            var repository = new RequestRepository();
            runtime = new SchedulerRuntime(repository, mock(EndpointRegistry.class), reporter,
                    mock(RequestSchedulerReporter.class), dispatcher, service, grpc);
            var availability = new PlacementAvailability();
            endpoints = new EndpointRegistry(service, repository, reporter, dispatcher, availability);
            ReflectionTestUtils.setField(runtime, "endpoints", endpoints);
            a = (PrefillEndpoint) publish(RoleType.PREFILL, "127.0.0.1");
            b = (PrefillEndpoint) publish(RoleType.PREFILL, "127.0.0.2");
            decode = separateDecode ? (org.flexlb.balance.endpoint.DecodeEndpoint) publish(RoleType.DECODE, "127.0.0.3") : null;
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
            var strategy = new CostBasedPrefillStrategy(endpoints, cache, mock(EngineHealthReporter.class),
                    mock(CacheMetricsReporter.class), service);
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(separateDecode ? List.of(RoleType.PREFILL, RoleType.DECODE) : List.of(RoleType.PREFILL));
            router = spy(new RequestWorkerSelector(strategy, new DecodeSelector(endpoints, service, SchedulingTestConfig.capacity(config)), mock(VitWorkerSelector.class), model));
            scheduler = (QueuedRequestScheduler) PlacementConfiguration.create(runtime, config, router, reporter,
                    eviction, availability, SchedulingTestConfig.capacity(config),
                    mock(RecentCacheKeyTraceReporter.class));
            runtime.registerScheduler(scheduler);
        }

        RequestContext context(long id) {
            var context = RequestProtocolTestSupport.context(config, id);
            context.data().setRequest(context.data().getRequest().toBuilder().seqLen(1000L).maxNewTokens(1)
                    .blockCacheKeys(List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)).cacheKeyBlockSize(100L).build());
            if (config.getDispatcher().requiresGenerateInput()) {
                context.data().setGenerateInputPb(org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB.newBuilder()
                        .setRequestId(id).setGenerateConfig(org.flexlb.engine.grpc.EngineRpcService.GenerateConfigPB.newBuilder())
                        .build().toByteString());
            }
            return context;
        }

        PrefillState.RouteReservation occupy(PrefillEndpoint endpoint, long id) {
            var context = context(id);
            var route = SchedulingTestConfig.createRoute(context, null, null, null, endpoint, null, null,
                    System.currentTimeMillis());
            var result = endpoint.ownership.reserveUnqueuedRoute(route, 1000, 1);
            assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
            return result.reservation();
        }

        org.flexlb.balance.endpoint.WorkerEndpoint publish(RoleType role, String ip) {
            var worker = WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, "test");
            endpoints.currentOrDiscover(role, worker.getIpPort(), () -> worker);
            worker.lock.lock();
            try {
                var status = new WorkerStatusResponse();
                status.setRole(role);
                status.setAlive(true);
                status.setStatusVersion(1L);
                status.setRunningTaskInfo(Map.of());
                status.setFinishedTaskInfo(Map.of());
                status.setTotalKvCacheTokens(1_000_000L);
                status.setAvailableKvCacheTokens(1_000_000L);
                return endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                        worker.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(worker, status)));
            } finally { worker.lock.unlock(); }
        }

        org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange observeTaskSnapshot(PrefillEndpoint endpoint,
                List<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB> running,
                List<org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB> finished) throws Exception {
            var worker = endpoint.getStatus();
            var proto = org.flexlb.engine.grpc.EngineRpcService.WorkerStatusPB.newBuilder()
                    .setRoleType(org.flexlb.engine.grpc.EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                    .setAlive(true).setStatusVersion(worker.appliedStatusCursor().statusVersion() + 1)
                    .setTotalKvCache(1_000_000L).setAvailableKvCache(1_000_000L)
                    .addAllRunningTaskInfo(running).addAllFinishedTaskList(finished).build();
            var observation = org.flexlb.service.grpc.EngineStatusConverter.convertToStatusObservation(worker,
                    org.flexlb.engine.grpc.EngineRpcService.WorkerStatusPB.parseFrom(proto.toByteArray()));
            org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange change;
            worker.lock.lock();
            try { change = endpoint.applyPreparedStatus(worker, worker.prepareNewStatus(observation)); }
            finally { worker.lock.unlock(); }
            endpoint.publishStatus(change);
            return change;
        }

        void updateWaitingTasks(PrefillEndpoint endpoint, int waiting, boolean heartbeat) throws Exception {
            var worker = endpoint.getStatus();
            var proto = org.flexlb.engine.grpc.EngineRpcService.WorkerStatusPB.newBuilder()
                    .setRoleType(org.flexlb.engine.grpc.EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                    .setAlive(true).setStatusVersion(worker.appliedStatusCursor().statusVersion() + (heartbeat ? 0 : 1))
                    .setTotalKvCache(1_000_000L).setAvailableKvCache(1_000_000L);
            for (int i = 0; i < waiting; i++) {
                proto.addRunningTaskInfo(org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId(1000 + i).setInputLength(1000).setIsWaiting(true)
                        .setPhase(org.flexlb.engine.grpc.EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));
            }
            var observation = org.flexlb.service.grpc.EngineStatusConverter.convertToStatusObservation(worker,
                    org.flexlb.engine.grpc.EngineRpcService.WorkerStatusPB.parseFrom(proto.build().toByteArray()));
            org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange change;
            worker.lock.lock();
            try {
                change = heartbeat ? endpoint.applyStatusHeartbeat(worker, observation)
                        : endpoint.applyPreparedStatus(worker, worker.prepareNewStatus(observation));
            } finally { worker.lock.unlock(); }
            endpoint.publishStatus(change);
        }

        void updateWaiting(PrefillEndpoint endpoint, long waiting, long availableKv) {
            var worker = endpoint.getStatus();
            worker.lock.lock();
            try {
                var status = new WorkerStatusResponse();
                status.setRole(RoleType.PREFILL);
                status.setAlive(true);
                status.setStatusVersion(worker.appliedStatusCursor().statusVersion() + 1L);
                status.setRunningTaskInfo(Map.of());
                status.setFinishedTaskInfo(Map.of());
                status.setTotalKvCacheTokens(1_000_000L);
                status.setAvailableKvCacheTokens(availableKv);
                status.setWaitingQueryLen(waiting);
                endpoint.publishStatus(endpoint.applyPreparedStatus(worker,
                        worker.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(worker, status))));
            } finally { worker.lock.unlock(); }
        }

        @Override public void close() { runtime.shutdown(); }
    }
}

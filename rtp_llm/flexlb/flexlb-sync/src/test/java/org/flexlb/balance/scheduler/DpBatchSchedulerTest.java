package org.flexlb.balance.scheduler;

import org.flexlb.balance.dp.DispatchContext;
import org.flexlb.balance.dp.DispatchPlan;
import org.flexlb.balance.dp.DispatchPlanner;
import org.flexlb.balance.dp.FailedRequest;
import org.flexlb.balance.dp.InflightBatchRegistry;
import org.flexlb.balance.dp.PendingRequest;
import org.flexlb.balance.dp.PrefillBatch;
import org.flexlb.balance.dp.QueuedRequest;
import org.flexlb.balance.dp.RoundRobinAssign;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.DpRankStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.dp.DpGrpcClient;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DpBatchSchedulerTest {

    private ConfigService configService;
    private FlexlbConfig cfg;
    private EngineWorkerStatus engineWorkerStatus;
    private DpGrpcClient grpcClient;
    private InflightBatchRegistry registry;
    private DispatchPlanner planner;
    private DpBatchScheduler scheduler;

    private final List<EngineRpcService.BatchEnqueueRequestPB> sentBatches = new CopyOnWriteArrayList<>();

    private static EngineRpcService.BatchEnqueueResponsePB buildAck(
            EngineRpcService.BatchEnqueueRequestPB req, boolean accept, String rejectMsg) {
        EngineRpcService.BatchEnqueueResponsePB.Builder rb =
                EngineRpcService.BatchEnqueueResponsePB.newBuilder().setBatchId(req.getBatchId());
        for (EngineRpcService.GenerateInputPB in : req.getInputsList()) {
            EngineRpcService.EnqueueResponsePB.Builder slot =
                    EngineRpcService.EnqueueResponsePB.newBuilder().setRequestId(in.getRequestId());
            if (!accept) {
                slot.setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                        .setErrorCode(1L)
                        .setErrorMessage(rejectMsg).build());
            }
            rb.addAcks(slot.build());
        }
        return rb.build();
    }

    private static final ServerStatus PREFILL = serverStatus(RoleType.PREFILL, "10.0.0.1", 8080, 9080, "g1");
    private static final ServerStatus DECODE  = serverStatus(RoleType.DECODE,  "10.0.0.2", 8081, 9081, "g1");

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        engineWorkerStatus = mock(EngineWorkerStatus.class);
        grpcClient = mock(DpGrpcClient.class);
        registry = new InflightBatchRegistry();
        planner = mock(DispatchPlanner.class);

        cfg = new FlexlbConfig();
        cfg.setDpBalanceEnabled(true);
        cfg.setDpBatchSizeMax(4);
        cfg.setDpBatchWindowMs(20);
        cfg.setDpBatchTimeoutMs(100);
        when(configService.loadBalanceConfig()).thenReturn(cfg);

        // Default planner: turn the drained chunk into a single PrefillBatch
        // landing on PREFILL/DECODE. Per-batch ranks are filled via the cursor
        // in RoundRobinAssign downstream.
        when(planner.plan(any(), any(DispatchContext.class))).thenAnswer(inv -> {
            List<QueuedRequest> drained = inv.getArgument(0);
            DispatchContext ctx = inv.getArgument(1);
            List<PendingRequest> placed = drained.stream()
                    .map(qr -> new PendingRequest(qr.ctx(), PREFILL, DECODE, qr.future(), qr.enqueuedAtMicros()))
                    .toList();
            PrefillBatch batch = new PrefillBatch(PREFILL, placed, ctx.dpSize());
            return DispatchPlan.of(List.of(batch));
        });

        // Default gRPC client: ack accepted, capture sent batches
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class)))
                .thenAnswer(inv -> {
                    EngineRpcService.BatchEnqueueRequestPB b = inv.getArgument(2);
                    sentBatches.add(b);
                    return CompletableFuture.completedFuture(buildAck(b, true, ""));
                });
        when(grpcClient.cancelPrefill(anyString(), anyInt(), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(null));
        when(grpcClient.cancelDecode(anyString(), anyInt(), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(null));

        scheduler = new DpBatchScheduler(configService, engineWorkerStatus, planner,
                List.of(new RoundRobinAssign()), grpcClient, registry,
                mock(CacheAwareService.class));
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void four_requests_form_one_batch_each_assigned_distinct_dpRank() throws Exception {
        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .toList();

        for (CompletableFuture<Response> f : futures) {
            Response resp = f.get(2, TimeUnit.SECONDS);
            assertTrue(resp.isSuccess());
            assertTrue(resp.isEnqueuedByMaster());
            assertEquals(2, resp.getServerStatus().size());
        }

        assertEquals(1, sentBatches.size(), "4 requests at dpSize=4 form one batch");
        EngineRpcService.BatchEnqueueRequestPB sent = sentBatches.get(0);
        assertEquals(4, sent.getInputsCount());

        List<Integer> ranks = IntStream.range(0, 4)
                .map(i -> sent.getInputs(i).getDpRank().getValue())
                .boxed().collect(Collectors.toList());
        assertEquals(List.of(0, 1, 2, 3), ranks, "fresh cursor + dp=batchSize=4 fills 0..3 in order");
    }

    @Test
    void per_request_failure_from_planner_completes_future_with_failure_response() throws Exception {
        // Assembler decides one of the four can't be placed (e.g., no decode pairing).
        when(planner.plan(any(), any(DispatchContext.class))).thenAnswer(inv -> {
            List<QueuedRequest> drained = inv.getArgument(0);
            DispatchContext ctx = inv.getArgument(1);
            QueuedRequest victim = drained.get(2);
            List<PendingRequest> placed = drained.stream()
                    .filter(qr -> qr != victim)
                    .map(qr -> new PendingRequest(qr.ctx(), PREFILL, DECODE, qr.future(), qr.enqueuedAtMicros()))
                    .toList();
            PrefillBatch batch = new PrefillBatch(PREFILL, placed, ctx.dpSize());
            return new DispatchPlan(
                    List.of(batch),
                    List.of(new FailedRequest(victim, StrategyErrorType.NO_DECODE_WORKER, "test")));
        });

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .toList();

        // The "victim" future receives a failure response synchronously from the batcher.
        Response victimResp = futures.get(2).get(2, TimeUnit.SECONDS);
        assertFalse(victimResp.isSuccess());
        assertEquals(StrategyErrorType.NO_DECODE_WORKER.getErrorCode(), victimResp.getCode());

        // The other three flow through the success path.
        for (int i : new int[]{0, 1, 3}) {
            Response r = futures.get(i).get(2, TimeUnit.SECONDS);
            assertTrue(r.isSuccess(), "request " + i + " should succeed");
        }

        long realInBatch = sentBatches.get(0).getInputsList().stream()
                .filter(in -> !in.getIsFakeQuery())
                .count();
        assertEquals(3, realInBatch, "victim must not appear in the dispatched batch (fake pads are accepted)");
    }

    @Test
    void no_dp_enabled_worker_fails_all_with_NO_PREFILL_WORKER() throws Exception {
        when(planner.plan(any(), any(DispatchContext.class))).thenAnswer(inv -> {
            List<QueuedRequest> drained = inv.getArgument(0);
            return DispatchPlan.allFailed(drained, StrategyErrorType.NO_PREFILL_WORKER, "no candidate");
        });

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .toList();
        for (CompletableFuture<Response> f : futures) {
            Response r = f.get(2, TimeUnit.SECONDS);
            assertFalse(r.isSuccess());
            assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), r.getCode());
        }
        verify(grpcClient, never()).enqueue(anyString(), anyInt(), any());
    }

    @Test
    void enqueue_rejection_fails_all_batched_futures() throws Exception {
        AtomicInteger callCount = new AtomicInteger();
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class)))
                .thenAnswer(inv -> {
                    callCount.incrementAndGet();
                    EngineRpcService.BatchEnqueueRequestPB b = inv.getArgument(2);
                    return CompletableFuture.completedFuture(buildAck(b, false, "queue full"));
                });

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .toList();
        for (CompletableFuture<Response> f : futures) {
            assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> f.get(2, TimeUnit.SECONDS));
        }
        assertEquals(1, callCount.get());
    }

    @Test
    void enqueue_rpc_failure_fails_futures() throws Exception {
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("UNAVAILABLE")));
        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .toList();
        for (CompletableFuture<Response> f : futures) {
            assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> f.get(2, TimeUnit.SECONDS));
        }
    }

    @Test
    void cancel_in_queue_yanks_request_and_completes_future_exceptionally() throws Exception {
        // Submit only 2 (under dpSize=4) so they sit in the queue.
        CompletableFuture<Response> f1 = scheduler.submit(makeCtx(1, "m1"));
        CompletableFuture<Response> f2 = scheduler.submit(makeCtx(2, "m1"));

        scheduler.cancel(1L);
        assertEquals(1, scheduler.totalQueueDepth(), "request 1 must be removed from the queue");

        // Cancelled future must be completed exceptionally, not left hanging — that
        // would otherwise hang the upstream Mono until client timeout.
        assertTrue(f1.isCompletedExceptionally(),
                "cancelInQueue must complete the future to avoid leaking a hung Mono");
        // CompletableFuture special-cases CancellationException: get() throws it directly,
        // not wrapped in ExecutionException.
        assertThrows(java.util.concurrent.CancellationException.class, () -> f1.get(1, TimeUnit.SECONDS));

        // Window timeout still flushes whatever is left (just request 2).
        Response r2 = f2.get(2, TimeUnit.SECONDS);
        assertTrue(r2.isSuccess());

        assertEquals(1, sentBatches.size());
        long realInBatch = sentBatches.get(0).getInputsList().stream()
                .filter(in -> !in.getIsFakeQuery())
                .count();
        assertEquals(1, realInBatch, "cancelled request must not be in the dispatched batch (fake pads are accepted)");
    }

    @Test
    void submit_completes_future_exceptionally_when_offer_throws() throws Exception {
        // Force every batcher's offer path to throw by shutting down the
        // timer executor that schedule-window-timer relies on. Submit must
        // never return a hanging future even in catastrophic conditions.
        scheduler.shutdown();   // kills timerExecutor → schedule() will throw RejectedExecutionException

        CompletableFuture<Response> f = scheduler.submit(makeCtx(99, "m1"));

        assertTrue(f.isCompletedExceptionally(),
                "submit must convert any upstream throw into an exceptionally completed future");
        assertThrows(java.util.concurrent.ExecutionException.class, () -> f.get(1, TimeUnit.SECONDS));
    }

    @Test
    void cancel_cascades_to_both_prefill_and_decode_when_in_flight() throws Exception {
        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(100 + i, "m1")))
                .toList();
        for (CompletableFuture<Response> f : futures) {
            f.get(2, TimeUnit.SECONDS);
        }

        scheduler.cancel(101L);

        verify(grpcClient, times(1)).cancelPrefill(anyString(), anyInt(), eq(101L));
        verify(grpcClient, times(1)).cancelDecode(anyString(), anyInt(), eq(101L));
        assertNull(registry.lookupByRequest(101L));
    }

    @Test
    void cancel_unknown_requestId_is_silent_noop() {
        scheduler.cancel(99999L);
        verify(grpcClient, never()).cancelPrefill(anyString(), anyInt(), anyLong());
        verify(grpcClient, never()).cancelDecode(anyString(), anyInt(), anyLong());
    }

    @Test
    void cancel_during_PENDING_ACK_then_enqueue_rejected_still_cascades_engine_cancel() throws Exception {
        CompletableFuture<EngineRpcService.BatchEnqueueResponsePB> ackFuture = new CompletableFuture<>();
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class)))
                .thenAnswer(inv -> {
                    sentBatches.add(inv.getArgument(2));
                    return ackFuture;
                });

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(300 + i, "m1")))
                .toList();
        Thread.sleep(50);

        scheduler.cancel(301L);
        assertEquals(InflightBatchRegistry.RequestState.CANCELLED, registry.getState(301L));

        ackFuture.completeExceptionally(new RuntimeException("DEADLINE_EXCEEDED"));

        for (CompletableFuture<Response> f : futures) {
            assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> f.get(2, TimeUnit.SECONDS));
        }
        verify(grpcClient, times(1)).cancelPrefill(anyString(), anyInt(), eq(301L));
        verify(grpcClient, times(1)).cancelDecode(anyString(), anyInt(), eq(301L));
        verify(grpcClient, never()).cancelPrefill(anyString(), anyInt(), eq(300L));
    }

    @Test
    void reentrant_cancel_is_idempotent() throws Exception {
        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(400 + i, "m1")))
                .toList();
        for (CompletableFuture<Response> f : futures) {
            f.get(2, TimeUnit.SECONDS);
        }
        scheduler.cancel(401L);
        scheduler.cancel(401L);
        verify(grpcClient, times(1)).cancelPrefill(anyString(), anyInt(), eq(401L));
        verify(grpcClient, times(1)).cancelDecode(anyString(), anyInt(), eq(401L));
    }

    @Test
    void window_timeout_flushes_partial_batch() throws Exception {
        CompletableFuture<Response> f1 = scheduler.submit(makeCtx(1, "m1"));
        CompletableFuture<Response> f2 = scheduler.submit(makeCtx(2, "m1"));
        Response r1 = f1.get(2, TimeUnit.SECONDS);
        Response r2 = f2.get(2, TimeUnit.SECONDS);
        assertTrue(r1.isSuccess() && r2.isSuccess());
        assertEquals(1, sentBatches.size());
        // dpSize=4 ⇒ partial 2-real batch is padded with 2 fake-query slots so
        // every DP rank gets a slot and the engine-side DeepEP all-to-all has
        // a peer on every rank.
        assertEquals(4, sentBatches.get(0).getInputsCount());
    }

    @Test
    void partial_batch_pads_remaining_dpRanks_with_fake_query_slots() throws Exception {
        // Submit one request only; window timer flushes a 1-real batch into
        // dpSize=4. buildPb must pad ranks 1..3 with is_fake_query=true slots.
        CompletableFuture<Response> f1 = scheduler.submit(makeCtx(42L, "m1"));
        Response r1 = f1.get(2, TimeUnit.SECONDS);
        assertTrue(r1.isSuccess(), "real request still completes successfully");

        assertEquals(1, sentBatches.size());
        EngineRpcService.BatchEnqueueRequestPB sent = sentBatches.get(0);
        assertEquals(4, sent.getInputsCount(), "padded to dpSize=4");

        // Exactly one slot is real (positive request_id, no is_fake_query); the
        // other three are fake (is_fake_query=true, negative synthetic id).
        long realCount = sent.getInputsList().stream()
                .filter(in -> !in.getIsFakeQuery())
                .count();
        long fakeCount = sent.getInputsList().stream()
                .filter(EngineRpcService.GenerateInputPB::getIsFakeQuery)
                .count();
        assertEquals(1, realCount);
        assertEquals(3, fakeCount);

        // All four dp_ranks [0,1,2,3] must be covered exactly once so DeepEP
        // collectives have a peer on every rank.
        List<Integer> ranks = sent.getInputsList().stream()
                .map(in -> in.getDpRank().getValue())
                .sorted()
                .collect(Collectors.toList());
        assertEquals(List.of(0, 1, 2, 3), ranks);

        // Fake request_ids must not collide with real ones (real is positive,
        // fake is negative); FlexLB never registers them, so handleAck's
        // assignment lookup naturally ignores them.
        for (EngineRpcService.GenerateInputPB in : sent.getInputsList()) {
            if (in.getIsFakeQuery()) {
                assertTrue(in.getRequestId() < 0,
                        "fake slot request_id must be negative, got " + in.getRequestId());
            } else {
                assertEquals(42L, in.getRequestId());
            }
        }
    }

    @Test
    void success_response_remaps_prefill_address_via_dpStatuses() throws Exception {
        // DP0 publishes per-rank addresses via WorkerStatusPB.dp_status[]; the
        // EngineWorkerStatus map has them under PREFILL/group=g1 keyed by
        // "<serverIp>:<httpPort>". buildSuccessResponse must rewrite each
        // request's prefill.serverIp/grpcPort to the dpRank-specific entry so
        // frontend FetchResponse hits the DP that actually owns the slot.
        WorkerStatus ws = new WorkerStatus();
        ws.setDpStatuses(List.of(
                new DpRankStatus(0, "10.0.0.1", 10101, 0, 0, 0, 0, true),
                new DpRankStatus(1, "10.0.0.1", 10109, 0, 0, 0, 0, true),
                new DpRankStatus(2, "10.0.0.1", 10117, 0, 0, 0, 0, true),
                new DpRankStatus(3, "10.0.0.1", 10125, 0, 0, 0, 0, true)));
        Map<String, WorkerStatus> roleMap = new HashMap<>();
        roleMap.put("10.0.0.1:8080", ws);
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), eq("g1")))
                .thenReturn(roleMap);

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .toList();

        // Build a dpRank → grpcPort map from the responses to verify per-request remap.
        Map<Long, Integer> dpRankToGrpcPort = new HashMap<>();
        for (CompletableFuture<Response> f : futures) {
            Response r = f.get(2, TimeUnit.SECONDS);
            assertTrue(r.isSuccess());
            ServerStatus prefill = r.getServerStatus().get(0);
            dpRankToGrpcPort.put(prefill.getDpRank(), prefill.getGrpcPort());
            assertEquals("10.0.0.1", prefill.getServerIp(),
                    "ip stays — only the per-DP grpc port flips per dpRank");
        }
        assertEquals(10101, dpRankToGrpcPort.get(0L).intValue());
        assertEquals(10109, dpRankToGrpcPort.get(1L).intValue());
        assertEquals(10117, dpRankToGrpcPort.get(2L).intValue());
        assertEquals(10125, dpRankToGrpcPort.get(3L).intValue());
    }

    @Test
    void success_response_keeps_original_prefill_address_when_dpStatuses_empty() throws Exception {
        // Legacy / V0 path: engine never published dp_status[], so dpStatuses
        // is empty. applyDpRankAddress must be a no-op and prefill.grpcPort
        // must stay at the pod-level entry (=9080). Otherwise V0 smokes that
        // share this code path would regress.
        WorkerStatus ws = new WorkerStatus();
        // dpStatuses defaults to List.of()
        Map<String, WorkerStatus> roleMap = new HashMap<>();
        roleMap.put("10.0.0.1:8080", ws);
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), eq("g1")))
                .thenReturn(roleMap);

        CompletableFuture<Response> f = scheduler.submit(makeCtx(1L, "m1"));
        // Need to fill dpSize=4 to flush sooner; submit 3 more.
        scheduler.submit(makeCtx(2L, "m1"));
        scheduler.submit(makeCtx(3L, "m1"));
        scheduler.submit(makeCtx(4L, "m1"));

        Response r = f.get(2, TimeUnit.SECONDS);
        assertTrue(r.isSuccess());
        ServerStatus prefill = r.getServerStatus().get(0);
        assertEquals("10.0.0.1", prefill.getServerIp());
        assertEquals(9080, prefill.getGrpcPort(),
                "empty dpStatuses ⇒ keep pod-level grpcPort untouched");
    }

    @Test
    void success_path_keeps_active_entry_until_worker_reports_finished() throws Exception {
        // Master holds the InflightBatchRegistry entry through the full request
        // lifetime so a late /rtp_llm/cancel can still cascade through master.
        // Cleanup runs when the worker reporter (GrpcWorkerStatusRunner) sees
        // the requestId in finishedTaskList — outside this test's scope; here
        // we verify the dispatcher does NOT prematurely drop the entry.
        IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .forEach(f -> {
                    try {
                        f.get(2, TimeUnit.SECONDS);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                });

        for (long id = 1; id <= 4; id++) {
            assertNotNull(registry.lookupByRequest(id),
                    "active entry " + id + " must stay until worker reports finished");
            assertEquals(InflightBatchRegistry.RequestState.ACTIVE, registry.getState(id));
        }
    }

    @Test
    void success_response_keeps_pod_grpcPort_when_dpRank_missing_from_dpStatuses() throws Exception {
        // Per design §10 invariant #9, only DP0 publishes dp_status[]. If a
        // request lands on a dpRank not in the published list (sparse publish,
        // partial sync, or peer-DP entry leaked into the role map),
        // applyDpRankAddress must be a no-op — leaving prefill.grpcPort at the
        // pod-level entry — instead of silently falling through to a wrong DP.
        WorkerStatus ws = new WorkerStatus();
        ws.setDpStatuses(List.of(
                new DpRankStatus(2, "10.0.0.1", 10117, 0, 0, 0, 0, true),
                new DpRankStatus(3, "10.0.0.1", 10125, 0, 0, 0, 0, true)));
        Map<String, WorkerStatus> roleMap = new HashMap<>();
        roleMap.put("10.0.0.1:8080", ws);
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), eq("g1")))
                .thenReturn(roleMap);

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .toList();

        Map<Long, Integer> dpRankToGrpcPort = new HashMap<>();
        for (CompletableFuture<Response> f : futures) {
            Response r = f.get(2, TimeUnit.SECONDS);
            assertTrue(r.isSuccess());
            ServerStatus prefill = r.getServerStatus().get(0);
            dpRankToGrpcPort.put((long) prefill.getDpRank(), prefill.getGrpcPort());
        }
        assertEquals(9080, dpRankToGrpcPort.get(0L).intValue(),
                "rank=0 missing from dpStatuses ⇒ keep pod-level grpcPort");
        assertEquals(9080, dpRankToGrpcPort.get(1L).intValue(),
                "rank=1 missing from dpStatuses ⇒ keep pod-level grpcPort");
        assertEquals(10117, dpRankToGrpcPort.get(2L).intValue(),
                "rank=2 present ⇒ remap to per-rank grpcPort");
        assertEquals(10125, dpRankToGrpcPort.get(3L).intValue(),
                "rank=3 present ⇒ remap to per-rank grpcPort");
    }

    @Test
    void different_models_use_independent_batchers() throws Exception {
        // 2 reqs of model A + 2 reqs of model B → neither alone reaches dpSize=4 so
        // both must wait for the window timer; they must NOT share a batch.
        CompletableFuture<Response> a1 = scheduler.submit(makeCtx(1, "modelA"));
        CompletableFuture<Response> a2 = scheduler.submit(makeCtx(2, "modelA"));
        CompletableFuture<Response> b1 = scheduler.submit(makeCtx(3, "modelB"));
        CompletableFuture<Response> b2 = scheduler.submit(makeCtx(4, "modelB"));

        a1.get(2, TimeUnit.SECONDS);
        a2.get(2, TimeUnit.SECONDS);
        b1.get(2, TimeUnit.SECONDS);
        b2.get(2, TimeUnit.SECONDS);

        assertEquals(2, sentBatches.size(), "two models ⇒ two batches");
        assertEquals(2, scheduler.batcherCount());
    }

    // ============== helpers ==============

    private BalanceContext makeCtx(long requestId, String model) {
        BalanceContext ctx = new BalanceContext();
        Request req = new Request();
        req.setRequestId(requestId);
        req.setSeqLen(100);
        req.setMaxNewTokens(128);
        req.setNumBeams(1);
        req.setBlockCacheKeys(List.of(1L, 2L, 3L));
        req.setModel(model);
        ctx.setRequest(req);
        ctx.setConfig(cfg);
        return ctx;
    }

    private static ServerStatus serverStatus(RoleType role, String ip, int httpPort, int grpcPort, String group) {
        ServerStatus s = new ServerStatus();
        s.setSuccess(true);
        s.setRole(role);
        s.setServerIp(ip);
        s.setHttpPort(httpPort);
        s.setGrpcPort(grpcPort);
        s.setGroup(group);
        return s;
    }
}

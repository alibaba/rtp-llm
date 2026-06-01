package org.flexlb.balance.scheduler;

import org.flexlb.balance.dp.DispatchPlanner;
import org.flexlb.balance.dp.InflightBatchRegistry;
import org.flexlb.balance.dp.LinearPrefillTimePredictor;
import org.flexlb.balance.dp.PendingRequest;
import org.flexlb.balance.dp.DispatchBatch;
import org.flexlb.balance.dp.QueuedRequest;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.DpRankStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.dp.DpGrpcClient;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
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
        cfg.setDpBatcherType("SIMPLE");
        when(configService.loadBalanceConfig()).thenReturn(cfg);

        // peekModelDpSize() reads from engineWorkerStatus
        WorkerStatus prefillWs = new WorkerStatus();
        prefillWs.setIp("10.0.0.1");
        prefillWs.setPort(8080);
        prefillWs.setDpSize(4);
        prefillWs.setAlive(true);
        prefillWs.setGroup("g1");
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any()))
                .thenReturn(Map.of("10.0.0.1:8080", prefillWs));

        // selectPrefillWorker returns the worker
        when(planner.selectPrefillWorker(anyString(), any(), any(), any()))
                .thenReturn(prefillWs);

        // Default decode selection: always succeed with DECODE ServerStatus
        when(planner.selectDecodeWorker(any(), any())).thenReturn(DECODE);

        // Default gRPC client: ack accepted, capture sent batches
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class), anyLong()))
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
                grpcClient, registry, new LinearPrefillTimePredictor());
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
    void per_request_decode_failure_completes_victim_future_with_failure_response() throws Exception {
        // selectDecodeWorker fails on the 3rd call (0-indexed: call #2)
        AtomicInteger decodeCallCount = new AtomicInteger();
        when(planner.selectDecodeWorker(any(), any())).thenAnswer(inv -> {
            int call = decodeCallCount.getAndIncrement();
            if (call == 2) {
                return ServerStatus.code(StrategyErrorType.NO_DECODE_WORKER);
            }
            return DECODE;
        });

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .toList();

        // The "victim" future receives a failure response from the batcher.
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
        when(planner.selectPrefillWorker(anyString(), any(), any(), any()))
                .thenReturn(null);

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .toList();
        for (CompletableFuture<Response> f : futures) {
            Response r = f.get(2, TimeUnit.SECONDS);
            assertFalse(r.isSuccess());
            assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), r.getCode());
        }
        verify(grpcClient, never()).enqueue(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void enqueue_rejection_fails_all_batched_futures() throws Exception {
        cfg.setMaxDpBatchRetryCount(0);
        AtomicInteger callCount = new AtomicInteger();
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class), anyLong()))
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
        cfg.setMaxDpBatchRetryCount(0);
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class), anyLong()))
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
    void enqueue_rejection_retries_then_succeeds() throws Exception {
        AtomicInteger callCount = new AtomicInteger();
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    int n = callCount.incrementAndGet();
                    EngineRpcService.BatchEnqueueRequestPB b = inv.getArgument(2);
                    // First call rejects, second call accepts
                    return CompletableFuture.completedFuture(buildAck(b, n > 1, "queue full"));
                });

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1, "m1")))
                .toList();
        for (CompletableFuture<Response> f : futures) {
            Response resp = f.get(3, TimeUnit.SECONDS);
            assertTrue(resp.isSuccess());
        }
        assertEquals(2, callCount.get());
    }

    @Test
    void cancel_in_queue_yanks_request_and_completes_future_exceptionally() throws Exception {
        // Submit only 2 (under dpSize=4) so they sit in the queue.
        CompletableFuture<Response> f1 = scheduler.submit(makeCtx(1, "m1"));
        CompletableFuture<Response> f2 = scheduler.submit(makeCtx(2, "m1"));

        scheduler.cancel(1L);

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
    void submit_completes_future_exceptionally_when_planner_throws() throws Exception {
        when(planner.selectPrefillWorker(anyString(), any(), any(), any()))
                .thenThrow(new RuntimeException("simulated planner crash"));

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
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class), anyLong()))
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
        assertEquals(8081, prefill.getGrpcPort(),
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
        assertEquals(8081, dpRankToGrpcPort.get(0L).intValue(),
                "rank=0 missing from dpStatuses ⇒ keep pod-level grpcPort");
        assertEquals(8081, dpRankToGrpcPort.get(1L).intValue(),
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
    }

    @Test
    void debugInfo_hit_cache_len_reflects_enriched_cache_matched_tokens() throws Exception {
        // selectPrefillWorker computes cacheMatchedTokens and stashes it on
        // BalanceContext; buildSuccessResponse copies it into
        // response.serverStatus[prefill].debugInfo.hitCacheLen so the Python
        // smoke check role_hit_cache_len can read it.
        long expectedHitTokens = 320L;

        WorkerStatus prefillWs = new WorkerStatus();
        prefillWs.setIp("10.0.0.1");
        prefillWs.setPort(8080);
        prefillWs.setDpSize(2);
        prefillWs.setAlive(true);
        prefillWs.setGroup("g1");
        prefillWs.setCacheStatus(CacheStatus.builder().blockSize(64L).build());
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any()))
                .thenReturn(Map.of("10.0.0.1:8080", prefillWs));

        when(planner.selectPrefillWorker(anyString(), any(), any(), any()))
                .thenAnswer(inv -> {
                    BalanceContext ctx = inv.getArgument(2);
                    ctx.setCacheMatchedTokens(expectedHitTokens);
                    return prefillWs;
                });

        DpBatchScheduler s = new DpBatchScheduler(configService, engineWorkerStatus, planner,
                grpcClient, registry,
                new LinearPrefillTimePredictor());
        try {
            List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                    .mapToObj(i -> s.submit(makeCtx(i + 1, "m1")))
                    .toList();

            for (CompletableFuture<Response> f : futures) {
                Response r = f.get(2, TimeUnit.SECONDS);
                assertTrue(r.isSuccess());
                ServerStatus prefill = r.getServerStatus().get(0);
                assertNotNull(prefill.getDebugInfo(),
                        "prefill ServerStatus must carry DebugInfo so Python smoke "
                                + "check can read role_hit_cache_len");
                assertEquals(expectedHitTokens, prefill.getDebugInfo().getHitCacheLen(),
                        "hitCacheLen must reflect cacheMatchedTokens set by selectPrefillWorker");
            }
        } finally {
            s.shutdown();
        }
    }

    @Test
    void debugInfo_hit_cache_len_is_zero_when_planner_does_not_set_cache() throws Exception {
        // When selectPrefillWorker does not set cacheMatchedTokens (e.g. cache-aware
        // scheduling is disabled), the dispatched response carries a well-formed
        // DebugInfo with hitCacheLen=0.
        DpBatchScheduler s = new DpBatchScheduler(configService, engineWorkerStatus, planner,
                grpcClient, registry,
                new LinearPrefillTimePredictor());
        try {
            CompletableFuture<Response> f0 = s.submit(makeCtx(1, "m1"));
            s.submit(makeCtx(2, "m1"));
            s.submit(makeCtx(3, "m1"));
            s.submit(makeCtx(4, "m1"));

            Response r = f0.get(2, TimeUnit.SECONDS);
            assertTrue(r.isSuccess());
            assertNotNull(r.getServerStatus().get(0).getDebugInfo());
            assertEquals(0L, r.getServerStatus().get(0).getDebugInfo().getHitCacheLen());
        } finally {
            s.shutdown();
        }
    }

    @Test
    void all_inputs_carry_batch_group_id_equal_to_batchId_and_force_batch_set() throws Exception {
        IntStream.range(0, 4).forEach(i -> scheduler.submit(makeCtx(i + 1, "m1")));
        // Wait for flush
        Thread.sleep(150);

        assertEquals(1, sentBatches.size());
        EngineRpcService.BatchEnqueueRequestPB sent = sentBatches.get(0);
        long batchId = sent.getBatchId();
        for (EngineRpcService.GenerateInputPB in : sent.getInputsList()) {
            assertTrue(in.hasBatchGroupId(),
                    "every input (real or fake) must carry batch_group_id so engine "
                            + "groups them in one forward step");
            assertEquals(batchId, in.getBatchGroupId().getValue(),
                    "batch_group_id must equal the FlexLB-assigned batchId");
            assertTrue(in.getGenerateConfig().hasForceBatch()
                            && in.getGenerateConfig().getForceBatch().getValue() == 1,
                    "force_batch must be 1 so FIFOScheduler refuses to mix this batch with anything else");
            assertTrue(in.getBatchGroupSize() > 0,
                    "batch_group_size must be set so FIFOScheduler knows how many local "
                            + "streams to wait for before scheduling the group");
            assertTrue(in.getGenerateConfig().hasBatchGroupTimeout(),
                    "batch_group_timeout must be set as a safety fallback");
        }
    }

    @Test
    void unfilled_rank_gets_fake_pad() throws Exception {
        // dpSize=2, submit 1 request → RR puts it on rank=0, rank=1 gets fake pad.
        WorkerStatus dpSize2Worker = new WorkerStatus();
        dpSize2Worker.setIp("10.0.0.1");
        dpSize2Worker.setPort(8080);
        dpSize2Worker.setDpSize(2);
        dpSize2Worker.setAlive(true);
        dpSize2Worker.setGroup("g1");
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any()))
                .thenReturn(Map.of("10.0.0.1:8080", dpSize2Worker));
        when(planner.selectPrefillWorker(anyString(), any(), any(), any()))
                .thenReturn(dpSize2Worker);

        cfg.setDpBatchWindowMs(10);
        DpBatchScheduler s = new DpBatchScheduler(configService, engineWorkerStatus, planner,
                grpcClient, registry, new LinearPrefillTimePredictor());
        try {
            sentBatches.clear();
            CompletableFuture<Response> f1 = s.submit(makeCtx(1, "m1"));
            f1.get(2, TimeUnit.SECONDS);

            assertEquals(1, sentBatches.size());
            EngineRpcService.BatchEnqueueRequestPB sent = sentBatches.get(0);

            long realOnRank0 = sent.getInputsList().stream()
                    .filter(in -> !in.getIsFakeQuery())
                    .filter(in -> in.getDpRank().getValue() == 0)
                    .count();
            long fakeOnRank0 = sent.getInputsList().stream()
                    .filter(EngineRpcService.GenerateInputPB::getIsFakeQuery)
                    .filter(in -> in.getDpRank().getValue() == 0)
                    .count();
            long fakeOnRank1 = sent.getInputsList().stream()
                    .filter(EngineRpcService.GenerateInputPB::getIsFakeQuery)
                    .filter(in -> in.getDpRank().getValue() == 1)
                    .count();

            assertEquals(1, realOnRank0, "rank=0 must carry the real input");
            assertEquals(0, fakeOnRank0, "rank=0 already filled — no fake pad expected");
            assertEquals(1, fakeOnRank1, "rank=1 has zero real requests — pad with exactly one fake slot");
            assertEquals(2, sent.getInputsCount(), "1 real on rank=0 + 1 fake on rank=1");
        } finally {
            s.shutdown();
        }
    }

    @Test
    void all_ranks_filled_by_real_requests_emits_zero_fake_pad() throws Exception {
        // dpSize=4 + 4 real requests via round-robin covers ranks 0..3 once each.
        // No fake-pad should appear.
        IntStream.range(0, 4).forEach(i -> scheduler.submit(makeCtx(i + 1, "m1")));
        Thread.sleep(150);

        assertEquals(1, sentBatches.size());
        EngineRpcService.BatchEnqueueRequestPB sent = sentBatches.get(0);
        long fakeCount = sent.getInputsList().stream()
                .filter(EngineRpcService.GenerateInputPB::getIsFakeQuery)
                .count();
        assertEquals(0, fakeCount, "all ranks covered by real requests ⇒ no fake-pad slots");
        assertEquals(4, sent.getInputsCount());
    }

    @Test
    void dispatch_emits_BatchDispatch_log_line_for_smoke_scanner() throws Exception {
        // The dpsize=1 smoke (FLEXLB_SMOKE_CHECK_DPSIZE1_PACK) greps the flexlb
        // log for "BatchDispatch batchId=N dpSize=N realRequests=N totalInputs=N"
        // to detect serialization regressions. If anyone refactors this log line,
        // this test fails LOUDLY at unit-test time instead of silently in CI smoke.
        ch.qos.logback.classic.Logger lb =
                (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger("flexlbLogger");
        ch.qos.logback.core.read.ListAppender<ch.qos.logback.classic.spi.ILoggingEvent> appender =
                new ch.qos.logback.core.read.ListAppender<>();
        appender.start();
        lb.addAppender(appender);
        try {
            IntStream.range(0, 4).forEach(i -> scheduler.submit(makeCtx(i + 1, "m1")));
            Thread.sleep(150);

            java.util.regex.Pattern p = java.util.regex.Pattern.compile(
                    "BatchDispatch batchId=\\d+ dpSize=\\d+ realRequests=\\d+ totalInputs=\\d+");
            long hits = appender.list.stream()
                    .map(ch.qos.logback.classic.spi.ILoggingEvent::getFormattedMessage)
                    .filter(msg -> p.matcher(msg).find())
                    .count();
            assertTrue(hits >= 1,
                    "expected at least one BatchDispatch log line for the dpsize=1 smoke scanner; got: "
                            + appender.list.stream()
                                    .map(ch.qos.logback.classic.spi.ILoggingEvent::getFormattedMessage)
                                    .collect(Collectors.joining("\n")));
        } finally {
            lb.detachAppender(appender);
        }
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

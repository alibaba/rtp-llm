package org.flexlb.balance.scheduler;

import org.flexlb.balance.dp.InflightBatchRegistry;
import org.flexlb.balance.dp.RoundRobinAssign;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.dp.DpGrpcClient;
import org.flexlb.enums.LoadBalanceStrategyEnum;
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
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.atLeastOnce;
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
    private LoadBalancer prefillSelector;
    private LoadBalancer decodeSelector;
    private DpBatchScheduler scheduler;

    private final List<EngineRpcService.BatchGenerateInputPB> sentBatches = new CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        engineWorkerStatus = mock(EngineWorkerStatus.class);
        grpcClient = mock(DpGrpcClient.class);
        registry = new InflightBatchRegistry();
        prefillSelector = mock(LoadBalancer.class);
        decodeSelector = mock(LoadBalancer.class);

        cfg = new FlexlbConfig();
        cfg.setDpBalanceEnabled(true);
        cfg.setDpBatchSizeMax(4);
        cfg.setDpBatchWindowMs(20);
        cfg.setDpBatchTimeoutMs(100);
        when(configService.loadBalanceConfig()).thenReturn(cfg);

        // Register our mock selectors with the static factory under the strategies
        // returned by FlexlbConfig.getStrategyForRoleType(...).
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, prefillSelector);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, decodeSelector);

        // Default: prefill selection succeeds
        when(prefillSelector.select(any(BalanceContext.class), eq(RoleType.PREFILL), isNull()))
                .thenAnswer(inv -> okPrefill());
        when(decodeSelector.select(any(BalanceContext.class), eq(RoleType.DECODE), anyString()))
                .thenAnswer(inv -> okDecode());

        // Default gRPC client: ack accepted, capture sent batches
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchGenerateInputPB.class)))
                .thenAnswer(inv -> {
                    EngineRpcService.BatchGenerateInputPB b = inv.getArgument(2);
                    sentBatches.add(b);
                    return CompletableFuture.completedFuture(EngineRpcService.EnqueueAckPB.newBuilder()
                            .setBatchId(b.getBatchId())
                            .setAccepted(true).build());
                });
        when(grpcClient.cancelPrefill(anyString(), anyInt(), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(null));
        when(grpcClient.cancelDecode(anyString(), anyInt(), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(null));

        scheduler = new DpBatchScheduler(configService, engineWorkerStatus,
                new RoundRobinAssign(), grpcClient, registry);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void four_requests_form_one_batch_each_assigned_distinct_dpRank() throws Exception {
        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1)))
                .toList();

        // dpBatchSizeMax=4 reached → flush immediately
        for (CompletableFuture<Response> f : futures) {
            Response resp = f.get(2, TimeUnit.SECONDS);
            assertTrue(resp.isSuccess(), "successfully-enqueued requests' futures must succeed");
            assertTrue(resp.isEnqueuedByMaster(),
                    "enqueued_by_master=true is the signal for frontend to switch to Decode.FetchResponse");
            assertEquals(2, resp.getServerStatus().size(),
                    "response must include both prefill and decode ServerStatus entries");
        }

        assertEquals(1, sentBatches.size(), "4 requests should form a single batch");
        EngineRpcService.BatchGenerateInputPB sent = sentBatches.get(0);
        assertEquals(4, sent.getInputsCount());

        // Every dp_rank must be distinct (RR + dpSize=4)
        List<Integer> ranks = IntStream.range(0, 4)
                .map(i -> (int) sent.getInputs(i).getDpRank())
                .boxed().collect(Collectors.toList());
        assertEquals(4, ranks.stream().distinct().count(), "4 requests must land on 4 distinct ranks");
        assertTrue(ranks.stream().allMatch(r -> r >= 0 && r < 4));
    }

    @Test
    void prefill_selection_failure_completes_future_with_NO_PREFILL_WORKER() throws Exception {
        ServerStatus failed = ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        failed.setMessage("no prefill in cluster");
        when(prefillSelector.select(any(BalanceContext.class), eq(RoleType.PREFILL), isNull()))
                .thenReturn(failed);

        Response resp = scheduler.submit(makeCtx(1)).get(1, TimeUnit.SECONDS);
        assertFalse(resp.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), resp.getCode());
        verify(grpcClient, never()).enqueue(anyString(), anyInt(), any());
    }

    @Test
    void decode_selection_failure_rolls_back_prefill_reservation() throws Exception {
        ServerStatus failed = ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        failed.setMessage("no decode in group");
        when(decodeSelector.select(any(BalanceContext.class), eq(RoleType.DECODE), anyString()))
                .thenReturn(failed);

        Response resp = scheduler.submit(makeCtx(1)).get(1, TimeUnit.SECONDS);
        assertFalse(resp.isSuccess());
        assertEquals(StrategyErrorType.NO_DECODE_WORKER.getErrorCode(), resp.getCode());
        // Critical: prefill rollBack must be called so the ShortestTTFT local-cache
        // reservation does not accumulate.
        verify(prefillSelector, atLeastOnce()).rollBack(anyString(), anyLong());
        verify(grpcClient, never()).enqueue(anyString(), anyInt(), any());
    }

    @Test
    void enqueue_rejection_fails_all_batched_futures_and_rolls_back() throws Exception {
        AtomicInteger callCount = new AtomicInteger();
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchGenerateInputPB.class)))
                .thenAnswer(inv -> {
                    callCount.incrementAndGet();
                    return CompletableFuture.completedFuture(EngineRpcService.EnqueueAckPB.newBuilder()
                            .setBatchId(inv.<EngineRpcService.BatchGenerateInputPB>getArgument(2).getBatchId())
                            .setAccepted(false)
                            .setErrorMessage("queue full").build());
                });

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1)))
                .toList();

        for (CompletableFuture<Response> f : futures) {
            assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> f.get(2, TimeUnit.SECONDS),
                    "a rejected ack must completeExceptionally every future in the batch");
        }
        assertEquals(1, callCount.get());
        // rollback should be called once per request
        verify(prefillSelector, times(4)).rollBack(anyString(), anyLong());
    }

    @Test
    void enqueue_rpc_failure_fails_futures() throws Exception {
        when(grpcClient.enqueue(anyString(), anyInt(), any(EngineRpcService.BatchGenerateInputPB.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("UNAVAILABLE")));

        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(i + 1)))
                .toList();

        for (CompletableFuture<Response> f : futures) {
            assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> f.get(2, TimeUnit.SECONDS));
        }
    }

    @Test
    void cancel_cascades_to_both_prefill_and_decode_when_in_flight() throws Exception {
        // Submit + flush first so the entries land in the inflight registry.
        List<CompletableFuture<Response>> futures = IntStream.range(0, 4)
                .mapToObj(i -> scheduler.submit(makeCtx(100 + i)))
                .toList();
        for (CompletableFuture<Response> f : futures) {
            f.get(2, TimeUnit.SECONDS);
        }

        // Now cancel request 101
        scheduler.cancel(101L);

        verify(grpcClient, times(1)).cancelPrefill(anyString(), anyInt(), eq(101L));
        verify(grpcClient, times(1)).cancelDecode(anyString(), anyInt(), eq(101L));
        // The corresponding registry entry must be cleared.
        assertNull(registry.lookupByRequest(101L));
    }

    @Test
    void cancel_unknown_requestId_is_silent_noop() {
        scheduler.cancel(99999L);
        verify(grpcClient, never()).cancelPrefill(anyString(), anyInt(), anyLong());
        verify(grpcClient, never()).cancelDecode(anyString(), anyInt(), anyLong());
    }

    @Test
    void window_timeout_flushes_partial_batch() throws Exception {
        // Only submit 2; rely on the window timeout to flush a partial batch.
        CompletableFuture<Response> f1 = scheduler.submit(makeCtx(1));
        CompletableFuture<Response> f2 = scheduler.submit(makeCtx(2));

        Response r1 = f1.get(2, TimeUnit.SECONDS);
        Response r2 = f2.get(2, TimeUnit.SECONDS);
        assertTrue(r1.isSuccess() && r2.isSuccess());
        assertEquals(1, sentBatches.size());
        assertEquals(2, sentBatches.get(0).getInputsCount(), "window timeout flushes a partial batch");
    }

    // ============== helpers ==============

    private BalanceContext makeCtx(long requestId) {
        BalanceContext ctx = new BalanceContext();
        Request req = new Request();
        req.setRequestId(requestId);
        req.setSeqLen(100);
        req.setMaxNewTokens(128);
        req.setNumBeams(1);
        req.setBlockCacheKeys(List.of(1L, 2L, 3L));
        req.setModel("m1");
        ctx.setRequest(req);
        return ctx;
    }

    private static ServerStatus okPrefill() {
        ServerStatus s = new ServerStatus();
        s.setSuccess(true);
        s.setRole(RoleType.PREFILL);
        s.setServerIp("10.0.0.1");
        s.setHttpPort(8080);
        s.setGrpcPort(9080);
        s.setGroup("g1");
        return s;
    }

    private static ServerStatus okDecode() {
        ServerStatus s = new ServerStatus();
        s.setSuccess(true);
        s.setRole(RoleType.DECODE);
        s.setServerIp("10.0.0.2");
        s.setHttpPort(8081);
        s.setGrpcPort(9081);
        s.setGroup("g1");
        return s;
    }
}

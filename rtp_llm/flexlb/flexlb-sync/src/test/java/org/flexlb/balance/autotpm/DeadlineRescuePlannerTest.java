package org.flexlb.balance.autotpm;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatcherContext;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.QueueSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link DeadlineRescuePlanner}.
 *
 * <p>Uses Mockito to mock {@link BatcherContext}, {@link FlexlbBatchScheduler},
 * {@link PriorityAdmissionScheduler}, {@link EndpointRegistry}, and
 * {@link ConfigService}. The {@link DecodeAdmissionTracker} is a real
 * instance (no mocking needed for the release call).
 */
class DeadlineRescuePlannerTest {

    private ConfigService configService;
    private EndpointRegistry endpointRegistry;
    private FlexlbBatchScheduler batchScheduler;
    private PriorityAdmissionScheduler admissionScheduler;
    private DecodeAdmissionTracker decodeTracker;
    private DeadlineRescuePlanner planner;

    private BatcherContext batcherCtx;
    private PrefillEndpoint prefillEp;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        endpointRegistry = mock(EndpointRegistry.class);
        batchScheduler = mock(FlexlbBatchScheduler.class);
        admissionScheduler = mock(PriorityAdmissionScheduler.class);
        decodeTracker = new DecodeAdmissionTracker();

        planner = new DeadlineRescuePlanner(
                configService, endpointRegistry, batchScheduler,
                admissionScheduler, decodeTracker);

        batcherCtx = mock(BatcherContext.class);
        prefillEp = mock(PrefillEndpoint.class);
        when(prefillEp.getBatcherContext()).thenReturn(batcherCtx);

        ConcurrentHashMap<String, PrefillEndpoint> endpoints = new ConcurrentHashMap<>();
        endpoints.put("10.0.0.1:8080", prefillEp);
        when(endpointRegistry.getPrefillEndpoints()).thenReturn(endpoints);

        // Default: admission succeeds
        when(admissionScheduler.submit(any())).thenReturn(CompletableFuture.completedFuture(successResp()));
    }

    // ==================== Danger zone rescue ====================

    @Test
    void dangerZoneRequest_getsRescued() {
        long now = System.currentTimeMillis();
        long requestId = 1001L;
        setupConfig(100L, 1, 32);
        setupSnapshot(requestId, 50, now, 100);
        setupTryRemove(requestId, 50, now);

        planner.scan();

        verify(batcherCtx, times(1)).tryRemove(Set.of(requestId), 0L);
        verify(batchScheduler, times(1)).removeInflightForRescue(requestId);
        verify(admissionScheduler, times(1)).submit(any());
    }

    // ==================== Non-danger request not rescued ====================

    @Test
    void nonDangerRequest_notRescued() {
        long now = System.currentTimeMillis();
        long requestId = 2002L;
        setupConfig(100L, 1, 32);
        // deadline far in the future — not in danger zone
        setupSnapshot(requestId, 50, now + 10_000, 100);

        planner.scan();

        verify(batcherCtx, never()).tryRemove(any(), anyLong());
        verify(batchScheduler, never()).removeInflightForRescue(anyLong());
    }

    // ==================== P30 not rescued ====================

    @Test
    void p30LowestPriority_notRescued() {
        long now = System.currentTimeMillis();
        long requestId = 3003L;
        setupConfig(100L, 1, 32);
        // P30 in danger zone — should be skipped
        setupSnapshot(requestId, 30, now, 100);

        planner.scan();

        verify(batcherCtx, never()).tryRemove(any(), anyLong());
        verify(batchScheduler, never()).removeInflightForRescue(anyLong());
    }

    // ==================== max_transfer cap ====================

    @Test
    void maxTransfer_secondRescueRejected() {
        long now = System.currentTimeMillis();
        long requestId = 4004L;
        setupConfig(100L, 1, 32);
        setupSnapshot(requestId, 50, now, 100);
        setupTryRemove(requestId, 50, now);

        // First scan: rescues the request
        planner.scan();
        verify(batcherCtx, times(1)).tryRemove(Set.of(requestId), 0L);

        // Second scan: same request still in snapshot (re-admitted to queue)
        // but transfer count = 1 >= max_transfer = 1, so NOT rescued
        planner.scan();

        // tryRemove should still be called only once (first scan)
        verify(batcherCtx, times(1)).tryRemove(Set.of(requestId), 0L);
    }

    // ==================== max_rescue_per_tick ====================

    @Test
    void maxRescuePerTick_stopsAfterLimit() {
        long now = System.currentTimeMillis();
        setupConfig(100L, 1, 2);

        // 5 items in danger zone
        long[] requestIds = {10L, 11L, 12L, 13L, 14L};
        List<QueueSnapshot.ItemSummary> items = new ArrayList<>();
        for (long id : requestIds) {
            items.add(new QueueSnapshot.ItemSummary(id, 50, now, 100));
        }
        setupSnapshot(items);

        for (long id : requestIds) {
            setupTryRemove(id, 50, now);
        }

        planner.scan();

        // Only 2 rescues should have happened (max_rescue_per_tick=2)
        verify(batchScheduler, times(2)).removeInflightForRescue(anyLong());
        verify(admissionScheduler, times(2)).submit(any());
    }

    // ==================== CAS version mismatch ====================

    @Test
    void casVersionMismatch_rescueFailsNoCrash() {
        long now = System.currentTimeMillis();
        long requestId = 5005L;
        setupConfig(100L, 1, 32);
        setupSnapshot(requestId, 50, now, 100);

        // tryRemove returns null (version mismatch)
        when(batcherCtx.tryRemove(any(), anyLong())).thenReturn(null);

        planner.scan();

        // Rescue failed: removeInflightForRescue should NOT be called
        verify(batchScheduler, never()).removeInflightForRescue(anyLong());
        verify(admissionScheduler, never()).submit(any());
    }

    // ==================== Re-admission failure ====================

    @Test
    void reAdmissionFailure_futureCompletedWithDeadLineRescueFailed() {
        long now = System.currentTimeMillis();
        long requestId = 6006L;
        setupConfig(100L, 1, 32);
        setupSnapshot(requestId, 50, now, 100);

        BatchItem item = makeBatchItem(requestId, 50, now);
        when(batcherCtx.tryRemove(Set.of(requestId), 0L))
                .thenReturn(List.of(item));

        // Re-admission returns a failed future (QUEUE_FULL)
        Response errorResp = new Response();
        errorResp.setSuccess(false);
        errorResp.setErrorMessage("queue full");
        when(admissionScheduler.submit(any()))
                .thenReturn(CompletableFuture.completedFuture(errorResp));

        planner.scan();

        // Original future should be completed with DEADLINE_RESCUE_FAILED
        assertTrue(item.future().isDone(), "original future must be completed");
        Response result = item.future().getNow(null);
        assertEquals("DEADLINE_RESCUE_FAILED", result.getErrorMessage().substring(0, "DEADLINE_RESCUE_FAILED".length()));
    }

    // ==================== Empty queue ====================

    @Test
    void emptyQueue_noRescue() {
        setupConfig(100L, 1, 32);
        setupSnapshot(List.of());

        planner.scan();

        verify(batcherCtx, never()).tryRemove(any(), anyLong());
        verify(batchScheduler, never()).removeInflightForRescue(anyLong());
    }

    // ==================== Deadline not set (0) ====================

    @Test
    void deadlineNotSet_notRescued() {
        long requestId = 7007L;
        setupConfig(100L, 1, 32);
        // deadline = 0 (not set)
        setupSnapshot(requestId, 50, 0L, 100);

        planner.scan();

        verify(batcherCtx, never()).tryRemove(any(), anyLong());
    }

    // ==================== Multiple endpoints ====================

    @Test
    void multipleEndpoints_rescuesFromAll() {
        long now = System.currentTimeMillis();
        setupConfig(100L, 1, 32);

        // Setup two endpoints
        BatcherContext ctx2 = mock(BatcherContext.class);
        PrefillEndpoint ep2 = mock(PrefillEndpoint.class);
        when(ep2.getBatcherContext()).thenReturn(ctx2);

        ConcurrentHashMap<String, PrefillEndpoint> endpoints = new ConcurrentHashMap<>();
        endpoints.put("10.0.0.1:8080", prefillEp);
        endpoints.put("10.0.0.2:8080", ep2);
        when(endpointRegistry.getPrefillEndpoints()).thenReturn(endpoints);

        // Both have a danger-zone item
        setupSnapshot(1L, 50, now, 100);
        setupTryRemove(1L, 50, now);

        List<QueueSnapshot.ItemSummary> items2 = List.of(
                new QueueSnapshot.ItemSummary(2L, 60, now, 100));
        when(ctx2.snapshot()).thenReturn(new QueueSnapshot(0L, 1, items2));
        BatchItem item2 = makeBatchItem(2L, 60, now);
        when(ctx2.tryRemove(Set.of(2L), 0L)).thenReturn(List.of(item2));

        planner.scan();

        verify(batchScheduler, times(1)).removeInflightForRescue(1L);
        verify(batchScheduler, times(1)).removeInflightForRescue(2L);
    }

    // ==================== Helpers ====================

    private void setupConfig(long dangerThreshold, int maxTransfer, int maxRescuePerTick) {
        FlexlbConfig config = new FlexlbConfig();
        config.setAutoTpmDangerThresholdMs(dangerThreshold);
        config.setAutoTpmMaxTransfer(maxTransfer);
        config.setAutoTpmMaxRescuePerTick(maxRescuePerTick);
        when(configService.loadBalanceConfig()).thenReturn(config);
    }

    private void setupSnapshot(long requestId, int priority, long deadlineMs, long seqLen) {
        List<QueueSnapshot.ItemSummary> items = List.of(
                new QueueSnapshot.ItemSummary(requestId, priority, deadlineMs, seqLen));
        setupSnapshot(items);
    }

    private void setupSnapshot(List<QueueSnapshot.ItemSummary> items) {
        when(batcherCtx.snapshot()).thenReturn(new QueueSnapshot(0L, items.size(), items));
    }

    private void setupTryRemove(long requestId, int priority, long deadlineMs) {
        BatchItem item = makeBatchItem(requestId, priority, deadlineMs);
        when(batcherCtx.tryRemove(Set.of(requestId), 0L))
                .thenReturn(List.of(item));
    }

    private static BatchItem makeBatchItem(long requestId, int priority, long deadlineMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(100);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setPriority(priority);
        ctx.setDeadlineMs(deadlineMs);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = new BatchItem(ctx, future, null, null, null,
                null, null, System.currentTimeMillis());
        item.setPriority(priority);
        item.setDeadlineMs(deadlineMs);
        return item;
    }

    private static Response successResp() {
        Response resp = new Response();
        resp.setSuccess(true);
        resp.setCode(200);
        return resp;
    }
}

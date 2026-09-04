package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.lang.reflect.Field;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Drop-valve gating tests for {@link SloBudgetBatcherAlgorithm}: the
 * deadline_expired and inflight_full_guard drops are gated solely on the
 * Auto-TPM switch — normalize() assigns every production request a 1-100
 * priority, so the former hasPriority() gate was removed (dead code); when
 * the switch is off the legacy drops stay active for everyone.
 */
class SloBudgetBatcherAlgorithmTest {

    // ---- deadline_expired valve ----

    @Test
    void autoTpmOnHeadWithoutPriorityFieldPastDeadlineFallsIntoDeadlineGuard() throws InterruptedException {
        // hasPriority gate removed: the exemption depends only on the switch,
        // so even a head whose priority field was never set (impossible in
        // production — normalize() always assigns 1-100) is not dropped.
        FlexlbConfig config = autoTpmOnConfig();
        PrefillEndpoint endpoint = endpoint(0);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 500, 100, 0);
        head.setSortKey(System.currentTimeMillis() - 100); // budget < 0
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler, never()).onExpired(any(BatchItem.class));
        ArgumentCaptor<DispatchMeta> meta = ArgumentCaptor.forClass(DispatchMeta.class);
        verify(handler).onBatchReady(anyList(), meta.capture());
        assertEquals("deadline_guard", meta.getValue().reason());
    }

    @Test
    void autoTpmOnPriorityHeadPastDeadlineIsNotDroppedAndFallsIntoDeadlineGuard() throws InterruptedException {
        FlexlbConfig config = autoTpmOnConfig();
        PrefillEndpoint endpoint = endpoint(0);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 500, 100, 50);
        head.setSortKey(System.currentTimeMillis() - 100); // budget < 0
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler, never()).onExpired(any(BatchItem.class));
        ArgumentCaptor<DispatchMeta> meta = ArgumentCaptor.forClass(DispatchMeta.class);
        verify(handler).onBatchReady(anyList(), meta.capture());
        assertEquals("deadline_guard", meta.getValue().reason());
    }

    @Test
    void autoTpmOffLegacyHeadPastDeadlineIsDroppedParity() throws InterruptedException {
        FlexlbConfig config = autoTpmOffConfig();
        PrefillEndpoint endpoint = endpoint(0);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 500, 100, 0);
        head.setSortKey(System.currentTimeMillis() - 100);
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler).onExpired(head);
        assertEquals(0, ctx.size());
    }

    @Test
    void autoTpmOffPriorityHeadPastDeadlineIsStillDroppedParity() throws InterruptedException {
        // Pre-fix behavior: with the switch off, priority carried on the item
        // has no effect — the legacy drop applies to everyone.
        FlexlbConfig config = autoTpmOffConfig();
        PrefillEndpoint endpoint = endpoint(0);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 500, 100, 50);
        head.setSortKey(System.currentTimeMillis() - 100);
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler).onExpired(head);
        assertEquals(0, ctx.size());
    }

    // ---- inflight_full_guard valve ----

    @Test
    void autoTpmOnHeadWithoutPriorityFieldUnderInflightGuardIsParkedNotDropped() throws InterruptedException {
        // hasPriority gate removed: the inflight_full_guard exemption also
        // depends only on the switch — the head is parked, never dropped.
        FlexlbConfig config = autoTpmOnConfig();
        config.setFlexlbBatchSloMaxInflightBatches(1);
        PrefillEndpoint endpoint = endpoint(1); // backpressure active
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 100, 100, 0);
        head.setSortKey(System.currentTimeMillis() + 10); // 0 < budget <= guard(40)
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler, never()).onExpired(any(BatchItem.class));
        verify(handler, never()).onBatchReady(anyList(), any(DispatchMeta.class));
        assertEquals(1, ctx.size(), "parked head must stay queued");
    }

    @Test
    void autoTpmOnPriorityHeadUnderInflightGuardIsParkedNotDropped() throws InterruptedException {
        FlexlbConfig config = autoTpmOnConfig();
        config.setFlexlbBatchSloMaxInflightBatches(1);
        PrefillEndpoint endpoint = endpoint(1);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 100, 100, 50);
        head.setSortKey(System.currentTimeMillis() + 10);
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler, never()).onExpired(any(BatchItem.class));
        verify(handler, never()).onBatchReady(anyList(), any(DispatchMeta.class));
        assertEquals(1, ctx.size(), "parked head must stay queued");
    }

    @Test
    void autoTpmOffLegacyHeadUnderInflightGuardIsDroppedParity() throws InterruptedException {
        FlexlbConfig config = autoTpmOffConfig();
        config.setFlexlbBatchSloMaxInflightBatches(1);
        PrefillEndpoint endpoint = endpoint(1);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 100, 100, 0);
        head.setSortKey(System.currentTimeMillis() + 10);
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler).onExpired(head);
        assertEquals(0, ctx.size());
    }

    // ---- park trace lifecycle on the drop path (F1 connected verification) ----

    @Test
    void dropPathRemovesParkTraceRecordedWhileParked() throws Exception {
        FlexlbConfig config = autoTpmOffConfig();
        config.setFlexlbBatchSloMaxInflightBatches(1);
        PrefillEndpoint endpoint = endpoint(1);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 100, 100, 0);
        head.setSortKey(System.currentTimeMillis() + 200); // budget > guard(40) → park
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));
        SloBudgetBatcherAlgorithm algorithm = new SloBudgetBatcherAlgorithm();

        algorithm.processQueue(ctx);
        assertFalse(parkTraces(algorithm).isEmpty(), "park must record a trace for the head");

        head.setSortKey(System.currentTimeMillis() - 10); // now expired → drop
        algorithm.processQueue(ctx);

        verify(handler).onExpired(head);
        assertTrue(parkTraces(algorithm).isEmpty(), "drop must remove the head's park trace");
        assertEquals(0, ctx.size());
    }

    // ---- fail-fast overestimate drop (P1-4, PR-D) ----

    /**
     * Behavioral pin (design decision, not a bug): when Auto-TPM is ON and the
     * head's dispatch deadline has passed, the fail-fast check compares
     * {@code remainingBudgetMs} against {@code estimatedPrefillMs}. An
     * overestimate causes a drop even though the request could theoretically
     * still meet its SLO — this is the intentional fail-fast trade-off.
     *
     * <p>Two cases at the threshold boundary:
     * <ul>
     *   <li>remainingBudgetMs(200) &lt; estimatedPrefillMs(250) → drop</li>
     *   <li>remainingBudgetMs(300) &ge; estimatedPrefillMs(250) → dispatch (deadline_guard)</li>
     * </ul>
     */
    @Test
    void failFast_dropAndPassThresholdAroundEstimatedPrefill() throws InterruptedException {
        // Case 1: remainingBudgetMs(200) < estimatedPrefillMs(250) → drop
        {
            FlexlbConfig config = autoTpmOnConfig();
            PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
            PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
            when(endpoint.getPredictor()).thenReturn(predictor);
            when(endpoint.getInflightBatchCount()).thenReturn(0);
            when(endpoint.realWaitTimeMs()).thenReturn(0L);
            when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(250L);
            when(predictor.predictBatchMs(anyList())).thenReturn(250.0);
            when(predictor.predictBatchMsUncached(anyList())).thenReturn(250.0);
            BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
            long now = System.currentTimeMillis();
            BatchItem head = item(1L, now - 500, 100, 50);
            head.ctx().setBudget(ScheduleBudget.forDeadline(50, now - 500, now + 200));
            head.setSortKey(now - 100);
            BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

            new SloBudgetBatcherAlgorithm().processQueue(ctx);

            verify(handler).onExpired(head);
            assertEquals(0, ctx.size(), "fail-fast must drop the head");
        }

        // Case 2: remainingBudgetMs(300) >= estimatedPrefillMs(250) → pass (deadline_guard)
        {
            FlexlbConfig config = autoTpmOnConfig();
            PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
            PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
            when(endpoint.getPredictor()).thenReturn(predictor);
            when(endpoint.getInflightBatchCount()).thenReturn(0);
            when(endpoint.realWaitTimeMs()).thenReturn(0L);
            when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(250L);
            when(predictor.predictBatchMs(anyList())).thenReturn(250.0);
            when(predictor.predictBatchMsUncached(anyList())).thenReturn(250.0);
            BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
            long now = System.currentTimeMillis();
            BatchItem head = item(2L, now - 500, 100, 50);
            head.ctx().setBudget(ScheduleBudget.forDeadline(50, now - 500, now + 300));
            head.setSortKey(now - 100);
            BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

            new SloBudgetBatcherAlgorithm().processQueue(ctx);

            verify(handler, never()).onExpired(any(BatchItem.class));
            ArgumentCaptor<DispatchMeta> meta = ArgumentCaptor.forClass(DispatchMeta.class);
            verify(handler).onBatchReady(anyList(), meta.capture());
            assertEquals("deadline_guard", meta.getValue().reason());
        }
    }

    @Test
    void longHeadBelowMaxSeqLenIsDispatchedAloneAboveBatchTokenBudget()
            throws InterruptedException {
        FlexlbConfig config = autoTpmOnConfig();
        config.setFlexlbBatchMaxCapacity(1_048_576);

        WorkerStatus status = new WorkerStatus();
        status.setMaxSeqLen(1_048_576L);
        status.setMaxBatchTokensSize(409_600L);
        PrefillEndpoint endpoint = endpoint(0);
        when(endpoint.getStatus()).thenReturn(status);

        long now = System.currentTimeMillis();
        BatchItem longHead = item(11L, now, 1_048_575L, 50);
        longHead.setSortKey(now - 1L);
        BatchItem next = item(12L, now + 1L, 1_024L, 50);
        next.setSortKey(now);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatcherContext ctx = context(
                endpoint, config, handler, queueWith(longHead, next));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        ArgumentCaptor<List<BatchItem>> dispatched = ArgumentCaptor.forClass(List.class);
        verify(handler).onBatchReady(dispatched.capture(), any(DispatchMeta.class));
        verify(handler, never()).onOfferFailure(any(), any());
        assertEquals(List.of(longHead), dispatched.getValue());
        assertEquals(List.of(next), ctx.sortedItems());
    }

    // ---- helpers ----

    private static FlexlbConfig autoTpmOnConfig() {
        FlexlbConfig config = autoTpmOffConfig();
        config.setAutoTpmEnabled(true);
        return config;
    }

    private static FlexlbConfig autoTpmOffConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("slo_budget");
        return config;
    }

    private static PrefillEndpoint endpoint(int inflightBatchCount) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getInflightBatchCount()).thenReturn(inflightBatchCount);
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(0L);
        when(predictor.predictBatchMs(anyList())).thenReturn(0.0);
        when(predictor.predictBatchMsUncached(anyList())).thenReturn(0.0);
        return endpoint;
    }

    private static BatchItem item(long requestId, long enqueuedAtMs, long seqLen, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        balanceContext.setBudget(ScheduleBudget.forDeadline(priority, enqueuedAtMs,
                enqueuedAtMs + 30_000));
        BatchItem item = new BatchItem(
                balanceContext, null, null, null, null, null, null, enqueuedAtMs);
        return item;
    }

    private static PriorityBlockingQueue<BatchItem> queueWith(BatchItem... items) {
        PriorityBlockingQueue<BatchItem> queue =
                new PriorityBlockingQueue<>(11, Comparator.comparingLong(BatchItem::sortKey));
        for (BatchItem item : items) {
            queue.add(item);
        }
        return queue;
    }

    private static BatcherContext context(PrefillEndpoint endpoint, FlexlbConfig config,
                                          BatchDecisionHandler handler,
                                          PriorityBlockingQueue<BatchItem> queue) {
        return new BatcherContext("test", endpoint, config, handler, queue,
                new AtomicInteger(queue.size()), mock(BatchSchedulerReporter.class));
    }

    @SuppressWarnings("unchecked")
    private static Map<Long, ?> parkTraces(SloBudgetBatcherAlgorithm algorithm) throws Exception {
        Field field = SloBudgetBatcherAlgorithm.class.getDeclaredField("lastParkByRequest");
        field.setAccessible(true);
        return (Map<Long, ?>) field.get(algorithm);
    }
}

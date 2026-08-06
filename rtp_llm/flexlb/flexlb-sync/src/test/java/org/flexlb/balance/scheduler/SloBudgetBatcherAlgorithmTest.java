package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.lang.reflect.Field;
import java.util.Comparator;
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
 * Drop-valve gating tests for {@link SloBudgetBatcherAlgorithm} (P0-1 fix):
 * the deadline_expired and inflight_full_guard drops are exempted only for
 * "Auto-TPM enabled AND head has priority"; no-priority (legacy) heads keep
 * the legacy drop protection regardless of the global switch.
 */
class SloBudgetBatcherAlgorithmTest {

    // ---- deadline_expired valve ----

    @Test
    void autoTpmOnLegacyHeadPastDeadlineIsDroppedAndReleasesQueueSlot() throws InterruptedException {
        FlexlbConfig config = autoTpmOnConfig();
        PrefillEndpoint endpoint = endpoint(0);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 500, 100, 0);
        head.setSortKey(System.currentTimeMillis() - 100); // budget < 0
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler).onExpired(head);
        verify(handler, never()).onBatchReady(anyList(), any(DispatchMeta.class));
        assertEquals(0, ctx.size(), "drop must release the reserved queue slot");
        assertTrue(ctx.isEmpty());
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
    void autoTpmOnLegacyHeadUnderInflightGuardIsDropped() throws InterruptedException {
        FlexlbConfig config = autoTpmOnConfig();
        config.setFlexlbBatchSloMaxInflightBatches(1);
        PrefillEndpoint endpoint = endpoint(1); // backpressure active
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 100, 100, 0);
        head.setSortKey(System.currentTimeMillis() + 10); // 0 < budget <= guard(40)
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler).onExpired(head);
        assertEquals(0, ctx.size(), "inflight_full_guard drop must release the queue slot");
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
        BatchItem item = new BatchItem(
                balanceContext, null, null, null, null, null, null, enqueuedAtMs);
        item.setPriority(priority);
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

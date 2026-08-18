package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.Comparator;
import java.util.List;
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

    @Test
    void unavailableWaitEstimateConsumesAllBatchingSlackWithoutOverflow() {
        FlexlbConfig config = autoTpmOnConfig();
        PrefillEndpoint endpoint = endpoint(0);
        PrefillTimePredictor predictor = endpoint.getPredictor();
        when(endpoint.realWaitTimeMs()).thenReturn(Long.MAX_VALUE);
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(Long.MAX_VALUE / 2);
        BatcherContext ctx = context(endpoint, config, mock(DecisionGroupHandler.class),
                queueWith());
        BatchItem item = item(1L, System.currentTimeMillis(), 100, 50);
        SloBudgetBatcherAlgorithm algorithm = new SloBudgetBatcherAlgorithm();

        long beforeMs = System.currentTimeMillis();
        long sortKey = algorithm.computeSortKey(ctx, item);
        long afterMs = System.currentTimeMillis();

        assertTrue(sortKey >= beforeMs && sortKey <= afterMs,
                "an incoherent worker wait must make the request immediately actionable");
    }

    // ---- deadline_expired valve ----

    @Test
    void autoTpmOnHeadWithoutPriorityFieldPastDeadlineFallsIntoDeadlineGuard() throws InterruptedException {
        // hasPriority gate removed: the exemption depends only on the switch,
        // so even a head whose priority field was never set (impossible in
        // production — normalize() always assigns 1-100) is not dropped.
        FlexlbConfig config = autoTpmOnConfig();
        PrefillEndpoint endpoint = endpoint(0);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 500, 100, 0);
        head.setSortKey(System.currentTimeMillis() - 100); // budget < 0
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler, never()).onExpired(any(BatchItem.class));
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(anyList(), meta.capture());
        assertEquals("deadline_guard", meta.getValue().reason());
    }

    @Test
    void autoTpmOnPriorityHeadPastDeadlineIsNotDroppedAndFallsIntoDeadlineGuard() throws InterruptedException {
        FlexlbConfig config = autoTpmOnConfig();
        PrefillEndpoint endpoint = endpoint(0);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 500, 100, 50);
        head.setSortKey(System.currentTimeMillis() - 100); // budget < 0
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler, never()).onExpired(any(BatchItem.class));
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(handler).onDecisionGroupReady(anyList(), meta.capture());
        assertEquals("deadline_guard", meta.getValue().reason());
    }

    @Test
    void autoTpmOffLegacyHeadPastDeadlineIsDroppedParity() throws InterruptedException {
        FlexlbConfig config = autoTpmOffConfig();
        PrefillEndpoint endpoint = endpoint(0);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
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
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
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
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 100, 100, 0);
        head.setSortKey(System.currentTimeMillis() + 10); // 0 < budget <= guard(40)
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler, never()).onExpired(any(BatchItem.class));
        verify(handler, never()).onDecisionGroupReady(anyList(), any(DecisionGroupMetadata.class));
        assertEquals(1, ctx.size(), "parked head must stay queued");
    }

    @Test
    void autoTpmOnPriorityHeadUnderInflightGuardIsParkedNotDropped() throws InterruptedException {
        FlexlbConfig config = autoTpmOnConfig();
        config.setFlexlbBatchSloMaxInflightBatches(1);
        PrefillEndpoint endpoint = endpoint(1);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 100, 100, 50);
        head.setSortKey(System.currentTimeMillis() + 10);
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        verify(handler, never()).onExpired(any(BatchItem.class));
        verify(handler, never()).onDecisionGroupReady(anyList(), any(DecisionGroupMetadata.class));
        assertEquals(1, ctx.size(), "parked head must stay queued");
    }

    @Test
    void autoTpmOffLegacyHeadUnderInflightGuardIsDroppedParity() throws InterruptedException {
        FlexlbConfig config = autoTpmOffConfig();
        config.setFlexlbBatchSloMaxInflightBatches(1);
        PrefillEndpoint endpoint = endpoint(1);
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
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
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        BatchItem head = item(1L, System.currentTimeMillis() - 100, 100, 0);
        head.setSortKey(System.currentTimeMillis() + 200); // budget > guard(40) → park
        BatcherContext ctx = context(endpoint, config, handler, queueWith(head));
        SloBudgetBatcherAlgorithm algorithm = new SloBudgetBatcherAlgorithm();

        algorithm.processQueue(ctx);
        assertTrue(head.hasParkTrace(), "park must record diagnostics on the head");

        head.setSortKey(System.currentTimeMillis() - 10); // now expired → drop
        algorithm.processQueue(ctx);

        verify(handler).onExpired(head);
        assertFalse(head.hasParkTrace(), "drop must consume the head's park trace");
        assertEquals(0, ctx.size());
    }

    @Test
    void externalRemovalClearsItemBoundParkTraceWithoutAlgorithmRetention()
            throws InterruptedException {
        FlexlbConfig config = autoTpmOffConfig();
        config.setFlexlbBatchSloMaxInflightBatches(1);
        PrefillEndpoint endpoint = endpoint(1);
        BatchItem head = item(1L, System.currentTimeMillis() - 100, 100, 0);
        head.setSortKey(System.currentTimeMillis() + 200);
        BatcherContext ctx = context(endpoint, config, mock(DecisionGroupHandler.class),
                queueWith(head));
        SloBudgetBatcherAlgorithm algorithm = new SloBudgetBatcherAlgorithm();

        algorithm.processQueue(ctx);
        assertTrue(head.hasParkTrace());
        assertTrue(ctx.remove(head));
        assertFalse(head.hasParkTrace());

        // Architectural assertion: request diagnostics cannot accumulate in
        // an algorithm-owned Map regardless of external-removal volume.
        assertTrue(java.util.Arrays.stream(algorithm.getClass().getDeclaredFields())
                .noneMatch(field -> java.util.Map.class.isAssignableFrom(field.getType())));
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
            DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
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
            DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
            long now = System.currentTimeMillis();
            BatchItem head = item(2L, now - 500, 100, 50);
            head.ctx().setBudget(ScheduleBudget.forDeadline(50, now - 500, now + 300));
            head.setSortKey(now - 100);
            BatcherContext ctx = context(endpoint, config, handler, queueWith(head));

            new SloBudgetBatcherAlgorithm().processQueue(ctx);

            verify(handler, never()).onExpired(any(BatchItem.class));
            ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
            verify(handler).onDecisionGroupReady(anyList(), meta.capture());
            assertEquals("deadline_guard", meta.getValue().reason());
        }
    }

    @Test
    void routeRequestCapLimitsDeliveryWithoutChangingSloBatchTarget()
            throws InterruptedException {
        FlexlbConfig config = autoTpmOnConfig();
        config.setFlexlbBatchSizeMax(4);
        config.setFlexlbBatchMinSize(3);
        config.setAutoTpmPrefillMaxInflightRequestsPerWorker(1);

        PrefillEndpoint endpoint = endpoint(0);
        when(endpoint.availableRequestSlots(1)).thenReturn(1, 0, 1);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        long now = System.currentTimeMillis();

        DecisionGroupHandler waitingHandler = mock(DecisionGroupHandler.class);
        BatcherContext waiting = context(endpoint, config, waitingHandler, queueWith(
                routeDecisionItem(1, now, now + 5_000),
                routeDecisionItem(2, now + 1, now + 5_000),
                routeDecisionItem(3, now + 2, now + 5_000)));

        new SloBudgetBatcherAlgorithm().processQueue(waiting);

        verify(waitingHandler, never()).onDecisionGroupReady(anyList(), any(DecisionGroupMetadata.class));
        assertEquals(3, waiting.size());

        DecisionGroupHandler readyHandler = mock(DecisionGroupHandler.class);
        BatcherContext ready = context(endpoint, config, readyHandler, queueWith(
                routeDecisionItem(11, now, now + 5_000),
                routeDecisionItem(12, now + 1, now + 5_000),
                routeDecisionItem(13, now + 2, now + 5_000),
                routeDecisionItem(14, now + 3, now + 5_000)));

        new SloBudgetBatcherAlgorithm().processQueue(ready);

        ArgumentCaptor<List<BatchItem>> delivered = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> meta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(readyHandler).onDecisionGroupReady(delivered.capture(), meta.capture());
        assertEquals(List.of(11L), delivered.getValue().stream()
                .map(BatchItem::requestId).toList());
        assertEquals("batch_size_max", meta.getValue().reason());
        assertEquals(3, ready.size());
        assertTrue(ready.isActiveEmpty());
        assertEquals(3, ready.readyDeliveryCount());

        assertEquals(BatcherContext.ReadyDeliveryResult.CAPACITY_BLOCKED,
                ready.deliverReadyRequests());
        assertEquals(BatcherContext.ReadyDeliveryResult.DELIVERED,
                ready.deliverReadyRequests());

        ArgumentCaptor<List<BatchItem>> allDeliveries = ArgumentCaptor.forClass(List.class);
        ArgumentCaptor<DecisionGroupMetadata> allMeta = ArgumentCaptor.forClass(DecisionGroupMetadata.class);
        verify(readyHandler, org.mockito.Mockito.times(2))
                .onDecisionGroupReady(allDeliveries.capture(), allMeta.capture());
        assertEquals(List.of(12L), allDeliveries.getAllValues().get(1).stream()
                .map(BatchItem::requestId).toList());
        assertEquals("batch_size_max", allMeta.getAllValues().get(1).reason());
        assertEquals(2, ready.size());
        assertEquals(2, ready.readyDeliveryCount());
    }

    @Test
    void routeHeadStopsAtModeBoundaryAndCannotBypassBatchInflightGate()
            throws InterruptedException {
        FlexlbConfig config = autoTpmOnConfig();
        config.setFlexlbBatchSizeMax(2);
        config.setFlexlbBatchMinSize(1);
        config.setFlexlbBatchSloMaxInflightBatches(1);
        config.setAutoTpmPrefillMaxInflightRequestsPerWorker(1);

        PrefillEndpoint endpoint = endpoint(1);
        when(endpoint.availableRequestSlots(1)).thenReturn(1);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        DecisionGroupHandler handler = mock(DecisionGroupHandler.class);
        long now = System.currentTimeMillis();
        BatchItem route = routeDecisionItem(1, now - 100, now - 2);
        BatchItem batch = item(2, now - 99, 1, 50);
        batch.setSortKey(now - 1);
        BatcherContext context = context(
                endpoint, config, handler, queueWith(route, batch));
        SloBudgetBatcherAlgorithm algorithm = new SloBudgetBatcherAlgorithm();

        algorithm.processQueue(context);

        ArgumentCaptor<List<BatchItem>> firstDelivery = ArgumentCaptor.forClass(List.class);
        verify(handler).onDecisionGroupReady(firstDelivery.capture(), any(DecisionGroupMetadata.class));
        assertEquals(List.of(1L), firstDelivery.getValue().stream()
                .map(BatchItem::requestId).toList());
        assertEquals(1, context.size());
        assertEquals(batch, context.peek());
        verify(endpoint, never()).getInflightBatchCount();

        // BATCH_ENQUEUE backpressure is evaluated when the batch item reaches the
        // head; a preceding ROUTE_DECISION group cannot bypass that gate.
        algorithm.processQueue(context);

        verify(handler, org.mockito.Mockito.times(1))
                .onDecisionGroupReady(anyList(), any(DecisionGroupMetadata.class));
        verify(endpoint, org.mockito.Mockito.atLeastOnce()).getInflightBatchCount();
        assertEquals(1, context.size());
        assertEquals(batch, context.peek());
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

    private static BatchItem routeDecisionItem(long requestId,
                                               long enqueuedAtMs,
                                               long sortKey) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(1);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setBudget(ScheduleBudget.forDeadline(
                50, enqueuedAtMs, enqueuedAtMs + 30_000));
        context.setScheduleMode(ScheduleModeEnum.QUEUE);
        BatchItem item = new BatchItem(
                context, null, null, null, null, null, null, enqueuedAtMs);
        item.setSortKey(sortKey);
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
                                          DecisionGroupHandler handler,
                                          PriorityBlockingQueue<BatchItem> queue) {
        return new BatcherContext("test", endpoint, config, handler, queue,
                new AtomicInteger(queue.size()), mock(BatchSchedulerReporter.class));
    }

}

package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;

/**
 * D10 guard — the two review-added priority metrics are emitted end-to-end
 * through the real scheduler + real gRPC pipeline:
 *
 * <ul>
 *   <li>{@code ttft_ms{priority}} — scheduler-side approximation, reported on
 *       successful settlement (submit arrival → engine enqueue ACK)</li>
 *   <li>{@code deadline_miss.count{priority}} — reported when the item is
 *       cleared on a queue-deadline path (legacy expiry drop branch; the
 *       yielded-queue-deadline branch is unit-locked in RejectionPolicyTest)</li>
 *   <li>0 sentinel parity: a no-priority request emits neither metric even
 *       with AUTO_TPM on (central guard in FlexlbMetricHelper)</li>
 *   <li>off-state parity: with AUTO_TPM_ENABLED=false at submit time no
 *       auto_tpm metric is emitted at all</li>
 * </ul>
 *
 * <p>The scheduler is wired with a real {@link FlexlbMetricHelper} wrapping a
 * mock {@link FlexMonitor}, so the central priority&lt;=0 guard stays active
 * and assertions run against the final on-the-wire metric shape.
 */
class AutoTpmD10MetricsE2ETest extends FlexLBMockTestBase {

    private static final int P50 = 50;

    private FlexMonitor monitor;

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = super.createConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(true);
        return cfg;
    }

    @Override
    protected FlexlbMetricHelper createMetricHelper() {
        monitor = mock(FlexMonitor.class);
        return new FlexlbMetricHelper(monitor, MetricConstant.PATH_BATCH);
    }

    // ---- ttft_ms{priority}: emitted on successful settlement ----

    @Test
    void successfulPriorityRequest_emitsTtftWithPriorityTag() throws Exception {
        CompletableFuture<Response> future = submitWithPriority(8101, P50);
        assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());

        ArgumentCaptor<FlexMetricTags> tags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor, timeout(2_000)).report(
                eq(MetricConstant.AUTO_TPM_TTFT_MS), tags.capture(), anyDouble());
        assertEquals(String.valueOf(P50),
                tags.getValue().getTags().get(MetricConstant.TAG_PRIORITY));
        assertEquals(MetricConstant.PATH_BATCH,
                tags.getValue().getTags().get(MetricConstant.TAG_PATH));

        // Success is not a deadline miss.
        verify(monitor, never()).report(
                eq(MetricConstant.AUTO_TPM_DEADLINE_MISS_COUNT), any(), anyDouble());

        simulatePrefillFinishedReport();
        assertEquals(0, inflightStore.activeCount());
    }

    // ---- deadline_miss.count{priority}: emitted on queue-deadline clearing ----

    @Test
    void queueDeadlineExpiry_emitsDeadlineMissWithPriorityTag() throws Exception {
        // Park the batcher (large window, batch never full) so the request
        // stalls in the queue past a short enqueue deadline → the legacy
        // QUEUE_DEADLINE_EXCEEDED drop branch clears it via failExpired().
        config.setFlexlbBatchSizeMax(4);
        config.setFlexlbBatchFixedWaitMs(10_000L);
        config.setFlexlbBatchEnqueueDeadlineMs(300L);

        CompletableFuture<Response> future = submitWithPriority(8102, P50);
        Response response = future.get(5, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode(),
                "the stalled request must be cleared on the legacy queue-deadline path");

        ArgumentCaptor<FlexMetricTags> tags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor, timeout(2_000)).report(eq(MetricConstant.AUTO_TPM_DEADLINE_MISS_COUNT),
                tags.capture(), eq(1.0));
        assertEquals(String.valueOf(P50),
                tags.getValue().getTags().get(MetricConstant.TAG_PRIORITY));
        assertEquals(MetricConstant.PATH_BATCH,
                tags.getValue().getTags().get(MetricConstant.TAG_PATH));

        // A failed request never emits the TTFT approximation.
        verify(monitor, never()).report(
                eq(MetricConstant.AUTO_TPM_TTFT_MS), any(), anyDouble());
    }

    // ---- parity: 0 sentinel and off-state emit nothing ----

    @Test
    void noPriorityRequest_emitsNeitherMetric() throws Exception {
        CompletableFuture<Response> future = submitRequest(8103);
        assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());

        verify(monitor, never()).report(
                eq(MetricConstant.AUTO_TPM_TTFT_MS), any(), anyDouble());
        verify(monitor, never()).report(
                eq(MetricConstant.AUTO_TPM_DEADLINE_MISS_COUNT), any(), anyDouble());

        simulatePrefillFinishedReport();
        assertEquals(0, inflightStore.activeCount());
    }

    @Test
    void autoTpmDisabledAtSubmit_emitsNoAutoTpmMetric() throws Exception {
        config.setAutoTpmEnabled(false);

        CompletableFuture<Response> future = submitWithPriority(8104, P50);
        assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());

        verify(monitor, never()).report(
                eq(MetricConstant.AUTO_TPM_TTFT_MS), any(), anyDouble());
        verify(monitor, never()).report(
                eq(MetricConstant.AUTO_TPM_DEADLINE_MISS_COUNT), any(), anyDouble());

        simulatePrefillFinishedReport();
        assertEquals(0, inflightStore.activeCount());
    }

    // ==================== helpers ====================

    private CompletableFuture<Response> submitWithPriority(long requestId, int priority) {
        BalanceContext ctx = createBalanceContext(requestId);
        ctx.setPriority(priority);
        ctx.getRequest().setPriority(priority);
        return scheduler.submit(ctx);
    }
}

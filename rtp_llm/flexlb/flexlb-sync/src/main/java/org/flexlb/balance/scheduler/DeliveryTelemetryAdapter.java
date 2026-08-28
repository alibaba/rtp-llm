package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.delivery.DeliveryTelemetry;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Objects;

/** Reporter-backed, no-throw delivery telemetry boundary. */
public final class DeliveryTelemetryAdapter implements DeliveryTelemetry {

    private static final String PREFILL_ROLE = RoleType.PREFILL.name();

    private final BatchSchedulerReporter reporter;

    public DeliveryTelemetryAdapter(BatchSchedulerReporter reporter) {
        this.reporter = Objects.requireNonNull(reporter, "reporter");
    }

    @Override
    public void routesDelivered(
            DeliveryMetadata metadata,
            List<DeliveryItem> exactItems) {
        try {
            if (exactItems.isEmpty()) {
                return;
            }
            BatchItem head = (BatchItem) exactItems.get(0);
            String engineIp = prefillIp(head);
            reporter.reportBatcherQueueSize(
                    PREFILL_ROLE, engineIp,
                    metadata.remainingQueueDepth());
            long nowMs = System.currentTimeMillis();
            for (DeliveryItem item : exactItems) {
                BatchItem exact = (BatchItem) item;
                reporter.reportBatchWaitTimeMs(
                        PREFILL_ROLE,
                        engineIp,
                        Math.max(0L, nowMs - exact.enqueuedAtMs()),
                        exact.priority());
            }
        } catch (Throwable failure) {
            Logger.warn("Route delivery telemetry isolated", failure);
        }
    }

    @Override
    public void batchDispatched(
            long batchId,
            DeliveryMetadata metadata,
            List<DeliveryItem> dispatched,
            long predictedMs) {
        try {
            if (dispatched.isEmpty()) {
                return;
            }
            String reason = metadata.decisionReason();
            BatchItem head = (BatchItem) dispatched.get(0);
            String engineIp = prefillIp(head);
            reporter.reportDispatchReason(
                    PREFILL_ROLE, engineIp, reason);
            reporter.reportBatcherQueueSize(
                    PREFILL_ROLE, engineIp,
                    metadata.remainingQueueDepth());
            long nowMs = System.currentTimeMillis();
            long hitTokens = 0L;
            long totalTokens = 0L;
            for (DeliveryItem item : dispatched) {
                BatchItem exact = (BatchItem) item;
                reporter.reportBatchWaitTimeMs(
                        PREFILL_ROLE,
                        engineIp,
                        Math.max(0L, nowMs - exact.enqueuedAtMs()),
                        exact.priority());
                hitTokens = saturatedAdd(hitTokens, exact.hitCache());
                totalTokens = saturatedAdd(totalTokens, exact.seqLen());
            }
            reporter.reportBatchCacheHitMetrics(
                    PREFILL_ROLE, engineIp, hitTokens, totalTokens);
            reporter.reportBatchSize(
                    PREFILL_ROLE, engineIp, reason,
                    dispatched.size());
            reporter.reportBatchTotalTokens(
                    PREFILL_ROLE, engineIp, reason, totalTokens);
            reporter.reportBatchPredictedTimeMs(
                    PREFILL_ROLE, engineIp, Math.max(0L, predictedMs));
        } catch (Throwable failure) {
            Logger.warn("Batch dispatch telemetry isolated", failure);
        }
    }

    private static String prefillIp(BatchItem item) {
        return item.prefillEp().getIp();
    }

    private static long saturatedAdd(long left, long right) {
        long nonNegative = Math.max(0L, right);
        return left > Long.MAX_VALUE - nonNegative
                ? Long.MAX_VALUE : left + nonNegative;
    }
}

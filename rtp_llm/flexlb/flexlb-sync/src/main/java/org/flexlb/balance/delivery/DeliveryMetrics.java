package org.flexlb.balance.delivery;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Objects;

/** No-throw metrics emitted after a delivery decision has committed. */
public final class DeliveryMetrics {

    private static final String PREFILL_ROLE = RoleType.PREFILL.name();

    private final BatchSchedulerReporter reporter;

    public DeliveryMetrics(BatchSchedulerReporter reporter) {
        this.reporter = Objects.requireNonNull(reporter, "reporter");
    }

    public void routesDelivered(
            DeliveryMetadata metadata,
            List<ScheduledRequest> exactItems) {
        try {
            if (exactItems.isEmpty()) {
                return;
            }
            ScheduledRequest head = exactItems.get(0);
            String engineIp = prefillIp(head);
            reporter.reportBatcherQueueSize(
                    PREFILL_ROLE, engineIp,
                    metadata.remainingQueueDepth());
            long nowMs = System.currentTimeMillis();
            for (ScheduledRequest item : exactItems) {
                reporter.reportBatchWaitTimeMs(
                        PREFILL_ROLE,
                        engineIp,
                        Math.max(0L, nowMs - item.enqueuedAtMs()),
                        item.priority());
            }
        } catch (Throwable failure) {
            Logger.warn("Route delivery telemetry isolated", failure);
        }
    }

    public void batchDispatched(
            long batchId,
            DeliveryMetadata metadata,
            List<ScheduledRequest> dispatched,
            long predictedMs) {
        try {
            if (dispatched.isEmpty()) {
                return;
            }
            String reason = metadata.decisionReason();
            ScheduledRequest head = dispatched.get(0);
            String engineIp = prefillIp(head);
            reporter.reportDispatchReason(PREFILL_ROLE, engineIp, reason);
            reporter.reportBatcherQueueSize(
                    PREFILL_ROLE, engineIp,
                    metadata.remainingQueueDepth());
            long nowMs = System.currentTimeMillis();
            long hitTokens = 0L;
            long totalTokens = 0L;
            for (ScheduledRequest item : dispatched) {
                reporter.reportBatchWaitTimeMs(
                        PREFILL_ROLE,
                        engineIp,
                        Math.max(0L, nowMs - item.enqueuedAtMs()),
                        item.priority());
                hitTokens = saturatedAdd(hitTokens, item.hitCache());
                totalTokens = saturatedAdd(totalTokens, item.seqLen());
            }
            reporter.reportBatchCacheHitMetrics(
                    PREFILL_ROLE, engineIp, hitTokens, totalTokens);
            reporter.reportBatchSize(
                    PREFILL_ROLE, engineIp, reason, dispatched.size());
            reporter.reportBatchTotalTokens(
                    PREFILL_ROLE, engineIp, reason, totalTokens);
            reporter.reportBatchPredictedTimeMs(
                    PREFILL_ROLE, engineIp, Math.max(0L, predictedMs));
        } catch (Throwable failure) {
            Logger.warn("Batch dispatch telemetry isolated", failure);
        }
    }

    private static String prefillIp(ScheduledRequest item) {
        return item.prefillEp().getIp();
    }

    private static long saturatedAdd(long left, long right) {
        long nonNegative = Math.max(0L, right);
        return left > Long.MAX_VALUE - nonNegative
                ? Long.MAX_VALUE : left + nonNegative;
    }
}

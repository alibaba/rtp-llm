package org.flexlb.balance.delivery;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Objects;

/** No-throw metrics emitted after a delivery decision has committed. */
public final class DeliveryMetrics {

    private final BatchSchedulerReporter reporter;

    public DeliveryMetrics(BatchSchedulerReporter reporter) {
        this.reporter = Objects.requireNonNull(reporter, "reporter");
    }

    public void routesDelivered(int remainingQueueDepth, List<ScheduledRequest> exactItems) {
        try {
            if (exactItems.isEmpty()) {
                return;
            }
            ScheduledRequest head = exactItems.get(0);
            String engineIp = prefillIp(head);
            String role = head.prefillEp().getStatus().getRole().name();
            var decisionGroup = head.ctx().getDecisionGroup();
            if (decisionGroup != null) {
                reporter.reportDispatchReason(role, engineIp, decisionGroup.reason());
                reporter.reportBatchSize(role, engineIp, decisionGroup.reason(), exactItems.size());
            }
            reporter.reportBatcherQueueSize(
                    role, engineIp,
                    remainingQueueDepth);
            long nowMs = System.currentTimeMillis();
            for (ScheduledRequest item : exactItems) {
                reporter.reportBatchWaitTimeMs(
                        role,
                        engineIp,
                        Math.max(0L, nowMs - item.enqueuedAtMs()),
                        item.priority());
            }
        } catch (Throwable failure) {
            Logger.warn("Route delivery telemetry isolated", failure);
        }
    }

    public void batchDispatched(long batchId,
                                String decisionReason,
                                int remainingQueueDepth,
                                List<ScheduledRequest> dispatched,
                                long predictedMs) {
        try {
            if (dispatched.isEmpty()) {
                return;
            }
            ScheduledRequest head = dispatched.get(0);
            String engineIp = prefillIp(head);
            String role = head.prefillEp().getStatus().getRole().name();
            reporter.reportDispatchReason(
                    role, engineIp, decisionReason);
            reporter.reportBatcherQueueSize(
                    role, engineIp,
                    remainingQueueDepth);
            long nowMs = System.currentTimeMillis();
            long hitTokens = 0L;
            long totalTokens = 0L;
            for (ScheduledRequest item : dispatched) {
                reporter.reportBatchWaitTimeMs(
                        role,
                        engineIp,
                        Math.max(0L, nowMs - item.enqueuedAtMs()),
                        item.priority());
                hitTokens = saturatedAdd(hitTokens, item.hitCache());
                totalTokens = saturatedAdd(totalTokens, item.seqLen());
            }
            reporter.reportBatchCacheHitMetrics(
                    role, engineIp, hitTokens, totalTokens);
            reporter.reportBatchSize(
                    role, engineIp, decisionReason, dispatched.size());
            reporter.reportBatchTotalTokens(
                    role, engineIp, decisionReason, totalTokens);
            reporter.reportBatchPredictedTimeMs(
                    role, engineIp, Math.max(0L, predictedMs));
        } catch (Throwable failure) {
            Logger.warn("Batch dispatch telemetry isolated", failure);
        }
    }

    private static String prefillIp(ScheduledRequest item) {
        return item.prefillEp().getStatus().getMetricIpPort();
    }

    private static long saturatedAdd(long left, long right) {
        long nonNegative = Math.max(0L, right);
        return left > Long.MAX_VALUE - nonNegative
                ? Long.MAX_VALUE : left + nonNegative;
    }
}

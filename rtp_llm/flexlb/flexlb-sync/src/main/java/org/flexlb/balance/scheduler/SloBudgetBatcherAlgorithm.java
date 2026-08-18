package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * SLO-deadline-aware batching algorithm with EMA arrival rate estimation,
 * budget-based greedy fill, and deadline-gated dispatch.
 *
 * <p>This is the original algorithm migrated from the now-refactored
 * {@link WorkerBatcher}. Arrival-rate state lives here; request-scoped park
 * diagnostics live on {@link BatchItem} so external queue removal cannot
 * retain request ids in an algorithm-wide map.
 */
public class SloBudgetBatcherAlgorithm implements BatcherAlgorithm {

    // ==================== Algorithm-specific mutable state ====================

    private volatile long lastOfferMs;
    private volatile double interArrivalEmaMs;

    // ==================== BatcherAlgorithm implementation ====================

    @Override
    public long computeSortKey(BatcherContext ctx, BatchItem item) {
        long sloMs = ctx.cfg().resolveSloMs(item.seqLen());
        PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
        long predMs = predictor.estimateMs(item.seqLen(), item.hitCache());
        long workerQueueMs = ctx.prefillEp().realWaitTimeMs();
        long nowMs = System.currentTimeMillis();
        if (workerQueueMs == Long.MAX_VALUE) {
            // The wait snapshot could not stabilize. Consume all batching slack
            // rather than subtracting the sentinel and risking signed overflow.
            return nowMs;
        }
        long nonNegativeSloMs = Math.max(0, sloMs);
        long nonNegativePredMs = Math.max(0, predMs);
        long nonNegativeQueueMs = Math.max(0, workerQueueMs);
        long remainingAfterPrefillMs = nonNegativePredMs >= nonNegativeSloMs
                ? 0 : nonNegativeSloMs - nonNegativePredMs;
        long batchingSlackMs = nonNegativeQueueMs >= remainingAfterPrefillMs
                ? 0 : remainingAfterPrefillMs - nonNegativeQueueMs;
        return nowMs > Long.MAX_VALUE - batchingSlackMs
                ? Long.MAX_VALUE : nowMs + batchingSlackMs;
    }

    @Override
    public void processQueue(BatcherContext ctx) throws InterruptedException {
        if (ctx.isActiveEmpty()) {
            return;
        }

        long windowMs = ctx.cfg().getFlexlbBatchWindowMs();
        int minBatchSize = ctx.cfg().getFlexlbBatchMinSize();
        long emergencyBudgetMs = ctx.cfg().getFlexlbBatchEmergencyBudgetMs();
        int maxScan = ctx.cfg().getFlexlbBatchScanAhead();
        long batchMaxTokens = ctx.batchTokenCapacity();
        int batchMaxCount = Math.max(1, ctx.cfg().getFlexlbBatchSizeMax());

        BatchItem head = ctx.peek();
        if (head == null) {
            return;
        }
        long now = ctx.now();
        long budgetMs = head.sortKey() - now;
        PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();

        // 1. expired → drop. When Auto-TPM is off the legacy drop stays.
        //    PR-D §2.3 fail-fast: when Auto-TPM is on and the head's dispatch
        //    deadline has passed, check whether the remaining coarse budget
        //    can still cover the estimated prefill time. If not, the request
        //    cannot meet its SLO — drive it through onExpired to the typed
        //    Auto-TPM admission result (8430/8431). The AdmissionLease.close()
        //    is an idempotent no-op here. If the coarse budget still has room,
        //    fall through to the deadline_guard dispatch below.
        if (budgetMs < 0) {
            if (!ctx.cfg().isAutoTpmEnabled()) {
                dropHead(ctx, head, now, budgetMs, "deadline_expired");
                return;
            }
            long remainingBudgetMs = head.deadlineMs() > 0
                    ? head.deadlineMs() - now : budgetMs;
            long estimatedPrefillMs = Math.max(1,
                    (long) predictor.estimateMs(head.seqLen(), head.hitCache()));
            if (remainingBudgetMs < estimatedPrefillMs) {
                dropHead(ctx, head, now, budgetMs, "deadline_expired_failfast");
                return;
            }
            // Still has enough coarse budget — fall through to dispatch.
        }

        if (!BatchShape.empty().add(head).fitsCompute(batchMaxTokens)) {
            head.clearParkTrace();
            ctx.rejectForBatchTokenCapacity(head, batchMaxTokens);
            return;
        }

        if (head.deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
            int maxInflightBatches = ctx.cfg().getFlexlbBatchSloMaxInflightBatches();
            if (maxInflightBatches > 0
                    && ctx.prefillEp().getInflightBatchCount() >= maxInflightBatches) {
                long inflightGuardMs = dispatchGuardMs(ctx, emergencyBudgetMs);
                // Auto-TPM never drops requests silently (design doc 8.3):
                // keep parking them until the engine backpressure clears.
                // When Auto-TPM is off, the legacy drop stays.
                if (budgetMs <= inflightGuardMs && !ctx.cfg().isAutoTpmEnabled()) {
                    dropHead(ctx, head, now, budgetMs, "inflight_full_guard");
                    return;
                }
                recordPark(ctx, head, "inflight_full", budgetMs, now);
                parkBriefly();
                return;
            }
        }

        long baseGuardMs = dispatchGuardMs(ctx, emergencyBudgetMs);
        BatchPick pick = pickWithinIncrementalBudget(
                ctx, head, predictor, Math.max(0, budgetMs - baseGuardMs), maxScan,
                batchMaxCount, batchMaxTokens, ctx.batchKvCapacity());
        List<BatchItem> picked = pick.items();
        long incrementalCostMs = Math.max(0, pick.predMs() - pick.headPredMs());
        long latestDispatchBudgetMs = latestDispatchBudgetMs(baseGuardMs, emergencyBudgetMs, incrementalCostMs);
        boolean insideWindow = budgetMs <= windowMs;
        int targetBatchSize = insideWindow
                ? targetBatchSize(ctx, minBatchSize, batchMaxCount,
                        budgetMs, latestDispatchBudgetMs, now)
                : batchMaxCount;
        double fillRatio = targetBatchSize > 0 ? (double) picked.size() / targetBatchSize : 1.0;
        boolean reachesMaxSize = picked.size() >= batchMaxCount;
        boolean reachesTarget = picked.size() >= targetBatchSize;
        boolean mustDispatch = budgetMs <= latestDispatchBudgetMs;
        boolean shouldWaitForMore = shouldWaitForMore(ctx,
                picked.size(), minBatchSize, batchMaxCount, targetBatchSize,
                budgetMs, latestDispatchBudgetMs, now);
        DecisionTrace trace = new DecisionTrace(
                targetBatchSize,
                budgetMs,
                latestDispatchBudgetMs,
                Math.max(0, budgetMs - latestDispatchBudgetMs),
                estimatedInterArrivalMs(ctx),
                estimatedTimeToNextArrivalMs(ctx, now),
                arrivalWaitGuardMs(ctx),
                ctx.deliveryInflightCount(head),
                now);

        // 2. Dispatch decision. Predictor is used for admission and deadline
        // protection; request count and arrival rate decide whether to keep
        // waiting for a more efficient batch.
        if (reachesMaxSize) {
            releaseDecisionGroup(ctx, picked, "batch_size_max", fillRatio, trace);
        } else if (mustDispatch) {
            releaseDecisionGroup(ctx, picked, "deadline_guard", fillRatio, trace);
        } else if (insideWindow && reachesTarget && !shouldWaitForMore) {
            releaseDecisionGroup(ctx, picked, "target_batch_size", fillRatio, trace);
        } else if (insideWindow && !shouldWaitForMore) {
            releaseDecisionGroup(ctx, picked, "arrival_guard", fillRatio, trace);
        } else {
            recordPark(ctx, head,
                    parkReason(insideWindow, picked.size(), minBatchSize, batchMaxCount),
                    budgetMs, now);
            parkBriefly();
        }
    }

    @Override
    public void onOffer(BatcherContext ctx, BatchItem item, long nowMs) {
        recordArrival(ctx, nowMs);
    }

    // ==================== Batch pick ====================

    private BatchPick pickWithinIncrementalBudget(BatcherContext ctx,
                                                  BatchItem head,
                                                  PrefillTimePredictor predictor,
                                                  long budgetMs,
                                                  int maxScan,
                                                  int batchMaxCount,
                                                  long batchMaxTokens,
                                                  long batchKvTokens) {
        List<BatchItem> picked = new ArrayList<>();
        picked.add(head);

        BatchShape shape = BatchShape.empty().add(head);
        long headPredMs = Math.max(0, (long) predictor.predictBatchMsUncached(picked));
        long maxPredMs = headPredMs + Math.max(0, budgetMs);
        int scanned = 0;

        for (BatchItem c : ctx.sortedItems()) {
            if (c == head) {
                continue;
            }
            // Logical groups are homogeneous. During a live delivery-mode
            // transition, the first different-mode item is a hard prefix
            // boundary; never skip it and reorder later requests into head's
            // group.
            if (c.deliveryMode() != head.deliveryMode()) {
                break;
            }
            if (scanned >= maxScan || picked.size() >= batchMaxCount) {
                break;
            }
            scanned++;

            BatchShape candidate = shape.add(c);
            if (!candidate.fitsCompute(batchMaxTokens) || !candidate.fitsKv(batchKvTokens)) {
                break;
            }

            List<BatchItem> trial = new ArrayList<>(picked.size() + 1);
            trial.addAll(picked);
            trial.add(c);
            long trialPredMs = Math.max(0, (long) predictor.predictBatchMsUncached(trial));
            if (trialPredMs <= maxPredMs) {
                picked.add(c);
                shape = candidate;
            }
        }
        return new BatchPick(picked, headPredMs, Math.max(headPredMs, (long) predictor.predictBatchMs(picked)));
    }

    // ==================== Target batch size ====================

    private static int minTargetBatchSize(int minBatchSize, int batchMaxCount) {
        return Math.max(1, Math.min(minBatchSize, batchMaxCount));
    }

    private int targetBatchSize(BatcherContext ctx,
                                int minBatchSize,
                                int batchMaxCount,
                                long budgetMs,
                                long latestDispatchBudgetMs,
                                long nowMs) {
        int minTarget = minTargetBatchSize(minBatchSize, batchMaxCount);
        if (batchMaxCount <= minTarget) {
            return batchMaxCount;
        }
        long slackMs = Math.max(0, budgetMs - latestDispatchBudgetMs);
        long usableSlackMs = Math.max(0, slackMs - arrivalWaitGuardMs(ctx));
        long arrivalMs = estimatedInterArrivalMs(ctx);
        long nextArrivalMs = estimatedTimeToNextArrivalMs(ctx, nowMs);
        if (arrivalMs <= 0 || nextArrivalMs > usableSlackMs) {
            return minTarget;
        }
        long expectedMore = 1 + (usableSlackMs - nextArrivalMs) / Math.max(1, arrivalMs);
        long target = (long) minTarget + expectedMore;
        return (int) Math.max(minTarget, Math.min(batchMaxCount, target));
    }

    // ==================== Wait decision ====================

    private boolean shouldWaitForMore(BatcherContext ctx,
                                      int pickedSize,
                                      int minBatchSize,
                                      int batchMaxCount,
                                      int targetBatchSize,
                                      long budgetMs,
                                      long latestDispatchBudgetMs,
                                      long nowMs) {
        if (pickedSize >= batchMaxCount) {
            return false;
        }
        long slackMs = budgetMs - latestDispatchBudgetMs;
        if (slackMs <= 1) {
            return false;
        }
        long nextArrivalMs = estimatedTimeToNextArrivalMs(ctx, nowMs);
        if (nextArrivalMs + arrivalWaitGuardMs(ctx) > slackMs) {
            return false;
        }
        if (pickedSize < minTargetBatchSize(minBatchSize, batchMaxCount)) {
            return true;
        }
        return pickedSize < targetBatchSize;
    }

    // ==================== Budget guards ====================

    private static long dispatchGuardMs(BatcherContext ctx, long emergencyBudgetMs) {
        long configured = Math.max(1, ctx.cfg().getFlexlbBatchDispatchGuardMs());
        return emergencyBudgetMs > 0 ? Math.min(configured, emergencyBudgetMs) : configured;
    }

    private static long latestDispatchBudgetMs(long baseGuardMs, long emergencyBudgetMs, long incrementalCostMs) {
        long latest = Math.max(baseGuardMs, baseGuardMs + incrementalCostMs);
        return emergencyBudgetMs > 0 ? Math.min(latest, emergencyBudgetMs) : latest;
    }

    // ==================== Arrival rate estimation (EMA) ====================

    private synchronized void recordArrival(BatcherContext ctx, long nowMs) {
        if (lastOfferMs > 0 && nowMs > lastOfferMs) {
            long intervalMs = Math.min(nowMs - lastOfferMs,
                    Math.max(1, ctx.cfg().getFlexlbBatchWindowMs()));
            double alpha = Math.max(0.01, Math.min(1.0, ctx.cfg().getFlexlbBatchArrivalEmaAlpha()));
            interArrivalEmaMs = interArrivalEmaMs <= 0
                    ? intervalMs
                    : alpha * intervalMs + (1.0 - alpha) * interArrivalEmaMs;
        }
        lastOfferMs = nowMs;
    }

    private long estimatedInterArrivalMs(BatcherContext ctx) {
        double ema = interArrivalEmaMs;
        if (ema > 0) {
            return Math.max(1, Math.round(ema));
        }
        long windowMs = Math.max(1, ctx.cfg().getFlexlbBatchWindowMs());
        int minBatchSize = Math.max(1, ctx.cfg().getFlexlbBatchMinSize());
        return Math.max(1, Math.round((double) windowMs / minBatchSize));
    }

    private long estimatedTimeToNextArrivalMs(BatcherContext ctx, long nowMs) {
        long intervalMs = estimatedInterArrivalMs(ctx);
        long lastMs = lastOfferMs;
        if (lastMs <= 0 || nowMs <= lastMs) {
            return intervalMs;
        }
        long elapsedMs = nowMs - lastMs;
        if (interArrivalEmaMs <= 0 || elapsedMs >= intervalMs * 2) {
            return intervalMs;
        }
        long remainderMs = elapsedMs % intervalMs;
        return remainderMs == 0 ? 1 : Math.max(1, intervalMs - remainderMs);
    }

    private static long arrivalWaitGuardMs(BatcherContext ctx) {
        return Math.max(0, ctx.cfg().getFlexlbBatchArrivalWaitGuardMs());
    }

    // ==================== Park tracking ====================

    private static String parkReason(boolean insideWindow,
                                     int pickedSize,
                                     int minBatchSize,
                                     int batchMaxCount) {
        if (!insideWindow) {
            return "outside_window";
        }
        int minTarget = minTargetBatchSize(minBatchSize, batchMaxCount);
        if (pickedSize < minTarget) {
            return "wait_for_min_batch";
        }
        return "wait_for_target_batch";
    }

    private void recordPark(BatcherContext ctx, BatchItem head, String reason, long budgetMs, long nowMs) {
        head.recordParkTrace(reason, budgetMs, nowMs - head.enqueuedAtMs(),
                ctx.size(), ctx.deliveryInflightCount(head));
    }

    // ==================== Drop ====================

    private void dropHead(BatcherContext ctx, BatchItem head, long nowMs, long budgetMs, String dropReason) {
        int queueBefore = ctx.size();
        int inflightCount = ctx.deliveryInflightCount(head);
        long waitMs = nowMs - head.enqueuedAtMs();
        long initialBudgetMs = head.sortKey() - head.enqueuedAtMs();
        BatchItem.ParkTrace parkTrace = head.consumeParkTrace();
        Logger.debug("flexlb_slo_drop delivery_mode={} req_id={} seq_len={} wait_ms={} budget_ms={} worker={} "
                        + "drop_reason={} initial_budget_ms={} deadline_ms={} enqueued_at_ms={} queue_size={} "
                        + "inflight_count={} last_park_reason={} last_park_budget_ms={} "
                        + "last_park_wait_ms={} last_park_queue_size={} last_park_inflight_count={}",
                head.deliveryMode(), head.requestId(), head.seqLen(), waitMs, budgetMs, ctx.key(),
                dropReason, initialBudgetMs, head.sortKey(), head.enqueuedAtMs(), queueBefore,
                inflightCount, parkTrace.reason(), parkTrace.budgetMs(),
                parkTrace.waitMs(), parkTrace.queueSize(), parkTrace.inflightCount());
        ctx.dropHead(head);
    }

    // ==================== Decision release ====================

    private void releaseDecisionGroup(BatcherContext ctx,
                               List<BatchItem> picked,
                               String reason,
                               double fillRatio,
                               DecisionTrace trace) {
        BatchItem head = picked.get(0);
        if (head.deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
            Logger.debug("flexlb_batch_decision reason={} picked_size={} target_batch_size={} "
                            + "fill_ratio={} wait_ms={} budget_ms={} slack_ms={} latest_dispatch_budget_ms={} "
                            + "arrival_ema_ms={} next_arrival_ms={} arrival_wait_guard_ms={} "
                            + "inflight_batches={} queue_before={} worker={} head_req_id={}",
                    reason, picked.size(), trace.targetBatchSize(), fillRatio,
                    trace.nowMs() - head.enqueuedAtMs(), trace.budgetMs(), trace.slackMs(),
                    trace.latestDispatchBudgetMs(), trace.arrivalEmaMs(), trace.nextArrivalMs(),
                    trace.arrivalWaitGuardMs(), trace.inflightCount(), ctx.size(), ctx.key(),
                    head.requestId());
        } else {
            Logger.debug("flexlb_route_decision reason={} picked_size={} target_group_size={} "
                            + "fill_ratio={} wait_ms={} budget_ms={} slack_ms={} latest_delivery_budget_ms={} "
                            + "arrival_ema_ms={} next_arrival_ms={} arrival_wait_guard_ms={} "
                            + "inflight_requests={} queue_before={} worker={} head_req_id={}",
                    reason, picked.size(), trace.targetBatchSize(), fillRatio,
                    trace.nowMs() - head.enqueuedAtMs(), trace.budgetMs(), trace.slackMs(),
                    trace.latestDispatchBudgetMs(), trace.arrivalEmaMs(), trace.nextArrivalMs(),
                    trace.arrivalWaitGuardMs(), trace.inflightCount(), ctx.size(), ctx.key(),
                    head.requestId());
        }
        for (BatchItem item : picked) {
            item.clearParkTrace();
        }
        ctx.stageDecisionGroup(picked,
                new DecisionGroupMetadata(reason, ctx.size() - picked.size()));
    }

    // ==================== Park ====================

    private static void parkBriefly() throws InterruptedException {
        TimeUnit.MILLISECONDS.sleep(1);
    }

    // ==================== Inner records ====================

    private record BatchPick(List<BatchItem> items, long headPredMs, long predMs) {
    }

    private record DecisionTrace(int targetBatchSize,
                                 long budgetMs,
                                 long latestDispatchBudgetMs,
                                 long slackMs,
                                 long arrivalEmaMs,
                                 long nextArrivalMs,
                                 long arrivalWaitGuardMs,
                                 int inflightCount,
                                 long nowMs) {
    }

}

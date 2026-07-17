package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/**
 * SLO-deadline-aware batching algorithm with EMA arrival rate estimation,
 * budget-based greedy fill, and deadline-gated dispatch.
 *
 * <p>This is the original algorithm migrated from the now-refactored
 * {@link WorkerBatcher}. All mutable algorithm-specific state lives here.
 */
public class SloBudgetBatcherAlgorithm implements BatcherAlgorithm {

    // ==================== Algorithm-specific mutable state ====================

    private volatile long lastOfferMs;
    private volatile double interArrivalEmaMs;
    private final Map<Long, ParkTrace> lastParkByRequest = new ConcurrentHashMap<>();

    // ==================== BatcherAlgorithm implementation ====================

    @Override
    public long computeSortKey(BatcherContext ctx, BatchItem item) {
        long sloMs = ctx.cfg().resolveSloMs(item.seqLen());
        PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
        long predMs = predictor.estimateMs(item.seqLen(), item.hitCache());
        long workerQueueMs = ctx.prefillEp().realWaitTimeMs();
        return System.currentTimeMillis() + Math.max(0, sloMs - predMs - workerQueueMs);
    }

    @Override
    public void processQueue(BatcherContext ctx) throws InterruptedException {
        if (ctx.isEmpty()) {
            return;
        }

        // 0. Clear cancelled/done items from the head before any dispatch logic.
        //    A cancelled request must never be dispatched. Completing its future
        //    here prevents BatchItem.future() from hanging when onExpired() cannot
        //    reach the inflight entry — the cancel path may have already finished
        //    the entry (removed from inflight, published to terminalStates), in
        //    which case onExpired() only rolls back decode resources and leaves
        //    the future pending forever.
        BatchItem head = ctx.peek();
        while (head != null && (head.future().isDone() || head.ctx().isCancelled())) {
            if (!head.future().isDone()) {
                Response cancelled = Response.error(StrategyErrorType.REQUEST_CANCELLED);
                cancelled.setErrorMessage("Request cancelled by client");
                head.future().complete(cancelled);
            }
            Logger.warn("flexlb_batch_drop request_id={} reason=request_cancelled_in_queue", head.requestId());
            ctx.dropHead(head);
            head = ctx.peek();
        }
        if (head == null) {
            return;
        }

        long windowMs = ctx.cfg().getFlexlbBatchWindowMs();
        int minBatchSize = ctx.cfg().getFlexlbBatchMinSize();
        long emergencyBudgetMs = ctx.cfg().getFlexlbBatchEmergencyBudgetMs();
        int maxScan = ctx.cfg().getFlexlbBatchScanAhead();
        int batchMaxTokens = ctx.cfg().getFlexlbBatchMaxCapacity();
        int batchMaxCount = Math.max(1, ctx.cfg().getFlexlbBatchSizeMax());

        long now = ctx.now();
        long budgetMs = head.deadlineMs() - now;

        // 1. expired → drop
        if (budgetMs < 0) {
            dropHead(ctx, head, now, budgetMs, "deadline_expired");
            return;
        }

        int maxInflightBatches = ctx.cfg().getFlexlbBatchSloMaxInflightBatches();
        if (maxInflightBatches > 0 && ctx.prefillEp().getInflightBatchCount() >= maxInflightBatches) {
            long inflightGuardMs = dispatchGuardMs(ctx, emergencyBudgetMs);
            if (budgetMs <= inflightGuardMs) {
                dropHead(ctx, head, now, budgetMs, "inflight_full_guard");
                return;
            }
            recordPark(ctx, head, "inflight_full", budgetMs, now);
            parkBriefly();
            return;
        }

        PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
        long effectiveTokens = Math.max(head.seqLen(), batchMaxTokens);
        long baseGuardMs = dispatchGuardMs(ctx, emergencyBudgetMs);
        BatchPick pick = pickWithinIncrementalBudget(
                ctx, head, predictor, Math.max(0, budgetMs - baseGuardMs), maxScan, batchMaxCount, effectiveTokens);
        List<BatchItem> picked = pick.items();
        long incrementalCostMs = Math.max(0, pick.predMs() - pick.headPredMs());
        long latestDispatchBudgetMs = latestDispatchBudgetMs(baseGuardMs, emergencyBudgetMs, incrementalCostMs);
        boolean insideWindow = budgetMs <= windowMs;
        int targetBatchSize = insideWindow
                ? targetBatchSize(ctx, minBatchSize, batchMaxCount, budgetMs, latestDispatchBudgetMs, now)
                : batchMaxCount;
        double fillRatio = targetBatchSize > 0 ? (double) picked.size() / targetBatchSize : 1.0;
        boolean reachesMaxSize = picked.size() >= batchMaxCount;
        boolean reachesTarget = picked.size() >= targetBatchSize;
        boolean mustDispatch = budgetMs <= latestDispatchBudgetMs;
        boolean shouldWaitForMore = shouldWaitForMore(ctx,
                picked.size(), minBatchSize, batchMaxCount, targetBatchSize, budgetMs, latestDispatchBudgetMs, now);
        DecisionTrace trace = new DecisionTrace(
                targetBatchSize,
                budgetMs,
                latestDispatchBudgetMs,
                Math.max(0, budgetMs - latestDispatchBudgetMs),
                estimatedInterArrivalMs(ctx),
                estimatedTimeToNextArrivalMs(ctx, now),
                arrivalWaitGuardMs(ctx),
                ctx.prefillEp().getInflightBatchCount(),
                now);

        // 2. Dispatch decision. Predictor is used for admission and deadline
        // protection; request count and arrival rate decide whether to keep
        // waiting for a more efficient batch.
        if (reachesMaxSize) {
            dispatchBatch(ctx, picked, "batch_size_max", fillRatio, batchMaxTokens, trace);
        } else if (mustDispatch) {
            dispatchBatch(ctx, picked, "deadline_guard", fillRatio, batchMaxTokens, trace);
        } else if (insideWindow && reachesTarget && !shouldWaitForMore) {
            dispatchBatch(ctx, picked, "target_batch_size", fillRatio, batchMaxTokens, trace);
        } else if (insideWindow && !shouldWaitForMore) {
            dispatchBatch(ctx, picked, "arrival_guard", fillRatio, batchMaxTokens, trace);
        } else {
            recordPark(ctx, head, parkReason(insideWindow, picked.size(), minBatchSize, batchMaxCount,
                    targetBatchSize, shouldWaitForMore), budgetMs, now);
            parkBriefly();
        }
    }

    @Override
    public void onOffer(BatcherContext ctx, BatchItem item, long nowMs) {
        recordArrival(ctx, nowMs);
    }

    @Override
    public void onShutdown(BatcherContext ctx) {
        lastParkByRequest.clear();
    }

    // ==================== Batch pick ====================

    private BatchPick pickWithinIncrementalBudget(BatcherContext ctx,
                                                  BatchItem head,
                                                  PrefillTimePredictor predictor,
                                                  long budgetMs,
                                                  int maxScan,
                                                  int batchMaxCount,
                                                  long batchMaxTokens) {
        List<BatchItem> picked = new ArrayList<>();
        picked.add(head);

        long sumTokens = head.seqLen();
        long headPredMs = Math.max(0, (long) predictor.predictBatchMsUncached(picked));
        long maxPredMs = headPredMs + Math.max(0, budgetMs);
        int scanned = 0;
        List<BatchItem> cancelled = new ArrayList<>();

        for (BatchItem c : ctx.sortedItems()) {
            if (c == head) {
                continue;
            }
            // Skip cancelled/done items; clean them up after the scan so they
            // are not dispatched and their futures do not hang.
            if (c.future().isDone() || c.ctx().isCancelled()) {
                cancelled.add(c);
                continue;
            }
            if (scanned >= maxScan || picked.size() >= batchMaxCount) {
                break;
            }
            scanned++;

            long nextTokens = sumTokens + c.seqLen();
            if (nextTokens > batchMaxTokens) {
                continue;
            }

            List<BatchItem> trial = new ArrayList<>(picked.size() + 1);
            trial.addAll(picked);
            trial.add(c);
            long trialPredMs = Math.max(0, (long) predictor.predictBatchMsUncached(trial));
            if (trialPredMs <= maxPredMs) {
                picked.add(c);
                sumTokens = nextTokens;
            }
        }
        // Remove cancelled/done items that were skipped during selection.
        // Complete the future explicitly (same rationale as the head-clearing
        // loop in processQueue) so onExpired() does not leave it pending.
        for (BatchItem item : cancelled) {
            if (!item.future().isDone()) {
                Response cancelledResp = Response.error(StrategyErrorType.REQUEST_CANCELLED);
                cancelledResp.setErrorMessage("Request cancelled by client");
                item.future().complete(cancelledResp);
            }
            ctx.remove(item);
            ctx.handler().onExpired(item);
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
                                     int batchMaxCount,
                                     int targetBatchSize,
                                     boolean shouldWaitForMore) {
        if (!insideWindow) {
            return "outside_window";
        }
        if (!shouldWaitForMore) {
            return "unknown";
        }
        int minTarget = minTargetBatchSize(minBatchSize, batchMaxCount);
        if (pickedSize < minTarget) {
            return "wait_for_min_batch";
        }
        if (pickedSize < targetBatchSize) {
            return "wait_for_target_batch";
        }
        return "wait_for_more";
    }

    private void recordPark(BatcherContext ctx, BatchItem head, String reason, long budgetMs, long nowMs) {
        lastParkByRequest.put(head.requestId(), new ParkTrace(
                reason,
                budgetMs,
                nowMs - head.enqueuedAtMs(),
                ctx.size(),
                ctx.prefillEp().getInflightBatchCount()));
    }

    // ==================== Drop ====================

    private void dropHead(BatcherContext ctx, BatchItem head, long nowMs, long budgetMs, String dropReason) {
        int queueBefore = ctx.size();
        int inflightBatches = ctx.prefillEp().getInflightBatchCount();
        long waitMs = nowMs - head.enqueuedAtMs();
        long initialBudgetMs = head.deadlineMs() - head.enqueuedAtMs();
        ParkTrace parkTrace = lastParkByRequest.remove(head.requestId());
        if (parkTrace == null) {
            parkTrace = ParkTrace.EMPTY;
        }
        Logger.warn("flexlb_batch_drop req_id={} seq_len={} wait_ms={} budget_ms={} worker={} "
                        + "drop_reason={} initial_budget_ms={} deadline_ms={} enqueued_at_ms={} queue_size={} "
                        + "inflight_batches={} last_park_reason={} last_park_budget_ms={} "
                        + "last_park_wait_ms={} last_park_queue_size={} last_park_inflight_batches={}",
                head.requestId(), head.seqLen(), waitMs, budgetMs, ctx.key(),
                dropReason, initialBudgetMs, head.deadlineMs(), head.enqueuedAtMs(), queueBefore,
                inflightBatches, parkTrace.reason(), parkTrace.budgetMs(),
                parkTrace.waitMs(), parkTrace.queueSize(), parkTrace.inflightBatches());
        ctx.dropHead(head);
    }

    // ==================== Dispatch ====================

    private void dispatchBatch(BatcherContext ctx,
                               List<BatchItem> picked,
                               String reason,
                               double fillRatio,
                               long batchMaxTokens,
                               DecisionTrace trace) {
        BatchItem head = picked.get(0);
        Logger.info("flexlb_batch_decision reason={} picked_size={} target_batch_size={} "
                        + "fill_ratio={} wait_ms={} budget_ms={} slack_ms={} latest_dispatch_budget_ms={} "
                        + "arrival_ema_ms={} next_arrival_ms={} arrival_wait_guard_ms={} "
                        + "inflight_batches={} queue_before={} worker={} head_req_id={}",
                reason, picked.size(), trace.targetBatchSize(), fillRatio,
                trace.nowMs() - head.enqueuedAtMs(), trace.budgetMs(), trace.slackMs(),
                trace.latestDispatchBudgetMs(), trace.arrivalEmaMs(), trace.nextArrivalMs(),
                trace.arrivalWaitGuardMs(), trace.inflightBatches(), ctx.size(), ctx.key(),
                head.requestId());
        for (BatchItem item : picked) {
            lastParkByRequest.remove(item.requestId());
        }
        ctx.dispatch(picked,
                new DispatchMeta(reason, fillRatio, batchMaxTokens, ctx.size() - picked.size()));
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
                                 int inflightBatches,
                                 long nowMs) {
    }

    private record ParkTrace(String reason,
                             long budgetMs,
                             long waitMs,
                             int queueSize,
                             int inflightBatches) {
        private static final ParkTrace EMPTY = new ParkTrace("none", -1, -1, -1, -1);
    }
}

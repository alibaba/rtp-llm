package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;

import java.util.ArrayList;
import java.util.List;
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
                dropHead(ctx, head);
                return;
            }
            long remainingBudgetMs = head.deadlineMs() > 0
                    ? head.deadlineMs() - now : budgetMs;
            long estimatedPrefillMs = Math.max(1,
                    (long) predictor.estimateMs(head.seqLen(), head.hitCache()));
            if (remainingBudgetMs < estimatedPrefillMs) {
                dropHead(ctx, head);
                return;
            }
            // Still has enough coarse budget — fall through to dispatch.
        }

        if (!BatchShape.empty().add(head).fitsCompute(batchMaxTokens)) {
            ctx.rejectForBatchTokenCapacity(head, batchMaxTokens);
            return;
        }

        int maxInflightBatches = ctx.cfg().getFlexlbBatchSloMaxInflightBatches();
        if (maxInflightBatches > 0 && ctx.prefillEp().getInflightBatchCount() >= maxInflightBatches) {
            long inflightGuardMs = dispatchGuardMs(ctx, emergencyBudgetMs);
            // Auto-TPM never drops requests silently (design doc 8.3):
            // keep parking them until the engine backpressure clears.
            // When Auto-TPM is off, the legacy drop stays.
            if (budgetMs <= inflightGuardMs && !ctx.cfg().isAutoTpmEnabled()) {
                dropHead(ctx, head);
                return;
            }
            parkBriefly();
            return;
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
                ? targetBatchSize(ctx, minBatchSize, batchMaxCount, budgetMs, latestDispatchBudgetMs, now)
                : batchMaxCount;
        boolean reachesMaxSize = picked.size() >= batchMaxCount;
        boolean reachesTarget = picked.size() >= targetBatchSize;
        boolean mustDispatch = budgetMs <= latestDispatchBudgetMs;
        boolean shouldWaitForMore = shouldWaitForMore(ctx,
                picked.size(), minBatchSize, batchMaxCount, targetBatchSize, budgetMs, latestDispatchBudgetMs, now);

        // 2. Dispatch decision. Predictor is used for admission and deadline
        // protection; request count and arrival rate decide whether to keep
        // waiting for a more efficient batch.
        if (reachesMaxSize) {
            dispatchBatch(ctx, picked, "batch_size_max");
        } else if (mustDispatch) {
            dispatchBatch(ctx, picked, "deadline_guard");
        } else if (insideWindow && reachesTarget && !shouldWaitForMore) {
            dispatchBatch(ctx, picked, "target_batch_size");
        } else if (insideWindow && !shouldWaitForMore) {
            dispatchBatch(ctx, picked, "arrival_guard");
        } else {
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

    // ==================== Drop ====================

    private void dropHead(BatcherContext ctx, BatchItem head) {
        ctx.dropHead(head);
    }

    // ==================== Dispatch ====================

    private void dispatchBatch(BatcherContext ctx,
                               List<BatchItem> picked,
                               String reason) {
        ctx.dispatch(picked,
                new DispatchMeta(reason, ctx.size() - picked.size()));
    }

    // ==================== Park ====================

    private static void parkBriefly() throws InterruptedException {
        TimeUnit.MILLISECONDS.sleep(1);
    }

    // ==================== Inner records ====================

    private record BatchPick(List<BatchItem> items, long headPredMs, long predMs) {
    }

}

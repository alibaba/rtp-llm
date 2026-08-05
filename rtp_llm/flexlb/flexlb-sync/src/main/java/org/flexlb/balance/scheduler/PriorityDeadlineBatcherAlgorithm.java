package org.flexlb.balance.scheduler;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Priority + deadline-aware batching algorithm for the Auto-TPM
 * {@code PriorityAdmissionScheduler}.
 *
 * <p><b>Sort key</b> is a composite encoding that guarantees:
 * <ol>
 *   <li>Higher priority (P70) always sorts before lower priority (P50, P30)</li>
 *   <li>Within the same priority tier, earlier deadline dequeued first</li>
 *   <li>Within the same priority and deadline, FIFO (arrival time)</li>
 * </ol>
 *
 * <p>Encoding: {@code (70 - priority) * 10^13 + deadlineMs}.
 * Since {@code 70 - priority} ranges from 0 (P70) to 40 (P30), the priority
 * band occupies the top digits and the timestamp occupies the lower 13 digits
 * (millisecond epoch fits within 10^13 until year 2286).
 *
 * <p><b>No silent drop on deadline expiry</b> — when the head item's deadline
 * has passed, the algorithm calls {@link BatcherContext#returnToScheduler}
 * which triggers {@link BatchDecisionHandler#onDeadlineExceeded} instead of
 * {@link BatchDecisionHandler#onExpired}. The scheduler can then decide
 * to retry or fail the request explicitly.
 */
public class PriorityDeadlineBatcherAlgorithm implements BatcherAlgorithm {

    private static final long DEADLINE_MULT = 10_000_000_000_000L; // 10^13
    private static final long DEFAULT_DISPATCH_INTERVAL_MS = 50L;

    // ==================== BatcherAlgorithm implementation ====================

    @Override
    public long computeSortKey(BatcherContext ctx, BatchItem item) {
        int priority = item.priority();
        long deadlineMs = item.deadlineMs();
        if (deadlineMs <= 0) {
            // Fallback: compute SLO deadline if no explicit deadline set
            deadlineMs = System.currentTimeMillis()
                    + ctx.cfg().resolveSloMs(item.seqLen());
        }
        long priorityPart = (long) (70 - priority);
        return priorityPart * DEADLINE_MULT + deadlineMs;
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
        long deadlineMs = head.deadlineMs();
        if (deadlineMs <= 0) {
            deadlineMs = head.sortKey() % DEADLINE_MULT;
        }
        long budgetMs = deadlineMs - now;

        // 1. Deadline expired → return to scheduler (NO silent drop)
        if (budgetMs < 0) {
            Logger.warn("flexlb_priority_batcher deadline_exceeded "
                            + "request_id={} priority={} deadline_ms={} now={} worker={}",
                    head.requestId(), head.priority(), deadlineMs, now, ctx.key());
            ctx.returnToScheduler(head);
            return;
        }

        // 2. Oversized head rejection
        if (!BatchShape.empty().add(head).fitsCompute(batchMaxTokens)) {
            ctx.rejectForBatchTokenCapacity(head, batchMaxTokens);
            return;
        }

        // 3. Inflight backpressure
        int maxInflightBatches = ctx.cfg().getFlexlbBatchSloMaxInflightBatches();
        if (maxInflightBatches > 0
                && ctx.prefillEp().getInflightBatchCount() >= maxInflightBatches) {
            long inflightGuardMs = dispatchGuardMs(ctx, emergencyBudgetMs);
            if (budgetMs <= inflightGuardMs) {
                // Inflight full and deadline approaching — return to scheduler
                Logger.warn("flexlb_priority_batcher inflight_full_guard "
                                + "request_id={} budget_ms={} inflight_guard_ms={} worker={}",
                        head.requestId(), budgetMs, inflightGuardMs, ctx.key());
                ctx.returnToScheduler(head);
                return;
            }
            // Still have budget — park and wait for inflight to drain
            TimeUnit.MILLISECONDS.sleep(1);
            return;
        }

        // 4. Batch assembly with predictor-based budget
        PrefillTimePredictor predictor = ctx.prefillEp().getPredictor();
        long baseGuardMs = dispatchGuardMs(ctx, emergencyBudgetMs);
        BatchPick pick = pickWithinIncrementalBudget(
                ctx, head, predictor, Math.max(0, budgetMs - baseGuardMs),
                maxScan, batchMaxCount, batchMaxTokens, ctx.batchKvCapacity());
        List<BatchItem> picked = pick.items();
        long incrementalCostMs = Math.max(0, pick.predMs() - pick.headPredMs());
        long latestDispatchBudgetMs = latestDispatchBudgetMs(
                baseGuardMs, emergencyBudgetMs, incrementalCostMs);

        boolean insideWindow = budgetMs <= windowMs;
        int targetBatchSize = insideWindow
                ? Math.max(1, Math.min(minBatchSize, batchMaxCount))
                : batchMaxCount;
        double fillRatio = targetBatchSize > 0
                ? (double) picked.size() / targetBatchSize : 1.0;

        boolean reachesMaxSize = picked.size() >= batchMaxCount;
        boolean mustDispatch = budgetMs <= latestDispatchBudgetMs;
        boolean reachesTarget = picked.size() >= targetBatchSize;

        // 5. Dispatch decision
        if (reachesMaxSize) {
            dispatchBatch(ctx, picked, "batch_size_max", fillRatio, now);
        } else if (mustDispatch) {
            dispatchBatch(ctx, picked, "deadline_guard", fillRatio, now);
        } else if (insideWindow && reachesTarget) {
            dispatchBatch(ctx, picked, "target_batch_size", fillRatio, now);
        } else if (insideWindow) {
            dispatchBatch(ctx, picked, "arrival_guard", fillRatio, now);
        } else {
            parkBriefly();
        }
    }

    @Override
    public void onOffer(BatcherContext ctx, BatchItem item, long nowMs) {
        // No arrival rate tracking needed for Phase 2
    }

    @Override
    public void onShutdown(BatcherContext ctx) {
        // No state to clean up
    }

    @Override
    public long queueWaitMs(BatcherContext ctx) {
        BatchItem head = ctx.peek();
        if (head == null) {
            return 0;
        }
        long deadlineMs = head.deadlineMs();
        if (deadlineMs <= 0) {
            deadlineMs = head.sortKey() % DEADLINE_MULT;
        }
        return Math.max(0, deadlineMs - ctx.now());
    }

    /**
     * Estimate the wait time for a specific incoming request based on
     * how many queued items would sort before it.
     *
     * @param ctx       the batcher context
     * @param incoming  the item to estimate wait for (need not be enqueued)
     * @return estimated wait in milliseconds
     */
    public long estimateWait(BatcherContext ctx, BatchItem incoming) {
        long incomingSortKey = computeSortKey(ctx, incoming);
        int aheadCount = 0;
        for (BatchItem item : ctx.sortedItems()) {
            if (item.sortKey() <= incomingSortKey) {
                aheadCount++;
            }
        }
        return (long) aheadCount * DEFAULT_DISPATCH_INTERVAL_MS;
    }

    // ==================== Internal helpers ====================

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
        return new BatchPick(picked, headPredMs,
                Math.max(headPredMs, (long) predictor.predictBatchMs(picked)));
    }

    private static long dispatchGuardMs(BatcherContext ctx, long emergencyBudgetMs) {
        long configured = Math.max(1, ctx.cfg().getFlexlbBatchDispatchGuardMs());
        return emergencyBudgetMs > 0
                ? Math.min(configured, emergencyBudgetMs) : configured;
    }

    private static long latestDispatchBudgetMs(long baseGuardMs,
                                                long emergencyBudgetMs,
                                                long incrementalCostMs) {
        long latest = Math.max(baseGuardMs, baseGuardMs + incrementalCostMs);
        return emergencyBudgetMs > 0
                ? Math.min(latest, emergencyBudgetMs) : latest;
    }

    private void dispatchBatch(BatcherContext ctx,
                                List<BatchItem> picked,
                                String reason,
                                double fillRatio,
                                long nowMs) {
        BatchItem head = picked.get(0);
        Logger.info("flexlb_priority_batch_decision reason={} picked_size={} "
                        + "fill_ratio={} budget_ms={} queue_before={} worker={} head_req_id={} head_priority={}",
                reason, picked.size(), fillRatio,
                nowMs - head.enqueuedAtMs(),
                ctx.size(), ctx.key(), head.requestId(), head.priority());
        ctx.dispatch(picked,
                new DispatchMeta(reason, ctx.size() - picked.size()));
    }

    private static void parkBriefly() throws InterruptedException {
        TimeUnit.MILLISECONDS.sleep(1);
    }

    // ==================== Inner records ====================

    private record BatchPick(List<BatchItem> items, long headPredMs, long predMs) {
    }
}

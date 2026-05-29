package org.flexlb.balance.dp;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.DpBatchReporter;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

/**
 * EDF (Earliest-Deadline-First) SLO-budget driven batcher used by
 * {@link org.flexlb.balance.scheduler.DpBatchScheduler} when the target
 * model's prefill workers all report {@code dpSize == 1}.
 *
 * <h3>Algorithm</h3>
 * <pre>
 * queue: priority queue sorted by deadline = enqueue + slo - pred_ttft(req)
 *
 * loop:
 *   head = queue.peek()  (most urgent)
 *   budget = head.deadline - now  (ms)
 *
 *   1. budget &lt; 0            -> fail head (SLO_DROPPED)
 *   2. budget &lt; margin       -> dispatch head alone (EDF_URGENT)
 *   3. binary search [headTokens, maxCapacity] for largest batch that fits budget
 *   4. greedy fill from queue up to computed batchMaxTokens
 *   5. fillRatio &gt;= threshold -> dispatch (BATCH_READY)
 *      else                   -> wait for new arrivals or deadline;
 *      budget shrinks each iteration, batchMaxTokens converges down,
 *      fillRatio rises naturally until dispatch or EDF_URGENT
 * </pre>
 */
public class SloBudgetBatcher {

    private final String model;
    private final ConfigService configService;
    private final DispatchPlanner planner;
    private final Consumer<PrefillBatch> dispatchCallback;
    private final PrefillTimePredictor predictor;
    private final DpBatchReporter dpBatchReporter;

    private final ReentrantLock lock = new ReentrantLock();
    private final Condition arrival = lock.newCondition();
    private final PriorityQueue<QueuedRequest> queue =
            new PriorityQueue<>(Comparator.comparingLong(QueuedRequest::sloDeadlineMicros));

    private final Thread worker;
    private volatile boolean shutdown = false;

    public SloBudgetBatcher(String model,
                            ConfigService configService,
                            DispatchPlanner planner,
                            Consumer<PrefillBatch> dispatchCallback,
                            PrefillTimePredictor predictor,
                            DpBatchReporter dpBatchReporter) {
        this.model = model;
        this.configService = configService;
        this.planner = planner;
        this.dispatchCallback = dispatchCallback;
        this.predictor = predictor;
        this.dpBatchReporter = dpBatchReporter;

        this.worker = new Thread(this::runLoop, "slo-batcher-" + (model == null ? "default" : model));
        this.worker.setDaemon(true);
    }

    public void start() {
        this.worker.start();
    }

    // ============== public API ==============

    public void offer(QueuedRequest req) {
        QueuedRequest enriched = enrichRequest(req);
        lock.lock();
        try {
            queue.add(enriched);
            arrival.signalAll();
        } finally {
            lock.unlock();
        }
    }

    public boolean cancelInQueue(long requestId) {
        lock.lock();
        try {
            Iterator<QueuedRequest> it = queue.iterator();
            while (it.hasNext()) {
                QueuedRequest qr = it.next();
                if (qr.requestId() == requestId) {
                    it.remove();
                    qr.future().completeExceptionally(
                            new CancellationException("Cancelled while queued in SLO batcher"));
                    return true;
                }
            }
        } finally {
            lock.unlock();
        }
        return false;
    }

    public int queueSize() {
        lock.lock();
        try {
            return queue.size();
        } finally {
            lock.unlock();
        }
    }

    public void shutdown() {
        shutdown = true;
        worker.interrupt();
    }

    // ============== main loop ==============

    private void runLoop() {
        while (!shutdown && !Thread.currentThread().isInterrupted()) {
            try {
                waitForNonEmpty();
                long loopStartNs = System.nanoTime();
                StepResult result = stepOnce();
                long durationUs = (System.nanoTime() - loopStartNs) / 1000;
                DpBatchReporter.LoopOutcome outcome = classifyOutcome(result);
                if (dpBatchReporter != null) {
                    dpBatchReporter.reportSloTickDuration(model, outcome, durationUs);
                }
                if (result == null) {
                    continue;
                }
                applyResult(result);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                Logger.error("SloBudgetBatcher[{}] loop iteration failed", model, t);
            }
        }
    }

    private void waitForNonEmpty() throws InterruptedException {
        lock.lock();
        try {
            while (queue.isEmpty()) {
                arrival.await();
                if (shutdown) {
                    throw new InterruptedException("shutdown");
                }
            }
        } finally {
            lock.unlock();
        }
    }

    private static DpBatchReporter.LoopOutcome classifyOutcome(StepResult r) {
        if (r == null) {
            return DpBatchReporter.LoopOutcome.PARK;
        }
        if (r.failOne != null) {
            return DpBatchReporter.LoopOutcome.FAIL;
        }
        return DpBatchReporter.LoopOutcome.DISPATCH;
    }

    private StepResult stepOnce() throws InterruptedException {
        lock.lock();
        try {
            if (queue.isEmpty()) {
                return null;
            }

            FlexlbConfig cfg = configService.loadBalanceConfig();
            long marginMs = cfg.getSloSafetyMargin();
            int maxScan = Math.max(1, cfg.getDpMaxScanAhead());
            double fillThreshold = cfg.getBatchFillThreshold();
            int bsIter = cfg.getBinarySearchMaxIter();
            int maxCapacity = cfg.getBatchMaxCapacity();

            reportQueueSnapshotLocked();

            QueuedRequest head = queue.peek();
            long nowMicros = System.nanoTime() / 1000;
            long budgetMs = (head.sloDeadlineMicros() - nowMicros) / 1000;

            // 1. impossible to complete -> drop
            if (budgetMs < 0) {
                queue.poll();
                return StepResult.fail(head, cfg);
            }

            // 2. urgent — no time to batch -> send head alone
            if (budgetMs < marginMs) {
                queue.poll();
                return StepResult.dispatch(List.of(head), DpBatchReporter.FlushReason.EDF_URGENT, cfg, computeTokens(head));
            }

            // 3. binary search for max batch tokens within budget
            long headTokens = computeTokens(head);
            long headHit = hitOf(head);
            long lo = headTokens;
            long hi = maxCapacity;
            for (int i = 0; i < bsIter && lo < hi; i++) {
                long mid = lo + (hi - lo + 1) / 2;
                if (predictor.estimateMs(mid, headHit) > budgetMs - marginMs) {
                    hi = mid - 1;
                } else {
                    lo = mid;
                }
            }
            long batchMaxTokens = Math.max(headTokens, lo);

            // 4. greedy fill from queue
            List<QueuedRequest> picked = new ArrayList<>();
            picked.add(head);
            long sumTokens = headTokens;

            int scanned = 0;
            List<QueuedRequest> snapshot = new ArrayList<>(queue);
            for (int i = 0; i < snapshot.size() && scanned < maxScan; i++) {
                QueuedRequest c = snapshot.get(i);
                if (c == head) {
                    continue;
                }
                scanned++;
                long cTok = computeTokens(c);
                if (sumTokens + cTok <= batchMaxTokens) {
                    picked.add(c);
                    sumTokens += cTok;
                }
            }

            // 5. dispatch or wait
            double fillRatio = batchMaxTokens > 0 ? (double) sumTokens / batchMaxTokens : 1.0;
            if (fillRatio >= fillThreshold) {
                for (QueuedRequest qr : picked) {
                    queue.remove(qr);
                }
                return StepResult.dispatch(picked, DpBatchReporter.FlushReason.BATCH_READY, cfg, batchMaxTokens);
            }

            // wait for new arrivals; budget shrinks each iteration -> converges
            long waitMs = 1;
            arrival.awaitNanos(waitMs * 1_000_000L);
            return null;
        } finally {
            lock.unlock();
        }
    }

    private void applyResult(StepResult result) {
        if (result.failOne != null) {
            reportFailedRequest(result.failOne, DpBatchReporter.FlushReason.SLO_DROPPED);
            if (dpBatchReporter != null) {
                dpBatchReporter.reportSloFailure(model, DpBatchReporter.FailureCause.SLO_DROPPED);
            }
            result.failOne.future().complete(failureResponse(
                    StrategyErrorType.NO_PREFILL_WORKER,
                    "request deadline expired — cannot meet TTFT SLO"));
            return;
        }
        if (result.drained != null && !result.drained.isEmpty()) {
            planAndDispatch(result.drained, result.cfg, result.reason, result.batchMaxTokens);
        }
    }

    private void reportQueueSnapshotLocked() {
        if (dpBatchReporter == null) {
            return;
        }
        long tokens = 0;
        for (QueuedRequest qr : queue) {
            tokens += computeTokens(qr);
        }
        dpBatchReporter.reportSloQueueSnapshot(model, queue.size(), tokens);
    }

    // ============== internals ==============

    private QueuedRequest enrichRequest(QueuedRequest raw) {
        if (raw.ctx() == null || raw.ctx().getRequest() == null) {
            return raw;
        }
        FlexlbConfig cfg = configService.loadBalanceConfig();
        long seqLen = raw.ctx().getRequest().getSeqLen();
        long hit = raw.ctx().getCacheMatchedTokens();
        int computeLen = (int) Math.max(0, seqLen - hit);

        long nowMicros = System.nanoTime() / 1000;
        long predMs = predictor.estimateMs(seqLen, hit);
        long sloMs = cfg.getDpTtftSloMs();
        long deadlineMicros = nowMicros + (sloMs - predMs) * 1000L;

        return QueuedRequest.of(raw.ctx(), raw.future(), computeLen, deadlineMicros, 0);
    }

    static long computeTokens(QueuedRequest qr) {
        if (qr.ctx() == null || qr.ctx().getRequest() == null) {
            return 0;
        }
        return qr.ctx().getRequest().getSeqLen();
    }

    private static long hitOf(QueuedRequest qr) {
        return qr.ctx() == null ? 0 : qr.ctx().getCacheMatchedTokens();
    }

    private void planAndDispatch(List<QueuedRequest> drained,
                                  FlexlbConfig cfg,
                                  DpBatchReporter.FlushReason reason,
                                  long batchMaxTokens) {
        long actualTokens = 0;
        for (QueuedRequest qr : drained) {
            actualTokens += computeTokens(qr);
        }
        if (dpBatchReporter != null) {
            dpBatchReporter.reportSloBatchFlush(model, reason, drained.size());
            dpBatchReporter.reportSloBatchTokens(model, reason, batchMaxTokens, actualTokens);
            long nowMicros = System.nanoTime() / 1000;
            for (QueuedRequest qr : drained) {
                long waitMs = (nowMicros - qr.enqueuedAtMicros()) / 1000;
                dpBatchReporter.reportSloQueueWait(model, reason, waitMs);
            }
        }

        DispatchPlan plan;
        try {
            plan = planner.plan(drained, new DispatchContext(model, 1, cfg, drained));
        } catch (Throwable t) {
            Logger.error("SloBudgetBatcher[{}] planner threw on {} requests; failing all",
                    model, drained.size(), t);
            if (dpBatchReporter != null) {
                for (int i = 0; i < drained.size(); i++) {
                    dpBatchReporter.reportSloFailure(model, DpBatchReporter.FailureCause.PLANNER_ERROR);
                }
            }
            for (QueuedRequest qr : drained) {
                qr.future().completeExceptionally(t);
            }
            return;
        }

        for (FailedRequest fr : plan.failures()) {
            if (dpBatchReporter != null) {
                dpBatchReporter.reportSloFailure(model, DpBatchReporter.FailureCause.DISPATCH_ERROR);
            }
            fr.request().future().complete(failureResponse(fr.reason(), fr.message()));
        }
        for (PrefillBatch batch : plan.batches()) {
            try {
                dispatchCallback.accept(batch);
            } catch (Throwable t) {
                Logger.error("SloBudgetBatcher[{}] dispatch callback threw on batch of {}; failing all in batch",
                        model, batch.size(), t);
                if (dpBatchReporter != null) {
                    for (int i = 0; i < batch.size(); i++) {
                        dpBatchReporter.reportSloFailure(model, DpBatchReporter.FailureCause.DISPATCH_ERROR);
                    }
                }
                for (PendingRequest pr : batch.requests()) {
                    pr.future().completeExceptionally(t);
                }
            }
        }
    }

    private void reportFailedRequest(QueuedRequest qr, DpBatchReporter.FlushReason reason) {
        if (dpBatchReporter == null) {
            return;
        }
        dpBatchReporter.reportSloBatchFlush(model, reason, 1);
        long waitMs = (System.nanoTime() / 1000 - qr.enqueuedAtMicros()) / 1000;
        dpBatchReporter.reportSloQueueWait(model, reason, waitMs);
    }

    private static Response failureResponse(StrategyErrorType type, String detail) {
        Response r = new Response();
        r.setSuccess(false);
        r.setCode(type.getErrorCode());
        r.setErrorMessage(type.getErrorMsg() + (detail != null ? ": " + detail : ""));
        return r;
    }

    static final class StepResult {
        final List<QueuedRequest> drained;
        final DpBatchReporter.FlushReason reason;
        final QueuedRequest failOne;
        final FlexlbConfig cfg;
        final long batchMaxTokens;

        private StepResult(List<QueuedRequest> drained,
                           DpBatchReporter.FlushReason reason,
                           QueuedRequest failOne,
                           FlexlbConfig cfg,
                           long batchMaxTokens) {
            this.drained = drained;
            this.reason = reason;
            this.failOne = failOne;
            this.cfg = cfg;
            this.batchMaxTokens = batchMaxTokens;
        }

        static StepResult dispatch(List<QueuedRequest> drained,
                                   DpBatchReporter.FlushReason reason,
                                   FlexlbConfig cfg,
                                   long batchMaxTokens) {
            return new StepResult(drained, reason, null, cfg, batchMaxTokens);
        }

        static StepResult fail(QueuedRequest one, FlexlbConfig cfg) {
            return new StepResult(null, null, one, cfg, 0);
        }
    }
}

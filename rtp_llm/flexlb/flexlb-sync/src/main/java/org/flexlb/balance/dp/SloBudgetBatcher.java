package org.flexlb.balance.dp;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.DpBatchReporter;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

/**
 * Single-FIFO SLO-budget driven batcher used by {@link
 * org.flexlb.balance.scheduler.DpBatchScheduler} when the target model's
 * prefill workers all report {@code dpSize == 1} (single-rank DP, no
 * cross-rank fan-out). Parallel to {@link GlobalPrefillBatcher}, which
 * handles the bucket-based multi-rank ({@code dpSize > 1}) case.
 *
 * <h3>Algorithm</h3>
 * <pre>
 * loop:
 *   head = queue.peekBlocking()
 *   budget = head.enqueueMs + dpTtftSloMs - sloSafetyMargin - now
 *
 *   1. budget <= 0           → force-dispatch up to batchMaxTokens (DEADLINE_FORCE)
 *   2. head alone &gt; budget   → fail head (SLO_EXCEEDED), loop
 *   3. FIFO + bounded backward scan: pick head + any later request that fits
 *      both batchMaxTokens and the SLO budget
 *   4. len(batch) &gt; 1 OR queue.size &gt; dpMaxScanAhead → dispatch (PACKED/SCAN_EXHAUSTED)
 *      else → park until next arrival or effectiveDeadline
 * </pre>
 *
 * <h3>Invariants</h3>
 * <ul>
 *   <li>Head always dispatched (FCFS, no starvation).</li>
 *   <li>SLO breach is handled immediately (force-flush or fail) — never silently delayed.</li>
 *   <li>Scan length capped at {@code dpMaxScanAhead} to bound per-iteration cost.</li>
 * </ul>
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
    private final ArrayList<QueuedRequest> queue = new ArrayList<>();

    private final Thread worker;
    private volatile boolean shutdown = false;

    /** Counts loop iterations since the last successful dispatch — emitted as loops-per-dispatch. */
    private int loopsSinceLastDispatch = 0;

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
            long loopStartNs = System.nanoTime();
            try {
                StepResult result = stepOnce();
                loopsSinceLastDispatch++;
                long durationUs = (System.nanoTime() - loopStartNs) / 1000;
                DpBatchReporter.LoopOutcome outcome = classifyOutcome(result);
                if (dpBatchReporter != null) {
                    dpBatchReporter.reportSloLoopDuration(model, outcome, durationUs);
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
            while (queue.isEmpty()) {
                arrival.await();
                if (shutdown) {
                    return null;
                }
            }

            FlexlbConfig cfg = configService.loadBalanceConfig();
            int batchMaxTokens = Math.max(1, cfg.getBatchMaxTokens());
            long sloMs = cfg.getDpTtftSloMs();
            long marginMs = cfg.getSloSafetyMargin();
            int maxScan = Math.max(1, cfg.getDpMaxScanAhead());

            reportQueueSnapshotLocked();

            QueuedRequest head = queue.get(0);
            long nowMicros = System.nanoTime() / 1000;
            long effectiveDeadlineMicros = head.enqueuedAtMicros() + (sloMs - marginMs) * 1000L;
            long budgetMs = (effectiveDeadlineMicros - nowMicros) / 1000;

            // 1. SLO already exhausted → force-dispatch up to batchMaxTokens (capacity-only)
            if (budgetMs <= 0) {
                List<QueuedRequest> drained = packByCapacityLocked(batchMaxTokens, maxScan);
                return StepResult.dispatch(drained, DpBatchReporter.FlushReason.DEADLINE_FORCE, cfg, batchMaxTokens);
            }

            // 2. head alone exceeds SLO budget → fail head, loop again
            long headTokens = tokensOf(head);
            long headHit = hitOf(head);
            if (predictor.estimateMs(headTokens, headHit) > budgetMs) {
                QueuedRequest failed = queue.remove(0);
                return StepResult.fail(failed, cfg);
            }

            // 3. FIFO + bounded backward scan
            int scanLimit = Math.min(queue.size() - 1, maxScan - 1);
            List<Integer> drainIndices = new ArrayList<>();
            List<QueuedRequest> picked = new ArrayList<>();
            drainIndices.add(0);
            picked.add(head);
            long sumTok = headTokens;
            long sumHit = headHit;

            for (int i = 1; i <= scanLimit; i++) {
                QueuedRequest c = queue.get(i);
                long cTok = tokensOf(c);
                if (sumTok + cTok > batchMaxTokens) {
                    continue;
                }
                if (predictor.estimateMs(sumTok + cTok, sumHit + hitOf(c)) > budgetMs) {
                    continue;
                }
                picked.add(c);
                drainIndices.add(i);
                sumTok += cTok;
                sumHit += hitOf(c);
            }

            // 4. decide: dispatch or park
            boolean queueExceedsScan = queue.size() > maxScan;
            if (picked.size() > 1 || queueExceedsScan) {
                DpBatchReporter.FlushReason r = picked.size() > 1
                        ? DpBatchReporter.FlushReason.PACKED
                        : DpBatchReporter.FlushReason.SCAN_EXHAUSTED;
                removeIndicesLocked(drainIndices);
                return StepResult.dispatch(picked, r, cfg, batchMaxTokens);
            }

            // park until earliest of: new arrival OR effectiveDeadline
            long waitNs = (effectiveDeadlineMicros - System.nanoTime() / 1000) * 1000L;
            if (waitNs > 0) {
                arrival.awaitNanos(waitNs);
            }
            return null;
        } finally {
            lock.unlock();
        }
    }

    private void applyResult(StepResult result) {
        if (result.failOne != null) {
            reportFailedRequest(result.failOne, DpBatchReporter.FlushReason.SLO_EXCEEDED);
            if (dpBatchReporter != null) {
                dpBatchReporter.reportSloFailure(model, DpBatchReporter.FailureCause.SLO_EXCEEDED);
            }
            result.failOne.future().complete(failureResponse(
                    StrategyErrorType.NO_PREFILL_WORKER,
                    "request alone exceeds TTFT SLO budget"));
            return;
        }
        if (result.drained != null && !result.drained.isEmpty()) {
            int loopsConsumed = loopsSinceLastDispatch;
            loopsSinceLastDispatch = 0;
            if (dpBatchReporter != null) {
                dpBatchReporter.reportSloLoopsPerDispatch(model, result.reason, loopsConsumed);
            }
            planAndDispatch(result.drained, result.cfg, result.reason, result.batchMaxTokens);
        }
    }

    private void reportQueueSnapshotLocked() {
        if (dpBatchReporter == null) {
            return;
        }
        int n = queue.size();
        long tokens = 0;
        for (int i = 0; i < n; i++) {
            tokens += tokensOf(queue.get(i));
        }
        dpBatchReporter.reportSloQueueSnapshot(model, n, tokens);
    }

    // ============== internals ==============

    private List<QueuedRequest> packByCapacityLocked(int batchMaxTokens, int maxScan) {
        int scanLimit = Math.min(queue.size(), maxScan);
        List<Integer> idx = new ArrayList<>();
        List<QueuedRequest> picked = new ArrayList<>();
        long sumTok = 0;
        for (int i = 0; i < scanLimit; i++) {
            QueuedRequest c = queue.get(i);
            long cTok = tokensOf(c);
            if (i > 0 && sumTok + cTok > batchMaxTokens) {
                continue;
            }
            picked.add(c);
            idx.add(i);
            sumTok += cTok;
        }
        removeIndicesLocked(idx);
        return picked;
    }

    private void removeIndicesLocked(List<Integer> indices) {
        for (int i = indices.size() - 1; i >= 0; i--) {
            queue.remove((int) indices.get(i));
        }
    }

    private QueuedRequest enrichRequest(QueuedRequest raw) {
        if (raw.ctx() == null || raw.ctx().getRequest() == null) {
            return raw;
        }
        long seqLen = raw.ctx().getRequest().getSeqLen();
        int computeTokenLength = (int) Math.max(0, seqLen - raw.ctx().getCacheMatchedTokens());
        return QueuedRequest.of(raw.ctx(), raw.future(), computeTokenLength, Long.MAX_VALUE, 0);
    }

    private static long tokensOf(QueuedRequest qr) {
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
                                  int batchMaxTokens) {
        long actualTokens = 0;
        for (QueuedRequest qr : drained) {
            actualTokens += tokensOf(qr);
        }
        if (dpBatchReporter != null) {
            dpBatchReporter.reportBatchFlush(reason, drained.size());
            dpBatchReporter.reportSloBatchTokens(model, reason, batchMaxTokens, actualTokens);
            long nowMicros = System.nanoTime() / 1000;
            for (QueuedRequest qr : drained) {
                long waitMs = (nowMicros - qr.enqueuedAtMicros()) / 1000;
                dpBatchReporter.reportRequestWaitTime(reason, waitMs);
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
            reportBatchDpSlots(batch);
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

    /**
     * dpSize=1 path: every request in this batch lands on dp_rank=0 of the chosen
     * prefill worker. Tag with model/role/group/endpoint composite so dashboards
     * can slice by deployment without consuming raw IPs.
     */
    private void reportBatchDpSlots(PrefillBatch batch) {
        if (dpBatchReporter == null || batch == null || batch.requests().isEmpty()) {
            return;
        }
        ServerStatus prefill = batch.prefillTarget();
        String endpoint = endpointOf(prefill);
        String role = prefill != null && prefill.getRole() != null ? prefill.getRole().name() : "PREFILL";
        String group = prefill != null ? prefill.getGroup() : null;

        long tokens = 0;
        for (PendingRequest pr : batch.requests()) {
            tokens += tokensOfPending(pr);
        }
        dpBatchReporter.reportSloBatchDpSlot(model, role, group, endpoint, 0,
                batch.requests().size(), tokens);
    }

    private static String endpointOf(ServerStatus s) {
        if (s == null) {
            return "unknown";
        }
        return s.getServerIp() + ":" + s.getGrpcPort();
    }

    private static long tokensOfPending(PendingRequest pr) {
        if (pr == null || pr.ctx() == null || pr.ctx().getRequest() == null) {
            return 0;
        }
        return pr.ctx().getRequest().getSeqLen();
    }

    private void reportFailedRequest(QueuedRequest qr, DpBatchReporter.FlushReason reason) {
        if (dpBatchReporter == null) {
            return;
        }
        dpBatchReporter.reportBatchFlush(reason, 1);
        long waitMs = (System.nanoTime() / 1000 - qr.enqueuedAtMicros()) / 1000;
        dpBatchReporter.reportRequestWaitTime(reason, waitMs);
    }

    private static Response failureResponse(StrategyErrorType type, String detail) {
        Response r = new Response();
        r.setSuccess(false);
        r.setCode(type.getErrorCode());
        r.setErrorMessage(type.getErrorMsg() + (detail != null ? ": " + detail : ""));
        return r;
    }

    /** Result of one main-loop iteration, applied outside the lock. */
    private static final class StepResult {
        final List<QueuedRequest> drained;
        final DpBatchReporter.FlushReason reason;
        final QueuedRequest failOne;
        final FlexlbConfig cfg;
        final int batchMaxTokens;

        private StepResult(List<QueuedRequest> drained,
                           DpBatchReporter.FlushReason reason,
                           QueuedRequest failOne,
                           FlexlbConfig cfg,
                           int batchMaxTokens) {
            this.drained = drained;
            this.reason = reason;
            this.failOne = failOne;
            this.cfg = cfg;
            this.batchMaxTokens = batchMaxTokens;
        }

        static StepResult dispatch(List<QueuedRequest> drained,
                                   DpBatchReporter.FlushReason reason,
                                   FlexlbConfig cfg,
                                   int batchMaxTokens) {
            return new StepResult(drained, reason, null, cfg, batchMaxTokens);
        }

        static StepResult fail(QueuedRequest one, FlexlbConfig cfg) {
            return new StepResult(null, null, one, cfg, 0);
        }
    }
}

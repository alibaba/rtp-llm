package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.BatcherSnapshot;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.balance.strategy.BatchRequest;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public final class WorkerBatcher {

    private final String key;
    private final PrefillEndpoint ep;
    private final FlexlbConfig cfg;
    private final BatchDecisionHandler handler;
    private final PriorityQueue<BatchItem> queue =
            new PriorityQueue<>(Comparator.comparingLong(BatchItem::deadlineMs));
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition arrival = lock.newCondition();
    private final Thread workerThread;
    private volatile boolean stopped;

    public WorkerBatcher(String key, PrefillEndpoint ep, FlexlbConfig cfg, BatchDecisionHandler handler) {
        this.key = key;
        this.ep = ep;
        this.cfg = cfg;
        this.handler = handler;
        this.workerThread = new Thread(this::runLoop, "flexlb-batcher-" + key);
        this.workerThread.setDaemon(true);
        this.workerThread.setUncaughtExceptionHandler((t, e) ->
                Logger.error("WorkerBatcher[{}] thread died unexpectedly", key, e));
    }

    public void start() {
        workerThread.start();
    }

    public void offer(BatchItem item) {
        lock.lock();
        try {
            if (stopped) {
                handler.onOfferFailure(item, new IllegalStateException("FlexLB batcher stopped"));
                return;
            }
            int maxSize = cfg.getFlexlbBatchQueueMaxSize();
            if (maxSize > 0 && queue.size() >= maxSize) {
                handler.onOfferFailure(item,
                        new IllegalStateException("FlexLB batcher queue full, maxSize=" + maxSize));
                return;
            }
            queue.add(item);
            arrival.signalAll();
        } finally {
            lock.unlock();
        }
    }

    public int queueSize() {
        lock.lock();
        try {
            return queue.size();
        } finally {
            lock.unlock();
        }
    }

    public BatcherSnapshot snapshot() {
        lock.lock();
        try {
            if (queue.isEmpty()) {
                return BatcherSnapshot.EMPTY;
            }
            List<BatchRequest> requests = new ArrayList<>(queue.size());
            long earliest = Long.MAX_VALUE;
            long headDeadline = queue.peek().deadlineMs();
            for (BatchItem item : queue) {
                requests.add(new BatchRequest(item.requestId(), item.seqLen(), item.hitCache()));
                if (item.ctx() != null) {
                    earliest = Math.min(earliest, item.ctx().getStartTime());
                }
            }
            return new BatcherSnapshot(queue.size(), requests, earliest, headDeadline);
        } finally {
            lock.unlock();
        }
    }

    public void shutdown() {
        stopped = true;
        workerThread.interrupt();
        lock.lock();
        try {
            arrival.signalAll();
            for (BatchItem item : queue) {
                handler.onOfferFailure(item,
                        new CancellationException("FlexLB batcher stopped: " + key));
            }
            queue.clear();
        } finally {
            lock.unlock();
        }
    }

    private void runLoop() {
        while (!stopped && !Thread.currentThread().isInterrupted()) {
            try {
                waitForNonEmpty();
                processQueue();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                Logger.error("WorkerBatcher[{}] loop failed", key, t);
            }
        }
    }

    private void waitForNonEmpty() throws InterruptedException {
        lock.lock();
        try {
            while (queue.isEmpty()) {
                arrival.await();
                if (stopped) {
                    throw new InterruptedException("stopped");
                }
            }
        } finally {
            lock.unlock();
        }
    }

    private void processQueue() throws InterruptedException {
        lock.lock();
        try {
            if (queue.isEmpty()) {
                return;
            }

            long marginMs = cfg.getCostSloRiskMarginMs();
            int maxScan = cfg.getFlexlbBatchScanAhead();
            double fillThreshold = cfg.getFlexlbBatchFillThreshold();
            int bsIter = cfg.getFlexlbBatchSearchIter();
            int maxCapacity = cfg.getFlexlbBatchMaxCapacity();
            int batchSizeMax = cfg.getFlexlbBatchSizeMax();

            BatchItem head = queue.peek();
            long budgetMs = head.deadlineMs() - System.currentTimeMillis();

            // 1. expired → drop
            if (budgetMs < 0) {
                queue.poll();
                long waitMs = System.currentTimeMillis() - head.enqueuedAtMs();
                Logger.warn("flexlb_batch_drop req_id={} seq_len={} wait_ms={} budget_ms={} worker={}",
                        head.requestId(), head.seqLen(), waitMs, budgetMs, key);
                handler.onExpired(head);
                return;
            }

            // 2. urgent → dispatch head alone
            if (budgetMs < marginMs) {
                queue.poll();
                handler.onUrgent(head,
                        new DispatchMeta("urgent", 1.0, head.seqLen(), queue.size()));
                return;
            }

            // 3. binary search for max batch tokens within budget
            PrefillTimePredictor predictor = ep.getPredictor();
            if (predictor == null) {
                predictor = new PrefillTimePredictor(
                        cfg.getCostAlpha0(), cfg.getCostAlpha1(), cfg.getCostAlpha2(),
                        cfg.getCostAlpha3(), cfg.getCostAlpha4(), cfg.getCostAlpha5());
            }
            long headTokens = head.seqLen();
            long headHit = head.hitCache();
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
            List<BatchItem> picked = new ArrayList<>();
            picked.add(head);
            long sumTokens = headTokens;
            int scanned = 0;
            for (BatchItem c : queue) {
                if (c == head) {
                    continue;
                }
                if (scanned >= maxScan) {
                    break;
                }
                scanned++;
                long cTok = c.seqLen();
                if (sumTokens + cTok <= batchMaxTokens) {
                    picked.add(c);
                    sumTokens += cTok;
                }
            }

            // 5. dispatch or wait
            double fillRatio = batchMaxTokens > 0 ? (double) sumTokens / batchMaxTokens : 1.0;
            long windowMs = cfg.getFlexlbBatchWindowMs();
            boolean windowExpired = (System.currentTimeMillis() - head.enqueuedAtMs()) >= windowMs;
            if (fillRatio >= fillThreshold || picked.size() >= batchSizeMax || windowExpired) {
                String reason = picked.size() >= batchSizeMax ? "batch_size_max"
                        : fillRatio >= fillThreshold ? "filled" : "window";
                for (BatchItem item : picked) {
                    queue.remove(item);
                }
                handler.onBatchReady(picked,
                        new DispatchMeta(reason, fillRatio, batchMaxTokens, queue.size()));
                return;
            }

            // park — budget shrinks each iteration, converges to dispatch
            arrival.awaitNanos(1_000_000L);
        } finally {
            lock.unlock();
        }
    }
}

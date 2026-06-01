package org.flexlb.balance.dp;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.DpBatchReporter;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

/**
 * Simple FIFO batcher for dpSize > 1. Drains when batch fills to dpSize or
 * timeout expires. DP rank assignment is performed inside the batcher after
 * drain, before dispatch.
 *
 * This is a placeholder — the batching algorithm will be refined later.
 */
public class SimpleDpBatcher implements DispatchBatcher {

    private final String model;
    private final int dpSize;
    private final ConfigService configService;
    private final DispatchPlanner planner;
    private final Consumer<DispatchBatch> dispatchCallback;
    private final DpBatchReporter dpBatchReporter;
    private final WorkerStatus targetWorker;
    private final AtomicInteger cursor = new AtomicInteger(0);

    private final ReentrantLock lock = new ReentrantLock();
    private final Condition arrival = lock.newCondition();
    private final LinkedList<QueuedRequest> queue = new LinkedList<>();

    private final Thread worker;
    private volatile boolean shutdown = false;

    public SimpleDpBatcher(String model,
                           int dpSize,
                           ConfigService configService,
                           DispatchPlanner planner,
                           Consumer<DispatchBatch> dispatchCallback,
                           DpBatchReporter dpBatchReporter,
                           WorkerStatus targetWorker) {
        if (dpSize <= 1) {
            throw new IllegalArgumentException(
                    "SimpleDpBatcher requires dpSize>1, got " + dpSize);
        }
        this.model = model;
        this.dpSize = dpSize;
        this.configService = configService;
        this.planner = planner;
        this.dispatchCallback = dispatchCallback;
        this.dpBatchReporter = dpBatchReporter;
        this.targetWorker = targetWorker;

        String suffix = (model == null ? "default" : model) + "-" + targetWorker.getIpPort();
        this.worker = new Thread(this::runLoop, "simple-dp-batcher-" + suffix);
        this.worker.setDaemon(true);
        this.worker.setUncaughtExceptionHandler((t, e) ->
                Logger.error("SimpleDpBatcher[{}] thread died unexpectedly", model, e));
    }

    public void start() {
        this.worker.start();
    }

    @Override
    public void offer(QueuedRequest req) {
        int maxSize = configService.loadBalanceConfig().getMaxBatcherQueueSize();
        if (maxSize > 0 && queueSize() >= maxSize) {
            req.future().complete(failureResponse(StrategyErrorType.QUEUE_FULL,
                    "batcher queue full (size=" + maxSize + ")"));
            return;
        }
        lock.lock();
        try {
            queue.add(req);
            arrival.signalAll();
        } finally {
            lock.unlock();
        }
    }

    @Override
    public boolean cancelInQueue(long requestId) {
        lock.lock();
        try {
            Iterator<QueuedRequest> it = queue.iterator();
            while (it.hasNext()) {
                QueuedRequest qr = it.next();
                if (qr.requestId() == requestId) {
                    it.remove();
                    qr.future().completeExceptionally(
                            new CancellationException("Cancelled while queued in SimpleDpBatcher"));
                    return true;
                }
            }
        } finally {
            lock.unlock();
        }
        return false;
    }

    @Override
    public int queueSize() {
        lock.lock();
        try {
            return queue.size();
        } finally {
            lock.unlock();
        }
    }

    @Override
    public boolean isAlive() {
        return worker.isAlive();
    }

    @Override
    public void shutdown() {
        shutdown = true;
        worker.interrupt();
        drainOnShutdown();
    }

    private void drainOnShutdown() {
        lock.lock();
        try {
            for (QueuedRequest qr : queue) {
                if (!qr.future().isDone()) {
                    qr.future().completeExceptionally(
                            new CancellationException("Batcher shutting down"));
                }
            }
            queue.clear();
        } finally {
            lock.unlock();
        }
    }

    private void runLoop() {
        while (!shutdown && !Thread.currentThread().isInterrupted()) {
            try {
                List<QueuedRequest> drained = waitAndDrain();
                if (drained != null && !drained.isEmpty()) {
                    planAndDispatch(drained);
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                Logger.error("SimpleDpBatcher[{}] loop iteration failed", model, t);
            }
        }
    }

    private List<QueuedRequest> waitAndDrain() throws InterruptedException {
        FlexlbConfig cfg = configService.loadBalanceConfig();
        long timeoutMs = cfg.getDpBatchWindowMs();

        lock.lock();
        try {
            while (queue.isEmpty()) {
                arrival.await();
                if (shutdown) throw new InterruptedException("shutdown");
            }

            if (queue.size() >= dpSize) {
                return drainLocked(dpSize);
            }

            long deadlineNs = System.nanoTime() + timeoutMs * 1_000_000L;
            while (queue.size() < dpSize) {
                long remainNs = deadlineNs - System.nanoTime();
                if (remainNs <= 0) break;
                arrival.awaitNanos(remainNs);
                if (shutdown) throw new InterruptedException("shutdown");
            }

            if (queue.isEmpty()) return null;
            return drainLocked(Math.min(queue.size(), dpSize));
        } finally {
            lock.unlock();
        }
    }

    private List<QueuedRequest> drainLocked(int count) {
        List<QueuedRequest> drained = new ArrayList<>(count);
        for (int i = 0; i < count && !queue.isEmpty(); i++) {
            drained.add(queue.poll());
        }
        return drained;
    }

    private void planAndDispatch(List<QueuedRequest> drained) {
        FlexlbConfig cfg = configService.loadBalanceConfig();

        if (dpBatchReporter != null) {
            dpBatchReporter.reportSloBatchFlush(model, DpBatchReporter.FlushReason.BATCH_READY, drained.size());
        }

        try {
            ServerStatus prefill = DefaultDispatchPlanner.toPrefillServerStatus(targetWorker);
            List<PendingRequest> placed = new ArrayList<>(drained.size());

            for (QueuedRequest qr : drained) {
                qr.ctx().setConfig(cfg);
                ServerStatus decode = planner.selectDecodeWorker(qr.ctx(), prefill.getGroup());
                if (decode == null || !decode.isSuccess()) {
                    qr.future().complete(failureResponse(StrategyErrorType.NO_DECODE_WORKER,
                            decode == null ? "decode selector returned null" : decode.getMessage()));
                    continue;
                }
                placed.add(new PendingRequest(qr.ctx(), prefill, decode, qr.future(), qr.enqueuedAtMicros()));
            }

            if (placed.isEmpty()) {
                return;
            }

            long blockSize = DefaultDispatchPlanner.getBlockSize(targetWorker);
            List<List<PendingRequest>> ranked = groupByRoundRobin(placed);
            DispatchBatch batch = new DispatchBatch(prefill, ranked, dpSize, blockSize);
            dispatchCallback.accept(batch);
        } catch (Throwable t) {
            Logger.error("SimpleDpBatcher[{}] planAndDispatch threw on {} requests; failing all",
                    model, drained.size(), t);
            for (QueuedRequest qr : drained) {
                if (!qr.future().isDone()) {
                    qr.future().completeExceptionally(t);
                }
            }
        }
    }

    private List<List<PendingRequest>> groupByRoundRobin(List<PendingRequest> placed) {
        List<List<PendingRequest>> ranked = new ArrayList<>(dpSize);
        for (int i = 0; i < dpSize; i++) {
            ranked.add(new ArrayList<>());
        }
        int start = cursor.getAndAdd(placed.size());
        for (int i = 0; i < placed.size(); i++) {
            ranked.get(Math.floorMod(start + i, dpSize)).add(placed.get(i));
        }
        return ranked;
    }

    private static Response failureResponse(StrategyErrorType type, String detail) {
        Response r = new Response();
        r.setSuccess(false);
        r.setCode(type.getErrorCode());
        r.setErrorMessage(type.getErrorMsg() + (detail != null ? ": " + detail : ""));
        return r;
    }
}

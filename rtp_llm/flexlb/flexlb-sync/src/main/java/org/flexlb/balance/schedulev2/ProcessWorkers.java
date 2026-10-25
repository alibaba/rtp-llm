package org.flexlb.balance.schedulev2;

import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestOrder;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.function.Supplier;

/**
 * 有界的并行准备 worker。执行中和刚完成、尚未被调度线程接收的任务合计不超过 limit，同一个请求不会重复提交。
 * 调度线程已接收的材料单独按 TTL 缓存，容量不足时复用，不占 worker 名额。
 * <p>
 * 锁只保护任务表和结果交接，缓存查询、耗时预测和关闭结果都在锁外执行。
 * 关闭后迟到的结果由 worker 自己丢弃。取消中的任务完成前仍占一个 worker 名额。
 */
public final class ProcessWorkers<T extends AutoCloseable> implements AutoCloseable {

    private final ExecutorService executor;
    private final int limit;
    private final LoopSignal signal;
    private final Comparator<Request> order;
    private final Map<Request, CompletableFuture<SelectionResult<T>>> tasks = new HashMap<>();
    /**
     * 已交给调度线程使用的材料；容量不足时保留，成功离开本阶段、过期或候选变化时移除。
     */
    private final Map<Request, SelectionResult<T>> cached = new HashMap<>();
    private final Set<Request> discardedTasks = new HashSet<>();
    private boolean closed;

    public ProcessWorkers(ExecutorService executor, int limit, LoopSignal signal, boolean priorityOrdering) {
        this.executor = executor;
        this.limit = limit;
        this.signal = signal;
        this.order = RequestOrder.of(priorityOrdering);
    }

    /**
     * 有空位且这个请求没有正在准备时提交任务，准备工作不占用调度线程。
     */
    public boolean submit(Request request, Supplier<T> prepare) {
        CompletableFuture<SelectionResult<T>> task;
        SelectionResult<T> removed = null;
        synchronized (this) {
            if (closed || tasks.containsKey(request) || cached.containsKey(request) || request.stage().isEnded()) {
                return false;
            }
            if (tasks.size() >= limit) {
                // 排在前面的请求不能被后面的待处理材料占住所有 worker 名额。
                // 只丢掉一个排在它后面的已完成结果，请求本身仍在队列里，之后重新准备。
                Iterator<Map.Entry<Request, CompletableFuture<SelectionResult<T>>>> iterator =
                        tasks.entrySet().iterator();
                while (iterator.hasNext()) {
                    var entry = iterator.next();
                    if (entry.getValue().isDone() && order.compare(request, entry.getKey()) < 0) {
                        removed = entry.getValue().join();
                        iterator.remove();
                        break;
                    }
                }
                if (tasks.size() >= limit) {
                    return false;
                }
            }
            task = new CompletableFuture<>();
            tasks.put(request, task);
        }
        if (removed != null) {
            closeResult(removed);
        }
        try {
            executor.execute(() -> {
                SelectionResult<T> result;
                try {
                    result = new SelectionResult<>(prepare.get(), null, System.nanoTime());
                } catch (Throwable failure) {
                    result = new SelectionResult<>(null, failure, System.nanoTime());
                }
                complete(request, task, result);
            });
        } catch (Throwable failure) {
            complete(request, task, new SelectionResult<>(null, failure, System.nanoTime()));
        }
        return true;
    }

    private void complete(Request request, CompletableFuture<SelectionResult<T>> task, SelectionResult<T> result) {
        boolean discard;
        synchronized (this) {
            boolean invalidated = discardedTasks.remove(request);
            discard = closed || request.stage().isEnded() || invalidated;
            if (discard) {
                tasks.remove(request);
            } else {
                task.complete(result);
            }
        }
        if (discard) {
            closeResult(result);
        }
        signal.wake();
    }

    /**
     * 结果已经完成时取走，由调用方处理并关闭。执行中的任务返回 null，继续等待 worker 唤醒。
     */
    public synchronized SelectionResult<T> take(Request request) {
        CompletableFuture<SelectionResult<T>> task = tasks.get(request);
        if (task == null || !task.isDone()) {
            return null;
        }
        tasks.remove(request);
        return task.join();
    }

    /**
     * 获取可复用材料，首次收到 worker 结果时转入缓存。调用方不得关闭返回值，由 discard 或过期清理关闭。
     */
    public SelectionResult<T> get(Request request, long ttlMs) {
        SelectionResult<T> result;
        synchronized (this) {
            result = cached.get(request);
            if (result == null) {
                result = take(request);
                if (result != null) {
                    cached.put(request, result);
                }
            }
            if (result == null || !result.expired(ttlMs)) {
                return result;
            }
            cached.remove(request);
        }
        closeResult(result);
        return null;
    }

    /**
     * 请求离开本阶段或材料失效时丢弃。执行中的旧任务先标记失效，完成后丢弃，期间仍占 worker 名额。
     */
    public void discard(Request request) {
        List<SelectionResult<T>> removed = new ArrayList<>();
        synchronized (this) {
            SelectionResult<T> result = cached.remove(request);
            if (result != null) {
                removed.add(result);
            }
            CompletableFuture<SelectionResult<T>> task = tasks.get(request);
            if (task != null) {
                if (task.isDone()) {
                    removed.add(task.join());
                    tasks.remove(request);
                } else {
                    discardedTasks.add(request);
                }
            }
        }
        removed.forEach(ProcessWorkers::closeResult);
    }

    /**
     * 每轮按最新 TTL 清理缓存及尚未取走的过期材料。
     */
    public void removeExpired(long ttlMs) {
        removeEnded();
        List<SelectionResult<T>> removed = new ArrayList<>();
        synchronized (this) {
            cached.entrySet().removeIf(entry -> {
                if (entry.getKey().stage().isEnded() || entry.getValue().expired(ttlMs)) {
                    removed.add(entry.getValue());
                    return true;
                }
                return false;
            });
            tasks.entrySet().removeIf(entry -> {
                if (entry.getValue().isDone() && entry.getValue().join().expired(ttlMs)) {
                    removed.add(entry.getValue().join());
                    return true;
                }
                return false;
            });
        }
        removed.forEach(ProcessWorkers::closeResult);
    }

    /**
     * 清掉已结束请求的待处理结果。尚未完成的任务由 worker 完成时清理。
     */
    public void removeEnded() {
        List<SelectionResult<T>> removed = new ArrayList<>();
        synchronized (this) {
            cached.entrySet().removeIf(entry -> {
                if (entry.getKey().stage().isEnded()) {
                    removed.add(entry.getValue());
                    return true;
                }
                return false;
            });
            Iterator<Map.Entry<Request, CompletableFuture<SelectionResult<T>>>> iterator = tasks.entrySet().iterator();
            while (iterator.hasNext()) {
                var entry = iterator.next();
                if (entry.getKey().stage().isEnded() && entry.getValue().isDone()) {
                    removed.add(entry.getValue().join());
                    iterator.remove();
                }
            }
        }
        removed.forEach(ProcessWorkers::closeResult);
    }

    @Override
    public void close() {
        List<SelectionResult<T>> removed = new ArrayList<>();
        synchronized (this) {
            closed = true;
            for (CompletableFuture<SelectionResult<T>> task : tasks.values()) {
                if (task.isDone()) {
                    removed.add(task.join());
                }
            }
            tasks.clear();
            removed.addAll(cached.values());
            cached.clear();
            discardedTasks.clear();
        }
        executor.shutdown();
        removed.forEach(ProcessWorkers::closeResult);
    }

    private static void closeResult(SelectionResult<?> result) {
        try {
            result.close();
        } catch (Throwable failure) {
            Logger.warn("V2 selection result close failed", failure);
        }
    }
}

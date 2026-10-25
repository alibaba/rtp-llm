package org.flexlb.mockengine;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/** BlockTreeTaskPool: completion first, at most four LOADs before a waiting BACKGROUND. */
final class MockCacheTaskPool implements AutoCloseable {
    private record Work(Runnable run, long deadline, Runnable expired) {}
    private final ArrayDeque<Work> loads = new ArrayDeque<>(), background = new ArrayDeque<>(), completions = new ArrayDeque<>();
    private final int capacity;
    private final List<Thread> workers = new ArrayList<>();
    // Java 21 monitor waits pin virtual-thread carriers, starving unrelated engine RPCs.
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition changed = lock.newCondition();
    private boolean closed, admissionStopped;
    private int pending;
    private int burst;
    MockCacheTaskPool(int count, int capacity) {
        if (count <= 0 || capacity < 0) throw new IllegalArgumentException("invalid cache task pool");
        this.capacity = capacity;
        for (int i = 0; i < count; i++) workers.add(Thread.ofVirtual().name("mock-cache-worker").start(this::work));
    }
    boolean submit(boolean load, Runnable run, long deadline, Runnable expired) {
        lock.lock();
        try {
            if (closed || admissionStopped) return false;
            if (deadline != Long.MAX_VALUE && System.nanoTime() >= deadline) { expired.run(); return true; }
            if (capacity > 0 && (loads.size() + background.size() >= capacity
                    || !load && background.size() >= (capacity > 64 ? capacity - 64 : capacity))) return false;
            (load ? loads : background).addLast(new Work(run, deadline, expired)); pending++; changed.signalAll(); return true;
        } finally { lock.unlock(); }
    }
    boolean complete(Runnable run) {
        lock.lock();
        try {
            if (closed) return false;
            completions.addLast(new Work(run, Long.MAX_VALUE, null)); pending++; changed.signalAll(); return true;
        } finally { lock.unlock(); }
    }
    int[] sizes() {
        lock.lock();
        try { return new int[]{loads.size(), background.size(), completions.size()}; }
        finally { lock.unlock(); }
    }
    private void work() {
        while (true) {
            Work next;
            lock.lock();
            try {
                while (!closed && completions.isEmpty() && loads.isEmpty() && background.isEmpty()) {
                    try { changed.await(); } catch (InterruptedException error) { if (closed) return; }
                }
                if (closed) return;
                if (!completions.isEmpty()) next = completions.removeFirst();
                else if (!background.isEmpty() && (loads.isEmpty() || burst >= 4)) { next = background.removeFirst(); burst = 0; }
                else { next = loads.removeFirst(); burst = Math.min(4, burst + 1); }
            } finally { lock.unlock(); }
            try {
                if (next.deadline() != Long.MAX_VALUE && System.nanoTime() >= next.deadline()) next.expired().run();
                else next.run().run();
            } catch (Throwable ignored) { /* workflow owns its terminal; keep the worker alive */ }
            finally {
                lock.lock();
                try { pending--; changed.signalAll(); }
                finally { lock.unlock(); }
            }
        }
    }
    void stopAdmission() {
        lock.lock();
        try { admissionStopped = true; }
        finally { lock.unlock(); }
    }
    void waitForIdle() {
        lock.lock();
        try {
            while (pending != 0) {
                try { changed.await(); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IllegalStateException("cache drain interrupted", error); }
            }
        } finally { lock.unlock(); }
    }
    @Override public void close() {
        lock.lock();
        try {
            if (closed) return;
            closed = admissionStopped = true;
            pending -= loads.size() + background.size() + completions.size();
            loads.clear(); background.clear(); completions.clear(); changed.signalAll();
        } finally { lock.unlock(); }
        for (var worker : workers) {
            try { worker.join(); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IllegalStateException("cache worker shutdown interrupted", error); }
        }
    }
}

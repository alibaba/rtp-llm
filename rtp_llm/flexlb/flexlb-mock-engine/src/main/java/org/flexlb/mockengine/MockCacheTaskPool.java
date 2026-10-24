package org.flexlb.mockengine;

import java.util.ArrayDeque;
import java.util.List;
import java.util.ArrayList;

/** BlockTreeTaskPool: completion first, at most four LOADs before a waiting BACKGROUND. */
final class MockCacheTaskPool implements AutoCloseable {
    private record Work(Runnable run, long deadline, Runnable expired) {}
    private final ArrayDeque<Work> loads = new ArrayDeque<>(), background = new ArrayDeque<>(), completions = new ArrayDeque<>();
    private final int capacity;
    private final List<Thread> workers = new ArrayList<>();
    private boolean closed, admissionStopped;
    private int pending;
    private int burst;
    MockCacheTaskPool(int count, int capacity) {
        if (count <= 0 || capacity < 0) throw new IllegalArgumentException("invalid cache task pool");
        this.capacity = capacity;
        for (int i = 0; i < count; i++) workers.add(Thread.ofVirtual().name("mock-cache-worker").start(this::work));
    }
    synchronized boolean submit(boolean load, Runnable run, long deadline, Runnable expired) {
        if (closed || admissionStopped) return false;
        if (deadline != Long.MAX_VALUE && System.nanoTime() >= deadline) { expired.run(); return true; }
        if (capacity > 0 && (loads.size() + background.size() >= capacity
                || !load && background.size() >= (capacity > 64 ? capacity - 64 : capacity))) return false;
        (load ? loads : background).addLast(new Work(run, deadline, expired)); pending++; notifyAll(); return true;
    }
    synchronized boolean complete(Runnable run) {
        if (closed) return false;
        completions.addLast(new Work(run, Long.MAX_VALUE, null)); pending++; notifyAll(); return true;
    }
    synchronized int[] sizes() { return new int[]{loads.size(), background.size(), completions.size()}; }
    private void work() {
        while (true) {
            Work next;
            synchronized (this) {
                while (!closed && completions.isEmpty() && loads.isEmpty() && background.isEmpty()) {
                    try { wait(); } catch (InterruptedException error) { if (closed) return; }
                }
                if (closed) return;
                if (!completions.isEmpty()) next = completions.removeFirst();
                else if (!background.isEmpty() && (loads.isEmpty() || burst >= 4)) { next = background.removeFirst(); burst = 0; }
                else { next = loads.removeFirst(); burst = Math.min(4, burst + 1); }
            }
            try {
                if (next.deadline() != Long.MAX_VALUE && System.nanoTime() >= next.deadline()) next.expired().run();
                else next.run().run();
            } catch (Throwable ignored) { /* workflow owns its terminal; keep the worker alive */ }
            finally { synchronized (this) { pending--; notifyAll(); } }
        }
    }
    synchronized void stopAdmission() { admissionStopped = true; }
    synchronized void waitForIdle() {
        while (pending != 0) {
            try { wait(); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IllegalStateException("cache drain interrupted", error); }
        }
    }
    @Override public void close() {
        synchronized (this) {
            if (closed) return;
            closed = admissionStopped = true;
            pending -= loads.size() + background.size() + completions.size();
            loads.clear(); background.clear(); completions.clear(); notifyAll();
        }
        for (var worker : workers) {
            try { worker.join(); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IllegalStateException("cache worker shutdown interrupted", error); }
        }
    }
}

package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Consumer;

/**
 * Serializes each owner's effects outside Context's monitor. Cleanup notifications coalesce;
 * a task carrying a reserved frontend response must run even when cleanup is already pending.
 */
public final class RequestCloseQueue {
    private final ThreadLocal<Boolean> executing = new ThreadLocal<>();
    private final ConcurrentMap<Object, Lane> lanes = new ConcurrentHashMap<>();

    void awaitIdle() {
        com.google.common.base.Preconditions.checkState(
                !isExecutingOnCurrentThread(), "request close cannot wait for itself");
        for (; ; ) {
            var pending = lanes.values().toArray(Lane[]::new);
            if (pending.length == 0) return;
            for (var lane : pending) lane.completed.join();
        }
    }

    boolean isExecutingOnCurrentThread() {
        return Boolean.TRUE.equals(executing.get());
    }

    /**
     * Runs one request's close operation; cleanup notifications coalesce, responses do not.
     */
    void submit(
            Object owner,
            Runnable work,
            Consumer<Runnable> start,
            boolean cleanupOnly,
            Consumer<Throwable> rejected) {
        Objects.requireNonNull(work, "work");
        for (; ; ) {
            Lane lane = lanes.get(owner);
            if (lane == null || lane.retired) {
                lane =
                        lanes.compute(
                                owner,
                                (key, old) -> old == null || old.retired ? new Lane(key) : old);
            }
            synchronized (lane) {
                if (lane.retired) {
                    continue;
                }
                if (cleanupOnly && !lane.pending.isEmpty()) {
                    return;
                } // One pending notification covers the latest facts.
                lane.pending.addLast(new CloseTask(work, rejected));
                if (lane.submitted) {
                    return;
                }
                lane.submitted = true;
            }
            try {
                start.accept(lane);
            } catch (RuntimeException | Error failure) {
                lane.reject(failure);
                throw failure;
            }
            return;
        }
    }

    /**
     * Drains one owner's queued work serially and retires when no accepted work remains.
     */
    private final class Lane implements Runnable {
        private final Object owner;
        private final java.util.concurrent.CompletableFuture<Void> completed =
                new java.util.concurrent.CompletableFuture<>();
        private final ArrayDeque<CloseTask> pending = new ArrayDeque<>();
        private boolean started;
        private boolean submitted;
        private volatile boolean retired;

        private Lane(Object owner) {
            this.owner = owner;
        }

        @Override
        public void run() {
            synchronized (this) {
                if (started || retired) {
                    return;
                }
                started = true;
            }
            var previous = executing.get();
            executing.set(true);
            try {
                Throwable failure = null;
                for (; ; ) {
                    CloseTask next;
                    synchronized (this) {
                        next = pending.pollFirst();
                        if (next == null) {
                            retired = true;
                        }
                    }
                    if (next == null) {
                        lanes.remove(owner, this);
                        Failures.rethrow(failure, "request closing failed");
                        return;
                    }
                    failure = Failures.run(failure, next);
                }
            } finally {
                if (previous == null) executing.remove();
                else executing.set(previous);
                completed.complete(null);
            }
        }

        /**
         * Rejects every queued close operation when its runner could not start.
         */
        private void reject(Throwable cause) {
            java.util.List<CloseTask> abandoned;
            synchronized (this) {
                if (started || retired) {
                    return;
                }
                retired = true;
                abandoned = new ArrayList<>(pending);
                pending.clear();
            }
            lanes.remove(owner, this);
            try {
                for (var work : abandoned) {
                    if (work.rejected != null)
                        Failures.run(cause, () -> work.rejected.accept(cause));
                }
            } finally {
                completed.complete(null);
            }
        }
    }

    /**
     * Runs one close action and retains its submission rejection callback.
     */
    private record CloseTask(Runnable action, Consumer<Throwable> rejected) implements Runnable {
        @Override
        public void run() {
            action.run();
        }
    }
}

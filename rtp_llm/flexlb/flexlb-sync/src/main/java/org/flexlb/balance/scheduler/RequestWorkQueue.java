package org.flexlb.balance.scheduler;

import org.flexlb.util.Failures;

import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Consumer;

/**
 * Serializes resource settlement only. Notifications wait here, never in RequestContext.
 */
final class RequestWorkQueue {
    private final ConcurrentMap<Object, Lane> lanes = new ConcurrentHashMap<>();

    boolean execute(
            Object owner, Runnable work, Consumer<Throwable> rejected, Consumer<Runnable> start) {
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
                if (lane.pending != null) {
                    return false;
                } // One pending notification covers the latest facts.
                lane.pending = new Task(work, rejected);
                if (lane.running) {
                    return true;
                }
                lane.running = true;
            }
            try {
                start.accept(lane);
            } catch (RuntimeException | Error failure) {
                lane.reject(failure);
                throw failure;
            }
            return true;
        }
    }

    private record Task(Runnable work, Consumer<Throwable> rejected) {}

    private final class Lane implements Runnable {
        private final Object owner;
        private Task pending;
        private boolean running;
        private boolean started;
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
            Throwable failure = null;
            for (; ; ) {
                Task next;
                synchronized (this) {
                    next = pending;
                    pending = null;
                    if (next == null) {
                        retired = true;
                    }
                }
                if (next == null) {
                    lanes.remove(owner, this);
                    Failures.rethrow(failure, "request settlement failed");
                    return;
                }
                failure = Failures.run(failure, next.work());
            }
        }

        /**
         * A failed start resolves every counted task, including a notification queued during handoff.
         */
        private void reject(Throwable cause) {
            Task abandoned;
            synchronized (this) {
                if (started || retired) {
                    return;
                }
                retired = true;
                abandoned = pending;
                pending = null;
            }
            lanes.remove(owner, this);
            if (abandoned != null) {
                Failures.run(cause, () -> abandoned.rejected().accept(cause));
            }
        }
    }
}

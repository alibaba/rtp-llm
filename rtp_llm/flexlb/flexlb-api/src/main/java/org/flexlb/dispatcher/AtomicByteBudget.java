package org.flexlb.dispatcher;

import java.util.concurrent.atomic.AtomicLong;

/** Request-scoped, overflow-safe byte budget shared by concurrent fanout calls. */
final class AtomicByteBudget {

    private final long limit;
    private final AtomicLong reserved = new AtomicLong();

    AtomicByteBudget(long limit) {
        if (limit <= 0) {
            throw new IllegalArgumentException("byte budget must be > 0, got " + limit);
        }
        this.limit = limit;
    }

    Reservation newReservation() {
        return new Reservation(this);
    }

    long limit() {
        return limit;
    }

    private boolean tryReserve(long bytes) {
        if (bytes < 0) {
            throw new IllegalArgumentException("bytes must be >= 0, got " + bytes);
        }
        while (true) {
            long current = reserved.get();
            if (bytes > limit - current) {
                return false;
            }
            if (reserved.compareAndSet(current, current + bytes)) {
                return true;
            }
        }
    }

    private void release(long bytes) {
        if (bytes > 0) {
            reserved.addAndGet(-bytes);
        }
    }

    /** Tracks one sub-call's share so failed/cancelled reads can release exactly once. */
    static final class Reservation {
        private final AtomicByteBudget owner;
        private long bytes;
        private boolean released;

        private Reservation(AtomicByteBudget owner) {
            this.owner = owner;
        }

        synchronized boolean tryReserve(long additionalBytes) {
            if (released) {
                return false;
            }
            if (!owner.tryReserve(additionalBytes)) {
                return false;
            }
            bytes += additionalBytes;
            return true;
        }

        synchronized boolean ensureTotal(long totalBytes) {
            long additional = totalBytes - bytes;
            return additional <= 0 || tryReserve(additional);
        }

        synchronized long bytes() {
            return bytes;
        }

        synchronized void release() {
            if (released) {
                return;
            }
            released = true;
            long held = bytes;
            bytes = 0;
            owner.release(held);
        }

        long limit() {
            return owner.limit();
        }
    }
}

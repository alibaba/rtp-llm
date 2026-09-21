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
        return new Reservation();
    }

    long limit() {
        return limit;
    }

    boolean tryReserve(long bytes) {
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

    /** Tracks one sub-call's share so failed/cancelled reads can release exactly once. */
    final class Reservation {
        private long bytes;
        private boolean released;

        synchronized boolean tryReserve(long additionalBytes) {
            if (released) {
                return false;
            }
            if (!AtomicByteBudget.this.tryReserve(additionalBytes)) {
                return false;
            }
            bytes += additionalBytes;
            return true;
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
            reserved.addAndGet(-held);
        }

        long limit() {
            return limit;
        }
    }
}

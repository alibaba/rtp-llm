package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;

import java.lang.invoke.VarHandle;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Consumer;
import java.util.function.LongSupplier;
import java.util.function.Predicate;

/**
 * Request-scoped Prefill accounting for frontend-delivered route decisions.
 *
 * <p>The ledger owns the complete lifecycle of a route request: capacity,
 * EngineFence protection, WorkerStatus progress, wait-time accounting, and TTL
 * eviction. Mutations for one request id are serialized by a fixed stripe. The
 * only nested lock order is request stripe followed by the running-wait lock.
 * No callback is invoked while either lock is held.
 *
 * <p>Entries deliberately retain no request context, response future, endpoint,
 * or scheduler object. This bounds their lifetime to compact accounting state
 * even when an Engine terminal is delayed until TTL eviction.
 */
final class PrefillRequestLedger {

    private static final int STRIPE_COUNT = 32;
    private static final int WAIT_SNAPSHOT_MAX_ATTEMPTS = 4;
    private static final long WAIT_CACHE_TTL_MS = 2;
    private static final Consumer<WaitSnapshotStage> NOOP_SNAPSHOT_OBSERVER = ignored -> {};

    enum WaitSnapshotStage {
        AFTER_QUEUED_READ,
        BEFORE_CACHE_PUBLISH
    }

    private final ConcurrentHashMap<String, Entry> entries = new ConcurrentHashMap<>();
    private final Stripe[] stripes = createStripes();
    private final RunningWaitState runningWait = new RunningWaitState();
    private final AtomicInteger count = new AtomicInteger();
    private final Runnable capacityAvailable;
    private final LongSupplier clock;
    private final Consumer<WaitSnapshotStage> snapshotObserver;

    /**
     * Concurrent-writer epoch for the queued and running wait components.
     * Equal started/completed counters identify a quiescent version even when
     * writers on different request stripes overlap.
     */
    private final AtomicLong waitMutationsStarted = new AtomicLong();
    private final AtomicLong waitMutationsCompleted = new AtomicLong();
    /** Serializes only cache publication by readers; never ledger mutation. */
    private final AtomicLong waitCacheSequence = new AtomicLong();
    private volatile long cachedWaitMs;
    private volatile long cachedWaitMutationVersion = -1;
    private volatile long cachedWaitExpireAtMs;

    PrefillRequestLedger(Runnable capacityAvailable) {
        this(capacityAvailable, System::currentTimeMillis, NOOP_SNAPSHOT_OBSERVER);
    }

    PrefillRequestLedger(Runnable capacityAvailable,
                         LongSupplier clock,
                         Consumer<WaitSnapshotStage> snapshotObserver) {
        this.capacityAvailable = Objects.requireNonNull(capacityAvailable, "capacityAvailable");
        this.clock = Objects.requireNonNull(clock, "clock");
        this.snapshotObserver = Objects.requireNonNull(snapshotObserver, "snapshotObserver");
    }

    /**
     * Atomically acquire accounting for one route request.
     *
     * <p>A non-positive limit disables the configured cap, while the integer
     * representation limit remains a hard bound. Re-acquiring a live request id
     * is idempotent and does not replace its original prediction.
     */
    boolean tryAcquire(String requestId, long predictMs, int maxPerWorker) {
        String requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        boolean capacityNeedsNotification = false;
        try {
            synchronized (stripe) {
                if (entries.containsKey(requestKey)) {
                    return true;
                }
                if (!tryAcquireSlot(maxPerWorker)) {
                    return false;
                }

                boolean queuedWorkAdded = false;
                boolean mutationBegun = false;
                boolean published = false;
                long accountedPredictMs = 0;
                try {
                    Entry entry = new Entry(predictMs, clock.getAsLong());
                    accountedPredictMs = entry.predictTimeMs();
                    beginWaitMutation();
                    mutationBegun = true;
                    stripe.addQueued(accountedPredictMs);
                    queuedWorkAdded = true;
                    entries.put(requestKey, entry);
                    published = true;
                    return true;
                } finally {
                    if (!published) {
                        if (queuedWorkAdded) {
                            stripe.removeQueued(accountedPredictMs);
                        }
                        count.decrementAndGet();
                        capacityNeedsNotification = true;
                    }
                    if (mutationBegun) {
                        endWaitMutation();
                    }
                }
            }
        } finally {
            if (capacityNeedsNotification) {
                capacityAvailable.run();
            }
        }
    }

    /** Remove an explicitly abandoned route request, idempotently. */
    boolean release(String requestId) {
        return remove(requestId);
    }

    /** Pin a live entry while EngineFence resolves ambiguous delivery ownership. */
    boolean protect(String requestId) {
        String requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        synchronized (stripe) {
            Entry entry = entries.get(requestKey);
            if (entry == null) {
                return false;
            }
            entry.protectWithEngineFence();
            return true;
        }
    }

    /** Clear EngineFence protection without refreshing the request's TTL age. */
    boolean unprotect(String requestId) {
        String requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        synchronized (stripe) {
            Entry entry = entries.get(requestKey);
            return entry != null && entry.clearEngineFenceProtection();
        }
    }

    /**
     * Observe the latest Engine phase for a request.
     *
     * @return whether the request is owned by this ledger
     */
    boolean observe(String requestId, boolean running, long observedAtMs) {
        String requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        synchronized (stripe) {
            Entry entry = entries.get(requestKey);
            if (entry == null) {
                return false;
            }
            if (running == entry.running()) {
                if (running) {
                    entry.markRunning(observedAtMs);
                } else {
                    entry.markQueued(observedAtMs);
                }
                return true;
            }

            beginWaitMutation();
            try {
                if (running) {
                    entry.markRunning(observedAtMs);
                    // Publish the running contribution first. A racing estimate
                    // may briefly overestimate, but cannot observe an unsafe gap.
                    runningWait.add(entry, observedAtMs);
                    stripe.removeQueued(entry.predictTimeMs());
                } else {
                    entry.markQueued(observedAtMs);
                    // The reverse transition follows the same conservative order.
                    stripe.addQueued(entry.predictTimeMs());
                    runningWait.remove(entry, observedAtMs);
                }
            } finally {
                endWaitMutation();
            }
            return true;
        }
    }

    /** Settle an authoritative WorkerStatus terminal, idempotently. */
    boolean settle(String requestId) {
        return remove(requestId);
    }

    /** Advisory capacity snapshot; {@link #tryAcquire} is the hard gate. */
    int available(int maxPerWorker) {
        if (maxPerWorker <= 0) {
            return Integer.MAX_VALUE;
        }
        return Math.max(0, maxPerWorker - count.get());
    }

    int count() {
        return count.get();
    }

    /**
     * Return the current quiescent mutation version, or {@code -1} while any
     * request writer is active. Endpoint uses this to compose request and batch
     * estimates without taking a cross-ledger lock.
     */
    long mutationVersion() {
        VarHandle.loadLoadFence();
        long started = waitMutationsStarted.get();
        long completed = waitMutationsCompleted.get();
        return started == completed ? completed : -1;
    }

    /**
     * Return a bounded, allocation-free, internally coherent wait estimate.
     * Continuous mutation is reported as {@link Long#MAX_VALUE} rather than an
     * admission-unsafe underestimate.
     */
    long estimate(long nowMs) {
        for (int attempt = 0; attempt < WAIT_SNAPSHOT_MAX_ATTEMPTS; attempt++) {
            long startedBefore = waitMutationsStarted.get();
            long completedBefore = waitMutationsCompleted.get();
            if (startedBefore != completedBefore) {
                Thread.onSpinWait();
                continue;
            }

            long cacheSequenceBefore = waitCacheSequence.get();
            if ((cacheSequenceBefore & 1L) == 0) {
                long cacheExpireAtMs = cachedWaitExpireAtMs;
                long cacheVersion = cachedWaitMutationVersion;
                long cacheValueMs = cachedWaitMs;
                VarHandle.loadLoadFence();
                long cacheSequenceAfter = waitCacheSequence.get();
                if (cacheSequenceBefore == cacheSequenceAfter
                        && cacheVersion == completedBefore
                        && nowMs < cacheExpireAtMs
                        && waitEpochUnchanged(startedBefore, completedBefore)) {
                    return cacheValueMs;
                }
            }

            long result = computeWaitMs(nowMs);
            if (!waitEpochUnchanged(startedBefore, completedBefore)) {
                Thread.onSpinWait();
                continue;
            }

            snapshotObserver.accept(WaitSnapshotStage.BEFORE_CACHE_PUBLISH);
            publishWaitCache(completedBefore, result, nowMs + WAIT_CACHE_TTL_MS);
            return result;
        }
        return Long.MAX_VALUE;
    }

    /** Evict stale, unprotected requests and release their capacity exactly once. */
    int evict(long ttlMs) {
        return evict(ttlMs, ignored -> false);
    }

    /** Evict only entries which are no longer owned by the scheduler. */
    int evict(long ttlMs, Predicate<String> schedulerOwnsRequest) {
        long nowMs = clock.getAsLong();
        int evicted = 0;
        for (Map.Entry<String, Entry> observed : entries.entrySet()) {
            Entry candidate = observed.getValue();
            if (nowMs - candidate.lastObservedAtMs() <= ttlMs) {
                continue;
            }

            String requestKey = observed.getKey();
            Stripe stripe = stripeFor(requestKey);
            synchronized (stripe) {
                Entry current = entries.get(requestKey);
                if (current != candidate
                        || current.engineFenceProtected()
                        || schedulerOwnsRequest.test(requestKey)
                        || nowMs - current.lastObservedAtMs() <= ttlMs) {
                    continue;
                }
                beginWaitMutation();
                try {
                    entries.remove(requestKey);
                    removeAccounting(stripe, current, nowMs);
                    evicted++;
                } finally {
                    endWaitMutation();
                }
            }
        }
        if (evicted > 0) {
            capacityAvailable.run();
        }
        return evicted;
    }

    /** Age of the oldest live route-request entry, for Endpoint metrics glue. */
    long maxAge(long nowMs) {
        return InflightEvictor.maxAgeMs(entries, nowMs);
    }

    private boolean remove(String requestId) {
        String requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        synchronized (stripe) {
            Entry entry = entries.get(requestKey);
            if (entry == null) {
                return false;
            }
            beginWaitMutation();
            try {
                entries.remove(requestKey);
                removeAccounting(stripe, entry, clock.getAsLong());
            } finally {
                endWaitMutation();
            }
        }
        capacityAvailable.run();
        return true;
    }

    private void removeAccounting(Stripe stripe, Entry entry, long nowMs) {
        count.decrementAndGet();
        if (entry.running()) {
            runningWait.remove(entry, nowMs);
        } else {
            stripe.removeQueued(entry.predictTimeMs());
        }
    }

    private boolean tryAcquireSlot(int maxPerWorker) {
        while (true) {
            int current = count.get();
            if ((maxPerWorker > 0 && current >= maxPerWorker)
                    || current == Integer.MAX_VALUE) {
                return false;
            }
            if (count.compareAndSet(current, current + 1)) {
                return true;
            }
        }
    }

    private long computeWaitMs(long nowMs) {
        if (entries.isEmpty()) {
            return 0;
        }
        long queuedMs = 0;
        for (Stripe stripe : stripes) {
            queuedMs = saturatedAdd(queuedMs, stripe.queuedPredictTimeMs());
        }
        snapshotObserver.accept(WaitSnapshotStage.AFTER_QUEUED_READ);
        return saturatedAdd(queuedMs, runningWait.estimate(nowMs));
    }

    private void beginWaitMutation() {
        waitMutationsStarted.incrementAndGet();
        VarHandle.storeStoreFence();
    }

    private void endWaitMutation() {
        cachedWaitExpireAtMs = 0;
        VarHandle.storeStoreFence();
        waitMutationsCompleted.incrementAndGet();
    }

    private boolean waitEpochUnchanged(long startedBefore, long completedBefore) {
        VarHandle.loadLoadFence();
        long startedAfter = waitMutationsStarted.get();
        long completedAfter = waitMutationsCompleted.get();
        return startedBefore == completedBefore
                && startedBefore == startedAfter
                && completedBefore == completedAfter
                && startedAfter == completedAfter;
    }

    private void publishWaitCache(long mutationVersion, long valueMs, long expireAtMs) {
        long sequence = waitCacheSequence.get();
        if ((sequence & 1L) != 0
                || !waitCacheSequence.compareAndSet(sequence, sequence + 1)) {
            return;
        }
        try {
            cachedWaitMs = valueMs;
            cachedWaitMutationVersion = mutationVersion;
            cachedWaitExpireAtMs = expireAtMs;
        } finally {
            VarHandle.storeStoreFence();
            waitCacheSequence.set(sequence + 2);
        }
    }

    private Stripe stripeFor(String requestId) {
        return stripes[requestId.hashCode() & (STRIPE_COUNT - 1)];
    }

    private static Stripe[] createStripes() {
        Stripe[] result = new Stripe[STRIPE_COUNT];
        for (int i = 0; i < result.length; i++) {
            result[i] = new Stripe();
        }
        return result;
    }

    private static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    private static final class Stripe {
        private volatile long queuedPredictTimeMs;

        void addQueued(long predictMs) {
            queuedPredictTimeMs += predictMs;
        }

        void removeQueued(long predictMs) {
            queuedPredictTimeMs -= predictMs;
        }

        long queuedPredictTimeMs() {
            return queuedPredictTimeMs;
        }
    }

    /** Compact entry; all mutable fields are guarded by the owning stripe or running lock. */
    private static final class Entry implements InflightEvictor.TtlTracked {

        // The count is an int, so this bound keeps the maximum aggregate
        // prediction representable in a signed long without saturation.
        private static final long MAX_ACCOUNTED_PREDICT_TIME_MS = Integer.MAX_VALUE;

        private final long predictTimeMs;
        private final long createdAtMs;
        private volatile long lastObservedAtMs;
        private boolean running;
        /** Guarded by the owning request stripe. */
        private boolean engineFenceProtected;
        /** Guarded by RunningWaitState.lock. */
        private long remainingPredictMs;
        private Entry serviceOrderPrevious;
        private Entry serviceOrderNext;
        private boolean inServiceOrder;

        Entry(long predictTimeMs, long nowMs) {
            this.predictTimeMs = Math.min(
                    MAX_ACCOUNTED_PREDICT_TIME_MS, Math.max(0, predictTimeMs));
            this.createdAtMs = nowMs;
            this.lastObservedAtMs = nowMs;
        }

        long predictTimeMs() {
            return predictTimeMs;
        }

        @Override
        public long createdAtMs() {
            return createdAtMs;
        }

        long lastObservedAtMs() {
            return lastObservedAtMs;
        }

        boolean running() {
            return running;
        }

        boolean engineFenceProtected() {
            return engineFenceProtected;
        }

        void protectWithEngineFence() {
            engineFenceProtected = true;
        }

        boolean clearEngineFenceProtection() {
            boolean wasProtected = engineFenceProtected;
            engineFenceProtected = false;
            return wasProtected;
        }

        boolean markQueued(long observedAtMs) {
            touch(observedAtMs);
            boolean wasRunning = running;
            running = false;
            return wasRunning;
        }

        boolean markRunning(long observedAtMs) {
            touch(observedAtMs);
            if (!running) {
                running = true;
                return true;
            }
            return false;
        }

        private void touch(long observedAtMs) {
            if (observedAtMs > lastObservedAtMs) {
                lastObservedAtMs = observedAtMs;
            }
        }
    }

    /**
     * Worker-wide running-work estimate with allocation-free O(1) arbitrary
     * removal through intrusive links stored on each entry.
     */
    private static final class RunningWaitState {
        private final StampedLock lock = new StampedLock();
        private Entry serviceOrderHead;
        private Entry serviceOrderTail;
        private long remainingMs;
        private int serviceableCount;
        private long progressBaseMs = Long.MAX_VALUE;

        void add(Entry entry, long nowMs) {
            long stamp = lock.writeLock();
            try {
                rollForward(nowMs);
                entry.remainingPredictMs = entry.predictTimeMs();
                append(entry);
                remainingMs = saturatedAdd(remainingMs, entry.remainingPredictMs);
                progressBaseMs = nowMs;
            } finally {
                lock.unlockWrite(stamp);
            }
        }

        void remove(Entry entry, long nowMs) {
            long stamp = lock.writeLock();
            try {
                rollForward(nowMs);
                long entryRemainingMs = entry.remainingPredictMs;
                if (unlink(entry)) {
                    remainingMs = Math.max(0, remainingMs - entryRemainingMs);
                }
                entry.remainingPredictMs = 0;
                if (serviceableCount <= 0) {
                    remainingMs = 0;
                    serviceableCount = 0;
                    progressBaseMs = Long.MAX_VALUE;
                } else {
                    progressBaseMs = nowMs;
                }
            } finally {
                lock.unlockWrite(stamp);
            }
        }

        long estimate(long nowMs) {
            long stamp = lock.tryOptimisticRead();
            long observedRemainingMs = remainingMs;
            int observedServiceableCount = serviceableCount;
            long observedProgressBaseMs = progressBaseMs;
            if (!lock.validate(stamp)) {
                stamp = lock.readLock();
                try {
                    observedRemainingMs = remainingMs;
                    observedServiceableCount = serviceableCount;
                    observedProgressBaseMs = progressBaseMs;
                } finally {
                    lock.unlockRead(stamp);
                }
            }
            if (observedServiceableCount > 0
                    && observedProgressBaseMs != Long.MAX_VALUE) {
                observedRemainingMs = Math.max(0, observedRemainingMs
                        - Math.max(0, nowMs - observedProgressBaseMs));
            }
            return observedRemainingMs;
        }

        private void rollForward(long nowMs) {
            if (serviceableCount <= 0 || progressBaseMs == Long.MAX_VALUE) {
                return;
            }
            long availableServiceMs = Math.max(0, nowMs - progressBaseMs);
            while (availableServiceMs > 0 && serviceOrderHead != null) {
                Entry head = serviceOrderHead;
                long consumedMs = Math.min(availableServiceMs, head.remainingPredictMs);
                head.remainingPredictMs = Math.max(0, head.remainingPredictMs - consumedMs);
                remainingMs = Math.max(0, remainingMs - consumedMs);
                availableServiceMs -= consumedMs;
                if (head.remainingPredictMs == 0) {
                    unlink(head);
                }
            }
            if (serviceableCount <= 0) {
                remainingMs = 0;
                serviceableCount = 0;
                progressBaseMs = Long.MAX_VALUE;
            } else {
                progressBaseMs = nowMs;
            }
        }

        private void append(Entry entry) {
            if (entry.inServiceOrder) {
                throw new IllegalStateException("request is already in running service order");
            }
            entry.serviceOrderPrevious = serviceOrderTail;
            entry.serviceOrderNext = null;
            entry.inServiceOrder = true;
            if (serviceOrderTail == null) {
                serviceOrderHead = entry;
            } else {
                serviceOrderTail.serviceOrderNext = entry;
            }
            serviceOrderTail = entry;
            serviceableCount++;
        }

        /** Remove one entry by identity in O(1), safely after exhaustion/removal. */
        private boolean unlink(Entry entry) {
            if (!entry.inServiceOrder) {
                return false;
            }
            Entry previous = entry.serviceOrderPrevious;
            Entry next = entry.serviceOrderNext;
            if (previous == null) {
                serviceOrderHead = next;
            } else {
                previous.serviceOrderNext = next;
            }
            if (next == null) {
                serviceOrderTail = previous;
            } else {
                next.serviceOrderPrevious = previous;
            }
            entry.serviceOrderPrevious = null;
            entry.serviceOrderNext = null;
            entry.inServiceOrder = false;
            serviceableCount--;
            return true;
        }
    }
}

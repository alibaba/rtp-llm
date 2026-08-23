package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.util.Logger;

import java.lang.invoke.VarHandle;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Consumer;
import java.util.function.LongPredicate;
import java.util.function.LongSupplier;

/**
 * Request-scoped Prefill accounting for individually delivered requests.
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

    private enum EntryKind {
        DIRECT_REQUEST,
        QUEUE_ROUTE
    }

    enum ProgressOwnership {
        NOT_TRACKED,
        DIRECT_REQUEST,
        QUEUE_ROUTE
    }

    /**
     * Result of reserving request capacity before route delivery begins.
     *
     * <p>Only {@link Status#ACQUIRED} carries a reservation. An existing
     * ledger owner is reported separately so a caller can preserve idempotent
     * admission without receiving authority to release that owner's entry.
     */
    record RequestCapacityReservationAcquisition(
            Status status,
            RequestCapacityReservation reservation) {

        enum Status {
            ACQUIRED,
            CAPACITY_FULL,
            REQUEST_ALREADY_TRACKED
        }

        RequestCapacityReservationAcquisition {
            Objects.requireNonNull(status, "status");
            if ((status == Status.ACQUIRED) != (reservation != null)) {
                throw new IllegalArgumentException(
                        "only an acquired capacity reservation may carry a token");
            }
        }

        static RequestCapacityReservationAcquisition acquired(
                RequestCapacityReservation reservation) {
            return new RequestCapacityReservationAcquisition(
                    Status.ACQUIRED, Objects.requireNonNull(reservation, "reservation"));
        }

        static RequestCapacityReservationAcquisition capacityFull() {
            return new RequestCapacityReservationAcquisition(Status.CAPACITY_FULL, null);
        }

        static RequestCapacityReservationAcquisition alreadyTracked() {
            return new RequestCapacityReservationAcquisition(
                    Status.REQUEST_ALREADY_TRACKED, null);
        }
    }

    /**
     * Exact-entry token for one newly reserved request-capacity slot.
     *
     * <p>{@link #prepareForDelivery()} verifies and pins the exact live ledger
     * entry while it remains compensable. After the final hard-capacity
     * ownership transition succeeds, {@link #completePreparedDeliveryTransfer()}
     * performs the local one-way handoff. {@link #abortBeforeDelivery()} can
     * remove only the exact entry created by this acquisition, never a
     * replacement request-id generation.
     */
    static final class RequestCapacityReservation {

        private enum State {
            RESERVED,
            PREPARED_FOR_DELIVERY,
            DELIVERY_OWNED,
            CLOSED
        }

        private final PrefillRequestLedger ledger;
        private final long requestId;
        private final Entry reservedEntry;
        private State state = State.RESERVED;

        private RequestCapacityReservation(PrefillRequestLedger ledger,
                                           long requestId,
                                           Entry reservedEntry) {
            this.ledger = ledger;
            this.requestId = requestId;
            this.reservedEntry = reservedEntry;
        }

        /**
         * Verify and pin the exact reserved entry against ordinary lifecycle
         * settlement. Capacity was already occupied during acquisition and is
         * not checked again here.
         *
         * @return {@code true} only for the first preparation while the exact
         *         reserved entry is still live
         */
        synchronized boolean prepareForDelivery() {
            if (state != State.RESERVED) {
                return false;
            }
            if (!ledger.prepareReservation(requestId, reservedEntry, this)) {
                state = State.CLOSED;
                return false;
            }
            state = State.PREPARED_FOR_DELIVERY;
            return true;
        }

        /**
         * Transfer this token to ordinary lifecycle ownership. The caller
         * invokes this only after the final hard-capacity ownership transition
         * succeeds, so this local transition performs no endpoint work and
         * cannot fail.
         */
        synchronized void completePreparedDeliveryTransfer() {
            if (state == State.PREPARED_FOR_DELIVERY) {
                // PREPARED pins the exact Entry against every ordinary removal.
                // The token monitor is therefore the only writer which can
                // complete this one-way handoff after Decode commits.
                reservedEntry.transferToDelivery();
                state = State.DELIVERY_OWNED;
            }
        }

        /**
         * Abandon this exact entry while delivery is still externally
         * invisible, whether it is reserved or prepared for delivery.
         */
        synchronized boolean abortBeforeDelivery() {
            if (state != State.RESERVED
                    && state != State.PREPARED_FOR_DELIVERY) {
                return false;
            }
            state = State.CLOSED;
            return ledger.releaseReservation(
                    requestId, reservedEntry, this, true);
        }

        /**
         * Release this reservation before ownership is prepared.
         *
         * @return {@code true} only when this call removed its exact entry
         */
        synchronized boolean release() {
            if (state != State.RESERVED) {
                return false;
            }
            state = State.CLOSED;
            return ledger.releaseReservation(
                    requestId, reservedEntry, this, false);
        }

    }

    private final ConcurrentHashMap<Long, Entry> entries = new ConcurrentHashMap<>();
    private final Stripe[] stripes = createStripes();
    private final RunningWaitState runningWait = new RunningWaitState();
    private final AtomicInteger requestCount = new AtomicInteger();
    /** Only QUEUE_ROUTE entries consume the configured route-request capacity. */
    private final AtomicInteger queueRouteCapacityUsage = new AtomicInteger();
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

    /** Atomically publish one DIRECT request as ordinary lifecycle-owned work. */
    boolean registerDirectRequest(long requestId, long predictMs) {
        Long requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        synchronized (stripe) {
            Entry existing = entries.get(requestKey);
            if (existing != null) {
                return existing.kind() == EntryKind.DIRECT_REQUEST
                        && existing.isDeliveryOwned();
            }
            if (!tryAcquireAccounting(EntryKind.DIRECT_REQUEST, 0)) {
                return false;
            }

            long nowMs = clock.getAsLong();
            Entry entry = new Entry(EntryKind.DIRECT_REQUEST, predictMs, nowMs);
            boolean waitAccountingAdded = false;
            boolean mutationBegun = false;
            boolean published = false;
            try {
                entry.startDirectDelivery();
                beginWaitMutation();
                mutationBegun = true;
                runningWait.add(entry, nowMs);
                waitAccountingAdded = true;
                entries.put(requestKey, entry);
                published = true;
                return true;
            } finally {
                if (!published) {
                    if (waitAccountingAdded) {
                        runningWait.remove(entry, nowMs);
                    }
                    releaseAccountingSlot(EntryKind.DIRECT_REQUEST);
                }
                if (mutationBegun) {
                    endWaitMutation();
                }
            }
        }
    }

    /**
     * Retire every DIRECT request owned by an endpoint generation.
     *
     * <p>The generation lifecycle has already rejected new handoffs and waited
     * for accepted DIRECT publications before invoking this method. Locking all
     * request stripes freezes ordinary request mutation, making this a complete
     * drain rather than a weakly consistent map scan. QUEUE_ROUTE entries remain
     * untouched because their hard-capacity and EngineFence lifecycle outlives
     * endpoint selection.
     */
    int retireDirectRequests() {
        return retireDirectRequestsWithStripesLocked(0);
    }

    private int retireDirectRequestsWithStripesLocked(int stripeIndex) {
        if (stripeIndex == stripes.length) {
            return removeDirectRequestsWhileStripesLocked();
        }
        synchronized (stripes[stripeIndex]) {
            return retireDirectRequestsWithStripesLocked(stripeIndex + 1);
        }
    }

    private int removeDirectRequestsWhileStripesLocked() {
        long nowMs = clock.getAsLong();
        int retired = 0;
        beginWaitMutation();
        try {
            for (Map.Entry<Long, Entry> observed : entries.entrySet()) {
                Entry entry = observed.getValue();
                if (entry.kind() != EntryKind.DIRECT_REQUEST
                        || !entries.remove(observed.getKey(), entry)) {
                    continue;
                }
                boolean routeCapacityReleased = removeAccounting(
                        stripeFor(observed.getKey()), entry, nowMs);
                if (routeCapacityReleased) {
                    throw new IllegalStateException(
                            "DIRECT retirement released QUEUE_ROUTE capacity");
                }
                retired++;
            }
        } finally {
            endWaitMutation();
        }
        return retired;
    }

    /**
     * Reserve one request-capacity slot without transferring it to the normal
     * delivery lifecycle yet.
     *
     * <p>Acquisition is the only capacity gate. A successful reservation is
     * immediately included in request count and wait accounting. The returned
     * token must subsequently be prepared and transferred, or released.
     */
    RequestCapacityReservationAcquisition acquireCapacityReservation(
            long requestId,
            long predictMs,
            int maxPerWorker) {
        return acquireCapacityReservation(
                requestId, predictMs, maxPerWorker, EntryKind.QUEUE_ROUTE);
    }

    private RequestCapacityReservationAcquisition acquireCapacityReservation(
            long requestId,
            long predictMs,
            int maxPerWorker,
            EntryKind kind) {
        Long requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        boolean capacityNeedsNotification = false;
        try {
            synchronized (stripe) {
                if (entries.containsKey(requestKey)) {
                    return RequestCapacityReservationAcquisition.alreadyTracked();
                }
                if (!tryAcquireAccounting(kind, maxPerWorker)) {
                    return RequestCapacityReservationAcquisition.capacityFull();
                }

                boolean waitAccountingAdded = false;
                boolean mutationBegun = false;
                boolean published = false;
                long accountedPredictMs = 0;
                long nowMs = 0;
                Entry entry = null;
                try {
                    nowMs = clock.getAsLong();
                    entry = new Entry(kind, predictMs, nowMs);
                    RequestCapacityReservation reservation =
                            new RequestCapacityReservation(this, requestId, entry);
                    entry.reserveFor(reservation);
                    RequestCapacityReservationAcquisition acquisition =
                            RequestCapacityReservationAcquisition.acquired(reservation);
                    accountedPredictMs = entry.predictTimeMs();
                    beginWaitMutation();
                    mutationBegun = true;
                    if (entry.waitProgressActive()) {
                        runningWait.add(entry, nowMs);
                    } else {
                        stripe.addQueued(accountedPredictMs);
                    }
                    waitAccountingAdded = true;
                    entries.put(requestKey, entry);
                    published = true;
                    return acquisition;
                } finally {
                    if (!published) {
                        if (waitAccountingAdded) {
                            if (entry.waitProgressActive()) {
                                runningWait.remove(entry, nowMs);
                            } else {
                                stripe.removeQueued(accountedPredictMs);
                            }
                        }
                        capacityNeedsNotification = releaseAccountingSlot(kind);
                    }
                    if (mutationBegun) {
                        endWaitMutation();
                    }
                }
            }
        } finally {
            if (capacityNeedsNotification) {
                notifyCapacityAvailable();
            }
        }
    }

    /** Atomically pin the exact entry against ordinary lifecycle settlement. */
    private boolean prepareReservation(long requestId,
                                       Entry reservedEntry,
                                       RequestCapacityReservation reservation) {
        Long requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        synchronized (stripe) {
            return entries.get(requestKey) == reservedEntry
                    && reservedEntry.prepareForDelivery(reservation);
        }
    }

    /** Release only the exact entry created for a still-compensable reservation. */
    private boolean releaseReservation(
            long requestId,
            Entry reservedEntry,
            RequestCapacityReservation reservation,
            boolean allowPrepared) {
        Long requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        boolean routeCapacityReleased;
        synchronized (stripe) {
            if (entries.get(requestKey) != reservedEntry
                    || !reservedEntry.isReleasableBy(reservation, allowPrepared)) {
                return false;
            }
            beginWaitMutation();
            try {
                if (!entries.remove(requestKey, reservedEntry)) {
                    return false;
                }
                routeCapacityReleased = removeAccounting(
                        stripe, reservedEntry, clock.getAsLong());
            } finally {
                endWaitMutation();
            }
        }
        if (routeCapacityReleased) {
            notifyCapacityAvailable();
        }
        return true;
    }

    /** Capacity accounting is already changed; a wakeup must not undo it. */
    private void notifyCapacityAvailable() {
        try {
            capacityAvailable.run();
        } catch (Throwable notificationFailure) {
            Logger.warn("Prefill request-capacity listener failed", notificationFailure);
        }
    }

    /** Remove an explicitly abandoned individually delivered request. */
    boolean release(long requestId) {
        return remove(requestId);
    }

    /** Pin a live entry while EngineFence resolves ambiguous delivery ownership. */
    boolean protect(long requestId) {
        Long requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        synchronized (stripe) {
            Entry entry = entries.get(requestKey);
            if (entry == null || !entry.isDeliveryOwned()) {
                return false;
            }
            entry.protectWithEngineFence();
            return true;
        }
    }

    /** Clear EngineFence protection without refreshing the request's TTL age. */
    boolean unprotect(long requestId) {
        Long requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        synchronized (stripe) {
            Entry entry = entries.get(requestKey);
            return entry != null && entry.isDeliveryOwned()
                    && entry.clearEngineFenceProtection();
        }
    }

    /**
     * Observe the latest Engine phase for a request.
     *
     * @return whether the request is owned by this ledger
     */
    ProgressOwnership observe(long requestId, boolean engineRunning, long observedAtMs) {
        Long requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        synchronized (stripe) {
            Entry entry = entries.get(requestKey);
            if (entry == null || !entry.isDeliveryOwned()) {
                return ProgressOwnership.NOT_TRACKED;
            }
            if (engineRunning == entry.waitProgressActive()) {
                if (engineRunning) {
                    entry.markRunning(observedAtMs);
                } else {
                    entry.markQueued(observedAtMs);
                }
                return progressOwnership(entry);
            }

            beginWaitMutation();
            try {
                if (engineRunning) {
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
            return progressOwnership(entry);
        }
    }

    private static ProgressOwnership progressOwnership(Entry entry) {
        return entry.kind() == EntryKind.DIRECT_REQUEST
                ? ProgressOwnership.DIRECT_REQUEST
                : ProgressOwnership.QUEUE_ROUTE;
    }

    /** Settle an authoritative WorkerStatus terminal, idempotently. */
    boolean settle(long requestId) {
        return remove(requestId);
    }

    /** Advisory QUEUE_ROUTE capacity snapshot; reservation is the hard gate. */
    int available(int maxPerWorker) {
        if (maxPerWorker <= 0) {
            return Integer.MAX_VALUE;
        }
        return Math.max(0, maxPerWorker - queueRouteCapacityUsage.get());
    }

    int count() {
        return requestCount.get();
    }

    int queueRouteCount() {
        return queueRouteCapacityUsage.get();
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
    int evict(long ttlMs, LongPredicate schedulerOwnsRequest) {
        long nowMs = clock.getAsLong();
        int evicted = 0;
        boolean routeCapacityReleased = false;
        for (Map.Entry<Long, Entry> observed : entries.entrySet()) {
            Entry candidate = observed.getValue();
            if (nowMs - candidate.lastObservedAtMs() <= ttlMs) {
                continue;
            }

            Long requestKey = observed.getKey();
            Stripe stripe = stripeFor(requestKey);
            synchronized (stripe) {
                Entry current = entries.get(requestKey);
                if (current != candidate
                        || current.isPreparedForDelivery()
                        || current.engineFenceProtected()
                        || schedulerOwnsRequest.test(requestKey)
                        || nowMs - current.lastObservedAtMs() <= ttlMs) {
                    continue;
                }
                beginWaitMutation();
                try {
                    entries.remove(requestKey);
                    routeCapacityReleased |= removeAccounting(
                            stripe, current, nowMs);
                    evicted++;
                } finally {
                    endWaitMutation();
                }
            }
        }
        if (routeCapacityReleased) {
            notifyCapacityAvailable();
        }
        return evicted;
    }

    /** Age of the oldest live DIRECT or QUEUE_ROUTE entry, for endpoint metrics. */
    long maxAge(long nowMs) {
        return InflightEvictor.maxAgeMs(entries, nowMs);
    }

    private boolean remove(long requestId) {
        Long requestKey = requestId;
        Stripe stripe = stripeFor(requestId);
        boolean routeCapacityReleased;
        synchronized (stripe) {
            Entry entry = entries.get(requestKey);
            if (entry == null || entry.isPreparedForDelivery()) {
                return false;
            }
            beginWaitMutation();
            try {
                entries.remove(requestKey);
                routeCapacityReleased = removeAccounting(
                        stripe, entry, clock.getAsLong());
            } finally {
                endWaitMutation();
            }
        }
        if (routeCapacityReleased) {
            notifyCapacityAvailable();
        }
        return true;
    }

    private boolean removeAccounting(Stripe stripe, Entry entry, long nowMs) {
        boolean routeCapacityReleased = releaseAccountingSlot(entry.kind());
        if (entry.waitProgressActive()) {
            runningWait.remove(entry, nowMs);
        } else {
            stripe.removeQueued(entry.predictTimeMs());
        }
        return routeCapacityReleased;
    }

    private boolean tryAcquireAccounting(EntryKind kind, int maxPerWorker) {
        if (!tryIncrement(requestCount, 0)) {
            return false;
        }
        if (kind == EntryKind.QUEUE_ROUTE
                && !tryIncrement(queueRouteCapacityUsage, maxPerWorker)) {
            decrementExact(requestCount, "Prefill request count");
            return false;
        }
        return true;
    }

    private static boolean tryIncrement(AtomicInteger counter, int limit) {
        while (true) {
            int current = counter.get();
            if ((limit > 0 && current >= limit)
                    || current == Integer.MAX_VALUE) {
                return false;
            }
            if (counter.compareAndSet(current, current + 1)) {
                return true;
            }
        }
    }

    private boolean releaseAccountingSlot(EntryKind kind) {
        decrementExact(requestCount, "Prefill request count");
        if (kind != EntryKind.QUEUE_ROUTE) {
            return false;
        }
        decrementExact(queueRouteCapacityUsage, "Prefill QUEUE_ROUTE capacity");
        return true;
    }

    private static void decrementExact(AtomicInteger counter, String name) {
        int remaining = counter.decrementAndGet();
        if (remaining < 0) {
            counter.incrementAndGet();
            throw new IllegalStateException(name + " released more than once");
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

    private Stripe stripeFor(long requestId) {
        return stripes[Long.hashCode(requestId) & (STRIPE_COUNT - 1)];
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

    /**
     * Compact entry. Request state is guarded by its stripe, except the final
     * PREPARED-to-DELIVERY_OWNED handoff: the exact reservation token performs
     * that one-way write and publishes it through the volatile ownership field.
     * Running-wait links remain guarded by {@link RunningWaitState#lock}.
     */
    private static final class Entry implements InflightEvictor.TtlTracked {

        private enum DeliveryOwnership {
            RESERVED,
            PREPARED_FOR_DELIVERY,
            DELIVERY_OWNED
        }

        // The count is an int, so this bound keeps the maximum aggregate
        // prediction representable in a signed long without saturation.
        private static final long MAX_ACCOUNTED_PREDICT_TIME_MS = Integer.MAX_VALUE;

        private final EntryKind kind;
        private final long predictTimeMs;
        private final long createdAtMs;
        private volatile long lastObservedAtMs;
        /** Prepared under the stripe; final ownership is published by the token. */
        private volatile DeliveryOwnership deliveryOwnership = DeliveryOwnership.RESERVED;
        /** Guarded by the stripe until PREPARED; then owned by the exact token. */
        private RequestCapacityReservation reservationOwner;
        private boolean waitProgressActive;
        /** Guarded by the owning request stripe. */
        private boolean engineFenceProtected;
        /** Guarded by RunningWaitState.lock. */
        private long remainingPredictMs;
        private Entry serviceOrderPrevious;
        private Entry serviceOrderNext;
        private boolean inServiceOrder;

        Entry(EntryKind kind, long predictTimeMs, long nowMs) {
            this.kind = Objects.requireNonNull(kind, "kind");
            this.predictTimeMs = Math.min(
                    MAX_ACCOUNTED_PREDICT_TIME_MS, Math.max(0, predictTimeMs));
            this.createdAtMs = nowMs;
            this.lastObservedAtMs = nowMs;
            // DIRECT has crossed local selection/accounting ownership when
            // published, so its wait prediction begins consuming service credit.
            this.waitProgressActive = kind == EntryKind.DIRECT_REQUEST;
        }

        EntryKind kind() {
            return kind;
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

        void reserveFor(RequestCapacityReservation reservation) {
            if (reservationOwner != null) {
                throw new IllegalStateException("Prefill entry already has a reservation owner");
            }
            reservationOwner = Objects.requireNonNull(reservation, "reservation");
        }

        void startDirectDelivery() {
            if (kind != EntryKind.DIRECT_REQUEST
                    || reservationOwner != null
                    || deliveryOwnership != DeliveryOwnership.RESERVED) {
                throw new IllegalStateException(
                        "Prefill entry is not available for DIRECT delivery");
            }
            deliveryOwnership = DeliveryOwnership.DELIVERY_OWNED;
        }

        boolean prepareForDelivery(RequestCapacityReservation reservation) {
            if (deliveryOwnership != DeliveryOwnership.RESERVED
                    || reservationOwner != reservation) {
                return false;
            }
            deliveryOwnership = DeliveryOwnership.PREPARED_FOR_DELIVERY;
            return true;
        }

        void transferToDelivery() {
            reservationOwner = null;
            deliveryOwnership = DeliveryOwnership.DELIVERY_OWNED;
        }

        boolean isReleasableBy(RequestCapacityReservation reservation,
                               boolean allowPrepared) {
            if (reservationOwner != reservation) {
                return false;
            }
            return deliveryOwnership == DeliveryOwnership.RESERVED
                    || (allowPrepared
                    && deliveryOwnership == DeliveryOwnership.PREPARED_FOR_DELIVERY);
        }

        boolean isPreparedForDelivery() {
            return deliveryOwnership == DeliveryOwnership.PREPARED_FOR_DELIVERY;
        }

        boolean isDeliveryOwned() {
            return deliveryOwnership == DeliveryOwnership.DELIVERY_OWNED;
        }

        boolean waitProgressActive() {
            return waitProgressActive;
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
            boolean wasRunning = waitProgressActive;
            waitProgressActive = false;
            return wasRunning;
        }

        boolean markRunning(long observedAtMs) {
            touch(observedAtMs);
            if (!waitProgressActive) {
                waitProgressActive = true;
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

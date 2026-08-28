package org.flexlb.balance.scheduler;

import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableSet;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Pull-based owner of requests which have not committed endpoint ownership.
 *
 * <p>A capacity notification activates one logical wait domain in O(1). The
 * coordinator then pulls ordered candidates. A pool-wide miss stops that
 * domain immediately; a request-specific miss may bypass to another runnable
 * request. Requests are never copied into a global retry wave.</p>
 */
final class PendingPlacementCoordinator implements AutoCloseable {

    private static final int ATTEMPTS_PER_TURN = 256;

    interface Work {
        int priority();

        boolean priorityOrdering();

        boolean done();

        AttemptResult attempt();

        void fail(Throwable failure);
    }

    sealed interface AttemptResult {

        enum Finished implements AttemptResult {
            INSTANCE
        }

        /**
         * @param poolUnavailable true when trying another request against the
         *                        same pool cannot succeed on the observed view
         */
        record Blocked(
                PlacementKey blocker,
                boolean poolUnavailable) implements AttemptResult {
            public Blocked {
                Objects.requireNonNull(blocker, "blocker");
            }
        }
    }

    final class Handle implements AutoCloseable {
        private final Entry entry;

        private Handle(Entry entry) {
            this.entry = entry;
        }

        @Override
        public void close() {
            discard(entry);
        }
    }

    private static final class Entry {
        private final Work work;
        private final long sequence;
        private final int orderingPriority;
        private volatile boolean discarded;
        private NavigableSet<Entry> container;

        private Entry(Work work, long sequence) {
            this.work = work;
            this.sequence = sequence;
            this.orderingPriority = work.priorityOrdering()
                    ? work.priority() : 0;
        }
    }

    private static final Comparator<Entry> ORDER = Comparator
            .comparingInt((Entry entry) -> entry.orderingPriority).reversed()
            .thenComparingLong(entry -> entry.sequence);

    /** Two sets keep one retry round explicit without copying the backlog. */
    private static final class WaitBucket {
        private NavigableSet<Entry> candidates = new TreeSet<>(ORDER);
        private NavigableSet<Entry> attempted = new TreeSet<>(ORDER);
        private boolean active;
        private boolean rescanRequested;

        private Entry peek() {
            return active && !candidates.isEmpty()
                    ? candidates.first() : null;
        }

        private void addWaiting(Entry entry) {
            entry.container = candidates;
            candidates.add(entry);
        }

        private void addAttempted(Entry entry) {
            entry.container = attempted;
            attempted.add(entry);
        }

        private void remove(Entry entry) {
            NavigableSet<Entry> owner = entry.container;
            if (owner != null) {
                owner.remove(entry);
            }
            entry.container = null;
        }

        private void activate() {
            if (active) {
                rescanRequested = true;
                return;
            }
            prepareNextRound();
        }

        private void prepareNextRound() {
            if (!attempted.isEmpty()) {
                if (candidates.isEmpty()) {
                    NavigableSet<Entry> swap = candidates;
                    candidates = attempted;
                    attempted = swap;
                } else {
                    for (Entry entry : attempted) {
                        entry.container = candidates;
                    }
                    candidates.addAll(attempted);
                    attempted.clear();
                }
            }
            active = !candidates.isEmpty();
        }

        private void finishRound() {
            active = false;
            if (rescanRequested) {
                rescanRequested = false;
                prepareNextRound();
            }
        }

        private boolean empty() {
            return candidates.isEmpty() && attempted.isEmpty();
        }
    }

    private record Candidate(
            Entry entry,
            WaitBucket source,
            long availabilitySequence) {
    }

    private final PlacementAvailability availability;
    private final PlacementAvailability.Listener availabilityListener =
            this::onCapacityChanged;
    private final ExecutorService executor = Executors.newSingleThreadExecutor(
            task -> {
                Thread thread = new Thread(task, "pending-placement");
                thread.setDaemon(true);
                return thread;
            });
    private final AtomicLong sequence = new AtomicLong();
    private final ReentrantLock lock = new ReentrantLock();
    private final NavigableSet<Entry> ready = new TreeSet<>(ORDER);
    private final Map<PlacementKey, WaitBucket> blocked = new HashMap<>();
    private final Set<Entry> owned = java.util.Collections.newSetFromMap(
            new IdentityHashMap<>());
    private boolean scheduled;
    private boolean closed;

    PendingPlacementCoordinator(PlacementAvailability availability) {
        this.availability = Objects.requireNonNull(
                availability, "availability");
        availability.addListener(availabilityListener);
    }

    /**
     * Retain a request after its caller already made the first placement
     * attempt. The observed sequence closes the check-then-park race without
     * manufacturing another capacity event.
     */
    Handle park(
            Work work,
            AttemptResult.Blocked reason,
            long observedAvailabilitySequence) {
        Objects.requireNonNull(reason, "reason");
        Entry entry = new Entry(
                Objects.requireNonNull(work, "work"),
                sequence.incrementAndGet());
        boolean accepted;
        lock.lock();
        try {
            accepted = !closed;
            if (!accepted) {
                entry.discarded = true;
            } else if (work.done()) {
                entry.discarded = true;
            } else {
                owned.add(entry);
                PlacementKey blocker = reason.blocker();
                if (availability.lastChangedSequence(blocker)
                        > observedAvailabilitySequence) {
                    ready.add(entry);
                    entry.container = ready;
                    scheduleUnderLock();
                } else {
                    WaitBucket bucket = blocked.computeIfAbsent(
                            blocker, ignored -> new WaitBucket());
                    if (bucket.active) {
                        bucket.addAttempted(entry);
                    } else {
                        bucket.addWaiting(entry);
                    }
                }
            }
        } finally {
            lock.unlock();
        }
        if (!accepted) {
            fail(entry, shutdownFailure());
        }
        return new Handle(entry);
    }

    long availabilitySequence() {
        return availability.sequence();
    }

    private void onCapacityChanged(
            PlacementKey key, long ignoredSequence) {
        lock.lock();
        try {
            WaitBucket bucket = blocked.get(key);
            if (bucket == null || bucket.empty()) {
                return;
            }
            bucket.activate();
            scheduleUnderLock();
        } finally {
            lock.unlock();
        }
    }

    private void runTurn() {
        for (int attempts = 0; attempts < ATTEMPTS_PER_TURN; attempts++) {
            Candidate candidate = takeNext();
            if (candidate == null) {
                break;
            }
            AttemptResult result = execute(candidate.entry());
            complete(candidate, result);
        }
        lock.lock();
        try {
            scheduled = false;
            scheduleUnderLock();
        } finally {
            lock.unlock();
        }
    }

    private Candidate takeNext() {
        lock.lock();
        try {
            Entry selected = ready.isEmpty() ? null : ready.first();
            WaitBucket source = null;
            for (WaitBucket bucket : blocked.values()) {
                Entry candidate = bucket.peek();
                if (candidate != null
                        && (selected == null
                        || ORDER.compare(candidate, selected) < 0)) {
                    selected = candidate;
                    source = bucket;
                }
            }
            if (selected == null) {
                return null;
            }
            if (source == null) {
                ready.remove(selected);
            } else {
                source.remove(selected);
            }
            selected.container = null;
            return new Candidate(
                    selected, source, availability.sequence());
        } finally {
            lock.unlock();
        }
    }

    private AttemptResult execute(Entry entry) {
        if (entry.discarded || entry.work.done()) {
            return AttemptResult.Finished.INSTANCE;
        }
        try {
            return Objects.requireNonNull(
                    entry.work.attempt(), "placement attempt result");
        } catch (Throwable failure) {
            Logger.error("Request placement attempt failed", failure);
            fail(entry, failure);
            return AttemptResult.Finished.INSTANCE;
        }
    }

    private void complete(Candidate candidate, AttemptResult result) {
        lock.lock();
        try {
            Entry entry = candidate.entry();
            if (entry.discarded || entry.work.done()
                    || result == AttemptResult.Finished.INSTANCE) {
                discardUnderLock(entry);
                finishEmptyRound(candidate.source());
                return;
            }

            AttemptResult.Blocked blockedResult =
                    (AttemptResult.Blocked) result;
            PlacementKey blocker = blockedResult.blocker();
            if (availability.lastChangedSequence(blocker)
                    > candidate.availabilitySequence()) {
                ready.add(entry);
                entry.container = ready;
                finishEmptyRound(candidate.source());
                scheduleUnderLock();
                return;
            }

            WaitBucket bucket = blocked.computeIfAbsent(
                    blocker, ignored -> new WaitBucket());
            if (bucket.active) {
                bucket.addAttempted(entry);
            } else {
                bucket.addWaiting(entry);
            }
            if (blockedResult.poolUnavailable()) {
                bucket.finishRound();
            } else if (bucket.active && bucket.candidates.isEmpty()) {
                bucket.finishRound();
            }
            finishEmptyRound(candidate.source());
            removeEmptyBuckets();
            scheduleUnderLock();
        } finally {
            lock.unlock();
        }
    }

    private void finishEmptyRound(WaitBucket bucket) {
        if (bucket != null && bucket.active
                && bucket.candidates.isEmpty()) {
            bucket.finishRound();
        }
    }

    private void scheduleUnderLock() {
        if (!closed && !scheduled && hasCandidateUnderLock()) {
            scheduled = true;
            executor.execute(this::runTurn);
        }
    }

    private boolean hasCandidateUnderLock() {
        if (!ready.isEmpty()) {
            return true;
        }
        for (WaitBucket bucket : blocked.values()) {
            if (bucket.peek() != null) {
                return true;
            }
        }
        return false;
    }

    int size() {
        lock.lock();
        try {
            return owned.size();
        } finally {
            lock.unlock();
        }
    }

    private void discard(Entry entry) {
        lock.lock();
        try {
            discardUnderLock(entry);
            removeEmptyBuckets();
        } finally {
            lock.unlock();
        }
    }

    private void discardUnderLock(Entry entry) {
        if (entry.discarded) {
            return;
        }
        entry.discarded = true;
        owned.remove(entry);
        NavigableSet<Entry> owner = entry.container;
        if (owner != null) {
            owner.remove(entry);
            entry.container = null;
        }
    }

    private void removeEmptyBuckets() {
        blocked.values().removeIf(bucket -> !bucket.active && bucket.empty());
    }

    @Override
    public void close() {
        List<Entry> abandoned;
        lock.lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            abandoned = new ArrayList<>(owned);
            for (Entry entry : abandoned) {
                entry.discarded = true;
            }
            owned.clear();
            ready.clear();
            blocked.clear();
        } finally {
            lock.unlock();
        }
        availability.removeListener(availabilityListener);
        executor.shutdownNow();
        for (Entry entry : abandoned) {
            fail(entry, shutdownFailure());
        }
    }

    private static void fail(Entry entry, Throwable failure) {
        try {
            entry.work.fail(failure);
        } catch (Throwable completionFailure) {
            Logger.warn("Failed to complete pending placement", completionFailure);
        }
    }

    private static IllegalStateException shutdownFailure() {
        return new IllegalStateException("request placement is shutting down");
    }
}

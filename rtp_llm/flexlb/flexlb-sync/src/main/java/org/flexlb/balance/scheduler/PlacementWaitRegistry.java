package org.flexlb.balance.scheduler;

import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableSet;
import java.util.Objects;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Ordered wait index for requests which own no endpoint resource.
 *
 * <p>One real capacity edge activates one resource lane. The lane retries its
 * ordered head and stops on the first same-resource miss; it never broadcasts
 * or scans a global pending queue.</p>
 */
final class PlacementWaitRegistry implements AutoCloseable {

    interface Work {
        boolean done();

        AttemptResult attempt();

        void fail(Throwable failure);
    }

    sealed interface AttemptResult {
        enum Finished implements AttemptResult {
            INSTANCE
        }

        record Blocked(PlacementKey blocker) implements AttemptResult {
            public Blocked {
                Objects.requireNonNull(blocker, "blocker");
            }
        }
    }

    /** Stable order allocated before the first placement attempt. */
    record PlacementOrder(int orderingPriority, long sequence) {
        PlacementOrder {
            if (sequence <= 0L) {
                throw new IllegalArgumentException(
                        "placement sequence must be positive");
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

    private static final Comparator<PlacementOrder> ORDER = Comparator
            .comparingInt(PlacementOrder::orderingPriority).reversed()
            .thenComparingLong(PlacementOrder::sequence);
    private static final Comparator<Entry> ENTRY_ORDER = Comparator
            .comparing((Entry entry) -> entry.order, ORDER);
    private static final int PLACEMENT_WORKERS = Math.min(
            8, Math.max(1, Runtime.getRuntime().availableProcessors() / 4));

    private static final class Entry {
        private final Work work;
        private final PlacementOrder order;
        private volatile boolean closed;
        private PlacementKey key;
        private WaitLane lane;
        /** Global availability sequence observed before the failed attempt. */
        private long blockedAtVersion;

        private Entry(Work work, PlacementOrder order) {
            this.work = work;
            this.order = order;
        }
    }

    private static final class WaitLane {
        private final NavigableSet<Entry> waiting =
                new TreeSet<>(ENTRY_ORDER);
        private Entry active;
        private boolean scheduled;

        private Entry orderedHead() {
            Entry waitingHead = waiting.isEmpty() ? null : waiting.first();
            Entry activeHead = active == null || active.closed ? null : active;
            if (activeHead == null) {
                return waitingHead;
            }
            return waitingHead == null
                    || ENTRY_ORDER.compare(activeHead, waitingHead) < 0
                    ? activeHead : waitingHead;
        }

        private boolean empty() {
            return active == null && waiting.isEmpty();
        }
    }

    private record Candidate(
            PlacementKey key,
            WaitLane lane,
            Entry entry,
            long observedAvailabilitySequence) {
    }

    private final Object monitor = new Object();
    private final PlacementAvailability availability;
    private final PlacementAvailability.Listener availabilityListener =
            this::onCapacityChanged;
    private final AtomicLong sequence = new AtomicLong();
    private final ExecutorService executor;
    private final Map<PlacementKey, WaitLane> lanes = new HashMap<>();
    private int size;
    private boolean closed;

    PlacementWaitRegistry(PlacementAvailability availability) {
        this.availability = Objects.requireNonNull(
                availability, "availability");
        AtomicInteger threadSequence = new AtomicInteger();
        this.executor = Executors.newFixedThreadPool(
                PLACEMENT_WORKERS,
                task -> {
                    Thread thread = new Thread(
                            task,
                            "placement-wait-"
                                    + threadSequence.incrementAndGet());
                    thread.setDaemon(true);
                    return thread;
                });
        availability.addListener(availabilityListener);
    }

    PlacementOrder newOrder(int priority, boolean priorityOrdering) {
        return new PlacementOrder(
                priorityOrdering ? priority : 0,
                sequence.incrementAndGet());
    }

    /** Retain one request without blocking its caller. */
    Handle park(
            Work work,
            PlacementOrder order,
            AttemptResult.Blocked reason,
            long observedAvailabilitySequence) {
        Objects.requireNonNull(reason, "reason");
        Entry entry = new Entry(
                Objects.requireNonNull(work, "work"),
                Objects.requireNonNull(order, "order"));
        boolean rejected;
        synchronized (monitor) {
            rejected = closed;
            if (rejected || work.done()) {
                entry.closed = true;
            } else {
                size++;
                waitOn(entry, reason.blocker(), observedAvailabilitySequence);
            }
        }
        if (rejected) {
            fail(entry, shutdownFailure());
        }
        return new Handle(entry);
    }

    long availabilitySequence() {
        return availability.sequence();
    }

    /** Return a resource lane whose older/higher request must go first. */
    PlacementKey blockingPredecessor(
            PlacementOrder incoming,
            PlacementKey resource) {
        Objects.requireNonNull(incoming, "incoming");
        synchronized (monitor) {
            return predecessorInDomain(incoming, resource);
        }
    }

    /** Return the first contended resource whose predecessor must go first. */
    PlacementKey blockingPredecessor(
            PlacementOrder incoming,
            PlacementKey firstResource,
            PlacementKey secondResource) {
        Objects.requireNonNull(incoming, "incoming");
        synchronized (monitor) {
            PlacementKey blocker = predecessorInDomain(
                    incoming, firstResource);
            return blocker != null ? blocker : predecessorInDomain(
                    incoming, secondResource);
        }
    }

    /** Caller holds monitor. */
    private PlacementKey predecessorInDomain(
            PlacementOrder incoming,
            PlacementKey resource) {
        PlacementKey blocker = predecessor(incoming, resource);
        return blocker != null
                ? blocker : predecessor(incoming, anyGroup(resource));
    }

    private PlacementKey predecessor(
            PlacementOrder incoming,
            PlacementKey key) {
        if (key == null) {
            return null;
        }
        WaitLane lane = lanes.get(key);
        Entry head = lane == null ? null : lane.orderedHead();
        return head != null && ORDER.compare(head.order, incoming) < 0
                ? key : null;
    }

    private static PlacementKey anyGroup(PlacementKey key) {
        return key == null || key.group() == null
                ? null : PlacementKey.anyGroup(key.role());
    }

    /** Caller holds monitor. */
    private void waitOn(
            Entry entry,
            PlacementKey blocker,
            long blockedAtVersion) {
        WaitLane lane = lanes.computeIfAbsent(
                blocker, ignored -> new WaitLane());
        entry.key = blocker;
        entry.lane = lane;
        entry.blockedAtVersion = blockedAtVersion;
        lane.waiting.add(entry);
        scheduleIfEligible(blocker, lane);
    }

    private void onCapacityChanged(
            PlacementKey key,
            long ignoredSequence) {
        synchronized (monitor) {
            WaitLane lane = lanes.get(key);
            if (lane != null) {
                scheduleIfEligible(key, lane);
            }
        }
    }

    /** Caller holds monitor. */
    private void scheduleIfEligible(PlacementKey key, WaitLane lane) {
        if (closed || lane.scheduled || lane.active != null
                || lane.waiting.isEmpty()) {
            return;
        }
        Entry head = lane.waiting.first();
        if (availability.lastChangedSequence(key)
                <= head.blockedAtVersion) {
            return;
        }
        lane.scheduled = true;
        executor.execute(() -> runLane(key, lane));
    }

    private void runLane(PlacementKey key, WaitLane lane) {
        while (true) {
            Candidate candidate = takeNext(key, lane);
            if (candidate == null) {
                return;
            }
            AttemptResult result = execute(candidate.entry());
            if (!complete(candidate, result)) {
                return;
            }
        }
    }

    private Candidate takeNext(PlacementKey key, WaitLane lane) {
        synchronized (monitor) {
            if (closed || lanes.get(key) != lane
                    || lane.active != null || lane.waiting.isEmpty()) {
                lane.scheduled = false;
                removeLaneIfEmpty(key, lane);
                return null;
            }
            // scheduleIfEligible validated the capacity edge for this lane.
            // Keep consuming that opportunity until an authoritative attempt
            // blocks: a follower may have parked behind the active head after
            // the edge and therefore has a newer blockedAtVersion of its own.
            Entry entry = lane.waiting.first();
            lane.waiting.remove(entry);
            lane.active = entry;
            return new Candidate(
                    key, lane, entry, availability.sequence());
        }
    }

    private AttemptResult execute(Entry entry) {
        if (entry.closed || entry.work.done()) {
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

    /** Continue while the same capacity edge may serve another ordered head. */
    private boolean complete(Candidate candidate, AttemptResult result) {
        synchronized (monitor) {
            Entry entry = candidate.entry();
            WaitLane source = candidate.lane();
            if (source.active != entry) {
                throw new IllegalStateException(
                        "placement lane lost its active request");
            }
            source.active = null;

            if (entry.closed || entry.work.done()
                    || result == AttemptResult.Finished.INSTANCE) {
                discardUnderLock(entry);
                return !closed;
            }

            PlacementKey blocker =
                    ((AttemptResult.Blocked) result).blocker();
            waitOn(
                    entry,
                    blocker,
                    candidate.observedAvailabilitySequence());
            if (!blocker.equals(candidate.key())) {
                return !closed;
            }
            if (availability.lastChangedSequence(blocker)
                    > candidate.observedAvailabilitySequence()) {
                return true;
            }
            source.scheduled = false;
            return false;
        }
    }

    int size() {
        synchronized (monitor) {
            return size;
        }
    }

    private void discard(Entry entry) {
        synchronized (monitor) {
            PlacementKey key = entry.key;
            WaitLane lane = entry.lane;
            discardUnderLock(entry);
            if (lane != null) {
                scheduleIfEligible(key, lane);
                removeLaneIfEmpty(key, lane);
            }
        }
    }

    /** Caller holds monitor. */
    private void discardUnderLock(Entry entry) {
        if (!entry.closed) {
            entry.closed = true;
            size--;
        }
        WaitLane lane = entry.lane;
        if (lane != null && lane.active != entry) {
            lane.waiting.remove(entry);
            entry.key = null;
            entry.lane = null;
        }
    }

    /** Caller holds monitor. */
    private void removeLaneIfEmpty(PlacementKey key, WaitLane lane) {
        if (!lane.scheduled && lane.empty()) {
            lanes.remove(key, lane);
        }
    }

    @Override
    public void close() {
        List<Entry> abandoned = new ArrayList<>();
        synchronized (monitor) {
            if (closed) {
                return;
            }
            closed = true;
            for (WaitLane lane : lanes.values()) {
                if (lane.active != null) {
                    abandoned.add(lane.active);
                }
                abandoned.addAll(lane.waiting);
            }
            abandoned.forEach(entry -> entry.closed = true);
            size = 0;
            lanes.clear();
        }
        availability.removeListener(availabilityListener);
        executor.shutdownNow();
        IllegalStateException failure = shutdownFailure();
        abandoned.forEach(entry -> fail(entry, failure));
    }

    private static void fail(Entry entry, Throwable failure) {
        try {
            entry.work.fail(failure);
        } catch (Throwable completionFailure) {
            Logger.warn(
                    "Failed to complete pending placement",
                    completionFailure);
        }
    }

    private static IllegalStateException shutdownFailure() {
        return new IllegalStateException(
                "request placement is shutting down");
    }
}

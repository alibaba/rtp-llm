package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.strategy.EncoderSelection;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;

/**
 * Orders Encoder decisions separately from Generation and waits for Encoder capacity changes.
 */
final class EncoderQueueCoordinator implements AutoCloseable {
    private record Entry(RequestContext context, long sequence) {}

    private final AbstractRequestScheduler owner;
    private final Function<RequestContext, PlacementResult<EncoderSelection, PlacementKey>>
            selectEncoder;
    private final PlacementAvailability availability;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition changed = lock.newCondition();
    private final PriorityQueue<Entry> ready;
    private final Set<Entry> blocked = Collections.newSetFromMap(new IdentityHashMap<>());
    private final IdentityHashMap<RequestContext, Entry> entries = new IdentityHashMap<>();
    private final PlacementAvailability.Listener capacityListener =
            key -> {
                if (key.role() == RoleType.ENCODER) {
                    capacityChanged();
                }
            };
    private final Thread decisionThread;
    private boolean closed;
    private long nextSequence;
    private long capacityVersion;

    EncoderQueueCoordinator(
            AbstractRequestScheduler owner,
            Function<RequestContext, PlacementResult<EncoderSelection, PlacementKey>> selectEncoder,
            PlacementAvailability availability,
            boolean priorityOrdering) {
        this.owner = owner;
        this.selectEncoder = selectEncoder;
        this.availability = availability;
        Comparator<Entry> order = Comparator.comparingLong(Entry::sequence);
        if (priorityOrdering) {
            order =
                    Comparator.<Entry>comparingInt(entry -> entry.context().data().getPriority())
                            .reversed()
                            .thenComparing(order);
        }
        ready = new PriorityQueue<>(order);
        decisionThread =
                Thread.ofPlatform()
                        .daemon()
                        .name("flexlb-encoder-decision")
                        .unstarted(this::runDecisions);
        availability.addListener(capacityListener);
        decisionThread.start();
    }

    boolean offer(RequestContext context) {
        lock.lock();
        try {
            if (closed || context.data().getFuture().isDone()) {
                return false;
            }
            Entry entry = new Entry(context, ++nextSequence);
            entries.put(context, entry);
            ready.add(entry);
            context.data().getFuture().whenComplete((ignored, failure) -> remove(context));
            changed.signal();
            return true;
        } finally {
            lock.unlock();
        }
    }

    void remove(RequestContext context) {
        lock.lock();
        try {
            Entry entry = entries.remove(context);
            if (entry != null) {
                ready.remove(entry);
                blocked.remove(entry);
            }
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    int size() {
        lock.lock();
        try {
            return entries.size();
        } finally {
            lock.unlock();
        }
    }

    int blockedSize() {
        lock.lock();
        try {
            return blocked.size();
        } finally {
            lock.unlock();
        }
    }

    private void capacityChanged() {
        lock.lock();
        try {
            capacityVersion++;
            ready.addAll(blocked);
            blocked.clear();
            changed.signal();
        } finally {
            lock.unlock();
        }
    }

    private void runDecisions() {
        try {
            while (true) {
                Entry entry;
                long observedCapacity;
                lock.lock();
                try {
                    while (!closed && ready.isEmpty()) {
                        changed.await();
                    }
                    if (closed) {
                        return;
                    }
                    entry = ready.remove();
                    observedCapacity = capacityVersion;
                } finally {
                    lock.unlock();
                }
                if (entry.context().data().requestExpired(System.currentTimeMillis())) {
                    owner.cancel(entry.context(), 0L, CancelReason.DEADLINE_EXCEEDED);
                    remove(entry.context());
                    continue;
                }
                if (owner.decideEncoder(entry.context(), selectEncoder)) {
                    remove(entry.context());
                    continue;
                }
                lock.lock();
                try {
                    if (!closed
                            && entries.get(entry.context()) == entry
                            && !entry.context().data().getFuture().isDone()) {
                        if (capacityVersion == observedCapacity) {
                            blocked.add(entry);
                        } else {
                            ready.add(entry);
                            changed.signal();
                        }
                    }
                } finally {
                    lock.unlock();
                }
            }
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
        } catch (Throwable failure) {
            Logger.error("Encoder queue decision thread failed", failure);
        } finally {
            close();
        }
    }

    @Override
    public void close() {
        ArrayList<RequestContext> abandoned;
        lock.lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            abandoned = new ArrayList<>(entries.keySet());
            entries.clear();
            ready.clear();
            blocked.clear();
            changed.signalAll();
        } finally {
            lock.unlock();
        }
        availability.removeListener(capacityListener);
        for (RequestContext context : abandoned) {
            owner.publish(
                    context,
                    Response.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED,
                            "request scheduler is shutting down"),
                    false);
        }
        if (Thread.currentThread() != decisionThread) {
            try {
                decisionThread.join();
            } catch (InterruptedException interruption) {
                Thread.currentThread().interrupt();
            }
        }
    }
}

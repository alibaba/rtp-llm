package org.flexlb.balance.schedule;

import org.flexlb.balance.queue.GlobalQueueEntry;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static com.google.common.base.Preconditions.checkState;

/**
 * Registers queued requests and tracks their unfinished decisions and preemption results under the scheduler lock.
 * A request has one work record even when both tasks overlap; removing its registration retains unfinished work
 * until callbacks return, without counting that request as waiting or affecting a replacement registration.
 */
final class SchedulingRequests {
    private final Map<RequestContext, GlobalQueueEntry> registered = new IdentityHashMap<>();
    private final Map<GlobalQueueEntry, Work> work = new IdentityHashMap<>();

    private int decisions;
    private int preemptions;

    GlobalQueueEntry find(RequestContext context) {
        return registered.get(context);
    }

    void register(GlobalQueueEntry entry) {
        checkState(
                registered.putIfAbsent(entry.context(), entry) == null,
                "duplicate global queue identity");
    }

    void remove(GlobalQueueEntry entry) {
        registered.remove(entry.context(), entry);
    }

    int waitingRequestCount() {
        int waiting = registered.size();
        for (var entry : work.keySet()) {
            if (registered.get(entry.context()) == entry) waiting--;
        }
        return waiting;
    }

    int decisionCount() {
        return decisions;
    }

    int preemptionCount() {
        return preemptions;
    }

    boolean hasWork() {
        return !work.isEmpty();
    }

    boolean isBusy(GlobalQueueEntry entry) {
        return work.containsKey(entry);
    }

    boolean isDeciding(GlobalQueueEntry entry) {
        var current = work.get(entry);
        return current != null && current.deciding;
    }

    void startDecisions(List<GlobalQueueEntry> entries) {
        for (var entry : entries) {
            checkState(
                    registered.get(entry.context()) == entry && !work.containsKey(entry),
                    "decision requires a registered idle request");
            var started = new Work();
            started.deciding = true;
            work.put(entry, started);
            decisions++;
        }
    }

    boolean tryStartPreemption(GlobalQueueEntry entry, int limit) {
        if (registered.get(entry.context()) != entry || preemptions >= limit) return false;
        var completion = new CompletableFuture<Void>();
        var current = work.computeIfAbsent(entry, ignored -> new Work());
        checkState(current.preemption == null, "request already has pending preemption");
        current.preemption = completion;
        preemptions++;
        return true;
    }

    void finishDecision(GlobalQueueEntry entry) {
        var current = work.get(entry);
        if (current == null || !current.deciding) return;
        current.deciding = false;
        decisions--;
        if (current.preemption == null) work.remove(entry);
    }

    CompletableFuture<Void> finishPreemption(GlobalQueueEntry entry) {
        var current = work.get(entry);
        if (current == null || current.preemption == null) return null;
        var completion = current.preemption;
        current.preemption = null;
        preemptions--;
        if (!current.deciding) work.remove(entry);
        return completion;
    }

    List<CompletableFuture<Void>> pendingPreemptions() {
        var pending = new ArrayList<CompletableFuture<Void>>(preemptions);
        for (var current : work.values()) {
            if (current.preemption != null) pending.add(current.preemption);
        }
        return pending;
    }

    List<GlobalQueueEntry> drainRegistered() {
        var entries = List.copyOf(registered.values());
        registered.clear();
        return entries;
    }

    /**
     * Decision execution and an asynchronous preemption result may overlap until the planner returns.
     */
    private static final class Work {
        private boolean deciding;
        private CompletableFuture<Void> preemption;
    }
}

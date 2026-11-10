package org.flexlb.balance.scheduler;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Independent test implementation: no intrusive links or default-queue delegation.
 */
final class ListSchedulingQueue implements SchedulingQueue {
    private final List<GlobalQueueEntry> entries = new ArrayList<>();
    private final Set<GlobalQueueEntry> pending = new HashSet<>();

    private long sequence;

    @Override
    public void add(GlobalQueueEntry entry) {
        entry.markQueued(++sequence);
        entries.add(entry);
        pending.add(entry);
    }

    @Override
    public void restore(GlobalQueueEntry entry) {
        entry.markQueued(entry.sequence());
        entries.add(entry);
        for (var candidate : entries) {
            if (candidate.sequence() >= entry.sequence()) pending.add(candidate);
        }
    }

    @Override
    public List<GlobalQueueEntry> scanForPlanningCandidates(
            int limit, int budget, Predicate<GlobalQueueEntry> eligible) {
        var ordered = new ArrayList<>(pending);
        ordered.sort(order());
        var result = new ArrayList<GlobalQueueEntry>();
        int examined = 0;
        for (var entry : ordered) {
            if (result.size() >= limit || examined++ >= budget) break;
            pending.remove(entry);
            if (eligible.test(entry)) result.add(entry);
        }
        return result;
    }

    @Override
    public Comparator<GlobalQueueEntry> order() {
        return Comparator.comparingLong(GlobalQueueEntry::sequence);
    }

    @Override
    public void markRequestReadyForRetry(GlobalQueueEntry entry) {
        if (entries.contains(entry)) pending.add(entry);
    }

    @Override
    public boolean hasUnscannedRequests() {
        return !pending.isEmpty();
    }

    @Override
    public int size() {
        return entries.size();
    }

    @Override
    public int[] priorityCounts() {
        int[] counts = new int[org.flexlb.util.PriorityNormalizer.MAX_PRIORITY + 1];
        for (var entry : entries) counts[entry.priority()]++;
        return counts;
    }

    @Override
    public List<GlobalQueueEntry> drain() {
        var result = new ArrayList<>(entries);
        for (var entry : result) remove(entry);
        return result;
    }

    @Override
    public boolean remove(GlobalQueueEntry entry) {
        if (!entries.remove(entry)) return false;
        pending.remove(entry);
        entry.markRemoved();
        return true;
    }
}

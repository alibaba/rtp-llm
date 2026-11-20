package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Independent test implementation: no intrusive links or default-queue delegation.
 */
public final class ListSchedulingQueue implements SchedulingQueue {
    private final List<RequestContext> entries = new ArrayList<>();
    private final Set<RequestContext> pending = new HashSet<>();

    private long sequence;

    @Override
    public boolean contains(RequestContext request) {
        return entries.contains(request);
    }

    @Override
    public void add(RequestContext entry) {
        entry.setPlacementSequence(++sequence);
        entries.add(entry);
        pending.add(entry);
    }

    @Override
    public void restore(RequestContext entry) {
        entries.add(entry);
        for (var candidate : entries) {
            if (candidate.getPlacementSequence() >= entry.getPlacementSequence())
                pending.add(candidate);
        }
    }

    @Override
    public List<RequestContext> scanForPlanningCandidates(
            int limit, int budget, Predicate<RequestContext> eligible) {
        var ordered = new ArrayList<>(pending);
        ordered.sort(order());
        var result = new ArrayList<RequestContext>();
        int examined = 0;
        for (var entry : ordered) {
            if (result.size() >= limit || examined++ >= budget) break;
            pending.remove(entry);
            if (eligible.test(entry)) result.add(entry);
        }
        return result;
    }

    @Override
    public Comparator<RequestContext> order() {
        return Comparator.comparingLong(RequestContext::getPlacementSequence);
    }

    @Override
    public void beginScan() {
        pending.addAll(entries);
    }

    @Override
    public void retry(RequestContext request) {
        if (entries.contains(request)) pending.add(request);
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
        for (var entry : entries) counts[SchedulingQueue.priority(entry)]++;
        return counts;
    }

    @Override
    public List<RequestContext> drain() {
        var result = new ArrayList<>(entries);
        for (var entry : result) remove(entry);
        return result;
    }

    @Override
    public boolean remove(RequestContext entry) {
        if (!entries.remove(entry)) return false;
        pending.remove(entry);
        return true;
    }
}

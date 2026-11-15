package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.util.PriorityNormalizer;

import java.util.Comparator;
import java.util.List;
import java.util.function.Predicate;

/**
 * Holds requests awaiting scheduling and their fixed ordering. The caller owns synchronization.
 * Scans count every examined entry against the budget; eligibility may remove an entry
 * but cannot restore a selected entry during the scan. Removal never settles a request.
 * Restore preserves its sequence and reopens the ordered scan from that position.
 * Membership is keyed by request identity; queue nodes are private implementation details.
 */
public interface SchedulingQueue {
    void add(RequestContext entry);

    boolean contains(RequestContext request);

    static int priority(RequestContext request) {
        int priority = request.getPriority();
        return PriorityNormalizer.isValid(priority)
                ? priority
                : PriorityNormalizer.DEFAULT_PRIORITY;
    }

    void restore(RequestContext entry);

    boolean remove(RequestContext entry);

    List<RequestContext> scanForPlanningCandidates(
            int candidateLimit, int scanBudget, Predicate<RequestContext> eligible);

    void beginScan();

    void retry(RequestContext request);

    boolean hasUnscannedRequests();

    int size();

    int[] priorityCounts();

    List<RequestContext> drain();

    Comparator<RequestContext> order();
}

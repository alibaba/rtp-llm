package org.flexlb.balance.queue;

import java.util.Comparator;
import java.util.List;
import java.util.function.Predicate;

/**
 * Holds requests awaiting scheduling and their fixed ordering. The caller owns synchronization.
 * Scans count every examined entry against the budget; eligibility may remove an entry
 * but cannot restore a selected entry during the scan. Removal never settles a request.
 * Restore preserves its sequence and reopens the ordered scan from that position.
 * Implementations must maintain exact entry membership
 * and the entry's removed flag, including during drain.
 */
public interface SchedulingQueue {
    void add(GlobalQueueEntry entry);

    void restore(GlobalQueueEntry entry);

    boolean remove(GlobalQueueEntry entry);

    List<GlobalQueueEntry> scanForPlanningCandidates(
            int candidateLimit, int scanBudget, Predicate<GlobalQueueEntry> eligible);

    void markRequestReadyForRetry(GlobalQueueEntry entry);

    boolean hasUnscannedRequests();

    int size();

    int[] priorityCounts();

    List<GlobalQueueEntry> drain();

    Comparator<GlobalQueueEntry> order();
}

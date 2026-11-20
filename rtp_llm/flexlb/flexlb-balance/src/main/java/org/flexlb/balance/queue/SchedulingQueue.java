package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.util.PriorityNormalizer;

import java.util.Comparator;
import java.util.List;
import java.util.function.Predicate;

/**
 * Holds requests awaiting scheduling and their fixed ordering. The caller owns synchronization.
 * FIFO and priority scans count each entry against the budget. Cost ordering ranks the full
 * eligible snapshot. Eligibility may remove an entry
 * but cannot restore a selected entry during the scan. Removal never settles a request.
 * Restore preserves the original sequence.
 * Membership is keyed by request identity; queue nodes are private implementation details.
 */
public interface SchedulingQueue {
    /** Claims refresh work under the scheduler lock. Complete it under the same lock. */
    default List<EstimateRequest> estimates(int limit) {
        return List.of();
    }

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

    /**
     * One exact refresh attempt; a later ticket for the same request is a different operation.
     */
    final class EstimateRequest {
        private final RequestContext request;
        private java.util.function.Consumer<java.util.OptionalLong> completion;

        public EstimateRequest(
                RequestContext request,
                java.util.function.Consumer<java.util.OptionalLong> completion) {
            this.request = request;
            this.completion = completion;
        }

        public RequestContext request() {
            return request;
        }

        public void complete(java.util.OptionalLong estimate) {
            if (completion == null) return;
            var action = completion;
            completion = null;
            action.accept(estimate);
        }
    }
}

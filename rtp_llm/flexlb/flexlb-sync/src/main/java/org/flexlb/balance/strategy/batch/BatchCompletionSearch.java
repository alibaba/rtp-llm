package org.flexlb.balance.strategy.batch;

import org.springframework.lang.NonNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Bounded depth-first repair for an incomplete greedy global Prefill plan.
 *
 * <p>Greedy placement is deliberately fast but can strand a request after an
 * early choice consumes a shared worker's KV or delivery-credit capacity. This
 * search revisits the entire assignment space without reserving any endpoint
 * state. It returns a candidate-index vector in original request order; the
 * caller materializes the selected placements only after planning completes.</p>
 *
 * <h2>Objective</h2>
 *
 * <p>The objective is lexicographic: first maximize the number of placed
 * requests, then minimize their aggregate virtual TTFT. Virtual TTFT uses the
 * same input-order, per-worker preceding-Prefill-work model as the rest of the
 * global planner. Thus a lower-cost tie breaker never trades away a request
 * that another feasible plan can place.</p>
 *
 * <h2>Traversal and pruning</h2>
 *
 * <p>Requests are visited in constrained-first order: fewer candidate workers,
 * then larger TTFT regret, then original index. At each depth the search tries
 * every candidate that satisfies the shared KV-token and delivery-credit
 * constraints, followed by an unplaced branch. It abandons a branch when even
 * placing every remaining request cannot exceed the best placed count found so
 * far. Trying placements before the unplaced branch usually raises that bound
 * early and makes the prune effective.</p>
 *
 * <h2>Evaluation budget</h2>
 *
 * <p>{@code globalDecision.maxPlanEvaluations} limits the work: each candidate
 * branch consumes one evaluation and each reached terminal assignment consumes
 * one more. Once exhausted, the current best plan is returned. The budget is a
 * latency guard, so this class is a deterministic bounded heuristic rather
 * than an exact bin-packing solver. It is invoked only after greedy placement
 * leaves at least one request unplaced; a complete greedy plan bypasses this
 * search entirely.</p>
 */
final class BatchCompletionSearch {

    /**
     * Immutable result of one bounded search, including the work consumed by
     * its evaluation budget. The plan accessor returns a defensive copy.
     */
    record SearchResult(int[] plan, int evaluations, boolean budgetExhausted) {
        SearchResult {
            plan = plan.clone();
        }

        @Override
        public int[] plan() {
            return plan.clone();
        }
    }

    private final List<BatchPlanningRequest> requests;
    private final List<Integer> order;
    private final int maxEvaluations;
    private int[] best;
    private int bestPlaced;
    private long bestCost;
    private int evaluations;
    private boolean budgetExhausted;

    private BatchCompletionSearch(List<BatchPlanningRequest> requests,
                                  int[] greedy,
                                  int maxEvaluations) {
        this.requests = List.copyOf(requests);
        this.order = constrainedFirstOrder(requests);
        this.maxEvaluations = maxEvaluations;
        this.best = greedy.clone();
        this.bestPlaced = placedCount(greedy);
        this.bestCost = BatchPlanner.total(BatchPlanner.ttfts(requests, greedy));
    }

    /**
     * Returns only the plan from {@link #search(List, int[], int)} for the planner.
     */
    static int[] complete(List<BatchPlanningRequest> requests,
                          int[] greedy,
                          int maxEvaluations) {
        return search(requests, greedy, maxEvaluations).plan();
    }

    /**
     * Searches for a better completion of a greedy candidate-index vector.
     *
     * <p>The input vector uses {@code -1} for an unplaced request and a
     * request-local candidate index otherwise. The input is never mutated. If
     * the budget is exhausted before an improvement is found, this returns a
     * value-equivalent copy of the greedy plan. The result's evaluation count
     * is the exact budget consumption, including candidate branches and
     * terminal assignments.</p>
     *
     * @param requests       candidate snapshots in original request order
     * @param greedy         feasible greedy plan to improve
     * @param maxEvaluations positive cap on candidate branches and terminal
     *                       assignment evaluations
     * @return the best plan and its evaluation count
     * @throws IllegalArgumentException when vector lengths differ or the
     *                                  evaluation budget is not positive
     */
    static SearchResult search(@NonNull List<BatchPlanningRequest> requests,
                               @NonNull int[] greedy,
                               int maxEvaluations) {
        if (requests.size() != greedy.length) {
            throw new IllegalArgumentException("requests and greedy must have equal size");
        }
        if (maxEvaluations <= 0) {
            throw new IllegalArgumentException("maxEvaluations must be positive");
        }
        BatchCompletionSearch search = new BatchCompletionSearch(requests, greedy, maxEvaluations);
        int[] trial = new int[requests.size()];
        Arrays.fill(trial, -1);
        search.search(0, trial, 0);
        return new SearchResult(search.best, search.evaluations, search.budgetExhausted);
    }

    /**
     * Explores one constrained-first prefix of the plan.
     *
     * <p>Candidate branches are intentionally visited before the unplaced
     * branch, so useful placements establish a stronger placement-count bound
     * for later pruning.</p>
     */
    private void search(int depth, int[] trial, int placed) {
        if (evaluations >= maxEvaluations) {
            budgetExhausted = true;
            return;
        }
        if (placed + order.size() - depth < bestPlaced) {
            return;
        }
        if (depth == order.size()) {
            evaluations++;
            long cost = BatchPlanner.total(BatchPlanner.ttfts(requests, trial));
            if (placed > bestPlaced || placed == bestPlaced && cost < bestCost) {
                best = trial.clone();
                bestPlaced = placed;
                bestCost = cost;
            }
            return;
        }
        int requestIndex = order.get(depth);
        BatchPlanningRequest request = requests.get(requestIndex);
        for (int candidate = 0;
             candidate < request.candidates().size() && evaluations < maxEvaluations;
             candidate++) {
            trial[requestIndex] = candidate;
            evaluations++;
            if (BatchPlanner.fits(requests, trial)) {
                search(depth + 1, trial, placed + 1);
            }
        }
        trial[requestIndex] = -1;
        search(depth + 1, trial, placed);
    }

    /**
     * Orders requests to expose infeasible prefixes early without changing the
     * input-order TTFT model used to score complete plans.
     */
    private static List<Integer> constrainedFirstOrder(List<BatchPlanningRequest> requests) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < requests.size(); i++) {
            result.add(i);
        }
        result.sort(Comparator
                .<Integer>comparingInt(index -> requests.get(index).candidates().size())
                .thenComparing(Comparator
                        .<Integer>comparingLong(index -> BatchPlanner.regret(requests.get(index)))
                        .reversed())
                .thenComparingInt(Integer::intValue));
        return List.copyOf(result);
    }

    private static int placedCount(int[] selected) {
        int placed = 0;
        for (int candidate : selected) {
            if (candidate >= 0) {
                placed++;
            }
        }
        return placed;
    }
}

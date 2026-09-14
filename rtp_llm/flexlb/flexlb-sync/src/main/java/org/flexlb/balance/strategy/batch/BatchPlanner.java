package org.flexlb.balance.strategy.batch;

import org.flexlb.config.GlobalDecisionConfig;
import org.springframework.lang.NonNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Produces a feasible candidate-index plan for one global Prefill decision tier.
 *
 * <p>The interface accepts immutable request snapshots and returns one
 * request-local candidate index per input request; {@code -1} means that no
 * feasible placement was found. It performs no endpoint discovery, telemetry,
 * reservation, or materialization, so callers can safely plan against an
 * observed capacity snapshot before committing ordinary route selections.</p>
 *
 * <h2>Planning sequence</h2>
 *
 * <ol>
 *   <li>A regret-ordered greedy pass builds a low-latency feasible plan. A
 *       request whose best and second-best TTFT differ most is handled first.</li>
 *   <li>If greedy leaves an entry unplaced, {@link BatchCompletionSearch}
 *       searches for a completion that first maximizes placed requests and
 *       then minimizes aggregate virtual TTFT.</li>
 *   <li>A local-improvement pass lowers aggregate virtual TTFT. When cache
 *       affinity is enabled, a second pass improves cache-hit tokens while
 *       preserving each request's TTFT allowance from that first-pass plan.</li>
 * </ol>
 *
 * <p>{@code maxPlanEvaluations} bounds completion search and each local
 * improvement invocation independently. The planner is consequently a
 * deterministic bounded heuristic, not an exact global optimizer.</p>
 */
public final class BatchPlanner {

    private BatchPlanner() {
    }

    /**
     * Produces a plan with the default global completion and improvement budget.
     */
    public static List<Integer> plan(@NonNull List<BatchPlanningRequest> requests) {
        return planWithTrace(requests, GlobalDecisionConfig.DEFAULT_MAX_PLAN_EVALUATIONS)
                .selections();
    }

    /**
     * Produces one feasible candidate-index vector for a collected tier.
     *
     * @param requests           candidate snapshots in original request order
     * @param maxPlanEvaluations positive limit for completion search and each
     *                           local-improvement phase
     * @return one candidate index or {@code -1} for every input request
     */
    public static List<Integer> plan(@NonNull List<BatchPlanningRequest> requests,
                                     int maxPlanEvaluations) {
        return planWithTrace(requests, maxPlanEvaluations).selections();
    }

    /**
     * Produces the final assignments and a phase-by-phase trace of changes from greedy.
     *
     * <p>The trace is intentionally derived from the same candidate snapshots and bounded
     * operations as the returned assignments. It therefore reports planner work without
     * introducing extra selection or reservation side effects.</p>
     */
    public static BatchPlan planWithTrace(@NonNull List<BatchPlanningRequest> requests,
                                          int maxPlanEvaluations) {
        int[] selected = new int[requests.size()];
        Arrays.fill(selected, -1);
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < requests.size(); i++) {
            order.add(i);
        }
        order.sort(Comparator.<Integer>comparingLong(i -> regret(requests.get(i))).reversed());
        for (int index : order) {
            long best = Long.MAX_VALUE;
            int choice = -1;
            for (int candidate = 0; candidate < requests.get(index).candidates().size(); candidate++) {
                selected[index] = candidate;
                if (!fits(requests, selected)) {
                    continue;
                }
                long cost = total(ttfts(requests, selected));
                if (choice < 0 || cost < best) {
                    best = cost;
                    choice = candidate;
                }
            }
            selected[index] = choice;
        }
        int[] greedy = selected.clone();
        List<EnumSet<BatchPlan.RequestChange>> requestChanges = requestChanges(requests.size());
        BatchPlan.CompletionSearch completion = new BatchPlan.CompletionSearch(false, 0, false, 0, 0);
        if (Arrays.stream(selected).anyMatch(candidate -> candidate < 0)) {
            BatchCompletionSearch.SearchResult result = BatchCompletionSearch.search(requests,
                    selected, maxPlanEvaluations);
            selected = result.plan();
            int recovered = 0;
            int reassigned = 0;
            for (int i = 0; i < selected.length; i++) {
                if (greedy[i] < 0 && selected[i] >= 0) {
                    requestChanges.get(i).add(BatchPlan.RequestChange.COMPLETION_REPAIR);
                    recovered++;
                } else if (greedy[i] >= 0 && selected[i] >= 0 && greedy[i] != selected[i]) {
                    requestChanges.get(i).add(BatchPlan.RequestChange.COMPLETION_REASSIGNMENT);
                    reassigned++;
                }
            }
            completion = new BatchPlan.CompletionSearch(true,
                    result.evaluations(), result.budgetExhausted(), recovered, reassigned);
        }
        BatchPlan.Optimization ttft = improve(requests,
                selected, null, maxPlanEvaluations, requestChanges,
                BatchPlan.RequestChange.TTFT_MOVE, BatchPlan.RequestChange.TTFT_SWAP);
        BatchPlan.Optimization cacheAffinity = BatchPlan.Optimization.notInvoked();
        if (requests.stream().anyMatch(BatchPlanningRequest::affinity)) {
            cacheAffinity = improve(requests,
                    selected, selected.clone(), maxPlanEvaluations, requestChanges,
                    BatchPlan.RequestChange.CACHE_AFFINITY_MOVE,
                    BatchPlan.RequestChange.CACHE_AFFINITY_SWAP);
        }
        return new BatchPlan(Arrays.stream(selected).boxed().toList(),
                placedCount(greedy),
                placedCount(selected),
                changedRequestCount(greedy, selected),
                completion,
                ttft,
                cacheAffinity,
                requestChanges.stream().map(changes -> List.copyOf(changes)).toList());
    }

    /**
     * Performs up to two passes of one-request moves and pairwise worker swaps.
     *
     * <p>With a {@code null} baseline, a move is accepted only when it lowers
     * aggregate virtual TTFT. With a baseline, a move must preserve every
     * affected request's TTFT allowance; it then prefers additional cache-hit
     * tokens and uses aggregate virtual TTFT to break a cache-hit tie. Each
     * invocation receives a separate bounded evaluation budget.</p>
     */
    private static BatchPlan.Optimization improve(List<BatchPlanningRequest> requests,
            int[] selected,
            int[] baseline,
            int maxPlanEvaluations,
            List<EnumSet<BatchPlan.RequestChange>> requestChanges,
            BatchPlan.RequestChange moveChange,
            BatchPlan.RequestChange swapChange) {
        int[] before = selected.clone();
        int evaluations = 0;
        int moves = 0;
        int swaps = 0;
        for (int pass = 0; pass < 2; pass++) {
            boolean changed = false;
            for (int i = 0; i < selected.length; i++) {
                if (selected[i] < 0) {
                    continue;
                }
                for (int c = 0; c < requests.get(i).candidates().size(); c++) {
                    if (evaluations >= maxPlanEvaluations) {
                        return optimization(requests, before, selected, true, evaluations, moves, swaps);
                    }
                    evaluations++;
                    int[] trial = selected.clone();
                    trial[i] = c;
                    if (better(requests, trial, selected, baseline)) {
                        System.arraycopy(trial, 0, selected, 0, selected.length);
                        changed = true;
                        moves++;
                        requestChanges.get(i).add(moveChange);
                    }
                }
                for (int j = i + 1; j < selected.length; j++) {
                    if (selected[j] < 0) {
                        continue;
                    }
                    if (evaluations >= maxPlanEvaluations) {
                        return optimization(requests, before, selected, true, evaluations, moves, swaps);
                    }
                    evaluations++;
                    int[] trial = selected.clone();
                    trial[i] = onWorker(requests.get(i), requests.get(j).candidates()
                            .get(selected[j]).worker());
                    trial[j] = onWorker(requests.get(j), requests.get(i).candidates()
                            .get(selected[i]).worker());
                    if (trial[i] >= 0 && trial[j] >= 0 && better(requests, trial, selected, baseline)) {
                        System.arraycopy(trial, 0, selected, 0, selected.length);
                        changed = true;
                        swaps++;
                        requestChanges.get(i).add(swapChange);
                        requestChanges.get(j).add(swapChange);
                    }
                }
            }
            if (!changed) {
                return optimization(requests, before, selected, false, evaluations, moves, swaps);
            }
        }
        return optimization(requests, before, selected, false, evaluations, moves, swaps);
    }

    private static BatchPlan.Optimization optimization(List<BatchPlanningRequest> requests,
                                                        int[] before,
                                                        int[] after,
                                                        boolean budgetExhausted,
                                                        int evaluations,
                                                        int moves,
                                                        int swaps) {
        return new BatchPlan.Optimization(true,
                evaluations,
                budgetExhausted,
                moves,
                swaps,
                changedRequestCount(before, after),
                signedDelta(total(ttfts(requests, before)), total(ttfts(requests, after))),
                signedDelta(hits(requests, after), hits(requests, before)));
    }

    private static List<EnumSet<BatchPlan.RequestChange>> requestChanges(int size) {
        List<EnumSet<BatchPlan.RequestChange>> result = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            result.add(EnumSet.noneOf(BatchPlan.RequestChange.class));
        }
        return result;
    }

    private static int placedCount(int[] selected) {
        int count = 0;
        for (int candidate : selected) {
            if (candidate >= 0) {
                count++;
            }
        }
        return count;
    }

    private static int changedRequestCount(int[] before, int[] after) {
        int changed = 0;
        for (int i = 0; i < before.length; i++) {
            if (before[i] != after[i]) {
                changed++;
            }
        }
        return changed;
    }

    private static long signedDelta(long later, long earlier) {
        return later - earlier;
    }

    private static int onWorker(BatchPlanningRequest request, String worker) {
        for (int i = 0; i < request.candidates().size(); i++) {
            if (request.candidates().get(i).worker().equals(worker)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Tests whether a feasible local move improves the active optimization phase.
     *
     * <p>TTFT optimization has no baseline and accepts only a lower aggregate
     * virtual TTFT. Cache-affinity optimization compares against the preceding
     * TTFT plan: it rejects non-affinity moves and any TTFT regression beyond
     * the per-request allowance before comparing cache hits.</p>
     */
    private static boolean better(List<BatchPlanningRequest> requests,
                                  int[] trial,
                                  int[] current,
                                  int[] baseline) {
        if (!fits(requests, trial)) {
            return false;
        }
        long[] times = ttfts(requests, trial);
        if (baseline == null) {
            return total(times) < total(ttfts(requests, current));
        }
        long[] baselineTimes = ttfts(requests, baseline);
        for (int i = 0; i < trial.length; i++) {
            if (trial[i] < 0) {
                continue;
            }
            BatchPlanningRequest request = requests.get(i);
            long allowance = request.affinity() ? Math.max(0, request.maxExtraTtftMs()) : 0;
            if (times[i] > add(baselineTimes[i], allowance)) {
                return false;
            }
            if (trial[i] != baseline[i] && (!request.affinity()
                    || !cacheEligible(request, request.candidates().get(trial[i]),
                    request.candidates().get(baseline[i])))) {
                return false;
            }
        }
        long trialHits = hits(requests, trial);
        long currentHits = hits(requests, current);
        return trialHits > currentHits || trialHits == currentHits
                && total(times) < total(ttfts(requests, current));
    }

    private static long hits(List<BatchPlanningRequest> requests, int[] selected) {
        long result = 0;
        for (int i = 0; i < selected.length; i++) {
            if (selected[i] >= 0) {
                result = add(result, requests.get(i).candidates().get(selected[i]).hitTokens());
            }
        }
        return result;
    }

    /**
     * Checks the hard aggregate capacity constraints of a candidate-index plan.
     *
     * <p>For each worker, the sum of selected requests' KV tokens must not
     * exceed its reported available KV tokens, and the number of selected
     * requests must not exceed its delivery credits. This is shared with
     * completion search so both phases apply identical constraints.</p>
     */
    static boolean fits(List<BatchPlanningRequest> requests, int[] selected) {
        Map<String, Long> usedKv = new HashMap<>();
        Map<String, Integer> usedRequests = new HashMap<>();
        for (int i = 0; i < selected.length; i++) {
            if (selected[i] >= 0) {
                BatchCandidate candidate = requests.get(i).candidates().get(selected[i]);
                usedKv.merge(candidate.worker(), candidate.kvTokens(), BatchPlanner::add);
                usedRequests.merge(candidate.worker(), 1, Integer::sum);
            }
        }
        for (int i = 0; i < selected.length; i++) {
            if (selected[i] >= 0) {
                BatchCandidate candidate = requests.get(i).candidates().get(selected[i]);
                if (usedKv.get(candidate.worker()) > candidate.availableKvTokens()
                        || usedRequests.get(candidate.worker()) > candidate.availableRequests()) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Returns the opportunity cost of not taking a request's best TTFT candidate.
     */
    static long regret(BatchPlanningRequest request) {
        long first = Long.MAX_VALUE;
        long second = Long.MAX_VALUE;
        for (BatchCandidate candidate : request.candidates()) {
            if (candidate.ttftMs() < first) {
                second = first;
                first = candidate.ttftMs();
            } else if (candidate.ttftMs() < second) {
                second = candidate.ttftMs();
            }
        }
        return first == Long.MAX_VALUE ? 0 : second - first;
    }

    /**
     * Calculates virtual TTFT for an input-order plan.
     *
     * <p>Each worker begins with zero virtual queued work. For every selected
     * request in original input order, its TTFT is the candidate's projected
     * standalone TTFT plus that worker's preceding selected Prefill work. An
     * unplaced entry remains zero and contributes no work.</p>
     */
    static long[] ttfts(List<BatchPlanningRequest> requests, int[] selected) {
        Map<String, Long> work = new HashMap<>();
        long[] result = new long[selected.length];
        for (int i = 0; i < selected.length; i++) {
            if (selected[i] < 0) {
                continue;
            }
            BatchCandidate candidate = requests.get(i).candidates().get(selected[i]);
            long preceding = work.getOrDefault(candidate.worker(), 0L);
            result[i] = add(candidate.ttftMs(), preceding);
            work.put(candidate.worker(), add(preceding, candidate.workMs()));
        }
        return result;
    }

    /**
     * Sums cost values with saturation so overflow cannot make a worse plan cheaper.
     */
    static long total(long[] values) {
        long sum = 0;
        for (long value : values) {
            sum = add(sum, value);
        }
        return sum;
    }

    static boolean cacheEligible(BatchPlanningRequest request,
                                 BatchCandidate candidate,
                                 BatchCandidate baseline) {
        return candidate.hitTokens() >= baseline.hitTokens()
                && (request.minPrefixHitPercent() <= 0.0
                || request.seqLen() > 0L
                && candidate.hitTokens() * 100.0 / request.seqLen()
                >= request.minPrefixHitPercent());
    }

    static long add(long left, long right) {
        return Long.MAX_VALUE - left < right ? Long.MAX_VALUE : left + right;
    }
}

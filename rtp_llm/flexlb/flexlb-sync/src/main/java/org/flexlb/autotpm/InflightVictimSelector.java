package org.flexlb.autotpm;

import java.util.Collection;
import java.util.Optional;
import java.util.function.LongPredicate;

/**
 * Pure-function victim selection for Auto-TPM running-decode preemption
 * (decision D3).
 *
 * <p>Eligibility (all must hold):
 * <ul>
 *   <li>{@code candidate.priority < incomingPriority} — strictly lower; equal
 *       priorities never preempt each other (iron rule 2)</li>
 *   <li>phase == RUNNING — guaranteed by construction: candidates come from
 *       {@code DecodeEndpoint#snapshotRunningCandidates}, which only emits
 *       RUNNING-phase tasks</li>
 *   <li>{@code nowMs - runningSinceMs >= criticalSectionMs} — grace period:
 *       freshly started work is not worth throwing away</li>
 *   <li>no pending cancel intent — a victim already being cancelled must not
 *       be picked twice</li>
 * </ul>
 *
 * <p>Selection order among eligible candidates (all ascending):
 * priority → iterateCount (shallow progress first) → kvTokens → requestId.
 * The lexicographic priority-first ordering subsumes the {@code 2^(priority
 * diff)} tie-break weight from D3: a strictly lower priority always wins
 * regardless of any weighting, and within one priority level the weight is
 * constant, so the remaining keys decide.
 */
public final class InflightVictimSelector {

    private InflightVictimSelector() {
    }

    /**
     * Pick the best eligible victim, or empty when none qualifies.
     *
     * @param candidates        RUNNING-phase snapshot (may be empty)
     * @param incomingPriority  priority of the request needing capacity
     * @param criticalSectionMs grace period; candidates running for less are skipped
     * @param nowMs             current epoch millis (injected for testability)
     * @param hasCancelIntent   requestIds with an in-flight cancel already issued
     */
    public static Optional<VictimCandidate> select(Collection<VictimCandidate> candidates,
                                                   int incomingPriority,
                                                   long criticalSectionMs,
                                                   long nowMs,
                                                   LongPredicate hasCancelIntent) {
        VictimCandidate best = null;
        for (VictimCandidate candidate : candidates) {
            if (!isEligible(candidate, incomingPriority, criticalSectionMs, nowMs, hasCancelIntent)) {
                continue;
            }
            if (best == null || compare(candidate, best) < 0) {
                best = candidate;
            }
        }
        return Optional.ofNullable(best);
    }

    /** Eligibility check — see class javadoc for the full rule set. */
    static boolean isEligible(VictimCandidate candidate,
                              int incomingPriority,
                              long criticalSectionMs,
                              long nowMs,
                              LongPredicate hasCancelIntent) {
        return candidate.priority() < incomingPriority
                && nowMs - candidate.runningSinceMs() >= criticalSectionMs
                && !hasCancelIntent.test(candidate.requestId());
    }

    /** D3 selection order: priority asc → iterateCount asc → kvTokens asc → requestId asc. */
    static int compare(VictimCandidate a, VictimCandidate b) {
        int cmp = Integer.compare(a.priority(), b.priority());
        if (cmp != 0) {
            return cmp;
        }
        cmp = Long.compare(a.iterateCount(), b.iterateCount());
        if (cmp != 0) {
            return cmp;
        }
        cmp = Long.compare(a.kvTokens(), b.kvTokens());
        if (cmp != 0) {
            return cmp;
        }
        return Long.compare(a.requestId(), b.requestId());
    }
}

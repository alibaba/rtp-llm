package org.flexlb.balance.autotpm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Pure function: {@link DecodeAdmissionTracker} state + incoming request →
 * list of RUNNING decode victims eligible for preemption.
 *
 * <p>Stateless and side-effect free — performs no tracker mutation, no cancel.
 * The caller ({@link RunningPreemptCommitter}) is responsible for executing
 * the preemption (cancel + wait for release).
 *
 * <h2>Hard rule</h2>
 * A victim is eligible only when {@code victim.priority < incoming.priority}.
 * Same-priority requests are never preempted.
 *
 * <h2>Critical section</h2>
 * Requests that entered RUNNING state less than {@code criticalSectionMs}
 * milliseconds ago are not preempted, giving them a grace period to make
 * progress before they can be cancelled.
 *
 * <h2>Sort order (first = most preferred to preempt)</h2>
 * <ol>
 *   <li>Priority ascending (lowest priority preempted first — minimizes the
 *       maximum victim priority)</li>
 *   <li>KV tokens descending (more KV released first — maximizes freed
 *       capacity per preemption)</li>
 *   <li>Request ID ascending (stable tie-break, earlier arrival proxy)</li>
 * </ol>
 */
public final class RunningPreemptPlanner {

    /**
     * Find RUNNING victims that can be preempted.
     *
     * @param tracker           the admission tracker (source of truth)
     * @param endpointKey       ip:port of the decode endpoint
     * @param incomingPriority  priority of the incoming request
     * @param neededSlots       decode slots needed (usually 1)
     * @param neededKv          KV tokens needed by the incoming request
     * @param criticalSectionMs don't preempt requests running &lt; this long (ms)
     * @param maxVictims        max victims to return
     * @return sorted list of candidate victims, or empty if none viable
     */
    public List<DecodeReservation> findPreemptCandidates(
            DecodeAdmissionTracker tracker, String endpointKey,
            int incomingPriority, int neededSlots, long neededKv,
            long criticalSectionMs, int maxVictims) {

        long now = System.currentTimeMillis();

        List<DecodeReservation> candidates = new ArrayList<>();
        for (DecodeReservation r : tracker.getReservations(endpointKey)) {
            if (r.state() != DecodeAdmissionState.RUNNING) {
                continue;
            }
            if (r.priority() >= incomingPriority) {
                continue; // hard rule: victim.priority < incoming.priority
            }
            if (r.runningSinceMs() == 0) {
                // Never transitioned to RUNNING via markRunning (edge case) —
                // treat as just started, protect with critical section
                continue;
            }
            if ((now - r.runningSinceMs()) < criticalSectionMs) {
                continue; // in critical section, don't preempt
            }
            candidates.add(r);
        }

        if (candidates.isEmpty()) {
            return Collections.emptyList();
        }

        // Sort: priority asc, KV desc, requestId asc
        candidates.sort(Comparator
                .comparingInt(DecodeReservation::priority)
                .thenComparing(Comparator
                        .comparingLong(DecodeReservation::kvTokensRequired)
                        .reversed())
                .thenComparingLong(DecodeReservation::requestId));

        int selectCount = Math.min(Math.max(0, maxVictims), candidates.size());
        return new ArrayList<>(candidates.subList(0, selectCount));
    }
}

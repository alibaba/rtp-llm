package org.flexlb.balance.autotpm;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Commits a {@link DecodeEvictionPlan} by removing victims from the
 * {@link DecodeAdmissionTracker}.
 *
 * <p>Unlike {@link PlanCommitter} (prefill), there is no global version CAS —
 * the decode tracker uses per-reservation CAS via
 * {@link DecodeAdmissionTracker#removeIfEvictable}. This means:
 * <ul>
 *   <li>Victims that transitioned to RUNNING between planning and committing
 *       are silently skipped (not evicted).</li>
 *   <li>Victims that were already released (completed/cancelled) are skipped.</li>
 * </ul>
 *
 * <p>The committer does NOT complete victims' futures or reserve the incoming —
 * that is the caller's responsibility (it has access to the inflight map /
 * lifecycle cleanup / {@link DecodeAdmissionTracker#reserve}). The removed
 * {@link DecodeReservation}s are returned via {@link CommitResult#victims()}
 * for the caller to fail with {@code PRIORITY_PREEMPTED}.
 *
 * <p>Stateless and reusable; safe to call concurrently from different threads.
 */
public final class DecodeEvictionCommitter {

    private static final Logger log = LoggerFactory.getLogger(DecodeEvictionCommitter.class);

    /**
     * Execute the eviction plan: remove victims from the tracker.
     *
     * @param plan    the eviction plan (carries victim reservations)
     * @param tracker the admission tracker to mutate
     * @return commit result; never {@code null}
     */
    public CommitResult execute(DecodeEvictionPlan plan, DecodeAdmissionTracker tracker) {
        if (plan == null || plan.isEmpty()) {
            return CommitResult.empty();
        }

        List<DecodeReservation> removed = new ArrayList<>();
        for (DecodeReservation victim : plan.victims()) {
            DecodeReservation r = tracker.removeIfEvictable(
                    victim.decodeEndpointKey(), victim.requestId());
            if (r != null) {
                removed.add(r);
            }
            // else: victim was already released or transitioned to RUNNING — skip
        }

        if (removed.isEmpty()) {
            log.debug("Decode eviction committed but no victims removed "
                            + "(all already released or RUNNING): ep={} planned={}",
                    plan.endpointKey(), plan.victimCount());
            return CommitResult.empty();
        }

        log.info("Decode eviction committed: {} victims removed from ep={}",
                removed.size(), plan.endpointKey());
        return CommitResult.success(removed);
    }

    /**
     * Result of a commit attempt.
     */
    public static final class CommitResult {
        private final boolean success;
        private final List<DecodeReservation> victims;
        private final String failureReason;

        private CommitResult(boolean success, List<DecodeReservation> victims,
                             String failureReason) {
            this.success = success;
            this.victims = victims == null ? Collections.emptyList() : victims;
            this.failureReason = failureReason;
        }

        /** CAS succeeded; victims were removed from the tracker. */
        public static CommitResult success(List<DecodeReservation> victims) {
            return new CommitResult(true, victims, null);
        }

        /** Empty plan or all victims already released/transitioned. */
        public static CommitResult empty() {
            return new CommitResult(false, Collections.emptyList(), "empty_or_all_released");
        }

        public boolean isSuccess() { return success; }
        public List<DecodeReservation> victims() { return victims; }
        public String failureReason() { return failureReason; }
    }
}

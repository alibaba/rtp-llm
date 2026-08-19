package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.scheduler.WorkerBatcher;
import org.springframework.stereotype.Component;

/**
 * Validates and applies an admission plan.
 *
 * <p>Two commit strategies (redesign N3 §3.2/3.3, gray switch
 * {@code autoTpmCommitStrategy}):
 *
 * <p><b>lockfree</b> (default) — the normal placement path carries no
 * optimistic version checks: the decode reservation was already atomically
 * booked inside {@code router.route()} (other reserve/release/calibrate
 * activity cannot invalidate it), and the prefill offer decides queue-full /
 * stopped locally inside the {@code queueLock} critical section. The plan's
 * version fields are retained for observability only (plan age / staleness).
 * {@link CommitResult#VERSION_MISMATCH} is never returned on this path.
 *
 * <p><b>versioned</b> (legacy fallback, kept for one version cycle):
 * <ol>
 *   <li>Re-check the target decode admission version against the value
 *       captured in the plan; a mismatch aborts with
 *       {@link CommitResult#VERSION_MISMATCH} (the caller releases the
 *       decode reservation and retries)</li>
 *   <li>Register the item as inflight, then offer it to the prefill batcher
 *       via {@code WorkerBatcher.offerAtVersion}, which re-checks the prefill
 *       queue version and enqueues in one {@code queueLock} critical section
 *       (task10 P1-4 — no window between version check and enqueue). A stale
 *       queue version undoes the registration and returns
 *       {@link CommitResult#VERSION_MISMATCH}; an offer failure (stopped /
 *       full queue) undoes the registration and returns
 *       {@link CommitResult#OFFER_FAILED} (the caller releases the decode
 *       reservation)</li>
 * </ol>
 */
@Component
public class PlanCommitter {

    public enum CommitResult {
        /** Plan applied; item queued on the prefill batcher. */
        SUCCESS,
        /** Snapshot versions stale; nothing applied (decode reservation still held by caller). */
        VERSION_MISMATCH,
        /** Offer to prefill batcher failed (stopped/full) or duplicate request id. */
        OFFER_FAILED
    }

    public CommitResult commit(NormalPlacementPlan plan, InflightRegistrar registrar,
                               boolean lockfree) {
        if (!lockfree && plan.decodeEp() != null
                && plan.decodeEp().admissionVersion() != plan.decodeAdmissionVersion()) {
            return CommitResult.VERSION_MISMATCH;
        }
        InflightRegistration registration =
                InflightRegistration.tryRegister(registrar, plan.item());
        if (registration == null) {
            return CommitResult.OFFER_FAILED;
        }
        try (registration) {
            if (lockfree) {
                // N3 §3.3: no version checks — queue-full/stopped are decided
                // locally and atomically inside the offer's queueLock section.
                if (!plan.prefillEp().getBatcher().tryOffer(plan.item())) {
                    return CommitResult.OFFER_FAILED;
                }
                registration.handoffToQueue();
                return CommitResult.SUCCESS;
            }
            // Version check + enqueue in one queueLock critical section.
            WorkerBatcher.OfferAtVersionResult offer = plan.prefillEp().getBatcher()
                    .offerAtVersion(plan.item(), plan.prefillQueueVersion());
            switch (offer) {
                case VERSION_MISMATCH -> {
                    return CommitResult.VERSION_MISMATCH;
                }
                case OFFER_FAILED -> {
                    return CommitResult.OFFER_FAILED;
                }
                default -> {
                    registration.handoffToQueue();
                    return CommitResult.SUCCESS;
                }
            }
        }
    }
}

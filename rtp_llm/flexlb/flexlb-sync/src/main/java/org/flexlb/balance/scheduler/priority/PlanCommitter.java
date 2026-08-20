package org.flexlb.balance.scheduler.priority;

import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

/**
 * Validates and applies an admission plan.
 *
 * <p>The decode reservation is already atomically booked by
 * {@code router.route()}. Commit therefore has one protocol: register the
 * request as inflight, then let the target worker queue atomically decide
 * whether it can accept the item. Unrelated queue mutations never invalidate
 * a valid reservation or force a full route retry.
 */
@Component
public class PlanCommitter {

    public enum CommitResult {
        /** Plan applied; item queued on the prefill batcher. */
        SUCCESS,
        /** Offer to prefill batcher failed (stopped/full) or duplicate request id. */
        OFFER_FAILED
    }

    public CommitResult commit(NormalPlacementPlan plan, InflightRegistrar registrar) {
        // The production future is PriorityScheduler's request-generation
        // gate. Holding it from registration through queue publication makes
        // cancellation/expiration linearize either before the whole commit or after
        // the item is externally visible to the batcher.
        synchronized (plan.item().future()) {
            // Registration refuses both a closed admission generation
            // (cancellation/expiration already won) and a duplicate request
            // id; the gate state is frozen for the whole critical section, so
            // the distinction is only needed to classify the failure.
            if (!registrar.registerInflight(plan.item())) {
                if (registrar.isAdmissionOpen(
                        plan.item().requestId(), plan.item().future())) {
                    Logger.warn("[priority-scheduler] commit failed: duplicate request_id={}",
                            plan.envelope().requestId());
                }
                return CommitResult.OFFER_FAILED;
            }
            if (!plan.prefillEp().getBatcher().tryOffer(plan.item())) {
                registrar.unregisterInflight(plan.item());
                Logger.debug("[priority-scheduler] commit offer failed (batcher stopped/full), request_id={}",
                        plan.envelope().requestId());
                return CommitResult.OFFER_FAILED;
            }
            return CommitResult.SUCCESS;
        }
    }
}

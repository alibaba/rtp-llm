package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.Objects;

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

    /** A closed generation has no new business failure to publish. */
    public record CommitResult(boolean committed, Response failure) {
        static final CommitResult COMMITTED = new CommitResult(true, null);
        static final CommitResult CLOSED = new CommitResult(false, null);

        public CommitResult {
            if (committed && failure != null) {
                throw new IllegalArgumentException("a committed plan cannot carry a rejection");
            }
        }

        static CommitResult rejected(Response failure) {
            return new CommitResult(false, Objects.requireNonNull(failure));
        }
    }

    public CommitResult commit(NormalPlacementPlan plan, InflightRegistrar registrar) {
        // The production future is PriorityScheduler's request-generation
        // gate. Holding it from registration through queue publication makes
        // cancellation/expiration linearize either before the whole commit or after
        // the item is externally visible to the batcher.
        synchronized (plan.item().future()) {
            CommitResult registration = register(plan.item(), registrar);
            if (!registration.committed()) {
                return registration;
            }
            Response failure = plan.prefillEp().getBatcher().tryOffer(plan.item());
            if (failure != null) {
                registrar.unregisterInflight(plan.item());
            }
            return failure == null ? CommitResult.COMMITTED : CommitResult.rejected(failure);
        }
    }

    /** Caller holds the request-generation monitor through queue publication. */
    static CommitResult register(BatchItem item, InflightRegistrar registrar) {
        if (!registrar.isAdmissionOpen(item.requestId(), item.future())) {
            return CommitResult.CLOSED;
        }
        if (registrar.registerInflight(item)) {
            return CommitResult.COMMITTED;
        }
        if (!registrar.isAdmissionOpen(item.requestId(), item.future())) {
            return CommitResult.CLOSED;
        }
        Logger.warn("[priority-scheduler] duplicate inflight registration: request_id={}", item.requestId());
        return CommitResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST,
                AdmissionRejectReason.UNSPECIFIED, "duplicate request_id: " + item.requestId()));
    }
}

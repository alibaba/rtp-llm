package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.loadbalance.Response;

/**
 * Phase 3 admission plan: free prefill queue slots by evicting strictly
 * lower-priority queued requests, then queue the incoming request in their
 * place (design doc 9.5).
 *
 * <p>The decode reservation is already held when the plan is constructed
 * (taken inside {@code router.route()}), so commit only needs the atomic
 * victim-replace on the prefill queue. On commit failure the caller releases
 * the decode reservation; on success the removed victims are terminated with
 * {@code PRIORITY_PREEMPTED}.
 */
public final class PrefillEvictionPlan {

    private final PriorityRequestEnvelope envelope;
    private final BatchItem item;
    private final Response routeResponse;
    private final PrefillEvictionProposal proposal;

    public PrefillEvictionPlan(PriorityRequestEnvelope envelope,
                               BatchItem item,
                               Response routeResponse,
                               PrefillEvictionProposal proposal) {
        this.envelope = envelope;
        this.item = item;
        this.routeResponse = routeResponse;
        this.proposal = proposal;
    }

    public PriorityRequestEnvelope envelope() {
        return envelope;
    }

    public BatchItem item() {
        return item;
    }

    public Response routeResponse() {
        return routeResponse;
    }

    public PrefillEvictionProposal proposal() {
        return proposal;
    }

    public PrefillEndpoint prefillEp() {
        return item.prefillEp();
    }

    public DecodeEndpoint decodeEp() {
        return item.decodeEp();
    }
}

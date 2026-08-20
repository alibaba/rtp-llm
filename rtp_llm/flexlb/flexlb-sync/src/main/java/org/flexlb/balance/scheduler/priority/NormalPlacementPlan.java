package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.loadbalance.Response;

/**
 * Phase 1 admission plan: place the incoming request on the router-selected
 * prefill/decode pair without any eviction.
 *
 * <p>The decode reservation is already held when the plan is constructed — it
 * happens inside {@code router.route()} — so commit only needs to register the
 * request and offer it to the target prefill queue. On commit failure the
 * caller must release the decode reservation.
 */
public final class NormalPlacementPlan {

    private final PriorityRequestEnvelope envelope;
    private final BatchItem item;
    private final Response routeResponse;
    private final long createdAtMs;

    /**
     * @param envelope               incoming request descriptor
     * @param item                   fully built batch item (endpoints resolved)
     * @param routeResponse successful route response backing the item
     */
    public NormalPlacementPlan(PriorityRequestEnvelope envelope,
                               BatchItem item,
                               Response routeResponse) {
        this.envelope = envelope;
        this.item = item;
        this.routeResponse = routeResponse;
        this.createdAtMs = System.currentTimeMillis();
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

    public PrefillEndpoint prefillEp() {
        return item.prefillEp();
    }

    public DecodeEndpoint decodeEp() {
        return item.decodeEp();
    }

    /**
     * Plan build timestamp for the {@code auto_tpm.plan_age_ms} observability
     * metric (redesign N3 §3.8): quantifies how stale the plan view was when
     * the commit succeeded.
     */
    public long createdAtMs() {
        return createdAtMs;
    }
}

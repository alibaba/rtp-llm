package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.loadbalance.Response;

import java.util.List;

/**
 * Phase 1 admission plan: place the incoming request on the router-selected
 * prefill/decode pair without any eviction.
 *
 * <p>The decode reservation ({@link PlanAction#RESERVE_DECODE_FOR_INCOMING})
 * is already held when the plan is constructed — it happens inside
 * {@code router.route()} — so commit only needs to validate versions and
 * perform {@link PlanAction#OFFER_PREFILL_FOR_INCOMING}. On commit failure the
 * caller must release the decode reservation.
 */
public final class NormalPlacementPlan implements AdmissionPlan {

    private static final List<PlanAction> ACTIONS = List.of(
            PlanAction.RESERVE_DECODE_FOR_INCOMING,
            PlanAction.OFFER_PREFILL_FOR_INCOMING);

    private final PriorityRequestEnvelope envelope;
    private final BatchItem item;
    private final Response routeResponse;
    private final long prefillQueueVersion;
    private final long decodeAdmissionVersion;
    private final long createdAtMs;

    /**
     * @param envelope               incoming request descriptor
     * @param item                   fully built batch item (endpoints resolved)
     * @param routeResponse          successful route response backing the item
     * @param prefillQueueVersion    target batcher queue version captured at
     *                               snapshot time (before route)
     * @param decodeAdmissionVersion target decode admission version captured
     *                               right after route's own reservation
     *                               (observational under the lockfree commit
     *                               strategy, redesign N3 §3.3)
     */
    public NormalPlacementPlan(PriorityRequestEnvelope envelope,
                               BatchItem item,
                               Response routeResponse,
                               long prefillQueueVersion,
                               long decodeAdmissionVersion) {
        this.envelope = envelope;
        this.item = item;
        this.routeResponse = routeResponse;
        this.prefillQueueVersion = prefillQueueVersion;
        this.decodeAdmissionVersion = decodeAdmissionVersion;
        this.createdAtMs = System.currentTimeMillis();
    }

    @Override
    public PriorityRequestEnvelope envelope() {
        return envelope;
    }

    @Override
    public List<PlanAction> actions() {
        return ACTIONS;
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

    public long prefillQueueVersion() {
        return prefillQueueVersion;
    }

    public long decodeAdmissionVersion() {
        return decodeAdmissionVersion;
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

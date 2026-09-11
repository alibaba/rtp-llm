package org.flexlb.dispatcher;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;

/** Defaults, DISPATCH_CONFIG JSON, then Spring dispatch.* overrides; validated at startup. */
@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class DispatchConfig {
    /** Empty discovery may retain the previous pool for this long. */
    private long discoveryFailureGraceMs = 300_000;

    /** count:N chunks (capped by item count), size:N items per chunk, or a bare size. */
    private String subBatch = "count:5";

    /** Supplying an FE discovery name enables dispatcher routes. Whitespace-only names fail startup. */
    private String fePoolServiceId = "";

    /** Passed to Reactor Netty responseTimeout for each FE sub-call. */
    private int batchTimeoutMs = 30_000;

    /** The whole sub-call, including response body, is capped at batchTimeoutMs + bodyReadMarginMs. */
    private long bodyReadMarginMs = 30_000;

    /** FE application health endpoint, independent of BE availability. */
    private String probePath = "/frontend_health";

    /** master shares the elected master's FE cursor; local uses this dispatcher's FE pool. */
    private String feAllocation = FeAllocationMode.MASTER.configValue();

    /**
     * Stateless BE placement for supported single-role deployments without traffic policies.
     * Default false retains FE request-aware LLM scheduling and admission. FE allocation is independent.
     */
    private boolean preAssignBe = false;

    /** Required on dispatcher and receiving FEs for BE preassignment; loaded only from DISPATCH_ROUTING_TOKEN. */
    @JsonIgnore
    private String trustedRoutingToken = "";

    /** Aggregate retained response bytes per batch, in addition to the per-FE response cap. */
    private long maxAggregateResponseBytes = 128L * 1024 * 1024;

    /** Aggregate outbound bytes, including the envelope repeated across chunks. */
    private long maxAggregateRequestBytes = 128L * 1024 * 1024;

    /** Serialized dry-run response limit, checked before allocating repeated envelopes. */
    private long maxDryRunResponseBytes = 64L * 1024 * 1024;

    /** Derived at startup; excluded from JSON binding despite Lombok's generated accessors. */
    @JsonIgnore
    private SubBatchSpec subBatchSpec;
}

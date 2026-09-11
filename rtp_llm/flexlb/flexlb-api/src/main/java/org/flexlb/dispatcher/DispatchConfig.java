package org.flexlb.dispatcher;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;

/** Spring dispatch.* properties, including DISPATCH_* environment variables; validated at startup. */
@Getter
@Setter
public class DispatchConfig {
    public enum FeAllocation { MASTER, LOCAL }

    /** Empty discovery may retain the previous pool for this long. */
    private long discoveryFailureGraceMs = 300_000;

    private String subBatch = "count:5";
    private String fePoolServiceId = "";

    /** Passed to Reactor Netty responseTimeout for each FE sub-call. */
    private int batchTimeoutMs = 30_000;

    /** The whole sub-call, including response body, is capped at batchTimeoutMs + bodyReadMarginMs. */
    private long bodyReadMarginMs = 30_000;

    private String probePath = "/frontend_health";

    private FeAllocation feAllocation = FeAllocation.MASTER;
    private boolean preAssignBe = false;

    /** Required on dispatcher and receiving FEs for BE preassignment; loaded only from DISPATCH_ROUTING_TOKEN. */
    @JsonIgnore
    private String trustedRoutingToken = "";

    /** Aggregate retained response bytes per batch, in addition to the per-FE response cap. */
    private long maxAggregateResponseBytes = 128L * 1024 * 1024;

    /** Aggregate outbound bytes, including the envelope repeated across chunks. */
    private long maxAggregateRequestBytes = 128L * 1024 * 1024;

    /** Derived at startup; excluded from JSON binding despite Lombok's generated accessors. */
    @JsonIgnore
    private SubBatchSpec subBatchSpec;
}

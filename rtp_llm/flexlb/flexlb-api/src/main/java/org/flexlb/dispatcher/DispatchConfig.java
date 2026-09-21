package org.flexlb.dispatcher;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;

/** Spring dispatch.* properties, including DISPATCH_* environment settings; validated at startup. */
@Getter
@Setter
public class DispatchConfig {
    private String subBatch = "count:5";
    private String fePoolServiceId = "";

    /** Passed to Reactor Netty responseTimeout for each FE sub-call. */
    private int batchTimeoutMs = 30_000;

    private String probePath = "/frontend_health";

    private boolean preAssignBe = true;

    /**
     * Required on dispatcher and receiving FEs for BE preassignment; loaded only from DISPATCH_ROUTING_TOKEN.
     */
    @Getter(onMethod_ = @JsonIgnore)
    private String trustedRoutingToken = "";

    /** Derived at startup, not a configuration property. */
    @Getter(onMethod_ = @JsonIgnore)
    private SubBatchSpec subBatchSpec;
}

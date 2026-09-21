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
     * Jackson merges the field and Lombok accessors into one property, so JsonIgnore also excludes the getter
     * from the serialized startup configuration; DispatcherStartupTest checks the actual application log.
     */
    @JsonIgnore
    private String trustedRoutingToken = "";

    /** Derived at startup; excluded from JSON binding despite Lombok's generated accessors. */
    @JsonIgnore
    private SubBatchSpec subBatchSpec;
}

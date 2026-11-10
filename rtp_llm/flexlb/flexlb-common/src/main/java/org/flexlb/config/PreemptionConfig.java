package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

import java.util.EnumSet;
import java.util.Set;

/**
 * Resource reclamation policy for priority-ordered queues; all stages are enabled by default.
 */
@Getter
@Setter
public final class PreemptionConfig {

    @com.fasterxml.jackson.annotation.JsonProperty("allowedVictimStages")
    private Set<PreemptionStage> allowedPreemptionStages = EnumSet.allOf(PreemptionStage.class);

    /** Maximum wait for the Engine terminal after the cancellation ACK phase. */
    private long timeoutMs = 1000;

    private EngineCancellationConfig engineCancellation = new EngineCancellationConfig();

    public boolean allows(PreemptionStage stage) {
        return allowedPreemptionStages != null && allowedPreemptionStages.contains(stage);
    }
}

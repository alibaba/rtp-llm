package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

import java.util.EnumSet;
import java.util.Set;

/** Resource reclamation policy for priority-ordered queues; all stages are enabled by default. */
@Getter
@Setter
public final class PreemptionConfig {

    private Set<VictimStage> allowedVictimStages = EnumSet.allOf(VictimStage.class);
    /** Maximum wait for the Engine terminal after the cancellation ACK phase. */
    private long timeoutMs = 1000;

    public boolean allows(VictimStage stage) {
        return allowedVictimStages != null && allowedVictimStages.contains(stage);
    }
}

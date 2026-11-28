package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSetter;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.EnumSet;
import java.util.Set;

/**
 * Decode resource reclamation policy for priority ordering; both stages are enabled by default.
 */
@Getter
@Setter
@Slf4j
public final class PreemptionConfig {

    private Set<VictimStage> allowedVictimStages = EnumSet.allOf(VictimStage.class);

    /** Maximum wait for the Engine terminal after the cancellation ACK phase. */
    private long timeoutMs = 1000;

    private EngineCancellationConfig engineCancellation = new EngineCancellationConfig();

    public boolean allows(VictimStage stage) {
        return allowedVictimStages != null && allowedVictimStages.contains(stage);
    }

    @JsonSetter("allowedVictimStages")
    private void readAllowedVictimStages(Set<String> configuredStages) {
        EnumSet<VictimStage> stages = EnumSet.noneOf(VictimStage.class);
        for (String stage : configuredStages) {
            if ("PREFILL_QUEUED".equals(stage)) {
                log.warn("Ignoring retired preemption stage PREFILL_QUEUED in allowedVictimStages");
            } else {
                stages.add(VictimStage.valueOf(stage));
            }
        }
        setAllowedVictimStages(stages);
    }
}

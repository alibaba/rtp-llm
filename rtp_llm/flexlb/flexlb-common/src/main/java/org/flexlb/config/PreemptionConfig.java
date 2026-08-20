package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

import java.util.EnumSet;
import java.util.Set;

@Getter
@Setter
public final class PreemptionConfig {

    private Set<VictimStage> allowedVictimStages = EnumSet.noneOf(VictimStage.class);
    private EngineCancellationConfig engineCancellation;

    public boolean allows(VictimStage stage) {
        return allowedVictimStages != null && allowedVictimStages.contains(stage);
    }
}

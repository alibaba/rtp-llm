package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class OptimizerRuntimeConfig {

    private boolean enabled;
    private long discoveryPollIntervalMs = 1000L;
}

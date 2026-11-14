package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/**
 * Controls optimizer queries and discovery polling intervals in milliseconds.
 */
@Getter
@Setter
public final class OptimizerRuntimeConfig {

    private boolean enabled;
    private long discoveryPollIntervalMs = 1000L;
}

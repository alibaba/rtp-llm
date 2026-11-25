package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/**
 * Controls P engine TTFT window reporting and whether to count remote KV loading before scheduling.
 */
@Getter
@Setter
public final class TtftStagesConfig {
    private boolean enabled = true;
    private boolean remoteKvBeforeScheduler;
}

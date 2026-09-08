package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Time bounds used only when preempting Decode requests owned by an Engine. */
@Getter
@Setter
public final class EngineCancellationConfig {

    private long ackTimeoutMs = 50;
    private long completionTimeoutMs = 1000;
}

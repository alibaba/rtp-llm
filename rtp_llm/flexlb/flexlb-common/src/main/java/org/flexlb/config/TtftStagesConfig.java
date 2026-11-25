package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/**
 * Collects per-dispatch P TTFT locally in standalone mode or on the elected leader when election is enabled.
 */
@Getter
@Setter
public final class TtftStagesConfig {
    private boolean enabled = true;
    private long rpcTimeoutMs = 800L;
    private int maxRecords = 4096;
    private int maximumEntries = 65_536;
    private long retentionMs = 120_000L;
    private long graceMs = 3000L;
    private boolean remoteKvBeforeScheduler;
}

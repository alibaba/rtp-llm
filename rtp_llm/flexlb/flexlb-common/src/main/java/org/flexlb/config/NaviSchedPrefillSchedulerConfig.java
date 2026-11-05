package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class NaviSchedPrefillSchedulerConfig implements PrefillSchedulerConfig {
    private String masterBizName;
    private long startupTimeoutMs = 60_000;
    private long stopTimeoutMs = 30_000;
}

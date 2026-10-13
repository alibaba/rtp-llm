package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class ObservabilityConfig {

    private CacheHitConfig cacheHit = new CacheHitConfig();

    @Getter
    @Setter
    public static final class CacheHitConfig {
        private RecentKeyWindowConfig recentKeyWindow = new RecentKeyWindowConfig();
        private boolean metricsEnabled = true;
        private boolean requestTraceLogEnabled;
        private TheoryLogConfig theoryLog;
    }

    @Getter
    @Setter
    public static final class RecentKeyWindowConfig {
        private boolean writeEnabled = true;
        private long durationMs = 30L * 60L * 1000L;
        private long maxKeyOccurrences = 10_000_000L;
    }

    @Getter
    @Setter
    public static final class TheoryLogConfig {
        private String path = "/home/admin/ai-whale/logs/master_theory_hit.log";
    }
}

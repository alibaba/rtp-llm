package org.flexlb.config.monitoring;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.util.log.LogLevel;

/**
 * Groups logging, cache-feedback metrics and recent-key observation settings.
 */
@Getter
@Setter
public final class ObservabilityConfig {

    private CacheHitConfig cacheHit = new CacheHitConfig();
    private LoggingConfig logging = new LoggingConfig();

    /**
     * Controls the FlexLB log level and console output switch.
     */
    @Getter
    @Setter
    public static final class LoggingConfig {
        private LogLevel level = LogLevel.INFO;
        private boolean stdoutEnabled;
    }

    /**
     * Controls cache prediction metrics, request traces and recent-key observations independently.
     */
    @Getter
    @Setter
    public static final class CacheHitConfig {
        private RecentKeyWindowConfig recentKeyWindow = new RecentKeyWindowConfig();
        private boolean metricsEnabled = true;
        private boolean requestTraceLogEnabled;
        private TheoryLogConfig theoryLog;
    }

    /**
     * Bounds recent-key history by retention time in milliseconds and occurrence capacity.
     */
    @Getter
    @Setter
    public static final class RecentKeyWindowConfig {
        private boolean writeEnabled = true;
        private long durationMs = 30L * 60L * 1000L;
        private long maxKeyOccurrences = 1_000_000L;
    }

    /**
     * Defines the destination for theoretical cache-hit trace records.
     */
    @Getter
    @Setter
    public static final class TheoryLogConfig {
        private String path = "/home/admin/ai-whale/logs/master_theory_hit.log";
    }
}

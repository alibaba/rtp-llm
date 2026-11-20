package org.flexlb.config.monitoring;

import lombok.Getter;
import lombok.Setter;

/**
 * Allocator scans are opt-in because Netty acquires arena locks while reading these metrics.
 */
@Getter
@Setter
public final class NettyMemoryReporterConfig {
    private boolean enabled = false;
    private long intervalMs = 10_000L;
    private AutoCaptureConfig autoCapture = new AutoCaptureConfig();

    /**
     * Controls memory diagnostics with a byte threshold and a cooldown in milliseconds.
     */
    @Getter
    @Setter
    public static final class AutoCaptureConfig {
        private boolean enabled = false;
        private long thresholdBytes = 5L * 1024 * 1024 * 1024;
        private long cooldownMs = 300_000L;
    }
}

package org.flexlb.service.lease;

import org.flexlb.config.PrefillLeaseConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Request-weighted TTFT from current serving workers' committed status telemetry.
 */
@Component
public final class PrefillTtftStatusReader {
    private final WorkerDirectory directory;

    public PrefillTtftStatusReader(WorkerDirectory directory) {
        this.directory = directory;
    }

    public record Measurement(double meanMs, long samples, long oldestStatusMs) {
    }

    public Measurement read(PrefillLeaseConfig config, List<String> workers, long nowMs, long minRequestReceivedMs) {
        double totalMs = 0;
        long count = 0;
        long oldest = nowMs;
        var snapshot = directory.statusSnapshot(RoleType.PREFILL);
        for (String address : workers) {
            var worker = snapshot.get(address);
            if (worker == null || !worker.isActiveGeneration() || !worker.isAlive()) {
                throw new IllegalStateException("Serving worker is unavailable");
            }
            var measurement = worker.prefillTtftWindow()
                    .read(nowMs - config.getWindowSeconds() * 1000, nowMs, config.getMaxMetricAgeSeconds() * 1000,
                            minRequestReceivedMs);
            totalMs += measurement.totalMs();
            count += measurement.count();
            oldest = Math.min(oldest, measurement.latestStatusMs());
        }
        if (count == 0) {
            throw new IllegalStateException("No valid TTFT samples from serving workers");
        }
        return new Measurement(totalMs / count, count, oldest);
    }
}

package org.flexlb.service.monitor;

import org.flexlb.config.TtftStagesConfig;
import org.flexlb.engine.grpc.core.GrpcTarget;

/**
 * Immutable settings for one collection session, independent of changes to the discovered instances.
 */
record RequestTimingSettings(boolean enabled, long timeoutMs,
                             int maxRecords, int maximumEntries, long retentionMs, long graceMs,
                             boolean remoteKvBeforeScheduler) {
    static RequestTimingSettings from(TtftStagesConfig config) {
        return new RequestTimingSettings(config.isEnabled(), config.getRpcTimeoutMs(), config.getMaxRecords(), config.getMaximumEntries(),
                config.getRetentionMs(), config.getGraceMs(), config.isRemoteKvBeforeScheduler());
    }

    /**
     * One physical daemon and the role its timing records must report.
     */
    record Source(GrpcTarget target, String role) {
        String key() {
            return "daemon/" + role + "/" + target.authority();
        }
    }
}

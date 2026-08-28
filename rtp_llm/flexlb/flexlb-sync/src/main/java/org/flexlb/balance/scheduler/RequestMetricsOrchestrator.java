package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.DependsOn;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Objects;

/** Isolated metrics traversal over scheduler and endpoint lifecycle views. */
@Component
@DependsOn("requestShutdownOrchestrator")
final class RequestMetricsOrchestrator {

    interface Lifecycle {
        boolean isShuttingDown();

        int liveRequestCount();

        /**
         * Age (ms) of the oldest live request slot, 0 when the ledger is
         * empty — the master-side leak signature gauge.
         */
        long oldestLiveSlotAgeMs();
    }

    private final Lifecycle lifecycle;
    private final EndpointRegistry registry;
    private final BatchSchedulerReporter reporter;

    RequestMetricsOrchestrator(
            Lifecycle lifecycle,
            EndpointRegistry registry,
            BatchSchedulerReporter reporter) {
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.registry = Objects.requireNonNull(registry, "registry");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    void report() {
        if (lifecycle.isShuttingDown()) {
            return;
        }
        reportSchedulerInflight();
        reportPrefillEndpoints();
        reportDecodeEndpoints();
    }

    private void reportSchedulerInflight() {
        try {
            reporter.reportSchedulerInflightSize(
                    lifecycle.liveRequestCount());
            // Age of the oldest scheduler-ledger inflight entry: with a
            // healthy TTL the size gauge alone cannot distinguish "busy"
            // from "leaking"; a max age creeping toward the TTL window is
            // the leak signature.
            reporter.reportSchedulerInflightMaxAgeMs(
                    lifecycle.oldestLiveSlotAgeMs());
        } catch (RuntimeException failure) {
            warnIsolated(
                    "Failed to report scheduler inflight metrics", failure);
        }
    }

    private void reportPrefillEndpoints() {
        final Map<String, PrefillEndpoint> endpoints;
        try {
            endpoints = registry.snapshotPrefillEndpoints();
        } catch (RuntimeException failure) {
            warnIsolated(
                    "Failed to snapshot Prefill endpoints for metrics",
                    failure);
            return;
        }
        for (Map.Entry<String, PrefillEndpoint> entry : endpoints.entrySet()) {
            try {
                entry.getValue().reportBatchMetrics(reporter);
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report Prefill endpoint metrics: endpoint="
                                + entry.getKey(),
                        failure);
            }
        }
    }

    private void reportDecodeEndpoints() {
        final Map<String, DecodeEndpoint> endpoints;
        try {
            endpoints = registry.snapshotDecodeEndpoints();
        } catch (RuntimeException failure) {
            warnIsolated(
                    "Failed to snapshot Decode endpoints for metrics",
                    failure);
            return;
        }
        for (Map.Entry<String, DecodeEndpoint> entry : endpoints.entrySet()) {
            try {
                entry.getValue().reportBatchMetrics(reporter);
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report Decode endpoint metrics: endpoint="
                                + entry.getKey(),
                        failure);
            }
        }
    }

    private static void warnIsolated(
            String message, RuntimeException failure) {
        try {
            Logger.warn(message, failure);
        } catch (Throwable ignored) {
            // Telemetry must never couple otherwise independent metric leaves.
        }
    }
}

package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Map;
import java.util.Objects;

/** Owns scheduler maintenance, metrics traversal, and ordered shutdown. */
@Component
final class SchedulerRuntime {

    private final RequestRegistry requests;
    private final EndpointRegistry endpoints;
    private final BatchSchedulerReporter reporter;
    private final RequestSchedulerReporter admissionReporter;
    private final Runnable closePlacement;

    SchedulerRuntime(
            RequestRegistry requests,
            EndpointRegistry endpoints,
            BatchSchedulerReporter reporter,
            RequestSchedulerReporter admissionReporter) {
        this(requests, endpoints, reporter, admissionReporter, () -> { });
    }

    @Autowired
    SchedulerRuntime(
            RequestRegistry requests,
            EndpointRegistry endpoints,
            BatchSchedulerReporter reporter,
            RequestSchedulerReporter admissionReporter,
            RequestScheduler scheduler) {
        this(requests, endpoints, reporter, admissionReporter,
                scheduler::closePlacement);
    }

    private SchedulerRuntime(
            RequestRegistry requests,
            EndpointRegistry endpoints,
            BatchSchedulerReporter reporter,
            RequestSchedulerReporter admissionReporter,
            Runnable closePlacement) {
        this.requests = Objects.requireNonNull(requests, "requests");
        this.endpoints = Objects.requireNonNull(endpoints, "endpoints");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.admissionReporter = Objects.requireNonNull(
                admissionReporter, "admissionReporter");
        this.closePlacement = Objects.requireNonNull(
                closePlacement, "closePlacement");
    }

    @Scheduled(fixedRate = 60000L)
    void maintainExpiration() {
        requests.maintainExpiration(endpoints::evictExpiredOrphans);
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    void report() {
        if (requests.isShuttingDown()) {
            return;
        }
        reportSchedulerInflight();
        reportPrefillEndpoints();
        reportDecodeEndpoints();
    }

    private void reportSchedulerInflight() {
        try {
            reporter.reportSchedulerInflightSize(
                    requests.liveRequestCount());
        } catch (RuntimeException failure) {
            warnIsolated(
                    "Failed to report scheduler inflight metrics", failure);
        }
    }

    private void reportPrefillEndpoints() {
        final Map<String, PrefillEndpoint> endpoints;
        try {
            endpoints = this.endpoints.snapshotPrefillEndpoints();
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
            try {
                admissionReporter.reportPrefillQueueDepth(
                        entry.getKey(), entry.getValue().queuedRequestCount());
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report Prefill admission metrics: endpoint="
                                + entry.getKey(),
                        failure);
            }
        }
    }

    private void reportDecodeEndpoints() {
        final Map<String, DecodeEndpoint> endpoints;
        try {
            endpoints = this.endpoints.snapshotDecodeEndpoints();
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
            try {
                entry.getValue().reportAdmissionMetrics(admissionReporter);
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report Decode admission metrics: endpoint="
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

    @PreDestroy
    void shutdown() {
        Throwable failure = null;
        boolean ownsShutdown;
        try {
            closePlacement.run();
        } catch (Throwable closeFailure) {
            failure = closeFailure;
        }
        try {
            ownsShutdown = requests.closeAdmissionAndAwaitMutations();
        } catch (Throwable closeFailure) {
            failure = append(failure, closeFailure);
            ownsShutdown = true;
        }
        if (!ownsShutdown) {
            return;
        }
        try {
            try {
                requests.closeOutstandingAndTerminalize();
            } catch (Throwable closeFailure) {
                failure = append(failure, closeFailure);
            }
            try {
                requests.closeExpiration();
            } catch (Throwable closeFailure) {
                failure = append(failure, closeFailure);
            }
            try {
                endpoints.close();
            } catch (Throwable closeFailure) {
                failure = append(failure, closeFailure);
            }
        } finally {
            try {
                requests.closePublisher();
            } catch (Throwable closeFailure) {
                failure = append(failure, closeFailure);
            }
        }
        rethrow(failure);
    }

    private static Throwable append(Throwable first, Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            try {
                first.addSuppressed(next);
            } catch (Throwable ignored) {
                // Preserve the primary failure and continue shutdown.
            }
        }
        return first;
    }

    private static void rethrow(Throwable failure) {
        if (failure == null) {
            return;
        }
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        throw new IllegalStateException("Scheduler shutdown failed", failure);
    }
}

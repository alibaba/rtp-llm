package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;

/** Owns scheduler maintenance, metrics traversal, and ordered shutdown. */
@Component
final class SchedulerRuntime {

    private static final long EXPIRATION_MAINTENANCE_INTERVAL_MS = 60_000L;

    private final RequestRegistry requests;
    private final EndpointRegistry endpoints;
    private final BatchSchedulerReporter reporter;
    private final RequestSchedulerReporter admissionReporter;
    private final Runnable closePlacement;
    /** Gauges survive removal from the endpoint directory and need one final unavailable sample. */
    private final Map<RoleType, Set<String>> computeGaugeIps = new EnumMap<>(RoleType.class);

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

    @Scheduled(fixedRate = EXPIRATION_MAINTENANCE_INTERVAL_MS)
    void maintainExpiration() {
        requests.maintainExpiration(endpoints::evictExpiredOrphans);
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    void report() {
        if (requests.isShuttingDown()) {
            return;
        }
        reportSchedulerInflight();
        reportPrefillEndpoints(RoleType.PREFILL);
        reportPrefillEndpoints(RoleType.PDFUSION);
        reportDecodeEndpoints();
    }

    private void reportSchedulerInflight() {
        try {
            reporter.reportSchedulerInflightSize(
                    requests.liveRequestCount());
            // Age of the oldest scheduler-ledger inflight entry: with a
            // healthy TTL the size gauge alone cannot distinguish "busy"
            // from "leaking"; a max age creeping toward the TTL window is
            // the leak signature.
            reporter.reportSchedulerInflightMaxAgeMs(
                    requests.oldestLiveSlotAgeMs());
        } catch (RuntimeException failure) {
            warnIsolated(
                    "Failed to report scheduler inflight metrics", failure);
        }
    }

    private void reportPrefillEndpoints(RoleType role) {
        List<EndpointRegistry.PrefillRoutingEntry> directory;
        try {
            directory = endpoints.prefillRoutingSnapshot(role);
        } catch (RuntimeException failure) {
            warnIsolated("Failed to snapshot " + role + " endpoints for metrics", failure);
            directory = List.of(); // No usable sample for previously published IPs in this role.
        }

        Map<String, Double> tokensByIp = new HashMap<>();
        for (var entry : directory) {
            PrefillEndpoint endpoint = entry.endpoint();
            if (role == RoleType.PREFILL) {
                reportPrefillSchedulingMetrics(entry.address(), endpoint);
            }
            String ip = endpoint.getIp();
            double tokens = Double.NaN;
            try {
                OptionalLong queued = endpoint.queuedComputeTokens();
                if (queued.isPresent()) {
                    tokens = queued.getAsLong();
                }
            } catch (RuntimeException failure) {
                warnIsolated("Failed to sample queued compute on " + entry.address(), failure);
            }
            // Ports on the same host share the engineIp label. A missing constituent makes the sum unknown.
            tokensByIp.merge(ip, tokens, Double::sum);
        }
        Set<String> published = computeGaugeIps.computeIfAbsent(role, ignored -> new HashSet<>());
        published.addAll(tokensByIp.keySet());
        published.removeIf(ip -> {
            boolean retired = !tokensByIp.containsKey(ip);
            try {
                reporter.reportPrefillQueuedComputeTokens(role, ip, tokensByIp.getOrDefault(ip, Double.NaN));
                return retired;
            } catch (RuntimeException failure) {
                warnIsolated("Failed to report queued compute on " + ip, failure);
                return false;
            }
        });
    }

    private void reportPrefillSchedulingMetrics(String address, PrefillEndpoint endpoint) {
        try {
            endpoint.reportBatchMetrics(reporter);
        } catch (RuntimeException failure) {
            warnIsolated("Failed to report Prefill endpoint metrics: endpoint=" + address, failure);
        }
        try {
            admissionReporter.reportPrefillQueueDepth(address, endpoint.queuedRequestCount());
        } catch (RuntimeException failure) {
            warnIsolated("Failed to report Prefill admission metrics: endpoint=" + address, failure);
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

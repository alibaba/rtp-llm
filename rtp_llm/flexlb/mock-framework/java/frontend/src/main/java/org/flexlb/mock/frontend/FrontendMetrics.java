package org.flexlb.mock.frontend;

import io.prometheus.client.*;
import org.flexlb.mock.metrics.MetricTags;
import java.util.*;

/**
 * Exports Frontend request counts and end-to-end latency measurements to Prometheus.
 */
final class FrontendMetrics {
    private final CollectorRegistry registry = new CollectorRegistry();
    private final Map<String, String> tags = MetricTags.fromEnvironment(System.getenv());
    private final Counter.Child issued = Counter.build().name("mock_frontend_requests_issued_total").help("Native calls started; not Master acceptance.")
        .labelNames("role").register(registry).labels("frontend");
    private final Counter.Child successful = Counter.build().name("py_rtp_success_qps_metric_total")
        .help("Generation requests that completed successfully.").labelNames("role").register(registry).labels("frontend");
    private final Map<BackendRpcTransport.Outcome, Counter.Child> finished = new EnumMap<>(BackendRpcTransport.Outcome.class);
    private final Histogram.Child latency = Histogram.build().name("mock_frontend_request_seconds").help("Frontend call lifetime.")
        .labelNames("role").register(registry).labels("frontend");
    private final Histogram.Child ttft = Histogram.build().name("mock_frontend_ttft_seconds").help("Call start to first received token.")
        .buckets(.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300, 3600)
        .labelNames("role").register(registry).labels("frontend");
    private final Histogram.Child tpot = Histogram.build().name("mock_frontend_tpot_seconds")
        .help("Time from first to last received token, divided by the number of tokens after the first, for successful calls.")
        .labelNames("role").register(registry).labels("frontend");
    FrontendMetrics() {
        var counter = Counter.build().name("mock_frontend_requests_finished_total").help("Frontend terminal observations.")
            .labelNames("role", "outcome").register(registry);
        for (var outcome : BackendRpcTransport.Outcome.values()) finished.put(outcome, counter.labels("frontend", outcome.name()));
    }
    void issued() { issued.inc(); }
    void finished(BackendRpcTransport.Result result) {
        finished.get(result.outcome()).inc(); latency.observe(result.elapsedNanos() / 1e9);
        if (result.ttftNanos() != null) ttft.observe(result.ttftNanos() / 1e9);
        if (result.outcome() == BackendRpcTransport.Outcome.success) {
            successful.inc();
            if (result.tpotNanos() != null) tpot.observe(result.tpotNanos() / 1e9);
        }
    }
    String scrape() { return MetricTags.scrape(registry, tags); }
}

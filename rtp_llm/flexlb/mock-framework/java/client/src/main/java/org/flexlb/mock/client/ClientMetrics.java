package org.flexlb.mock.client;

import io.prometheus.client.*;
import org.flexlb.mock.metrics.MetricTags;
import java.util.*;

final class ClientMetrics {
    private final CollectorRegistry registry = new CollectorRegistry();
    private final Map<String, String> tags = MetricTags.fromEnvironment(System.getenv());
    private final Counter.Child issued = Counter.build().name("mock_client_requests_issued_total").help("Native calls started; not Master acceptance.")
        .labelNames("role").register(registry).labels("client");
    private final Map<NativeClient.Outcome, Counter.Child> finished = new EnumMap<>(NativeClient.Outcome.class);
    private final Histogram.Child latency = Histogram.build().name("mock_client_request_seconds").help("Client call lifetime.")
        .labelNames("role").register(registry).labels("client");
    private final Histogram.Child ttft = Histogram.build().name("mock_client_ttft_seconds").help("Call start to first received token.")
        .buckets(.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300, 3600)
        .labelNames("role").register(registry).labels("client");
    private final Histogram.Child tpot = Histogram.build().name("mock_client_tpot_seconds")
        .help("Time from first to last received token per token after the first, for successful calls.")
        .labelNames("role").register(registry).labels("client");
    ClientMetrics() {
        var counter = Counter.build().name("mock_client_requests_finished_total").help("Client terminal observations.")
            .labelNames("role", "outcome").register(registry);
        for (var outcome : NativeClient.Outcome.values()) finished.put(outcome, counter.labels("client", outcome.name()));
    }
    void issued() { issued.inc(); }
    void finished(NativeClient.Result result) {
        finished.get(result.outcome()).inc(); latency.observe(result.elapsedNanos() / 1e9);
        if (result.ttftNanos() != null) ttft.observe(result.ttftNanos() / 1e9);
        if (result.outcome() == NativeClient.Outcome.success && result.tpotNanos() != null)
            tpot.observe(result.tpotNanos() / 1e9);
    }
    String scrape() { return MetricTags.scrape(registry, tags); }
}

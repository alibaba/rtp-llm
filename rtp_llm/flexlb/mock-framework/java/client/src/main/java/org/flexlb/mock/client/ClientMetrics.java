package org.flexlb.mock.client;

import io.prometheus.client.*;
import io.prometheus.client.exporter.common.TextFormat;
import java.io.*;
import java.util.*;

final class ClientMetrics {
    private final CollectorRegistry registry = new CollectorRegistry();
    private final Counter.Child issued = Counter.build().name("mock_client_requests_issued_total").help("Native calls started; not Master acceptance.")
        .labelNames("role").register(registry).labels("client");
    private final Map<NativeClient.Outcome, Counter.Child> finished = new EnumMap<>(NativeClient.Outcome.class);
    private final Histogram.Child latency = Histogram.build().name("mock_client_request_seconds").help("Client call lifetime.")
        .labelNames("role").register(registry).labels("client");
    private final Histogram.Child ttft = Histogram.build().name("mock_client_ttft_seconds").help("Call start to first received token.")
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
    }
    String scrape() {
        var out = new StringWriter();
        try { TextFormat.write004(out, registry.metricFamilySamples()); }
        catch (IOException impossible) { throw new UncheckedIOException(impossible); }
        return out.toString();
    }
}

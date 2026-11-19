package org.flexlb.service.lease;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.config.PrefillLeaseConfig;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 * Reads the existing TTFT histogram sum/count from KMon's PromQL query API.
 * No node mean or readiness count is used as a weight.
 */
@Component
public final class PrefillTtftMetricReader {
    private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
    private final ObjectMapper mapper = new ObjectMapper();

    public record Measurement(double meanMs, double samples, long oldestSampleMs) {
    }

    public Measurement read(PrefillLeaseConfig config, List<String> workers, long nowMs)
            throws IOException, InterruptedException {
        Set<String> nodes = new HashSet<>();
        for (String worker : workers) {
            String node = config.getWorkerMetricNodes().get(worker);
            if (node == null || node.isBlank() || !nodes.add(node)) {
                throw new IllegalArgumentException("Missing or ambiguous metric identity for " + worker);
            }
        }
        if (nodes.isEmpty()) {
            throw new IllegalArgumentException("No serving workers to measure");
        }
        String selector = selector(config, nodes);
        String sum = "{__name__=" + quote(config.getTtftMetric() + "_sum") + "," + selector + "}";
        String count = "{__name__=" + quote(config.getTtftMetric() + "_count") + "," + selector + "}";
        String window = "[" + config.getWindowSeconds() + "s]";
        Map<String, Double> sums =
                query(config, "sum by (lease_node) (" + label(config, "increase(" + sum + window + ")") + ")", nowMs);
        Map<String, Double> counts =
                query(config, "sum by (lease_node) (" + label(config, "increase(" + count + window + ")") + ")", nowMs);
        // Expression result timestamps are query time; use timestamps of the original samples instead.
        Map<String, Double> timestamps =
                query(config, "min by (lease_node) (" + label(config, "timestamp(" + count + ")") + ")", nowMs);
        Map<String, Double> sumTimestamps =
                query(config, "min by (lease_node) (" + label(config, "timestamp(" + sum + ")") + ")", nowMs);
        if (!nodes.equals(sumTimestamps.keySet()) || !nodes.equals(timestamps.keySet())) {
            throw new IllegalArgumentException("Missing original TTFT sample timestamps");
        }
        for (String node : nodes) {
            double countTime = timestamps.get(node);
            double sumTime = sumTimestamps.get(node);
            if (!Double.isFinite(countTime) || !Double.isFinite(sumTime) || countTime * 1000 > nowMs + 1000
                    || sumTime * 1000 > nowMs + 1000) {
                throw new IllegalArgumentException("Invalid original TTFT sample timestamp");
            }
            timestamps.put(node, Math.min(countTime, sumTime));
        }
        return combine(nodes, sums, counts, timestamps, nowMs, config.getMaxMetricAgeSeconds());
    }

    static Measurement combine(Set<String> nodes,
                               Map<String, Double> sums,
                               Map<String, Double> counts,
                               Map<String, Double> timestamps,
                               long nowMs,
                               long maxAgeSeconds) {
        if (!nodes.equals(sums.keySet()) || !nodes.equals(counts.keySet()) || !nodes.equals(timestamps.keySet())) {
            throw new IllegalArgumentException("TTFT metrics do not cover exactly the serving worker set");
        }
        double totalSeconds = 0;
        double totalSamples = 0;
        long oldest = nowMs;
        for (String node : nodes) {
            double seconds = sums.get(node);
            double samples = counts.get(node);
            double sampleTime = timestamps.get(node) * 1000;
            if (!Double.isFinite(seconds) || seconds < 0 || !Double.isFinite(samples) || samples < 0
                    || !Double.isFinite(sampleTime) || sampleTime > nowMs + 1000
                    || nowMs - sampleTime > maxAgeSeconds * 1000.0 || samples == 0 && seconds != 0) {
                throw new IllegalArgumentException("Invalid or stale TTFT metric for " + node);
            }
            totalSeconds += seconds;
            totalSamples += samples;
            oldest = Math.min(oldest, (long) sampleTime);
        }
        if (totalSamples <= 0 || !Double.isFinite(totalSeconds) || !Double.isFinite(totalSamples)) {
            throw new IllegalArgumentException("No valid TTFT samples");
        }
        double meanMs = totalSeconds / totalSamples * 1000;
        if (!Double.isFinite(meanMs)) {
            throw new IllegalArgumentException("Invalid TTFT mean");
        }
        return new Measurement(meanMs, totalSamples, oldest);
    }

    private static String label(PrefillLeaseConfig config, String expression) {
        return "label_replace(" + expression + ",\"lease_node\",\"$1\"," + quote(config.getMetricNodeLabel())
                + ",\"(.+)\")";
    }

    static String selector(PrefillLeaseConfig config, Set<String> nodes) {
        String regex = nodes.stream()
                .sorted()
                .map(PrefillTtftMetricReader::escapeRegex)
                .collect(Collectors.joining("|", "^(", ")$"));
        String result = config.getMetricNodeLabel() + "=~" + quote(regex);
        for (var entry : new TreeMap<>(config.getMetricLabels()).entrySet()) {
            result += "," + entry.getKey() + "=" + quote(entry.getValue());
        }
        return result;
    }

    private static String escapeRegex(String value) {
        return value.replaceAll("([\\\\.\\[\\]{}()*+?^$|])", "\\\\$1");
    }

    private static String quote(String value) {
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r")
                + "\"";
    }

    private Map<String, Double> query(PrefillLeaseConfig config, String expression, long nowMs)
            throws IOException, InterruptedException {
        String base = config.getMetricStoreUrl().replaceAll("/+$", "");
        String url = base + "/api/v1/query?query=" + encode(expression) + "&time=" + nowMs / 1000.0
                + "&version=1.0.0&tenant=" + encode(config.getMetricTenant()) + "&token="
                + encode(config.getMetricToken());
        HttpRequest request = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofMillis(config.getQueryTimeoutMs()))
                .GET()
                .build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new IOException("TTFT metric query returned HTTP " + response.statusCode());
        }
        JsonNode json = mapper.readTree(response.body());
        if (!"success".equals(json.path("status").asText())
                || !"vector".equals(json.path("data").path("resultType").asText())) {
            throw new IOException("TTFT metric query did not return a successful vector");
        }
        Map<String, Double> values = new HashMap<>();
        for (JsonNode item : json.path("data").path("result")) {
            String node = item.path("metric").path("lease_node").asText();
            JsonNode value = item.path("value");
            if (node.isEmpty() || !value.isArray() || value.size() != 2
                    || values.put(node, Double.parseDouble(value.get(1).asText())) != null) {
                throw new IOException("Malformed or duplicate TTFT query result");
            }
        }
        return values;
    }

    private static String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
    }
}

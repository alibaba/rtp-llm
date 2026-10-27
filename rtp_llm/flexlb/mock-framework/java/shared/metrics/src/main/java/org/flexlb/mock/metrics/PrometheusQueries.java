package org.flexlb.mock.metrics;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.math.*;
import java.net.*;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.*;
import org.flexlb.mock.api.engine.EngineMetricsApi;
import org.flexlb.mock.api.metrics.PrometheusApi;
import org.flexlb.mock.api.model.Types.*;
import org.flexlb.mock.runtime.InvocationContext;

/** Fixed-time instant queries. PromQL remains here, not in ordinary case checks. */
public final class PrometheusQueries implements EngineMetricsApi, PrometheusApi, AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper().enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
    private final HttpClient http;
    private final URI endpoint;
    private final String scope;
    private record Window(Instant end, long millis, boolean latest) {}

    public PrometheusQueries(URI base, String job, String engine) {
        if ((!"http".equals(base.getScheme()) && !"https".equals(base.getScheme())) || base.getHost() == null
                || base.getRawQuery() != null || base.getRawFragment() != null || base.getUserInfo() != null
                || job == null || job.isBlank() || engine == null || engine.isBlank())
            throw new IllegalArgumentException("Prometheus needs an HTTP(S) base URL, nonempty job and engine");
        endpoint = URI.create(base.toString().replaceAll("/+$", "") + "/api/v1/query");
        scope = "job=" + quote(job) + ",engine=" + quote(engine);
        http = HttpClient.newBuilder().connectTimeout(java.time.Duration.ofSeconds(5)).build();
    }

    /** Optional Mock dotted configuration; malformed supplied configuration is never ignored. */
    public static PrometheusQueries fromConfig(JsonNode config, String engine) {
        if (config.isMissingNode()) return null;
        var fields = new HashSet<String>(); config.fieldNames().forEachRemaining(fields::add);
        if (!config.isObject() || !fields.equals(Set.of("url", "job"))
                || !config.path("url").isTextual() || !config.path("job").isTextual())
            throw new IllegalArgumentException("prometheus requires exactly url and job");
        return new PrometheusQueries(URI.create(config.get("url").textValue()), config.get("job").textValue(), engine);
    }

    @Override public QueryResult query(Promql args) {
        var context = InvocationContext.current();
        if (context.observation() != null) throw new IllegalArgumentException("Raw PromQL has no outer range");
        return instant(args.promql(), context.queryTime());
    }

    @Override public MetricValue hitRate(CacheRange args) {
        var window = window(args.range());
        String filter = scope + ",role=\"prefill\"";
        if (args.engines() != null) {
            if (args.engines().instances().isEmpty()) throw new IllegalArgumentException("Engine selection is empty");
            var ids = args.engines().instances().stream().map(EngineRef::id).distinct().sorted().map(PrometheusQueries::regex).toList();
            filter += ",node=~" + quote(String.join("|", ids));
        }
        String query = aggregate(EngineMetrics.CACHE_HITS, filter, window) + " / "
            + aggregate(EngineMetrics.CACHE_LOOKUPS, filter, window);
        return metric(query, window, "ratio");
    }

    @Override public MetricValue prefillTps(EngineRange args) { return prefill(args.range(), "compute"); }
    @Override public MetricValue cachedTps(EngineRange args) { return prefill(args.range(), "input"); }
    private MetricValue prefill(TimeRange range, String kind) {
        var window = window(range);
        String filter = scope + ",role=\"prefill\"";
        return metric(aggregate(EngineMetrics.FORWARD_TOKENS, filter + ",kind=" + quote(kind), window)
            + " / " + aggregate(EngineMetrics.FORWARD_SECONDS + "_sum", filter, window), window, "tokens/s");
    }

    @Override public MetricValue decodeTps(EngineRange args) {
        var window = window(args.range());
        // Output throughput uses elapsed wall time; prefill compute TPS uses forward time.
        String query = aggregate(EngineMetrics.FORWARD_TOKENS, scope + ",role=\"decode\",kind=\"output\"", window);
        if (!window.latest()) query += " / " + BigDecimal.valueOf(window.millis(), 3).toPlainString();
        return metric(query, window, "tokens/s");
    }

    private MetricValue metric(String expression, Window window, String unit) {
        var samples = instant(expression, window.end()).samples();
        if (samples.size() != 1) throw new IllegalStateException("Metric query must return one finite sample; found " + samples.size());
        return new MetricValue(samples.getFirst().value(), unit);
    }
    private static String aggregate(String name, String labels, Window window) {
        return "sum(" + (window.latest() ? "irate" : "increase") + "(" + name + "{" + labels + "}[" + window.millis() + "ms]))";
    }

    private static Window window(TimeRange range) {
        var context = InvocationContext.current();
        var observed = context.observation();
        if (observed == null) throw new IllegalArgumentException("Missing resolved observation range");
        if (range instanceof Latest) {
            if (!"latest".equals(observed.get("selector")) || !(observed.get("at") instanceof String at)
                    || !context.queryTime().equals(Instant.parse(at)))
                throw new IllegalArgumentException("Latest observation does not match invocation time");
            // Latest completed scrape interval, not an average over the whole lookback.
            return new Window(context.queryTime(), 60_000, true);
        }
        if (!(range instanceof Interval interval) || !List.of(interval.start().value(), interval.end().value()).equals(observed.get("selector"))
                || !(observed.get("start") instanceof Number start) || !(observed.get("end") instanceof Number end))
            throw new IllegalArgumentException("Invalid resolved Engine observation range");
        long first = epochMillis(start), last = epochMillis(end);
        long millis = Math.subtractExact(last, first);
        if (millis <= 0 || last > context.queryTime().toEpochMilli())
            throw new IllegalArgumentException("Observation must cover at least 1ms and cannot end in the future");
        return new Window(Instant.ofEpochMilli(last), millis, false);
    }

    private QueryResult instant(String expression, Instant at) {
        if (expression == null || expression.isBlank()) throw new IllegalArgumentException("PromQL must not be blank");
        var context = InvocationContext.current();
        if (context.timeoutMs() <= 0) throw new IllegalArgumentException("Query budget must be positive");
        String body = "query=" + encode(expression) + "&time=" + encode(at.toString()) + "&timeout=" + context.timeoutMs() + "ms";
        var request = HttpRequest.newBuilder(endpoint).timeout(java.time.Duration.ofMillis(context.timeoutMs()))
            .header("Content-Type", "application/x-www-form-urlencoded").POST(HttpRequest.BodyPublishers.ofString(body)).build();
        try {
            var response = http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() != 200) throw new IllegalStateException("Prometheus HTTP " + response.statusCode());
            var root = JSON.readTree(response.body());
            if (!root.path("status").asText().equals("success")) throw new IllegalStateException("Prometheus query failed: " + root.path("error").asText());
            if (root.has("warnings") && (!root.get("warnings").isArray() || !root.get("warnings").isEmpty()))
                throw new IllegalStateException("Prometheus returned query warnings; refusing partial results");
            var data = root.path("data");
            var result = data.path("result");
            var samples = new ArrayList<QuerySample>();
            if (data.path("resultType").asText().equals("scalar")) samples.add(sample(Map.of(), result));
            else if (data.path("resultType").asText().equals("vector") && result.isArray()) {
                for (var value : result) {
                    if (!value.path("metric").isObject()) throw new IllegalStateException("Invalid Prometheus labels");
                    var labels = new TreeMap<String, String>();
                    var fields = value.get("metric").fields();
                    while (fields.hasNext()) {
                        var field = fields.next();
                        if (!field.getValue().isTextual()) throw new IllegalStateException("Invalid Prometheus label value");
                        labels.put(field.getKey(), field.getValue().textValue());
                    }
                    samples.add(sample(Collections.unmodifiableMap(labels), value.path("value")));
                }
            } else throw new IllegalStateException("PromQL must return a scalar or float instant vector");
            return new QueryResult(List.copyOf(samples), context.queryTime().toString());
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt(); throw new IllegalStateException("Prometheus query interrupted", error);
        } catch (IOException error) { throw new UncheckedIOException("Prometheus query failed", error); }
    }

    private static QuerySample sample(Map<String, String> labels, JsonNode value) {
        if (!value.isArray() || value.size() != 2 || !value.get(0).isNumber() || !value.get(1).isTextual())
            throw new IllegalStateException("Invalid Prometheus float sample");
        double number = Double.parseDouble(value.get(1).textValue());
        if (!Double.isFinite(number)) throw new IllegalStateException("Prometheus returned a nonfinite sample (no observations or zero denominator)");
        return new QuerySample(labels, number, Instant.ofEpochMilli(epochMillis(value.get(0).numberValue())).toString());
    }
    private static long epochMillis(Number seconds) {
        return new BigDecimal(seconds.toString()).movePointRight(3).setScale(0, RoundingMode.FLOOR).longValueExact();
    }
    private static String encode(String text) { return URLEncoder.encode(text, StandardCharsets.UTF_8); }
    private static String quote(String text) {
        try { return JSON.writeValueAsString(text); }
        catch (IOException impossible) { throw new UncheckedIOException(impossible); }
    }
    private static String regex(String text) {
        var result = new StringBuilder();
        for (char ch : text.toCharArray()) {
            if ("\\.+*?()|[]{}^$".indexOf(ch) >= 0) result.append('\\');
            result.append(ch);
        }
        return result.toString();
    }
    @Override public void close() { http.close(); }
}

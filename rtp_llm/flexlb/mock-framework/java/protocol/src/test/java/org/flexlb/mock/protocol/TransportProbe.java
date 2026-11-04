package org.flexlb.mock.protocol;

import java.net.InetSocketAddress;
import java.nio.file.*;
import java.time.Instant;
import java.util.*;
import org.flexlb.mock.api.*;

/** Test-only transport peer. Does not simulate Master, Engine, traffic or Prometheus. */
public final class TransportProbe {
    public record Request(@Argument(optional=true, minimum=0) Long delay_ms,
                          @Argument(optional=true) Double value,
                          @Argument(optional=true) Boolean empty,
                          @Argument(optional=true) Boolean invalid_result,
                          @Argument(optional=true) Boolean crash) {}
    public record Echo(String query_time, String run, String observation) {}
    public record None() {}
    public record MasterRef(String id) {}
    public record MasterSet(List<MasterRef> instances) {}
    public record PerMasterEngines(Map<String, List<String>> instances) {}
    public record MasterReceipt(MasterSet affected) {}
    public record MetricValue(double value, String unit) {}
    public record QuerySample(Map<String, String> labels, double value, String sample_time) {}
    public record QueryResult(List<QuerySample> samples, String query_time) {}
    public record EngineRange(@Argument(ranges={"latest", "interval"}) TimeRange range) {}
    public record Prepare(
        @Argument(optional=true) OutputTokens output_tokens,
        @Argument(optional=true) Integer priority) {}

    public static final class Implementation {
        public MasterReceipt action(Request args) {
            delay(args);
            return new MasterReceipt(new MasterSet(List.of(new MasterRef("master-0"))));
        }
        public MasterSet masters(None args) {
            return new MasterSet(List.of(new MasterRef("master-0")));
        }
        public MetricValue metric(Request args) {
            delay(args);
            return new MetricValue(Boolean.TRUE.equals(args.invalid_result()) ? Double.NaN : args.value() == null ? 0.25 : args.value(), "s");
        }
        public QueryResult query(Request args, ControlServer.Context context) {
            delay(args);
            return new QueryResult(Boolean.TRUE.equals(args.empty()) ? List.of() : List.of(
                new QuerySample(Map.of("node", "a"), 1, context.queryTime().toString()),
                new QuerySample(Map.of("node", "b"), 2, context.queryTime().toString())), context.queryTime().toString());
        }
        public PerMasterEngines engines(None args) {
            var engines = List.of("127.0.0.1:9002", "127.0.0.1:9001");
            return new PerMasterEngines(Map.of("master-0", engines, "master-1", engines));
        }
        public Echo echo(EngineRange args, ControlServer.Context context) {
            return new Echo(context.queryTime().toString(), context.run(), String.valueOf(context.observation()));
        }
        public MetricValue rangeMetric(EngineRange args, ControlServer.Context context) {
            var observation = context.observation();
            double start = ((Number) observation.get("start")).doubleValue();
            double end = ((Number) observation.get("end")).doubleValue();
            return new MetricValue(end - start, "s");
        }
        public None prepare(Prepare args) { return new None(); }
        private void delay(Request args) {
            if (Boolean.TRUE.equals(args.crash())) System.exit(17);
            if (args.delay_ms() != null) {
                try { Thread.sleep(args.delay_ms()); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IllegalStateException("Interrupted", error); }
            }
        }
    }

    static ControlServer server() throws Exception {
        var probe = new Implementation();
        return new ControlServer(new InetSocketAddress("127.0.0.1", 0))
            .route("probe/action", Request.class, MasterReceipt.class, true, (args, context) -> probe.action(args))
            .route("probe/masters", None.class, MasterSet.class, probe::masters)
            .route("probe/metric", Request.class, MetricValue.class, probe::metric)
            .route("probe/query", Request.class, QueryResult.class, false, probe::query)
            .route("probe/engines", None.class, PerMasterEngines.class, probe::engines)
            .route("probe/echo", EngineRange.class, Echo.class, false, probe::echo)
            .route("probe/range_metric", EngineRange.class, MetricValue.class, false, probe::rangeMetric)
            .route("probe/prepare", Prepare.class, None.class, probe::prepare)
            .start();
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 1 && args[0].equals("exit")) System.exit(19);
        try (var server = server()) {
            Runtime.getRuntime().addShutdownHook(new Thread(server::close));
            server.publishReady(Path.of(System.getenv("MOCK_READY_FILE")));
            Thread.currentThread().join();
        }
    }
}

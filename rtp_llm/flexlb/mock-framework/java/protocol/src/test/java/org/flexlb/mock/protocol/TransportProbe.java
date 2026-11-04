package org.flexlb.mock.protocol;

import java.net.InetSocketAddress;
import java.nio.file.*;
import java.time.Instant;
import java.util.*;
import org.flexlb.mock.api.*;
import org.flexlb.mock.api.model.Types;

/** Test-only transport peer. Does not simulate Master, Engine, traffic or Prometheus. */
public final class TransportProbe {
    public record Request(@Argument(optional=true, minimum=0) Long delay_ms,
                          @Argument(optional=true) Double value,
                          @Argument(optional=true) Boolean empty,
                          @Argument(optional=true) Boolean invalid_result,
                          @Argument(optional=true) Boolean crash) {}
    public record Echo(String query_time, String run, String observation) {}
    public static final class Implementation {
        public Types.MasterReceipt action(Request args) {
            delay(args);
            return new Types.MasterReceipt(new Types.MasterSet(List.of(new Types.MasterRef("master-0"))));
        }
        public Types.MasterSet masters(Types.None args) {
            return new Types.MasterSet(List.of(new Types.MasterRef("master-0")));
        }
        public Types.MetricValue metric(Request args) {
            delay(args);
            return new Types.MetricValue(Boolean.TRUE.equals(args.invalid_result()) ? Double.NaN : args.value() == null ? 0.25 : args.value(), "s");
        }
        public Types.QueryResult query(Request args, ControlServer.Context context) {
            delay(args);
            return new Types.QueryResult(Boolean.TRUE.equals(args.empty()) ? List.of() : List.of(
                new Types.QuerySample(Map.of("node", "a"), 1, context.queryTime().toString()),
                new Types.QuerySample(Map.of("node", "b"), 2, context.queryTime().toString())), context.queryTime().toString());
        }
        public Types.PerMasterEngines engines(Types.None args) {
            var engines = List.of("127.0.0.1:9002", "127.0.0.1:9001");
            return new Types.PerMasterEngines(Map.of("master-0", engines, "master-1", engines));
        }
        public Echo echo(Types.EngineRange args, ControlServer.Context context) {
            return new Echo(context.queryTime().toString(), context.run(), String.valueOf(context.observation()));
        }
        public Types.MetricValue rangeMetric(Types.EngineRange args, ControlServer.Context context) {
            var observation = context.observation();
            double start = ((Number) observation.get("start")).doubleValue();
            double end = ((Number) observation.get("end")).doubleValue();
            return new Types.MetricValue(end - start, "s");
        }
        public Types.None prepare(Types.Prepare args) { return new Types.None(); }
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
            .route("probe/action", Request.class, Types.MasterReceipt.class, true, (args, context) -> probe.action(args))
            .route("probe/masters", Types.None.class, Types.MasterSet.class, probe::masters)
            .route("probe/metric", Request.class, Types.MetricValue.class, probe::metric)
            .route("probe/query", Request.class, Types.QueryResult.class, false, probe::query)
            .route("probe/engines", Types.None.class, Types.PerMasterEngines.class, probe::engines)
            .route("probe/echo", Types.EngineRange.class, Echo.class, false, probe::echo)
            .route("probe/range_metric", Types.EngineRange.class, Types.MetricValue.class, false, probe::rangeMetric)
            .route("probe/prepare", Types.Prepare.class, Types.None.class, probe::prepare)
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

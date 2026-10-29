package org.flexlb.mock.runtime;

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
    public interface Api {
        @Endpoint(value="probe/action", kind=Endpoint.Kind.action, background=true)
        Types.MasterReceipt action(Request args);
        @Endpoint(value="probe/masters", kind=Endpoint.Kind.read)
        Types.MasterSet masters(Types.None args);
        @Endpoint(value="probe/metric", kind=Endpoint.Kind.read, unit="s")
        Types.MetricValue metric(Request args);
        @Endpoint(value="probe/query", kind=Endpoint.Kind.read)
        Types.QueryResult query(Request args);
        @Endpoint(value="probe/engines", kind=Endpoint.Kind.read)
        Types.PerMasterEngines engines(Types.None args);
        @Endpoint(value="probe/echo", kind=Endpoint.Kind.read)
        Echo echo(Types.EngineRange args);
        @Endpoint(value="probe/range_metric", kind=Endpoint.Kind.read, unit="s")
        Types.MetricValue rangeMetric(Types.EngineRange args);
        @Endpoint(value="probe/prepare", kind=Endpoint.Kind.action)
        Types.None prepare(Types.Prepare args);
    }

    public static final class Implementation implements Api {
        @Override public Types.MasterReceipt action(Request args) {
            delay(args);
            return new Types.MasterReceipt(new Types.MasterSet(List.of(new Types.MasterRef("master-0"))));
        }
        @Override public Types.MasterSet masters(Types.None args) {
            return new Types.MasterSet(List.of(new Types.MasterRef("master-0")));
        }
        @Override public Types.MetricValue metric(Request args) {
            delay(args);
            return new Types.MetricValue(Boolean.TRUE.equals(args.invalid_result()) ? Double.NaN : args.value() == null ? 0.25 : args.value(), "s");
        }
        @Override public Types.QueryResult query(Request args) {
            delay(args);
            var context = InvocationContext.current();
            return new Types.QueryResult(Boolean.TRUE.equals(args.empty()) ? List.of() : List.of(
                new Types.QuerySample(Map.of("node", "a"), 1, context.queryTime().toString()),
                new Types.QuerySample(Map.of("node", "b"), 2, context.queryTime().toString())), context.queryTime().toString());
        }
        @Override public Types.PerMasterEngines engines(Types.None args) {
            var engines = List.of("127.0.0.1:9002", "127.0.0.1:9001");
            return new Types.PerMasterEngines(Map.of("master-0", engines, "master-1", engines));
        }
        @Override public Echo echo(Types.EngineRange args) {
            var context = InvocationContext.current();
            return new Echo(context.queryTime().toString(), context.run(), String.valueOf(context.observation()));
        }
        @Override public Types.MetricValue rangeMetric(Types.EngineRange args) {
            var observation = InvocationContext.current().observation();
            double start = ((Number) observation.get("start")).doubleValue();
            double end = ((Number) observation.get("end")).doubleValue();
            return new Types.MetricValue(end - start, "s");
        }
        @Override public Types.None prepare(Types.Prepare args) { return new Types.None(); }
        private void delay(Request args) {
            if (Boolean.TRUE.equals(args.crash())) System.exit(17);
            if (args.delay_ms() != null) {
                try { Thread.sleep(args.delay_ms()); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IllegalStateException("Interrupted", error); }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 1 && args[0].equals("exit")) System.exit(19);
        try (var server = new ApiServer(new InetSocketAddress("127.0.0.1", 0)).bind(Api.class, new Implementation()).start()) {
            Runtime.getRuntime().addShutdownHook(new Thread(server::close));
            server.publishReady(Path.of(System.getenv("MOCK_READY_FILE")));
            Thread.currentThread().join();
        }
    }
}

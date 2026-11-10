package org.flexlb.mock.protocol;

import java.net.InetSocketAddress;
import java.nio.file.*;
import java.util.*;
import org.flexlb.mock.protocol.ControlServer.OptionalArgument;

/**
 * Starts a minimal control peer for transport tests.
 */
public final class TransportProbe {
    public static void main(String[] args) throws Exception {
        if (args.length == 1 && args[0].equals("exit")) System.exit(19);
        try (var server = server()) {
            Runtime.getRuntime().addShutdownHook(new Thread(server::close));
            server.publishReady(Path.of(System.getenv("MOCK_READY_FILE")));
            Thread.currentThread().join();
        }
    }

    static ControlServer server() throws Exception {
        var probe = new Implementation();
        return new ControlServer(new InetSocketAddress("127.0.0.1", 0))
            .route("probe/action", Request.class, MasterReceipt.class, probe::action)
            .route("probe/masters", None.class, MasterSet.class, probe::masters)
            .route("probe/metric", Request.class, MetricValue.class, probe::metric)
            .route("probe/engines", None.class, PerMasterEngines.class, probe::engines)
            .start();
    }

    public record Request(@OptionalArgument Long delay_ms,
                          @OptionalArgument Double value,
                          @OptionalArgument Boolean empty,
                          @OptionalArgument Boolean invalid_result,
                          @OptionalArgument Boolean crash) {}

    public record None() {}

    public record MasterRef(String id) {}

    public record MasterSet(List<MasterRef> instances) {}

    public record PerMasterEngines(Map<String, List<String>> instances) {}

    public record MasterReceipt(MasterSet affected) {}

    public record MetricValue(double value, String unit) {}

    /**
     * Supplies deterministic responses and failures for the control test peer.
     */
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
        public PerMasterEngines engines(None args) {
            var engines = List.of("127.0.0.1:9002", "127.0.0.1:9001");
            return new PerMasterEngines(Map.of("master-0", engines, "master-1", engines));
        }
        private void delay(Request args) {
            if (Boolean.TRUE.equals(args.crash())) System.exit(17);
            if (args.delay_ms() != null) {
                try { Thread.sleep(args.delay_ms()); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IllegalStateException("Interrupted", error); }
            }
        }
    }
}

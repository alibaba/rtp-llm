package org.flexlb.mock.host;

import java.nio.file.*;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

/** One P/D/V engine per JVM; Whale owns the Pod lifecycle. */
public final class EngineMain {
    public static void main(String[] args) throws Exception {
        if (args.length != 0) throw new IllegalArgumentException("Use instance configuration variables");
        Path configuration = Path.of(required("MOCK_CONFIG_FILE"));
        Path output = Path.of(required("MOCK_OUTPUT_DIR"));
        Path ready = Path.of(required("MOCK_READY_FILE"));
        int rpcPort = port("MOCK_RPC_PORT"), controlPort = port("MOCK_CONTROL_PORT");
        if (rpcPort != 0 && rpcPort == controlPort)
            throw new IllegalArgumentException("RPC and control ports must differ");
        Files.createDirectories(output);
        try (var node = EngineInstances.start(configuration, output, rpcPort, controlPort)) {
            var hook = new Thread(() -> {
                try { node.close(); } finally { deleteReady(ready); }
            }, "engine-shutdown");
            Runtime.getRuntime().addShutdownHook(hook);
            try {
                EngineInstances.publish(output.resolve("endpoints.json"), node.endpoints());
                EngineInstances.publish(ready, Map.of("url", node.endpoints().get("control")));
                new CountDownLatch(1).await();
            } finally {
                deleteReady(ready);
                try { Runtime.getRuntime().removeShutdownHook(hook); } catch (IllegalStateException stopping) { }
            }
        }
    }

    private static int port(String name) {
        int value = Integer.parseInt(System.getenv().getOrDefault(name, "0"));
        if (value < 0 || value > 65535) throw new IllegalArgumentException("Invalid " + name);
        return value;
    }
    private static String required(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) throw new IllegalArgumentException("Missing " + name);
        return value;
    }
    private static void deleteReady(Path ready) {
        try { Files.deleteIfExists(ready); }
        catch (java.io.IOException error) { throw new IllegalStateException("Cannot remove Engine readiness", error); }
    }
}

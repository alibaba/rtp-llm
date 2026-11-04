package org.flexlb.mock.engine;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Foreground single-engine service. The platform owns discovery and process replacement. */
public final class EngineMain {
    public static void main(String[] args) throws Exception {
        if (args.length != 0) throw new IllegalArgumentException("Use instance configuration variables");
        int port = Integer.parseInt(System.getenv().getOrDefault("START_PORT", "8088"));
        if (port < 1 || port > 65534) throw new IllegalArgumentException("START_PORT must be in 1..65534");
        String configuration = System.getenv("MOCK_CONFIG_FILE");
        String readyFile = System.getenv("MOCK_READY_FILE");
        Path ready = readyFile == null ? null : Path.of(readyFile);
        if (ready != null && configuration == null)
            throw new IllegalArgumentException("Local Engine launch requires MOCK_CONFIG_FILE");
        int rpcPort = Integer.parseInt(System.getenv().getOrDefault("MOCK_RPC_PORT", Integer.toString(port + 1)));
        var started = configuration == null
            ? EngineInstances.startEnvironment(System.getenv(), port + 1, port)
            : EngineInstances.start(Path.of(configuration), rpcPort, port);
        var stopped = new CountDownLatch(1);
        try (var node = started.node()) {
            var hook = new Thread(() -> {
                stopped.countDown();
                node.close();
            }, "engine-shutdown");
            Runtime.getRuntime().addShutdownHook(hook);
            try {
                node.checkHealthy();
                System.out.println(new ObjectMapper().writeValueAsString(Map.of(
                    "event", "engine_started", "endpoints", node.endpoints(), "effective_config", started.effective())));
                if (ready != null) {
                    Path output = Path.of(System.getenv("MOCK_OUTPUT_DIR"));
                    publish(output.resolve("effective-config.json"), started.effective());
                    publish(output.resolve("endpoints.json"), node.endpoints());
                    publish(ready, Map.of("url", node.endpoints().get("control")));
                }
                while (!stopped.await(1, TimeUnit.SECONDS)) node.checkHealthy();
            } finally {
                if (ready != null) Files.deleteIfExists(ready);
                try { Runtime.getRuntime().removeShutdownHook(hook); } catch (IllegalStateException stopping) { }
            }
        }
    }
    private static void publish(Path target, Object value) throws Exception {
        Path temporary = Files.createTempFile(target.toAbsolutePath().getParent(), ".publish-", ".json");
        try {
            new ObjectMapper().writeValue(temporary.toFile(), value);
            Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Files.deleteIfExists(temporary);
        }
    }
}

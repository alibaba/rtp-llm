package org.flexlb.mock.host;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
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
        var started = configuration == null
            ? EngineInstances.startEnvironment(System.getenv(), port + 1, port)
            : EngineInstances.start(Path.of(configuration), port + 1, port);
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
                while (!stopped.await(1, TimeUnit.SECONDS)) node.checkHealthy();
            } finally {
                try { Runtime.getRuntime().removeShutdownHook(hook); } catch (IllegalStateException stopping) { }
            }
        }
    }
}

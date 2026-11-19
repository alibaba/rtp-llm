package org.flexlb.mock.engine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.execution.LLMMockEngine;
import org.flexlb.mock.engine.vit.VitConfig;
import org.flexlb.mock.engine.vit.VitNode;
import org.flexlb.mock.protocol.ConfigInput;
import org.flexlb.mock.protocol.JsonFiles;

/**
 * Foreground single-engine service. The platform owns discovery and process replacement.
 */
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
        if (rpcPort < 1 || rpcPort > 65535 || rpcPort == port)
            throw new IllegalArgumentException("Engine listeners must use distinct valid ports");
        if (ready != null) {
            Path output = Path.of(System.getenv("MOCK_OUTPUT_DIR"));
            for (Path artifact : java.util.List.of(ready, output.resolve("effective-config.json"), output.resolve("endpoints.json")))
                if (Files.exists(artifact)) throw new IllegalArgumentException("Engine startup artifact already exists: " + artifact);
        }
        var started = configuration == null
            ? startEnvironment(System.getenv(), port + 1, port)
            : start(ConfigInput.read(Path.of(configuration)), rpcPort, port);
        var stopped = new CountDownLatch(1);
        try (var node = started) {
            var hook = new Thread(() -> {
                try { node.close(); }
                finally { stopped.countDown(); }
            }, "engine-shutdown");
            Runtime.getRuntime().addShutdownHook(hook);
            try {
                node.health().run();
                var endpoints = node.endpoints().get();
                System.out.println(new ObjectMapper().writeValueAsString(Map.of(
                    "event", "engine_started", "endpoints", endpoints, "effective_config", started.effective())));
                if (ready != null) {
                    Path output = Path.of(System.getenv("MOCK_OUTPUT_DIR"));
                    JsonFiles.writeAtomically(output.resolve("effective-config.json"), started.effective());
                    JsonFiles.writeAtomically(output.resolve("endpoints.json"), endpoints);
                    JsonFiles.writeAtomically(ready, Map.of("url", endpoints.get("control")));
                }
                while (!stopped.await(1, TimeUnit.SECONDS)) node.health().run();
            } finally {
                if (ready != null) Files.deleteIfExists(ready);
                try { Runtime.getRuntime().removeShutdownHook(hook); } catch (IllegalStateException stopping) { }
            }
        }
    }

    private static Started startEnvironment(Map<String, String> env, int rpcPort, int controlPort) throws Exception {
        String role = env.getOrDefault("ROLE_TYPE", "").toLowerCase(Locale.ROOT);
        String id = env.getOrDefault("MOCK_NODE_ID", role + "-" + Address.local().advertise());
        if (id == null || !id.startsWith(role + "-")) throw new IllegalArgumentException("Missing/invalid MOCK_NODE_ID for ROLE_TYPE");
        if (role.equals("vit")) {
            return startVit(VitConfig.fromEnvironment(id, env), rpcPort, controlPort);
        }
        if (!role.equals("prefill") && !role.equals("decode")) throw new IllegalArgumentException("Unknown ROLE_TYPE: " + role);
        return startRtp(RtpConfig.fromEnvironment(id, Role.valueOf(role), env), rpcPort, controlPort);
    }

    private static Started startVit(VitConfig config, int rpcPort, int controlPort) throws Exception {
        var node = VitNode.start(config, rpcPort, controlPort);
        return new Started(node::close, node::checkHealthy, node::endpoints, config.effective());
    }

    private static Started startRtp(RtpConfig config, int rpcPort, int controlPort) throws Exception {
        var node = LLMMockEngine.start(config, rpcPort, controlPort);
        return new Started(node::close, node::checkHealthy, node::endpoints, config.effective());
    }

    private static Started start(JsonNode root, int rpcPort, int controlPort) throws Exception {
        String role = root.path("component_role").asText();
        if (role.isBlank() || !root.path("id").asText().startsWith(role + "-"))
            throw new IllegalArgumentException("Instance identity differs from role");
        if (role.equals("vit")) {
            return startVit(VitConfig.load(root), rpcPort, controlPort);
        }
        if (!role.equals("prefill") && !role.equals("decode"))
            throw new IllegalArgumentException("Unknown Engine role: " + role);
        return startRtp(RtpConfig.load(root), rpcPort, controlPort);
    }

    /**
     * Owns the Engine listeners and startup artifacts until the process stops.
     */
    private record Started(Runnable stop, Runnable health, Supplier<Map<String, Object>> endpoints,
                           JsonNode effective) implements AutoCloseable {
        @Override public void close() { stop.run(); }
    }

    /**
     * Parses the host and port used by an Engine listener.
     */
    public record Address(String bind, String advertise) {
        public static Address local() {
            String bind=System.getenv().getOrDefault("MOCK_BIND_HOST","127.0.0.1");
            String advertise=System.getenv("MOCK_ADVERTISE_HOST");
            if (advertise == null) {
                if (bind.equals("0.0.0.0") || bind.equals("::")) {
                    try { advertise = java.net.InetAddress.getLocalHost().getHostAddress(); }
                    catch (java.net.UnknownHostException error) { throw new IllegalStateException("Cannot determine Engine advertise address", error); }
                } else advertise = bind;
            }
            if(bind.isBlank() || advertise.isBlank() || advertise.equals("0.0.0.0") || advertise.equals("::") || advertise.contains("/"))
                throw new IllegalArgumentException("Invalid Engine bind/advertise host");
            return new Address(bind,advertise);
        }

        public InetSocketAddress listener(int port) { return new InetSocketAddress(bind,port); }

        public String http(int port) { return "http://"+rpc(port); }

        public String rpc(int port) { return (advertise.contains(":")?"["+advertise+"]":advertise)+":"+port; }
    }
}

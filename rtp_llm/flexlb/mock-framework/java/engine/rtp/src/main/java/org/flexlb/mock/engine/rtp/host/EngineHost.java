package org.flexlb.mock.engine.rtp.host;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.InetSocketAddress;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import org.flexlb.mock.api.model.Types.None;
import org.flexlb.mock.engine.rtp.*;
import org.flexlb.mock.runtime.ApiServer;

/** Owns Engine objects, never their scheduling or cluster membership policy. */
public final class EngineHost implements EngineHostApi, AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final Path directory;
    private final Map<String, RtpNode> nodes = new LinkedHashMap<>();
    private final Set<String> used = new HashSet<>(), addresses = new HashSet<>();
    private boolean closed;
    public EngineHost(Path directory) { this.directory = directory.toAbsolutePath().normalize(); }

    @Override public synchronized Instance create(Create args) throws Exception {
        if (closed || nodes.size() >= 1024) throw new IllegalStateException("Host closed or full");
        if (!used.add(args.id())) throw new IllegalArgumentException("Instance id has already been used");
        var output = directory.resolve(args.id());
        var config = RtpConfig.load(output.resolve("config.json"));
        if (!config.id().equals(args.id()) || !args.id().startsWith(config.role().name() + "-"))
            throw new IllegalArgumentException("Instance identity differs from configuration");
        var unavailable = new HashSet<>(args.unavailable());
        unavailable.addAll(addresses);
        for (int attempt = 0; attempt < 32; attempt++) {
            var node = RtpNode.start(config);
            if (unavailable.contains(node.rpcAddress())) { node.close(); continue; }
            try {
                write(output.resolve("effective-config.json"), config.effective());
                write(output.resolve("endpoints.json"), node.endpoints());
                write(output.resolve("ready.json"), Map.of("url", node.endpoints().get("control")));
                nodes.put(args.id(), node);
                addresses.add(node.rpcAddress());
                return instance(node);
            } catch (Exception error) {
                node.close();
                Files.deleteIfExists(output.resolve("ready.json"));
                throw error;
            }
        }
        throw new IllegalStateException("Cannot allocate a fresh Engine service address");
    }
    @Override public synchronized Instance stop(Stop args) throws Exception {
        var node = nodes.get(args.id());
        if (node == null) throw new IllegalArgumentException("Unknown active instance");
        var result = instance(node);
        node.close();
        Files.deleteIfExists(directory.resolve(args.id()).resolve("ready.json"));
        nodes.remove(args.id());
        return result;
    }
    @Override public synchronized List<Instance> status(None args) {
        if (closed) throw new IllegalStateException("Host closed");
        return nodes.values().stream().map(node -> { node.checkHealthy(); return instance(node); }).toList();
    }
    private static Instance instance(RtpNode node) {
        var e = node.endpoints();
        return new Instance((String)e.get("id"), (String)e.get("role"), (String)e.get("rpc"),
            (String)e.get("control"), (String)e.get("metrics"));
    }
    private static void write(Path path, Object value) throws Exception {
        var temporary = Files.createTempFile(path.getParent(), ".publish-", ".json");
        try {
            JSON.writeValue(temporary.toFile(), value);
            Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally { Files.deleteIfExists(temporary); }
    }
    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        RuntimeException failed = null;
        for (var entry : nodes.entrySet()) try {
            entry.getValue().close();
            Files.deleteIfExists(directory.resolve(entry.getKey()).resolve("ready.json"));
        } catch (Exception error) {
            if (failed == null) failed = new IllegalStateException("Engine cleanup failed", error);
            else failed.addSuppressed(error);
        }
        nodes.clear();
        if (failed != null) throw failed;
    }
    public static void main(String[] args) throws Exception {
        if (args.length != 0) throw new IllegalArgumentException("Use launcher configuration");
        try (var host = new EngineHost(Path.of(required("MOCK_COMPONENTS_DIR")));
             var api = new ApiServer(new InetSocketAddress("127.0.0.1", 0)).bind(EngineHostApi.class, host).start()) {
            var hook = new Thread(() -> { try { host.close(); } finally { api.close(); } }, "engine-host-close");
            Runtime.getRuntime().addShutdownHook(hook);
            try {
                api.publishReady(Path.of(required("MOCK_READY_FILE")));
                new CountDownLatch(1).await();
            } finally {
                try { Runtime.getRuntime().removeShutdownHook(hook); } catch (IllegalStateException closing) { }
            }
        }
    }
    private static String required(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) throw new IllegalArgumentException("Missing " + name);
        return value;
    }
}

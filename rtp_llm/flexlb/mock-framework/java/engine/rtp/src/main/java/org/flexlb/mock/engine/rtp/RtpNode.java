package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.flexlb.mock.engine.rtp.cache.DeviceCache;
import org.flexlb.mock.runtime.ApiServer;

/** One owned RTP P/D node per JVM. Does not contain the case runner or a cluster controller. */
public final class RtpNode implements AutoCloseable {
    private final RtpConfig config;
    private final DeviceCache cache;
    private final RtpStatusService service;
    private final AtomicBoolean closed = new AtomicBoolean();
    private Server rpc;
    private ApiServer api;

    private RtpNode(RtpConfig config) {
        this.config = config;
        cache = new DeviceCache(config.deviceBlocks(), config.reserveBlocks(), true);
        service = new RtpStatusService(config, cache);
    }

    public static RtpNode start(RtpConfig config) throws IOException {
        var node = new RtpNode(config);
        try {
            node.rpc = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0)).addService(node.service).build().start();
            node.api = new ApiServer(new InetSocketAddress("127.0.0.1", 0)).start();
            return node;
        } catch (IOException | RuntimeException error) { node.close(); throw error; }
    }

    public String rpcAddress() { return "127.0.0.1:" + rpc.getPort(); }
    public DeviceCache cache() { return cache; }
    public Map<String, Object> endpoints() {
        return Map.of("id", config.id(), "role", config.role().name(), "engine", "rtp", "rpc", rpcAddress(),
            "control", api.url(), "supported_rpc", List.of("CheckHealth", "GetWorkerStatus", "GetCacheStatus"));
    }

    @Override public void close() {
        if (!closed.compareAndSet(false, true)) return;
        service.close();
        if (rpc != null) {
            rpc.shutdownNow();
            try { rpc.awaitTermination(3, TimeUnit.SECONDS); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); }
        }
        if (api != null) api.close();
        cache.close();
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 0) throw new IllegalArgumentException("Use runner configuration variables, not legacy CLI flags");
        var config = RtpConfig.load(Path.of(required("MOCK_CONFIG_FILE")));
        Path directory = Path.of(required("MOCK_OUTPUT_DIR"));
        Files.createDirectories(directory);
        try (var node = start(config)) {
            var hook = new Thread(node::close, "rtp-node-shutdown");
            Runtime.getRuntime().addShutdownHook(hook);
            try {
                var json = new ObjectMapper();
                json.writerWithDefaultPrettyPrinter().writeValue(directory.resolve("effective-config.json").toFile(), config.effective());
                json.writerWithDefaultPrettyPrinter().writeValue(directory.resolve("endpoints.json").toFile(), node.endpoints());
                node.api.publishReady(Path.of(required("MOCK_READY_FILE")));
                node.rpc.awaitTermination();
            } finally {
                try { Runtime.getRuntime().removeShutdownHook(hook); } catch (IllegalStateException ignored) { }
            }
        }
    }

    private static String required(String key) {
        var value = System.getenv(key);
        if (value == null || value.isBlank()) throw new IllegalArgumentException("Missing " + key);
        return value;
    }
}

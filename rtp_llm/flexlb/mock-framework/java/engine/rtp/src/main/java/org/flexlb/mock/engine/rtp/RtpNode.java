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
import org.flexlb.mock.metrics.PrometheusQueries;
import org.flexlb.mock.api.engine.EngineMetricsApi;
import org.flexlb.mock.api.metrics.PrometheusApi;

/** One owned RTP P/D node per JVM. Does not contain the case runner or a cluster controller. */
public final class RtpNode implements AutoCloseable {
    private final RtpConfig config;
    private final DeviceCache cache;
    private final RtpRpcService service;
    private final PrometheusQueries queries;
    private final AtomicBoolean closed = new AtomicBoolean();
    private Server rpc;
    private ApiServer api;

    private RtpNode(RtpConfig config) {
        this.config = config;
        queries = PrometheusQueries.fromConfig(config.business().path("prometheus"), "rtp");
        cache = new DeviceCache(config.deviceBlocks(), config.reserveBlocks(), true);
        service = new RtpRpcService(config, cache);
    }

    public static RtpNode start(RtpConfig config) throws IOException {
        var node = new RtpNode(config);
        try {
            node.rpc = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0)).addService(node.service).build().start();
            node.api = new ApiServer(new InetSocketAddress("127.0.0.1", 0)).metrics(node.service::scrapeMetrics);
            if (node.queries != null) node.api.bind(EngineMetricsApi.class, node.queries).bind(PrometheusApi.class, node.queries);
            node.api.start();
            return node;
        } catch (IOException | RuntimeException error) { node.close(); throw error; }
    }

    public String rpcAddress() { return "127.0.0.1:" + rpc.getPort(); }
    public DeviceCache cache() { return cache; }
    public Map<String, Object> endpoints() {
        return Map.of("id", config.id(), "role", config.role().name(), "engine", "rtp", "rpc", rpcAddress(),
            "control", api.url(), "metrics", api.url() + "/metrics", "supported_rpc", config.role() == org.flexlb.mock.api.model.Types.Role.prefill
                ? List.of("CheckHealth", "GetWorkerStatus", "GetCacheStatus", "GenerateStreamCall")
                : List.of("CheckHealth", "GetWorkerStatus", "GetCacheStatus", "RemoteGenerate"),
            "local_generation", config.role() == org.flexlb.mock.api.model.Types.Role.prefill
                ? Map.of("max_new_tokens", 1, "text_only", true) : Map.of(),
            "remote_generation", config.role() == org.flexlb.mock.api.model.Types.Role.decode
                ? Map.of("stages", List.of("ALLOCATE"), "text_only", true, "load", false) : Map.of());
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
        if (queries != null) queries.close();
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

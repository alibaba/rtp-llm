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
    private final RtpRpcService service;
    private final AtomicBoolean closed = new AtomicBoolean();
    private Server rpc;
    private ApiServer api;

    private RtpNode(RtpConfig config) {
        this.config = config;
        cache = new DeviceCache(config.deviceBlocks(), config.reserveBlocks(), config.evictionPolicy());
        service = new RtpRpcService(config, cache);
    }

    public static RtpNode start(RtpConfig config) throws IOException {
        var node = new RtpNode(config);
        try {
            var builder = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0)).addService(node.service);
            var transfer = node.service.transfer();
            if (transfer != null) builder.addService(transfer);
            node.rpc = builder.build().start();
            node.service.rpcAddress(node.rpcAddress());
            node.api = new ApiServer(new InetSocketAddress("127.0.0.1", 0)).metrics(node.service::scrapeMetrics);
            node.api.start();
            return node;
        } catch (IOException | RuntimeException error) { node.close(); throw error; }
    }

    public String rpcAddress() { return "127.0.0.1:" + rpc.getPort(); }
    public void checkHealthy() {
        if (closed.get()) throw new IllegalStateException("Engine closed");
        service.scrapeMetrics();
    }
    public DeviceCache cache() { return cache; }
    public Map<String, Object> endpoints() {
        boolean prefill = config.role() == org.flexlb.mock.api.model.Types.Role.prefill;
        var result = new LinkedHashMap<String, Object>();
        result.put("id", config.id()); result.put("role", config.role().name()); result.put("engine", "rtp");
        result.put("rpc", rpcAddress()); result.put("control", api.url()); result.put("metrics", api.url() + "/metrics");
        result.put("supported_rpc", prefill ? List.of("CheckHealth", "GetWorkerStatus", "GetCacheStatus", "GenerateStreamCall", "EnqueueBatch", "EnqueueGroup", "FetchResponse", "Cancel")
            : List.of("CheckHealth", "GetWorkerStatus", "GetCacheStatus", "RemoteGenerate"));
        result.put("local_generation", prefill ? Map.of("max_new_tokens", 1, "text_only", true) : Map.of());
        result.put("pd_generation", prefill ? Map.of("rpc", "GenerateStreamCall", "decode_target", "role_addrs", "text_only", true) : Map.of());
        result.put("remote_generation", prefill ? Map.of() : Map.of("stages", List.of("ALLOCATE", "LOAD", "GENERATE", "PREFILL_COMPLETE"), "text_only", true, "load", true));
        result.put("mock_transfer", prefill ? Map.of("address", rpcAddress(), "service", "mock.rtp.PrefillTransfer", "metadata_only", true) : Map.of());
        return Map.copyOf(result);
    }

    @Override public void close() {
        if (!closed.compareAndSet(false, true)) return;
        RuntimeException failure = null;
        try { service.close(); } catch (RuntimeException error) { failure = error; }
        try {
            if (rpc != null) {
                rpc.shutdownNow();
                if (!rpc.awaitTermination(3, TimeUnit.SECONDS)) throw new IllegalStateException("Engine RPC did not terminate");
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            if (failure == null) failure = new IllegalStateException("Interrupted closing Engine", error);
            else failure.addSuppressed(error);
        } catch (RuntimeException error) {
            if (failure == null) failure = error; else failure.addSuppressed(error);
        } finally {
            try { if (api != null) api.close(); } finally { cache.close(); }
        }
        if (failure != null) throw failure;
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

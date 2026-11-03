package org.flexlb.mock.engine.rtp;

import org.flexlb.mock.admission.RpcService;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.flexlb.mock.cache.KvCache;
import org.flexlb.mock.cache.KvcmReporter;
import org.flexlb.mock.engine.Scheduler;
import org.flexlb.mock.runtime.ApiServer;

/** One owned RTP P/D object with independent listeners and resources. Does not contain the case runner or a cluster controller. */
public final class RtpNode implements org.flexlb.mock.runtime.EngineInstance {
    private final RtpConfig config;
    private final KvCache cache;
    private final RpcService service;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final org.flexlb.mock.runtime.EngineAddress address = org.flexlb.mock.runtime.EngineAddress.local();
    private Server rpc;
    private ApiServer api;
    private KvcmReporter publisher;

    private RtpNode(RtpConfig config, Scheduler.Factory schedulers) {
        this.config = Objects.requireNonNull(config);
        Objects.requireNonNull(schedulers);
        cache = new KvCache(config.blockSize(), config.deviceBlocks(), config.reserveBlocks(), config.evictionPolicy(),
            config.deviceCache(), config.hostBlocks(), config.hostPolicy(), config.deviceWatermark(), config.hostWatermark());
        try { service = new RpcService(config.endpoint(), cache, schedulers); }
        catch (RuntimeException error) { cache.close(); throw error; }
    }

    public static RtpNode start(RtpConfig config) throws IOException {
        return start(config, 0, 0);
    }

    public static RtpNode start(RtpConfig config, int rpcPort, int controlPort) throws IOException {
        return start(config, rpcPort, controlPort, RtpScheduler.factory(config));
    }

    public static RtpNode start(RtpConfig config, Scheduler.Factory schedulers) throws IOException {
        return start(config, 0, 0, schedulers);
    }

    public static RtpNode start(RtpConfig config, int rpcPort, int controlPort, Scheduler.Factory schedulers) throws IOException {
        var node = new RtpNode(config, schedulers);
        try {
            var builder = NettyServerBuilder.forAddress(node.address.listener(rpcPort)).addService(node.service);
            node.rpc = builder.build().start();
            node.api = new ApiServer(node.address.listener(controlPort)).health(node::checkHealthy).metrics(node.service::scrapeMetrics);
            node.api.start();
            node.publisher = node.startMetadataReporter();
            return node;
        } catch (IOException | RuntimeException error) { node.close(); throw error; }
    }

    private KvcmReporter startMetadataReporter() {
        var values = config.business();
        String type = values.path("kv_cache_event_publisher_type").asText("none");
        if (!Set.of("none", "kvcm").contains(type)) throw new IllegalArgumentException("Invalid kv_cache_event_publisher_type");
        if (type.equals("none") && !RtpConfig.nativeBool(values, "enable_remote_cache", false)) return null;
        String instance = values.path("kv_cache_event_instance_id").asText();
        String host = values.path("kv_cache_event_host_ip_port").asText(address.rpc(rpc.getPort() - 1) + "@0");
        if (instance.isBlank() || host.isBlank())
            throw new IllegalArgumentException("KVCM requires a shared instance_id and a nonempty Master-visible host_ip_port");
        var identity = new KvcmReporter.Identity(
            instance,
            values.path("kv_cache_event_instance_group").asText("default"),
            host,
            values.path("model_name").asText("mock-full-kv"), "FP16", false,
            config.blockSize(), config.tpSize(), config.dpSize(), "rtp_llm_hbm_" + config.blockSize() + "_full");
        int snapshotMillis = RtpConfig.nativeInt(values, "kv_cache_event_snapshot_interval_ms", 600000, 1, Integer.MAX_VALUE);
        String override = values.path("kv_cache_event_manager_endpoint").asText();
        boolean vip = RtpConfig.nativeBool(values, "kvcm_enable_vipserver", false);
        var endpoint = override.isBlank()
            ? org.flexlb.mock.cache.KvcmDiscovery.discover(
                values.path(vip ? "kvcm_vipserver_domain" : "kvcm_server_address").asText(), instance, vip)
            : org.flexlb.mock.cache.KvcmDiscovery.direct(java.net.URI.create(override));
        var settings = new KvcmReporter.Configuration(endpoint, type.equals("kvcm"),
            snapshotMillis);
        return new KvcmReporter(settings, identity, cache);
    }

    public String rpcAddress() { return address.rpc(rpc.getPort()); }
    public void checkHealthy() {
        if (closed.get()) throw new IllegalStateException("Engine closed");
        if (rpc == null || rpc.isShutdown()) throw new IllegalStateException("Engine RPC stopped");
        service.checkHealthy();
    }
    public KvCache cache() { return cache; }
    public Map<String, Object> endpoints() {
        boolean prefill = config.role() == org.flexlb.mock.api.model.Types.Role.prefill;
        var result = new LinkedHashMap<String, Object>();
        result.put("id", config.id()); result.put("role", config.role().name()); result.put("engine", "rtp");
        result.put("rpc", rpcAddress()); result.put("control", address.http(api.port())); result.put("metrics", address.http(api.port()) + "/metrics");
        result.put("supported_rpc", prefill ? List.of("CheckHealth", "GetWorkerStatus", "GetCacheStatus", "GenerateStreamCall", "EnqueueBatch", "EnqueueGroup", "FetchResponse", "Cancel")
            : List.of("CheckHealth", "GetWorkerStatus", "GetCacheStatus", "RemoteGenerate"));
        if (publisher != null) result.put("kvcm",publisher.status());
        result.put("local_generation", prefill ? Map.of("max_new_tokens", 1, "multimodal", true) : Map.of());
        result.put("pd_generation", prefill ? Map.of("rpc", "GenerateStreamCall", "decode_target", "role_addrs", "multimodal", true) : Map.of());
        result.put("remote_generation", prefill ? Map.of() : Map.of("stages", List.of("ALLOCATE", "LOAD", "GENERATE", "PREFILL_COMPLETE"), "multimodal", true, "load", true));
        return Map.copyOf(result);
    }

    @Override public void close() {
        if (!closed.compareAndSet(false, true)) return;
        RuntimeException failure = null;
        try { service.close(); } catch (RuntimeException error) { failure = error; }
        try {
            if (rpc != null) {
                rpc.shutdown();
                if (!rpc.awaitTermination(3, TimeUnit.SECONDS)) {
                    rpc.shutdownNow();
                    if (!rpc.awaitTermination(3, TimeUnit.SECONDS)) throw new IllegalStateException("Engine RPC did not terminate");
                }
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            if (failure == null) failure = new IllegalStateException("Interrupted closing Engine", error);
            else failure.addSuppressed(error);
        } catch (RuntimeException error) {
            if (failure == null) failure = error; else failure.addSuppressed(error);
        } finally {
            try { if (publisher != null) publisher.close(); } finally {
                try { if (api != null) api.close(); } finally { cache.close(); }
            }
        }
        if (failure != null) throw failure;
    }

}

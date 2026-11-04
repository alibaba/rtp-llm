package org.flexlb.mock.engine.rtp;

import io.prometheus.client.Collector.MetricFamilySamples;
import org.flexlb.mock.admission.RpcService;
import org.flexlb.mock.metrics.MetricTags;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.flexlb.mock.cache.KvCache;
import org.flexlb.mock.cache.remote.RemoteCache;
import org.flexlb.mock.cache.KvcmReporter;
import org.flexlb.mock.engine.Scheduler;
import org.flexlb.mock.runtime.ApiServer;

/** One owned RTP P/D object with independent listeners and resources. Does not contain the case runner or a cluster controller. */
public final class RtpNode implements AutoCloseable {
    private final RtpConfig config;
    private final KvCache cache;
    private final RpcService service;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final org.flexlb.mock.runtime.EngineAddress address = org.flexlb.mock.runtime.EngineAddress.local();
    private Server rpc;
    private ApiServer api;
    private org.flexlb.mock.metrics.MetricsOutput metrics;
    private KvcmReporter publisher;
    private MetricSource metricSource;

    private record MetricSource(Object lock, RtpMetrics metrics) {}

    private RtpNode(RtpConfig config, Scheduler.Factory schedulers, RemoteCache remote) {
        this.config = Objects.requireNonNull(config);
        Objects.requireNonNull(schedulers);
        boolean enabled = RtpConfig.nativeBool(config.business(), "enable_remote_cache", false);
        if (enabled != (remote != null))
            throw new IllegalArgumentException(enabled ? "enable_remote_cache requires a remote KV adapter; no storage adapter is installed"
                : "Remote KV adapter requires enable_remote_cache=true");
        if (remote != null && config.groupScheduling())
            throw new IllegalArgumentException("Remote KV requires mock.scheduler.mode=fifo");
        cache = new KvCache(config.blockSize(), config.deviceBlocks(), config.reserveBlocks(), config.evictionPolicy(),
            config.deviceCache(), config.hostBlocks(), config.hostPolicy(), config.deviceWatermark(), config.hostWatermark(), remote);
        try {
            service = new RpcService(config.endpoint(), cache, (lock, storage, finished) -> {
                var scheduler = schedulers.create(lock, storage, finished);
                if (scheduler instanceof RtpSchedulerBase rtp)
                    metricSource = new MetricSource(lock, rtp.metrics());
                return scheduler;
            }, config.groupScheduling());
        }
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
        return start(config, rpcPort, controlPort, schedulers, null);
    }

    /** Explicit adapter assembly; production entrypoints install no remote storage implementation. */
    public static RtpNode start(RtpConfig config, int rpcPort, int controlPort, Scheduler.Factory schedulers,
                                RemoteCache remote) throws IOException {
        var node = new RtpNode(config, schedulers, remote);
        try {
            var builder = NettyServerBuilder.forAddress(node.address.listener(rpcPort)).addService(node.service);
            node.rpc = builder.build().start();
            node.metrics = org.flexlb.mock.metrics.MetricsOutput.start(node::collectMetrics);
            node.api = new ApiServer(node.address.listener(controlPort)).health(node::checkHealthy);
            if (node.metrics.prometheus()) node.api.metrics(() -> MetricTags.scrape(node.collectMetrics()));
            node.api.start();
            node.publisher = node.startMetadataReporter();
            return node;
        } catch (IOException | RuntimeException | Error error) {
            try { node.close(); } catch (RuntimeException cleanup) { error.addSuppressed(cleanup); }
            throw error;
        }
    }

    private List<MetricFamilySamples> collectMetrics() {
        if (metricSource == null) return List.of();
        synchronized (metricSource.lock()) {
            service.checkHealthy();
            return metricSource.metrics().collect();
        }
    }

    private KvcmReporter startMetadataReporter() {
        var values = config.business();
        String type = values.path("kv_cache_event_publisher_type").asText("none");
        if (!Set.of("none", "kvcm").contains(type)) throw new IllegalArgumentException("Invalid kv_cache_event_publisher_type");
        if (type.equals("none") && !RtpConfig.nativeBool(values, "enable_remote_cache", false)) return null;
        String instance = values.path("kv_cache_event_instance_id").asText();
        String host = values.path("kv_cache_event_host_ip_port").asText(address.rpc(rpc.getPort() - 1));
        if (instance.isBlank() || host.isBlank())
            throw new IllegalArgumentException("KVCM requires a shared instance_id and a nonempty Master-visible host_ip_port");
        var identity = new KvcmReporter.Identity(
            instance,
            values.path("kv_cache_event_instance_group").asText("default"),
            host,
            values.path("model_name").asText("mock-full-kv"), "FP16", false,
            config.blockSize(), config.tpSize(), config.dpSize(), "rtp_llm_hbm_" + config.blockSize() + "_full");
        int snapshotMillis = RtpConfig.nativeInt(values, "kv_cache_event_snapshot_interval_ms", 600000, Integer.MIN_VALUE, Integer.MAX_VALUE);
        String override = values.path("kv_cache_event_manager_endpoint").asText();
        boolean vip = RtpConfig.nativeBool(values, "kvcm_enable_vipserver", false);
        var endpoint = override.isBlank()
            ? org.flexlb.mock.cache.KvcmDiscovery.discover(
                values.path(vip ? "kvcm_vipserver_domain" : "kvcm_server_address").asText(), instance, vip)
            : org.flexlb.mock.cache.KvcmDiscovery.direct(java.net.URI.create(override));
        var settings = new KvcmReporter.Configuration(endpoint, type.equals("kvcm") && snapshotMillis > 0,
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
        result.put("rpc", rpcAddress()); result.put("control", address.http(api.port()));
        result.put("metrics", metrics.prometheus() ? address.http(api.port()) + "/metrics" : "");
        result.put("metrics_backend", metrics.prometheus() ? "prometheus" : "kmonitor");
        var supported = new ArrayList<>(List.of("ExecuteFunction", "CheckHealth", "GetWorkerStatus", "GetCacheStatus"));
        if (prefill) {
            supported.addAll(List.of("GenerateStreamCall", "FetchResponse", "Cancel"));
            if (config.groupScheduling()) supported.addAll(List.of("EnqueueBatch", "EnqueueGroup"));
        } else supported.add("RemoteGenerate");
        result.put("supported_rpc", List.copyOf(supported));
        if (publisher != null) result.put("kvcm",publisher.status());
        result.put("local_generation", prefill ? Map.of("max_new_tokens", 1, "multimodal", true) : Map.of());
        result.put("pd_generation", prefill ? Map.of("rpc", "GenerateStreamCall", "decode_target", "role_addrs", "multimodal", true) : Map.of());
        result.put("remote_generation", prefill ? Map.of() : Map.of("stages", List.of("ALLOCATE", "LOAD", "GENERATE", "PREFILL_COMPLETE"), "multimodal", true, "load", true));
        return Map.copyOf(result);
    }

    @Override public void close() {
        if (!closed.compareAndSet(false, true)) return;
        RuntimeException failure = null;
        try { if (metrics != null) metrics.close(); } catch (RuntimeException error) { failure = error; }
        try { service.close(); } catch (RuntimeException error) {
            if (failure == null) failure = error; else failure.addSuppressed(error);
        }
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

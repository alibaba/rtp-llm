package org.flexlb.mock.engine.execution;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.prometheus.client.Collector.MetricFamilySamples;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.scheduler.AbstractRtpScheduler;
import org.flexlb.mock.engine.scheduler.EngineScheduler;
import org.flexlb.mock.engine.scheduler.TaskHistory;
import org.flexlb.mock.engine.service.RtpRpcService;
import org.flexlb.mock.metrics.MetricTags;
import org.flexlb.mock.protocol.ControlServer;

/**
 * One LLM Mock P/D Engine with independent listeners and resources. Does not contain the case runner or a cluster controller.
 */
public final class LLMMockEngine implements AutoCloseable {
    private final RtpConfig config;
    private final KvCacheManager cacheManager;
    private final RtpRpcService service;
    private final EngineLoop engine;
    private final AtomicBoolean closed = new AtomicBoolean();
    private final org.flexlb.mock.engine.EngineMain.Address address = org.flexlb.mock.engine.EngineMain.Address.local();

    private Server rpc;
    private ControlServer api;
    private org.flexlb.mock.metrics.MetricsOutput metrics;
    private java.util.function.Supplier<List<MetricFamilySamples>> metricSource;

    private LLMMockEngine(RtpConfig config, EngineScheduler.Factory schedulers) {
        this.config = Objects.requireNonNull(config);
        Objects.requireNonNull(schedulers);
        cacheManager = new KvCacheManager(config.tokensPerCacheKey(), config.deviceBlocks(), config.totalCacheBlocks(), config.reserveBlocks(), config.deviceCache());
        var lock = new Object();
        var finished = new TaskHistory();
        try {
            var scheduler = schedulers.create(lock, cacheManager, finished);
            var samples = AbstractRtpScheduler.metricsFor(scheduler);
            if (samples != null) metricSource = samples;
            engine = new EngineLoop(lock, scheduler, finished, cacheManager, config.compute(), config.role(), config.id());
            try {
                service = new RtpRpcService(RtpRpcService.Config.from(config), cacheManager, lock, finished, engine, config.groupScheduling());
            } catch (RuntimeException | Error failure) {
                engine.close();
                throw failure;
            }
        } catch (RuntimeException | Error failure) {
            cacheManager.close();
            throw failure;
        }
    }

    public static LLMMockEngine start(RtpConfig config) throws IOException {
        return start(config, 0, 0);
    }

    public static LLMMockEngine start(RtpConfig config, int rpcPort, int controlPort) throws IOException {
        return start(config, rpcPort, controlPort, AbstractRtpScheduler.factory(config));
    }

    public static LLMMockEngine start(RtpConfig config, EngineScheduler.Factory schedulers) throws IOException {
        return start(config, 0, 0, schedulers);
    }

    public static LLMMockEngine start(RtpConfig config, int rpcPort, int controlPort, EngineScheduler.Factory schedulers) throws IOException {
        var node = new LLMMockEngine(config, schedulers);
        try {
            var builder = NettyServerBuilder.forAddress(node.address.listener(rpcPort)).addService(node.service);
            node.rpc = builder.build().start();
            node.metrics = org.flexlb.mock.metrics.MetricsOutput.start(node::collectMetrics);
            node.api = new ControlServer(node.address.listener(controlPort)).health(node::checkHealthy);
            if (node.metrics.prometheus()) node.api.metrics(() -> MetricTags.scrape(node.collectMetrics()));
            node.api.start();
            node.cacheManager.startKvcmReporter(config, node.address.rpc(node.rpc.getPort() - 1));
            return node;
        } catch (IOException | RuntimeException | Error error) {
            try { node.close(); } catch (RuntimeException cleanup) { error.addSuppressed(cleanup); }
            throw error;
        }
    }

    public void checkHealthy() {
        if (closed.get()) throw new IllegalStateException("Engine closed");
        if (rpc == null || rpc.isShutdown()) throw new IllegalStateException("Engine RPC stopped");
        service.checkHealthy();
    }

    public KvCacheManager cacheManager() { return cacheManager; }

    public Map<String, Object> endpoints() {
        boolean prefill = config.role() == org.flexlb.mock.protocol.Role.prefill;
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
        var kvcm = cacheManager.kvcmStatus();
        if (!kvcm.isEmpty()) result.put("kvcm", kvcm);
        result.put("local_generation", prefill ? Map.of("max_new_tokens", 1, "multimodal", true) : Map.of());
        result.put("pd_generation", prefill ? Map.of("rpc", "GenerateStreamCall", "decode_target", "role_addrs", "multimodal", true) : Map.of());
        result.put("remote_generation", prefill ? Map.of() : Map.of("stages", List.of("ALLOCATE", "LOAD", "GENERATE", "PREFILL_COMPLETE"), "multimodal", true, "load", true));
        return Map.copyOf(result);
    }

    public String rpcAddress() { return address.rpc(rpc.getPort()); }

    @Override public void close() {
        if (!closed.compareAndSet(false, true)) return;
        RuntimeException failure = null;
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
            try { if (metrics != null) metrics.close(); }
            catch (RuntimeException error) { if (failure == null) failure = error; else failure.addSuppressed(error); }
            try { engine.close(); } finally {
                try { if (api != null) api.close(); } finally { cacheManager.close(); }
            }
        }
        if (failure != null) throw failure;
    }

    private List<MetricFamilySamples> collectMetrics() {
        if (metricSource == null) return service.collectMetrics();
        service.checkHealthy();
        var samples = new java.util.ArrayList<>(metricSource.get());
        samples.addAll(service.collectMetrics());
        return samples;
    }

}

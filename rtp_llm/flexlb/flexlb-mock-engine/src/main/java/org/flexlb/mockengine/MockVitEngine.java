package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import io.grpc.*;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/** One V worker in the cluster runtime. A separate JVM is a launcher decision. */
final class MockVitEngine implements MockEngineLifecycle {
    private final JavaMockEngineCluster.Config config;
    private final String name, host;
    private final int port;
    private final EventLoopGroup boss, workers;
    final MockVitService service;
    private final MockRpcExecutors runtime;
    private final MockRpcExecutors.Engine rpc;
    private final MockVitMonitor monitor;
    private final MockPerformanceModel performance;
    private final AtomicBoolean closed = new AtomicBoolean();
    private volatile boolean stopped, accepting = true;

    MockVitEngine(JavaMockEngineCluster.Config config, MockPerformanceModel template, String name, String host,
                  int port, EventLoopGroup boss, EventLoopGroup workers) throws IOException {
        this.config = config; this.name = name; this.host = host; this.port = port; this.boss = boss; this.workers = workers;
        performance = template.forEngine(name + ":" + port);
        JsonNode mock = (JsonNode) performance.calibrationState().get("mock");
        var vit = mock.path("vit"); var defaults = MockVitService.Config.defaults();
        var limits = config.engineConfig.role("vit");
        var settings = new MockVitService.Config(vit.path("hidden_size").asInt(defaults.hiddenSize()),
                limits.path("vit_concurrency").intValue(), limits.path("vit_max_queue_size").intValue(),
                limits.path("mm_cache_cpu_max_bytes").longValue(), limits.path("mm_hash_key_cache_max_bytes").longValue(),
                MockEngineConfig.vitImageBytes(limits), vit.path("max_output_bytes").asLong(defaults.maxOutputBytes()),
                mock.path("forward").path("vit").path("fixed_ms").asDouble(defaults.computeDelayMs()));
        MockV41Image.Config model = null;
        if (config.vitModelPath != null) {
            try { model = MockV41Image.Config.fromModel(java.nio.file.Path.of(config.vitModelPath)); }
            catch (Exception error) { throw new IOException("cannot load V model metadata",error); }
        }
        runtime = config.rpcExecutors(workers);
        service = new MockVitService(settings, model, config.engineConfig.snapshot());
        service.timing(performance.timing());
        service.alive = () -> accepting && !stopped && !closed.get();
        rpc = runtime.engine();
        rpc.timing(performance.timing(), service::reportTiming);
        MockVitMonitor created = null;
        try {
            if (config.kmonitor) {
                var tags = new java.util.HashMap<>(WhaleMockMonitor.engineTags(System.getenv(),
                        config.whaleBundle ? config.host : host, "ROLE_TYPE_VIT"));
                tags.putAll(Map.of("source", "mock_vit", "engine", name, "engine_ip", host, "engine_port", Integer.toString(port), "generation", service.instance, "backend", "mock"));
                created = new MockVitMonitor(WhaleMockMonitor.createSink(), tags);
                created.start(service);
            }
        } catch (Exception error) {
            rpc.stop(); service.close(); if (created != null) created.close(); throw new IOException("V monitor startup failed",error);
        }
        monitor = created;
    }

    synchronized Server start() throws IOException {
        if (closed.get()) throw new IllegalStateException("V worker already closed");
        ServerInterceptor admission = new ServerInterceptor() {
            @Override public <Q, S> ServerCall.Listener<Q> interceptCall(ServerCall<Q, S> call, Metadata headers, ServerCallHandler<Q, S> next) {
                String method = call.getMethodDescriptor().getBareMethodName();
                if ((!accepting || stopped) && !method.equals("GetWorkerStatus") && !method.equals("GetCacheStatus")) {
                    call.close(Status.UNAVAILABLE.withDescription("mock V admission closed"), new Metadata());
                    return new ServerCall.Listener<>() {};
                }
                return next.startCall(call, headers);
            }
        };
        var server = NettyServerBuilder.forAddress(new java.net.InetSocketAddress(config.bindHost, port))
                .bossEventLoopGroup(boss).workerEventLoopGroup(workers).channelType(NioServerSocketChannel.class)
                .withOption(ChannelOption.SO_REUSEADDR, true).withChildOption(ChannelOption.SO_REUSEADDR, true)
                .executor(runtime.controlExecutor()).callExecutor(rpc).intercept(rpc).intercept(admission)
                .maxInboundMessageSize(config.grpcMaxInboundMessageBytes)
                .addService(ServerInterceptors.intercept(service, monitor == null ? java.util.List.of() : java.util.List.of(monitor)))
                .build().start();
        stopped = false; accepting = true;
        return server;
    }
    synchronized void setPerformance(JsonNode patch) {
        var forward = patch.path("mock").path("forward");
        if (!forward.isMissingNode()) MockEngineConfig.fields(forward,"vit");
        performance.applyOverrides(patch);
        var mock = (JsonNode) performance.calibrationState().get("mock");
        service.forwardMs(mock.path("forward").path("vit").path("fixed_ms").asDouble(MockVitService.Config.defaults().computeDelayMs()));
    }
    @Override public String getEngineName() { return name; }
    @Override public String getHost() { return host; }
    @Override public String getRoleName() { return "VIT"; }
    @Override public int getGrpcPort() { return port; }
    @Override public int getRunningCount() { return service.snapshot().get("active").intValue(); }
    @Override public int getWaitingCount() { return service.snapshot().get("queued").intValue(); }
    @Override public boolean isStopped() { return stopped; }
    @Override public boolean isShuttingDown() { return closed.get() || !accepting; }
    @Override public boolean hasInflightWork() { return inflightWorkSnapshot().values().stream().anyMatch(value -> value > 0); }
    @Override public Map<String, Integer> inflightWorkSnapshot() {
        var s=service.snapshot();
        return Map.of("active",s.get("active").intValue(),"queued",s.get("queued").intValue(),
                "gpu_queued",s.get("gpu_queued").intValue(),"pending",s.get("pending").intValue(),"rpc_calls",rpc.inFlight());
    }
    @Override public Map<String, Object> admissionSnapshot() { return Map.of("accepting", accepting && !stopped, "rpc_runtime", rpc.snapshot()); }
    @Override public void closeAdmission() { accepting = false; }
    @Override public void reopenAdmissionAfterFailedRemoval() { if (!closed.get()) accepting = true; }
    @Override public void setStopped(boolean value) { stopped = value; }
    @Override public void drainAndShutdown() { closeAdmission(); }
    @Override public void shutdown() {
        if (closed.compareAndSet(false,true)) {
            accepting=false; stopped=true; rpc.stop();
            ShutdownTasks.runAll(java.util.List.of(service::close, () -> { if(monitor!=null) monitor.close(); }));
        }
    }
    @Override public Map<String, Object> getTopologySnapshot() {
        return Map.of("name",name,"role","vit","grpc_addr",host+":"+port,"http_addr",host+":"+(port-1),"engine_incarnation",service.instance);
    }
    @Override public Map<String, Object> getSnapshot() {
        var result = new LinkedHashMap<String,Object>(getTopologySnapshot());
        result.put("healthy",!stopped && accepting && !closed.get()); result.put("stopped",stopped);
        result.put("running",getRunningCount()); result.put("waiting",getWaitingCount());
        result.put("work_unit","image_computations");
        result.put("vit",service.snapshot()); result.putAll(admissionSnapshot()); return result;
    }
    @Override public Map<String, Object> effectiveConfiguration() {
        var result = new LinkedHashMap<String,Object>(service.effectiveConfiguration());
        result.put("engine",name); result.put("calibration",performance.calibrationState()); return result;
    }
}

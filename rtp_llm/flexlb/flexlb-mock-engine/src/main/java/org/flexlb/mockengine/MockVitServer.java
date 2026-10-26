package org.flexlb.mockengine;

import io.grpc.netty.NettyServerBuilder;

/** Independent ViT process; logical workers must have independent listening endpoints. */
public final class MockVitServer {
    private MockVitServer() {}
    public static void main(String[] args) throws Exception {
        int port = 62000;
        int httpPort = 0;
        boolean kmonitor = false;
        java.nio.file.Path model = null;
        String engineFile = null, calibrationFile = null;
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 == args.length) throw new IllegalArgumentException("missing value for " + args[i]);
            switch (args[i]) {
                case "--mock.kmonitor" -> kmonitor = parseKmonitor(args[i + 1]);
                case "--mock.model_path" -> model = java.nio.file.Path.of(args[i + 1]);
                case "--mock.port" -> port = Integer.parseInt(args[i + 1]);
                case "--mock.http_port" -> httpPort = Integer.parseInt(args[i + 1]);
                case "--mock.engine_config" -> engineFile = args[i + 1];
                case "--mock.calibration" -> calibrationFile = args[i + 1];
                default -> throw new IllegalArgumentException("unknown argument " + args[i]);
            }
        }
        if (engineFile == null || calibrationFile == null) throw new IllegalArgumentException("--mock.engine_config and --mock.calibration are required");
        var effective = MockEngineConfig.load(engineFile);
        var limits = effective.role("vit");
        var document = new com.fasterxml.jackson.databind.ObjectMapper().readTree(java.nio.file.Path.of(calibrationFile).toFile());
        var timing = MockPerformanceModel.loadCalibration(document).timing().forEngine("vit:" + port);
        var calibration = document.path("mock").path("vit");
        MockEngineConfig.fields(calibration, "hidden_size", "max_output_bytes");
        var config = new MockVitService.Config(MockEngineConfig.integer(calibration, "hidden_size", 1),
                limits.path("vit_concurrency").intValue(), limits.path("vit_max_queue_size").intValue(),
                limits.path("mm_cache_cpu_max_bytes").longValue(), limits.path("mm_hash_key_cache_max_bytes").longValue(),
                MockEngineConfig.vitImageBytes(limits), MockEngineConfig.integer(calibration, "max_output_bytes", 1),
                document.path("mock").path("forward").path("vit").path("fixed_ms").asDouble(1));
        var service = new MockVitService(config, model == null ? null : MockV41Image.Config.fromModel(model), effective.snapshot());
        service.timing(timing);
        var rpcRuntime = new MockRpcExecutors(64, 32768);
        var rpc = rpcRuntime.engine();
        rpc.timing(timing, service::reportTiming);
        MockVitMonitor monitor = null;
        if (kmonitor) {
            var tags = new java.util.HashMap<>(WhaleMockMonitor.engineTags(System.getenv(),
                    System.getenv().getOrDefault("POD_IP", java.net.InetAddress.getLocalHost().getHostAddress()), "ROLE_TYPE_VIT"));
            tags.put("source", "mock_vit");
            monitor = new MockVitMonitor(WhaleMockMonitor.createSink(), tags);
            monitor.start(service);
        }
        var server = NettyServerBuilder.forPort(port).executor(rpcRuntime.controlExecutor()).callExecutor(rpc).intercept(rpc).maxInboundMessageSize(config.maxInputBytes() * 2)
                .addService(io.grpc.ServerInterceptors.intercept(service,
                        monitor == null ? java.util.List.of() : java.util.List.of(monitor))).build().start();
        if (httpPort < 0 || httpPort > 65535 || httpPort == port) {
            server.shutdownNow();
            if (monitor != null) monitor.close();
            service.close();
            rpcRuntime.close();
            throw new IllegalArgumentException("invalid HTTP health port");
        }
        com.sun.net.httpserver.HttpServer health = null;
        if (httpPort != 0) {
            health = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress(httpPort), 16);
            health.createContext("/effective_config", exchange -> {
                byte[] body = new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsBytes(service.effectiveConfiguration());
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, body.length);
                try (var output = exchange.getResponseBody()) { output.write(body); }
            });
            health.createContext("/health", exchange -> {
                boolean ready = !server.isShutdown() && !server.isTerminated();
                byte[] body = (ready ? "ok" : "not ready").getBytes(java.nio.charset.StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(ready ? 200 : 503, body.length);
                try (var output = exchange.getResponseBody()) { output.write(body); }
            });
            health.start();
        }
        var healthServer = health;
        var vitMonitor = monitor;
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (healthServer != null) healthServer.stop(0);
            server.shutdownNow();
            if (vitMonitor != null) vitMonitor.close();
            service.close();
            rpcRuntime.close();
        }));
        System.out.println("Mock ViT ready port=" + server.getPort() + " protocol=main geometry=qwen2_vl/prepared_v41");
        server.awaitTermination();
    }

    static boolean parseKmonitor(String value) {
        if (!value.equals("true") && !value.equals("false"))
            throw new IllegalArgumentException("--mock.kmonitor requires true or false");
        return Boolean.parseBoolean(value);
    }
}

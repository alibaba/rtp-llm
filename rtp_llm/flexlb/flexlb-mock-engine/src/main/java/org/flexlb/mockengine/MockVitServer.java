package org.flexlb.mockengine;

import io.grpc.netty.NettyServerBuilder;

/** Independent ViT process; logical workers must have independent listening endpoints. */
public final class MockVitServer {
    private MockVitServer() {}
    public static void main(String[] args) throws Exception {
        int port = 62000;
        int httpPort = 0;
        java.nio.file.Path model = null;
        var defaults = MockVitService.Config.defaults();
        int hidden = defaults.hiddenSize(), concurrency = defaults.concurrency(), queue = defaults.queueCapacity();
        long embeddings = defaults.embeddingCacheBytes(), hashes = defaults.hashCacheBytes(), delay = defaults.computeDelayMs();
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 == args.length) throw new IllegalArgumentException("missing value for " + args[i]);
            switch (args[i]) {
                case "--model-path" -> model = java.nio.file.Path.of(args[i + 1]);
                case "--port" -> port = Integer.parseInt(args[i + 1]);
                case "--http-port" -> httpPort = Integer.parseInt(args[i + 1]);
                case "--hidden-size" -> hidden = Integer.parseInt(args[i + 1]);
                case "--concurrency" -> concurrency = Integer.parseInt(args[i + 1]);
                case "--queue-capacity" -> queue = Integer.parseInt(args[i + 1]);
                case "--embedding-cache-bytes" -> embeddings = Long.parseLong(args[i + 1]);
                case "--hash-cache-bytes" -> hashes = Long.parseLong(args[i + 1]);
                case "--compute-delay-ms" -> delay = Long.parseLong(args[i + 1]);
                default -> throw new IllegalArgumentException("unknown argument " + args[i]);
            }
        }
        var service = new MockVitService(new MockVitService.Config(hidden, concurrency, queue, embeddings, hashes,
                defaults.maxInputBytes(), defaults.maxOutputBytes(), delay),
                model == null ? null : MockV41Image.Config.fromModel(model));
        var server = NettyServerBuilder.forPort(port).maxInboundMessageSize(defaults.maxInputBytes() * 2)
                .addService(service).build().start();
        if (httpPort < 0 || httpPort > 65535 || httpPort == port) {
            server.shutdownNow();
            service.close();
            throw new IllegalArgumentException("invalid HTTP health port");
        }
        com.sun.net.httpserver.HttpServer health = null;
        if (httpPort != 0) {
            health = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress(httpPort), 16);
            health.createContext("/health", exchange -> {
                boolean ready = !server.isShutdown() && !server.isTerminated();
                byte[] body = (ready ? "ok" : "not ready").getBytes(java.nio.charset.StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(ready ? 200 : 503, body.length);
                try (var output = exchange.getResponseBody()) { output.write(body); }
            });
            health.start();
        }
        var healthServer = health;
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (healthServer != null) healthServer.stop(0);
            server.shutdownNow();
            service.close();
        }));
        System.out.println("Mock ViT ready port=" + server.getPort() + " protocol=main geometry=qwen2_vl/prepared_v41");
        server.awaitTermination();
    }
}

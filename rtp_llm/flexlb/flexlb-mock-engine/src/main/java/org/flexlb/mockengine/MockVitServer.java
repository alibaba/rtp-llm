package org.flexlb.mockengine;

import io.grpc.netty.NettyServerBuilder;

/** Independent ViT process; logical workers must have independent listening endpoints. */
public final class MockVitServer {
    private MockVitServer() {}
    public static void main(String[] args) throws Exception {
        int port = 62000;
        var defaults = MockVitService.Config.defaults();
        int hidden = defaults.hiddenSize(), concurrency = defaults.concurrency(), queue = defaults.queueCapacity();
        long embeddings = defaults.embeddingCacheBytes(), hashes = defaults.hashCacheBytes(), delay = defaults.computeDelayMs();
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 == args.length) throw new IllegalArgumentException("missing value for " + args[i]);
            switch (args[i]) {
                case "--port" -> port = Integer.parseInt(args[i + 1]);
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
                defaults.maxInputBytes(), defaults.maxOutputBytes(), delay));
        var server = NettyServerBuilder.forPort(port).maxInboundMessageSize(defaults.maxInputBytes() * 2)
                .addService(service).build().start();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> { server.shutdownNow(); service.close(); }));
        System.out.println("Mock ViT ready port=" + server.getPort() + " protocol=main geometry=qwen2_vl");
        server.awaitTermination();
    }
}

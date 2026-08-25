package org.flexlb.it.fixture.kvcm;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.concurrent.TimeUnit;

/** Lifecycle owner for the loopback KVCM gRPC server used by VLLM and SGLang scenarios. */
final class ScriptedKvcm implements AutoCloseable {

    private final ScriptedKvcmService service = new ScriptedKvcmService();
    private Server server;

    synchronized void start() throws IOException {
        if (server == null) {
            server = NettyServerBuilder.forAddress(new InetSocketAddress(IntegrationTestFixtures.WORKER_IP, 0))
                    .addService(service)
                    .build()
                    .start();
            service.setLeaderPort(server.getPort());
        }
    }

    int port() {
        if (server == null) {
            throw new IllegalStateException("Scripted KVCM has not started");
        }
        return server.getPort();
    }

    ScriptedKvcmService service() {
        return service;
    }

    @Override
    public synchronized void close() throws InterruptedException {
        if (server != null) {
            server.shutdownNow();
            server.awaitTermination(5, TimeUnit.SECONDS);
            server = null;
        }
    }
}

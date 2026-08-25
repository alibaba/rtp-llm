package org.flexlb.it.fixture.engine;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import org.flexlb.dao.route.RoleType;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.concurrent.TimeUnit;

/**
 * One loopback engine endpoint with a real worker-status gRPC server.
 *
 * <p>The HTTP identity is one less than the dynamically assigned status port, preserving the
 * production default {@code httpPort -> workerStatusPort} mapping while no model server is needed.
 */
final class ScriptedWorker implements AutoCloseable {

    private final String host;
    private final ScriptedWorkerStatusService service;
    private Server server;

    ScriptedWorker(String host, RoleType roleType) {
        this.host = host;
        this.service = new ScriptedWorkerStatusService(roleType);
    }

    synchronized void start() throws IOException {
        if (server == null) {
            server = NettyServerBuilder.forAddress(new InetSocketAddress(host, 0))
                    .addService(service)
                    .build()
                    .start();
        }
    }

    int workerStatusPort() {
        if (server == null) {
            throw new IllegalStateException("Scripted worker has not started");
        }
        return server.getPort();
    }

    ScriptedWorkerStatusService service() {
        return service;
    }

    int httpPort() {
        return workerStatusPort() - 1;
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

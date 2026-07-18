package org.flexlb.mock;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * Abstract base class managing a mock gRPC worker server lifecycle.
 *
 * <p>Starts a real Netty gRPC server ({@link NettyServerBuilder}) on a configurable
 * port.  The server hosts a {@link MockRpcService} that records all incoming
 * RPC calls for later assertion.
 *
 * <p>Usage:
 * <pre>{@code
 * MockPrefillWorker worker = new MockPrefillWorker(behavior);
 * worker.start(0);             // auto-assign port
 * int grpcPort = worker.getPort();
 * int httpPort = grpcPort - 1; // CommonConstants.GRPC_PORT_OFFSET = 1
 * // ... run test ...
 * worker.stop();
 * }</pre>
 *
 * <p>Port convention: gRPC port = HTTP port + 1 (see
 * {@code CommonConstants.GRPC_PORT_OFFSET}).  Mock workers always serve gRPC
 * on the requested port; the FlexLB scheduler derives the HTTP port internally.
 */
public abstract class MockWorker {

    private static final Logger log = LoggerFactory.getLogger(MockWorker.class);

    protected final MockRpcService rpcService;
    protected final MockWorkerBehavior behavior;
    private Server server;
    private int actualPort = -1;

    protected MockWorker(MockWorkerBehavior behavior) {
        this.behavior = behavior;
        this.rpcService = new MockRpcService();
        this.rpcService.setBehavior(behavior);
    }

    /**
     * Start the gRPC server on the given port.
     *
     * @param port port to bind; 0 lets the OS auto-assign a free port
     * @throws IOException if the server cannot start
     */
    public void start(int port) throws IOException {
        server = NettyServerBuilder.forPort(port)
                .addService(rpcService)
                .maxInboundMessageSize(16 * 1024 * 1024)
                .build()
                .start();
        actualPort = server.getPort();
        log.info("{} started on gRPC port {} (http port would be {})",
                getClass().getSimpleName(), actualPort, actualPort - 1);
    }

    /**
     * Stop the gRPC server, waiting up to 5 seconds for graceful shutdown.
     */
    public void stop() {
        if (server != null) {
            server.shutdown();
            try {
                if (!server.awaitTermination(5, TimeUnit.SECONDS)) {
                    server.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                server.shutdownNow();
            }
            log.info("{} stopped", getClass().getSimpleName());
        }
    }

    /**
     * @return the actual gRPC port the server is listening on, or -1 if not started.
     */
    public int getPort() {
        return actualPort;
    }

    /**
     * @return the HTTP port that corresponds to this worker (gRPC port - 1).
     */
    public int getHttpPort() {
        if (actualPort < 0) {
            throw new IllegalStateException("Worker not started");
        }
        return actualPort - 1;
    }

    // ==================== Delegated accessors ====================

    public MockRpcService getRpcService() {
        return rpcService;
    }

    public MockWorkerBehavior getBehavior() {
        return behavior;
    }

    /**
     * Hot-swap the worker behavior at runtime.
     */
    public void setBehavior(MockWorkerBehavior behavior) {
        rpcService.setBehavior(behavior);
    }

    // ==================== Convenience accessors for call records ====================

    public int getEnqueueCount() {
        return rpcService.getEnqueueCount();
    }

    public int getCancelCount() {
        return rpcService.getCancelCount();
    }

    public long getWorkerStatusCallCount() {
        return rpcService.getWorkerStatusCallCount();
    }

    public void resetRecords() {
        rpcService.resetRecords();
    }
}

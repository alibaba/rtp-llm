package org.flexlb.engine.grpc;

import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;

class EngineGrpcClientBatchDeliveryTest {
    private ThreadPoolExecutor executor;
    private NioEventLoopGroup eventLoop;
    private EngineGrpcClient client;
    private Server server;

    @BeforeEach
    void setUp() {
        executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2);
        eventLoop = new NioEventLoopGroup(1);
        client = new EngineGrpcClient(mock(CustomNameResolver.class), executor, eventLoop,
                mock(EngineLocalView.class), mock(GlobalCacheIndex.class), mock(GrpcReporter.class));
    }

    @AfterEach
    void tearDown() throws Exception {
        client.shutdownChannelPool();
        if (server != null) {
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
        executor.shutdownNow();
        executor.awaitTermination(5, TimeUnit.SECONDS);
        eventLoop.shutdownGracefully(0, 5, TimeUnit.SECONDS).sync();
    }

    @Test
    void connectionRefusedIsDefinitelyNotSent() throws Exception {
        int port;
        try (ServerSocket socket = new ServerSocket()) {
            socket.bind(new InetSocketAddress("127.0.0.1", 0));
            port = socket.getLocalPort();
        }

        Throwable failure = enqueueFailure(port, 5000);

        EngineGrpcClient.BatchNotSentException notSent =
                assertInstanceOf(EngineGrpcClient.BatchNotSentException.class, failure);
        assertEquals(Status.Code.UNAVAILABLE, Status.fromThrowable(notSent.getCause()).getCode());
    }

    @Test
    void unavailableAfterServerReceivesRequestRemainsAmbiguousAndIsNotReplayed() throws Exception {
        AtomicInteger received = new AtomicInteger();
        server = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0))
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void enqueueBatch(EngineRpcService.EnqueueBatchRequestPB request,
                                             StreamObserver<EngineRpcService.EnqueueBatchResponsePB> observer) {
                        received.incrementAndGet();
                        observer.onError(Status.UNAVAILABLE.withDescription("ACK lost").asRuntimeException());
                    }
                }).build().start();

        Throwable failure = enqueueFailure(server.getPort(), 5000);

        assertInstanceOf(StatusRuntimeException.class, failure);
        assertEquals(Status.Code.UNAVAILABLE, Status.fromThrowable(failure).getCode());
        assertEquals(1, received.get());
    }

    @Test
    void deadlineAfterServerReceivesRequestRemainsAmbiguousAndIsNotReplayed() throws Exception {
        AtomicInteger received = new AtomicInteger();
        server = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0))
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void enqueueBatch(EngineRpcService.EnqueueBatchRequestPB request,
                                             StreamObserver<EngineRpcService.EnqueueBatchResponsePB> observer) {
                        received.incrementAndGet();
                    }
                }).build().start();

        Throwable failure = enqueueFailure(server.getPort(), 1000);

        assertInstanceOf(StatusRuntimeException.class, failure);
        assertEquals(Status.Code.DEADLINE_EXCEEDED, Status.fromThrowable(failure).getCode());
        assertEquals(1, received.get());
    }

    private Throwable enqueueFailure(int port, long deadlineMs) {
        ExecutionException failure = assertThrows(ExecutionException.class, () ->
                client.batchEnqueueAsync("127.0.0.1", port,
                        EngineRpcService.EnqueueBatchRequestPB.newBuilder().setBatchId(1).build(), deadlineMs)
                        .get(10, TimeUnit.SECONDS));
        return failure.getCause();
    }
}

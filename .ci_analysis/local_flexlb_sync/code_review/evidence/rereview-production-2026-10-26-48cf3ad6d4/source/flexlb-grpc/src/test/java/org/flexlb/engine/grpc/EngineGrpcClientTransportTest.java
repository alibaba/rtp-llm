package org.flexlb.engine.grpc;

import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

/** Exercises transport observations through the production NIO channel builder. */
class EngineGrpcClientTransportTest {
    private static final String LOOPBACK = "127.0.0.1";
    private final ThreadPoolExecutor executor =
            new ThreadPoolExecutor(1, 1, 1, TimeUnit.MINUTES, new LinkedBlockingQueue<>());
    private final NioEventLoopGroup eventLoopGroup = new NioEventLoopGroup(1);
    private final GrpcReporter reporter = mock(GrpcReporter.class);
    private final EngineGrpcClient client =
            new EngineGrpcClient(
                    mock(CustomNameResolver.class), executor, eventLoopGroup, reporter, 2000, 5000);
    private final CountDownLatch releaseCallbacks = new CountDownLatch(1);
    private Server server;

    @AfterEach
    void close() throws InterruptedException {
        releaseCallbacks.countDown();
        client.shutdownChannelPool();
        if (server != null) {
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
        executor.shutdownNow();
        executor.awaitTermination(5, TimeUnit.SECONDS);
        eventLoopGroup.shutdownGracefully(0, 1, TimeUnit.SECONDS).await(5, TimeUnit.SECONDS);
    }

    @Test
    void successfulResponseIncludesReceivedTrailers() throws Exception {
        startServer(EngineGrpcClientTransportTest::succeed);

        assertEquals(
                response(),
                client.getWorkerStatusAsync(LOOPBACK, server.getPort(), request(), 5000)
                        .get(5, TimeUnit.SECONDS));

        var trace = captureDelivery(false);
        assertEquals("RESPONSE_TRAILERS_RECEIVED", trace.stage().toString());
        assertTrue(trace.callbackDelayMs() >= 0L);
        assertTrue(trace.responseDeliveryMs() >= 0L);
        verify(reporter).reportCallMetrics(eq("GetWorkerStatus"), anyLong(), eq(false));
        verify(reporter, never())
                .reportCallFailure(anyString(), anyBoolean(), anyString(), any(), any());
    }

    @Test
    void delayedClientCallbackRemainsSuccessfulAndReportsLocalDeliveryTime() throws Exception {
        CountDownLatch callbacksBlocked = new CountDownLatch(1);
        CountDownLatch responseSent = new CountDownLatch(1);
        startServer(
                observer -> {
                    executor.execute(
                            () -> {
                                callbacksBlocked.countDown();
                                await(releaseCallbacks);
                            });
                    await(callbacksBlocked);
                    succeed(observer);
                    responseSent.countDown();
                });

        var result = client.getWorkerStatusAsync(LOOPBACK, server.getPort(), request(), 5000);
        assertTrue(responseSent.await(5, TimeUnit.SECONDS));
        // The gate, rather than a scheduler sleep, proves the completed server reply cannot
        // yet reach the caller. The bounded wait also makes its local delay measurable.
        assertThrows(TimeoutException.class, () -> result.get(100, TimeUnit.MILLISECONDS));
        releaseCallbacks.countDown();

        assertEquals(response(), result.get(5, TimeUnit.SECONDS));
        var trace = captureDelivery(false);
        assertEquals("RESPONSE_TRAILERS_RECEIVED", trace.stage().toString());
        assertTrue(trace.callbackDelayMs() > 0L, "the blocked client executor must be observable");
        assertTrue(trace.responseDeliveryMs() >= trace.callbackDelayMs());
        assertTrue(
                trace.responseDeliveryMs() >= 50.0,
                "the transport-to-callback interval must include the deliberately blocked delivery");
        verify(reporter, never())
                .reportCallFailure(anyString(), anyBoolean(), anyString(), any(), any());
    }

    @Test
    void silentWorkerTimesOutAfterRequestWasSent() throws Exception {
        CountDownLatch requestReceived = new CountDownLatch(1);
        startServer(observer -> requestReceived.countDown());

        var result = client.getWorkerStatusAsync(LOOPBACK, server.getPort(), request(), 1500);
        assertTrue(requestReceived.await(5, TimeUnit.SECONDS));
        ExecutionException failure =
                assertThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        assertEquals(
                Status.Code.DEADLINE_EXCEEDED,
                assertInstanceOf(StatusRuntimeException.class, failure.getCause())
                        .getStatus()
                        .getCode());

        var trace = ArgumentCaptor.forClass(GrpcCallTrace.Snapshot.class);
        verify(reporter)
                .reportCallFailure(
                        eq("worker"), eq(false), eq("CALL_RESULT"), any(), trace.capture());
        // This handler never calls onNext/sendHeaders: grpc-java sends no response headers.
        assertEquals("WAITING_RESPONSE", trace.getValue().stage().toString());
        assertEquals(-1.0, trace.getValue().responseDeliveryMs());
        verify(reporter, never()).reportCallMetrics(anyString(), anyLong(), anyBoolean());
    }

    @Test
    void refusedConnectionIsReportedOnceAfterRetryFails() throws Exception {
        int port;
        try (ServerSocket unused = new ServerSocket(0, 1, InetAddress.getByName(LOOPBACK))) {
            port = unused.getLocalPort();
        }

        var result = client.getWorkerStatusAsync(LOOPBACK, port, request(), 5000);
        ExecutionException failure =
                assertThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        assertEquals(GrpcFailure.Kind.CONNECT_REFUSED, GrpcFailure.classify(failure.getCause()));
        assertEquals(1, failure.getCause().getSuppressed().length);

        var detail = ArgumentCaptor.forClass(GrpcFailure.Analysis.class);
        verify(reporter)
                .reportCallFailure(
                        eq("worker"), eq(true), eq("CALL_RESULT"), detail.capture(), any());
        assertEquals(GrpcFailure.Kind.CONNECT_REFUSED, detail.getValue().kind());
        verify(reporter, never()).reportCallMetrics(anyString(), anyLong(), anyBoolean());
    }

    @Test
    void peerEofRetriesOnANewConnectionAndReportsOnlyTheRecoveredSuccess() throws Exception {
        AtomicInteger attempts = new AtomicInteger();
        startPeerClosingServer(attempts, false);

        assertEquals(
                response(),
                client.getWorkerStatusAsync(LOOPBACK, server.getPort(), request(), 5000)
                        .get(5, TimeUnit.SECONDS));

        assertEquals(2, attempts.get());
        verify(reporter).reportCallMetrics(eq("GetWorkerStatus"), anyLong(), eq(true));
        verify(reporter, never())
                .reportCallFailure(anyString(), anyBoolean(), anyString(), any(), any());
        assertEquals(GrpcCallTrace.Stage.RESPONSE_TRAILERS_RECEIVED, captureDelivery(true).stage());
    }

    @Test
    void peerEofProducesBareUnavailableWithSocketEvidenceAfterTheSingleRetry() throws Exception {
        AtomicInteger attempts = new AtomicInteger();
        startPeerClosingServer(attempts, true);

        ExecutionException failure =
                assertThrows(
                        ExecutionException.class,
                        () ->
                                client.getWorkerStatusAsync(
                                                LOOPBACK, server.getPort(), request(), 5000)
                                        .get(5, TimeUnit.SECONDS));

        StatusRuntimeException rpcFailure =
                assertInstanceOf(StatusRuntimeException.class, failure.getCause());
        assertEquals(Status.Code.UNAVAILABLE, rpcFailure.getStatus().getCode());
        assertNull(rpcFailure.getCause(), "clean peer EOF supplies no typed transport exception");
        assertEquals(2, attempts.get());
        var detail = ArgumentCaptor.forClass(GrpcFailure.Analysis.class);
        var trace = ArgumentCaptor.forClass(GrpcCallTrace.Snapshot.class);
        verify(reporter)
                .reportCallFailure(
                        eq("worker"),
                        eq(true),
                        eq("CALL_RESULT"),
                        detail.capture(),
                        trace.capture());
        assertEquals(GrpcFailure.Kind.RPC_FAILURE, detail.getValue().kind());
        assertEquals("PEER_EOF", trace.getValue().transportEvent());
        assertEquals(GrpcFailure.Kind.UNEXPECTED_EOF, trace.getValue().transportCauseKind());
        verify(reporter, never()).reportCallMetrics(anyString(), anyLong(), anyBoolean());
    }

    private void startPeerClosingServer(AtomicInteger attempts, boolean closeEveryAttempt)
            throws Exception {
        AtomicReference<SocketChannel> connection = new AtomicReference<>();
        server =
                NettyServerBuilder.forAddress(new InetSocketAddress(LOOPBACK, 0))
                        .bossEventLoopGroup(eventLoopGroup)
                        .workerEventLoopGroup(eventLoopGroup)
                        .channelFactory(
                                () ->
                                        new NioServerSocketChannel() {
                                            @Override
                                            protected int doReadMessages(List<Object> messages)
                                                    throws Exception {
                                                int read = super.doReadMessages(messages);
                                                if (read > 0) {
                                                    connection.set(
                                                            (SocketChannel)
                                                                    messages.get(
                                                                            messages.size() - 1));
                                                }
                                                return read;
                                            }
                                        })
                        .directExecutor()
                        .addService(
                                new RpcServiceGrpc.RpcServiceImplBase() {
                                    @Override
                                    public void getWorkerStatus(
                                            EngineRpcService.StatusVersionPB request,
                                            StreamObserver<EngineRpcService.WorkerStatusPB>
                                                    observer) {
                                        if (attempts.incrementAndGet() == 1 || closeEveryAttempt) {
                                            // Send TCP FIN without a gRPC response, GOAWAY, or
                                            // RST_STREAM.
                                            connection.get().shutdownOutput();
                                        } else {
                                            succeed(observer);
                                        }
                                    }
                                })
                        .build()
                        .start();
    }

    private GrpcCallTrace.Snapshot captureDelivery(boolean retry) {
        var trace = ArgumentCaptor.forClass(GrpcCallTrace.Snapshot.class);
        verify(reporter).reportDeliveryDelay(eq("worker"), eq(retry), trace.capture());
        return trace.getValue();
    }

    private void startServer(Consumer<StreamObserver<EngineRpcService.WorkerStatusPB>> reply)
            throws Exception {
        server =
                NettyServerBuilder.forAddress(new InetSocketAddress(LOOPBACK, 0))
                        .directExecutor()
                        .addService(
                                new RpcServiceGrpc.RpcServiceImplBase() {
                                    @Override
                                    public void getWorkerStatus(
                                            EngineRpcService.StatusVersionPB request,
                                            StreamObserver<EngineRpcService.WorkerStatusPB>
                                                    observer) {
                                        reply.accept(observer);
                                    }
                                })
                        .build()
                        .start();
    }

    private static EngineRpcService.StatusVersionPB request() {
        return EngineRpcService.StatusVersionPB.getDefaultInstance();
    }

    private static EngineRpcService.WorkerStatusPB response() {
        return EngineRpcService.WorkerStatusPB.newBuilder()
                .setAlive(true)
                .setStatusVersion(1L)
                .build();
    }

    private static void succeed(StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
        observer.onNext(response());
        observer.onCompleted();
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(5, TimeUnit.SECONDS), "test gate was not released");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError("test gate interrupted", interrupted);
        }
    }
}

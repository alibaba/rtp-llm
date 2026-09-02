package org.flexlb.httpserver;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.ServerInterceptors;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Real Netty boundaries for self-target and one-hop forwarding guards. */
class FlexlbForwardHopGuardNettyTest {

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void selfTargetReturnsImmediatelyWithoutRecursiveRpc() throws Exception {
        try (Node node = Node.start("127.0.0.1")) {
            node.masterAddress.set(node.httpAddress());

            List<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responses =
                    new ArrayList<>();
            try (Client client = Client.connect(node.grpcPort())) {
                long started = System.nanoTime();
                for (int i = 0; i < 8; i++) {
                    responses.add(client.stub.schedule(request(71_000L + i)));
                }
                assertTrue(Duration.ofNanos(System.nanoTime() - started)
                                .compareTo(Duration.ofSeconds(2)) < 0,
                        "self-target guard must not wait on a recursive RPC");
            }

            assertEquals(8, node.inboundCalls.get());
            assertEquals(0, node.rejections.get());
            assertTrue(responses.stream().noneMatch(
                    FlexlbScheduleProtocol.FlexlbScheduleResponsePB::getSuccess));
            assertTrue(responses.stream().allMatch(response ->
                    response.getErrorMessage().contains("SELF_FORWARD_BLOCKED")));
            verify(node.routeService, never()).route(any());
            node.awaitExecutorIdle();
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void twoStaleFollowersForwardOnlyOnce() throws Exception {
        try (Node first = Node.start("10.0.0.1");
             Node second = Node.start("10.0.0.2")) {
            first.masterAddress.set(second.httpAddress());
            second.masterAddress.set(first.httpAddress());

            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response;
            long started = System.nanoTime();
            try (Client client = Client.connect(first.grpcPort())) {
                response = client.stub.schedule(request(72_001L));
            }

            assertTrue(Duration.ofNanos(System.nanoTime() - started)
                            .compareTo(Duration.ofSeconds(2)) < 0,
                    "hop guard must terminate stale follower ping-pong");
            assertFalse(response.getSuccess());
            assertTrue(response.getErrorMessage().contains("FORWARD_HOP_LIMIT"));
            assertEquals(1, first.inboundCalls.get(),
                    "request must not return to the first follower");
            assertEquals(1, second.inboundCalls.get(),
                    "only one forwarded RPC is allowed");
            assertEquals(0, first.rejections.get());
            assertEquals(0, second.rejections.get());
            verify(first.routeService, never()).route(any());
            verify(second.routeService, never()).route(any());
            first.awaitExecutorIdle();
            second.awaitExecutorIdle();
        }
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleRequestPB request(long requestId) {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(requestId)
                .setSeqLen(1024)
                .build();
    }

    private static final class Client implements AutoCloseable {
        private final ManagedChannel channel;
        private final FlexlbServiceGrpc.FlexlbServiceBlockingStub stub;

        private Client(ManagedChannel channel) {
            this.channel = channel;
            this.stub = FlexlbServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(5, TimeUnit.SECONDS);
        }

        static Client connect(int port) {
            return new Client(NettyChannelBuilder
                    .forAddress("127.0.0.1", port)
                    .usePlaintext()
                    .build());
        }

        @Override
        public void close() throws InterruptedException {
            channel.shutdownNow();
            channel.awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    private static final class Node implements AutoCloseable {
        private final AtomicReference<String> masterAddress = new AtomicReference<>();
        private final AtomicInteger inboundCalls = new AtomicInteger();
        private final AtomicInteger rejections = new AtomicInteger();
        private final LBStatusConsistencyService consistency;
        private final RouteService routeService;
        private final FlexlbGrpcForwarder forwarder;
        private final EventLoopGroup channelEventLoop;
        private final ExecutorService channelExecutor;
        private final ThreadPoolExecutor serverExecutor;
        private final Server server;

        private Node(String localIdentity) throws Exception {
            consistency = mock(LBStatusConsistencyService.class);
            when(consistency.isNeedConsistency()).thenReturn(true);
            when(consistency.isMaster()).thenReturn(false);
            when(consistency.getLocalHostIp()).thenReturn(localIdentity);
            when(consistency.getMasterHostIpPort()).thenAnswer(
                    invocation -> masterAddress.get());

            ConfigService configService = mock(ConfigService.class);
            when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
            routeService = mock(RouteService.class);
            EngineHealthReporter healthReporter = mock(EngineHealthReporter.class);
            ActiveRequestCounter activeRequestCounter = mock(ActiveRequestCounter.class);
            when(activeRequestCounter.acquire()).thenReturn(
                    mock(ActiveRequestCounter.RequestToken.class));

            channelEventLoop = new NioEventLoopGroup(1);
            channelExecutor = Executors.newFixedThreadPool(2);
            forwarder = new FlexlbGrpcForwarder(
                    consistency, configService, healthReporter,
                    channelEventLoop, channelExecutor);
            FlexlbServiceImpl service = new FlexlbServiceImpl(
                    routeService,
                    consistency,
                    healthReporter,
                    activeRequestCounter,
                    forwarder,
                    configService,
                    mock(BatchSchedulerReporter.class),
                    mock(ServerScheduleLatencyRecorder.class),
                    mock(RequestSchedulerReporter.class));

            serverExecutor = new ThreadPoolExecutor(
                    4, 4, 0L, TimeUnit.MILLISECONDS,
                    new ArrayBlockingQueue<>(16),
                    runnable -> new Thread(runnable, "hop-guard-netty-test"),
                    (runnable, executor) -> {
                        rejections.incrementAndGet();
                        new ThreadPoolExecutor.AbortPolicy()
                                .rejectedExecution(runnable, executor);
                    });
            ServerInterceptor countCalls = new ServerInterceptor() {
                @Override
                public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
                        ServerCall<ReqT, RespT> call,
                        io.grpc.Metadata headers,
                        ServerCallHandler<ReqT, RespT> next) {
                    inboundCalls.incrementAndGet();
                    return next.startCall(call, headers);
                }
            };
            server = NettyServerBuilder
                    .forAddress(new InetSocketAddress("127.0.0.1", 0))
                    .executor(serverExecutor)
                    .addService(ServerInterceptors.intercept(service, countCalls))
                    .build()
                    .start();
        }

        static Node start(String localIdentity) throws Exception {
            return new Node(localIdentity);
        }

        int grpcPort() {
            return server.getPort();
        }

        String httpAddress() {
            return "127.0.0.1:"
                    + (grpcPort() - FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET);
        }

        void awaitExecutorIdle() throws InterruptedException {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while ((serverExecutor.getActiveCount() != 0
                    || !serverExecutor.getQueue().isEmpty())
                    && System.nanoTime() < deadline) {
                Thread.sleep(10);
            }
            assertEquals(0, serverExecutor.getActiveCount());
            assertTrue(serverExecutor.getQueue().isEmpty());
        }

        @Override
        public void close() throws InterruptedException {
            forwarder.shutdown();
            server.shutdownNow();
            server.awaitTermination(5, TimeUnit.SECONDS);
            serverExecutor.shutdownNow();
            channelExecutor.shutdownNow();
            channelEventLoop.shutdownGracefully().sync();
        }
    }
}

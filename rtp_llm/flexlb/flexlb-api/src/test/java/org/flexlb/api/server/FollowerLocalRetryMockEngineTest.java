package org.flexlb.api.server;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.api.ServerScheduleLatencyRecorder;
import org.flexlb.api.protocol.FlexlbScheduleProtocol;
import org.flexlb.api.protocol.FlexlbServiceGrpc;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.master.MasterStatusService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks follower recovery returns a route through real Netty RPCs and local scheduling.
 */
class FollowerLocalRetryMockEngineTest extends FlexLBMockTestBase {
    @ParameterizedTest
    @ValueSource(strings = {"NOT_MASTER", "SHUTDOWN", "DEAD_BEFORE_CONNECT", "UNRESOLVABLE_HOST"})
    @Timeout(value = 20, unit = TimeUnit.SECONDS)
    void followerRecoveryReturnsOneRouteWithoutBatchSubmission(String failure) throws Exception {
        AbstractRequestScheduler remoteRoutes = mock(AbstractRequestScheduler.class);
        try (Node oldMaster = new Node("10.0.0.1", remoteRoutes);
                Node follower = new Node("10.0.0.2", scheduler);
                var deadAddress = java.nio.channels.SocketChannel.open()) {
            when(follower.leadership.getMasterHostIpPort()).thenReturn(oldMaster.httpAddress());
            if (failure.equals("SHUTDOWN")) {
                oldMaster.server.shutdown().awaitTermination(3, TimeUnit.SECONDS);
            } else if (failure.equals("DEAD_BEFORE_CONNECT")) {
                oldMaster.server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            } else if (failure.equals("UNRESOLVABLE_HOST")) {
                // The reserved .invalid domain exercises actual DNS resolution failure.
                when(follower.leadership.getMasterHostIpPort())
                        .thenReturn("flexlb-unresolvable.invalid.:7001");
            }
            if (failure.equals("SHUTDOWN") || failure.equals("DEAD_BEFORE_CONNECT")) {
                // Reserve a fresh non-listening address exclusively. A closed listener's
                // port can still be occupied by connections from an earlier parameter case.
                deadAddress.bind(new java.net.InetSocketAddress("127.0.0.1", 0));
                int deadPort =
                        ((java.net.InetSocketAddress) deadAddress.getLocalAddress()).getPort();
                when(follower.leadership.getMasterHostIpPort())
                        .thenReturn(
                                "127.0.0.1:"
                                        + (deadPort - FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET));
                follower.useBoundedConnectFailure(deadPort);
            }
            long requestId = 91_001L;
            var originalInput =
                    org.flexlb.worker.rpc.protocol.EngineRpcService.GenerateInputPB.newBuilder()
                            .setRequestId(requestId)
                            .setGenerateConfig(
                                    org.flexlb.worker.rpc.protocol.EngineRpcService.GenerateConfigPB
                                            .newBuilder()
                                            .setMaxNewTokens(8))
                            .build()
                            .toByteString();
            var request =
                    FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                            .setRequestId(requestId)
                            .setSeqLen(128)
                            .setMaxNewTokens(8)
                            .setNumBeams(1)
                            .setModel("mock-model")
                            .setGenerateInput(originalInput)
                            .build();
            ManagedChannel frontend =
                    NettyChannelBuilder.forAddress("127.0.0.1", follower.server.getPort())
                            .usePlaintext()
                            .build();
            try {
                GrpcReceivingGuardRepro.awaitReady(frontend);
                var response =
                        FlexlbServiceGrpc.newBlockingStub(frontend)
                                .withDeadlineAfter(8, TimeUnit.SECONDS)
                                .schedule(request);
                assertTrue(response.getSuccess(), response.getErrorMessage());
                org.junit.jupiter.api.Assertions.assertFalse(response.getEnqueuedByMaster());
                assertTrue(response.getServerStatusCount() > 0);
                assertEquals(0, mockPrefillWorker.getEnqueueCount());
                assertEquals(0, mockDecodeWorker.getEnqueueCount());
                verify(remoteRoutes, never()).submit(any());
            } finally {
                frontend.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            }
        }
    }

    /**
     * Owns a mock-engine test node and its HTTP endpoint.
     */
    private final class Node implements AutoCloseable {
        final MasterStatusService leadership = mock(MasterStatusService.class);
        final NioEventLoopGroup channelLoop = new NioEventLoopGroup(1);
        final FlexlbGrpcForwarder forwarder;
        final FlexlbServiceImpl service;
        final Server server;

        Node(String identity, AbstractRequestScheduler routes) throws Exception {
            when(leadership.isNeedConsistency()).thenReturn(true);
            when(leadership.isMaster()).thenReturn(false);
            when(leadership.getLocalHostIp()).thenReturn(identity);
            var health = mock(ServiceMetricsReporter.class);
            forwarder =
                    new FlexlbGrpcForwarder(
                            leadership, configService, health, channelLoop, Runnable::run);
            service =
                    FlexlbServiceTestSupport.create(
                            routes,
                            routes == scheduler
                                    ? requestRegistry()
                                    : mock(org.flexlb.balance.schedule.RequestRepository.class),
                            leadership,
                            health,
                            forwarder,
                            configService,
                            reporter,
                            mock(ServerScheduleLatencyRecorder.class),
                            mock(RequestSchedulerReporter.class));
            server =
                    NettyServerBuilder.forAddress(new java.net.InetSocketAddress("127.0.0.1", 0))
                            .channelType(io.netty.channel.socket.nio.NioServerSocketChannel.class)
                            .bossEventLoopGroup(channelLoop)
                            .workerEventLoopGroup(channelLoop)
                            .addService(service)
                            .build()
                            .start();
        }

        @Override
        public void close() throws Exception {
            server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            forwarder.shutdown();
            channelLoop.shutdownGracefully(0, 1, TimeUnit.SECONDS).sync();
        }

        @SuppressWarnings("unchecked")
        void useBoundedConnectFailure(int port) {
            // A bound but non-listening socket can drop SYNs on macOS instead of
            // refusing them. Keep that real connect failure inside the RPC deadline.
            ManagedChannel channel =
                    NettyChannelBuilder.forAddress("127.0.0.1", port)
                            .channelType(io.netty.channel.socket.nio.NioSocketChannel.class)
                            .eventLoopGroup(channelLoop)
                            .withOption(io.netty.channel.ChannelOption.CONNECT_TIMEOUT_MILLIS, 500)
                            .usePlaintext()
                            .disableRetry()
                            .build();
            var channels =
                    (java.util.Map<String, ManagedChannel>)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    forwarder, "channels");
            channels.put("127.0.0.1:" + port, channel);
        }

        String httpAddress() {
            return "127.0.0.1:" + (server.getPort() - FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET);
        }
    }
}

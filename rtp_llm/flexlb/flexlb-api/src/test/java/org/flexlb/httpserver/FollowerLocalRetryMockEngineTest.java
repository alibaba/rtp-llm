package org.flexlb.httpserver;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.consistency.MasterStatusService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
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

/** Real scheduler, dispatcher and Netty engine RPCs; only engine compute and discovery are simulated. */
class FollowerLocalRetryMockEngineTest extends FlexLBMockTestBase {
    @ParameterizedTest
    @ValueSource(strings = {"NOT_MASTER", "SHUTDOWN", "DEAD_BEFORE_CONNECT", "UNRESOLVABLE_HOST"})
    @Timeout(value = 20, unit = TimeUnit.SECONDS)
    void followerRecoveryDispatchesExactlyOnceToMockEngine(String failure) throws Exception {
        RequestScheduler remoteRoutes = mock(RequestScheduler.class);
        try (Node oldMaster = new Node("10.0.0.1", remoteRoutes);
             Node follower = new Node("10.0.0.2", scheduler);
             var deadAddress = java.nio.channels.SocketChannel.open()) {
            int oldMasterPort = oldMaster.server.getPort();
            when(follower.leadership.getMasterHostIpPort()).thenReturn(oldMaster.httpAddress());
            if (failure.equals("SHUTDOWN")) {
                oldMaster.server.shutdown().awaitTermination(3, TimeUnit.SECONDS);
            } else if (failure.equals("DEAD_BEFORE_CONNECT")) {
                oldMaster.server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            } else if (failure.equals("UNRESOLVABLE_HOST")) {
                // The reserved .invalid domain exercises actual DNS resolution failure.
                when(follower.leadership.getMasterHostIpPort()).thenReturn("flexlb-unresolvable.invalid.:7001");
            }
            if (failure.equals("SHUTDOWN") || failure.equals("DEAD_BEFORE_CONNECT")) {
                // Reserve the dead listener's address without listening, so another
                // local service cannot turn a connect-failure test into an HTTP response.
                deadAddress.bind(new java.net.InetSocketAddress("127.0.0.1", oldMasterPort));
                follower.useBoundedConnectFailure(oldMasterPort);
            }
            long requestId = 91_001L;
            var originalInput = org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB.newBuilder()
                    .setRequestId(requestId)
                    .setGenerateConfig(org.flexlb.engine.grpc.EngineRpcService.GenerateConfigPB.newBuilder()
                            .setMaxNewTokens(8)).build().toByteString();
            var request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                    .setRequestId(requestId).setSeqLen(128).setMaxNewTokens(8)
                    .setNumBeams(1).setModel("mock-model")
                    .setGenerateInput(originalInput).build();
            ManagedChannel frontend = NettyChannelBuilder.forAddress("127.0.0.1", follower.server.getPort())
                    .usePlaintext().build();
            try {
                GrpcReceivingGuardRepro.awaitReady(frontend);
                var response = FlexlbServiceGrpc.newBlockingStub(frontend)
                        .withDeadlineAfter(8, TimeUnit.SECONDS).schedule(request);
                assertTrue(response.getSuccess(), response.getErrorMessage());
                assertTrue(response.getEnqueuedByMaster());
                assertTrue(response.getLifecycle().getBatchId() > 0);
                assertEquals(requestId, response.getLifecycle().getRequestId());
                assertEquals(1, mockPrefillWorker.getEnqueueCount(), "recovery must dispatch exactly once");
                assertEquals(0, mockDecodeWorker.getEnqueueCount());
                verify(remoteRoutes, never()).submit(any());
            } finally {
                frontend.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            }
        }
    }

    private final class Node implements AutoCloseable {
        final MasterStatusService leadership = mock(MasterStatusService.class);
        final NioEventLoopGroup channelLoop = new NioEventLoopGroup(1);
        final FlexlbGrpcForwarder forwarder;
        final FlexlbServiceImpl service;
        final Server server;

        Node(String identity, RequestScheduler routes) throws Exception {
            when(leadership.isNeedConsistency()).thenReturn(true);
            when(leadership.isMaster()).thenReturn(false);
            when(leadership.getLocalHostIp()).thenReturn(identity);
            var health = mock(EngineHealthReporter.class);
            forwarder = new FlexlbGrpcForwarder(leadership, configService, health, channelLoop, Runnable::run);
            service = FlexlbServiceTestSupport.create(routes, routes == scheduler ? requestRegistry() : mock(org.flexlb.balance.scheduler.RequestRepository.class), leadership, health, forwarder, configService, reporter, mock(ServerScheduleLatencyRecorder.class), mock(RequestSchedulerReporter.class));
            server = NettyServerBuilder.forAddress(new java.net.InetSocketAddress("127.0.0.1", 0))
                    .channelType(io.netty.channel.socket.nio.NioServerSocketChannel.class)
                    .bossEventLoopGroup(channelLoop).workerEventLoopGroup(channelLoop)
                    .addService(service).build().start();
        }

        @SuppressWarnings("unchecked")
        void useBoundedConnectFailure(int port) {
            // A bound but non-listening socket can drop SYNs on macOS instead of
            // refusing them. Keep that real connect failure inside the RPC deadline.
            ManagedChannel channel = NettyChannelBuilder.forAddress("127.0.0.1", port)
                    .channelType(io.netty.channel.socket.nio.NioSocketChannel.class)
                    .eventLoopGroup(channelLoop)
                    .withOption(io.netty.channel.ChannelOption.CONNECT_TIMEOUT_MILLIS, 500)
                    .usePlaintext().disableRetry().build();
            var channels = (java.util.Map<String, ManagedChannel>)
                    org.springframework.test.util.ReflectionTestUtils.getField(forwarder, "channels");
            channels.put("127.0.0.1:" + port, channel);
        }

        String httpAddress() {
            return "127.0.0.1:" + (server.getPort() - FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET);
        }

        @Override
        public void close() throws Exception {
            server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            forwarder.shutdown();
            channelLoop.shutdownGracefully(0, 1, TimeUnit.SECONDS).sync();
        }
    }
}

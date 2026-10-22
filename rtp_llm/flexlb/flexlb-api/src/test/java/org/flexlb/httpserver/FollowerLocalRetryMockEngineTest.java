package org.flexlb.httpserver;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.RouteService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/** Real scheduler, dispatcher and Netty engine RPCs; only engine compute and discovery are simulated. */
class FollowerLocalRetryMockEngineTest extends FlexLBMockTestBase {
    @ParameterizedTest
    @ValueSource(strings = {"NOT_MASTER", "SHUTDOWN", "DEAD_BEFORE_CONNECT", "UNRESOLVABLE_HOST"})
    @Timeout(value = 20, unit = TimeUnit.SECONDS)
    void followerRecoveryDispatchesExactlyOnceToMockEngine(String failure) throws Exception {
        RouteService remoteRoutes = mock(RouteService.class);
        RouteService localRoutes = new RouteService(scheduler, mock(RecentCacheKeyTraceReporter.class));
        try (Node oldMaster = new Node("10.0.0.1", remoteRoutes);
             Node follower = new Node("10.0.0.2", localRoutes)) {
            when(follower.leadership.getMasterHostIpPort()).thenReturn(oldMaster.httpAddress());
            if (failure.equals("SHUTDOWN")) {
                oldMaster.server.shutdown().awaitTermination(3, TimeUnit.SECONDS);
            } else if (failure.equals("DEAD_BEFORE_CONNECT")) {
                oldMaster.server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            } else if (failure.equals("UNRESOLVABLE_HOST")) {
                // The reserved .invalid domain exercises actual DNS resolution failure.
                when(follower.leadership.getMasterHostIpPort()).thenReturn("flexlb-unresolvable.invalid.:7001");
            }
            long requestId = 91_001L;
            var original = createBalanceContext(requestId);
            var request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                    .setRequestId(requestId).setSeqLen(128).setMaxNewTokens(8)
                    .setNumBeams(1).setModel("mock-model")
                    .setGenerateInput(original.getGenerateInputPb()).build();
            ManagedChannel frontend = NettyChannelBuilder.forAddress("127.0.0.1", follower.server.getPort())
                    .usePlaintext().build();
            try {
                var response = FlexlbServiceGrpc.newBlockingStub(frontend)
                        .withDeadlineAfter(8, TimeUnit.SECONDS).schedule(request);
                assertTrue(response.getSuccess(), response.getErrorMessage());
                assertTrue(response.getEnqueuedByMaster());
                assertTrue(response.getLifecycle().getBatchId() > 0);
                assertEquals(requestId, response.getLifecycle().getRequestId());
                assertEquals(1, mockPrefillWorker.getEnqueueCount(), "recovery must dispatch exactly once");
                assertEquals(0, mockDecodeWorker.getEnqueueCount());
                verify(remoteRoutes, never()).route(any());
            } finally {
                frontend.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            }
        }
    }

    private final class Node implements AutoCloseable {
        final LBStatusConsistencyService leadership = mock(LBStatusConsistencyService.class);
        final NioEventLoopGroup channelLoop = new NioEventLoopGroup(1);
        final FlexlbGrpcForwarder forwarder;
        final FlexlbServiceImpl service;
        final Server server;

        Node(String identity, RouteService routes) throws Exception {
            when(leadership.isNeedConsistency()).thenReturn(true);
            when(leadership.isMaster()).thenReturn(false);
            when(leadership.getLocalHostIp()).thenReturn(identity);
            var health = mock(EngineHealthReporter.class);
            forwarder = new FlexlbGrpcForwarder(leadership, configService, health, channelLoop, Runnable::run);
            service = new FlexlbServiceImpl(routes, leadership, health, forwarder, configService,
                    reporter, mock(ServerScheduleLatencyRecorder.class), mock(RequestSchedulerReporter.class));
            server = NettyServerBuilder.forPort(0).addService(service).build().start();
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

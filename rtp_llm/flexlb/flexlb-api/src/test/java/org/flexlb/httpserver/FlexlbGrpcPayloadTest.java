package org.flexlb.httpserver;

import com.google.protobuf.ByteString;
import io.grpc.Server;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.mock.TestFlexlbConfigs;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.core.env.Environment;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class FlexlbGrpcPayloadTest {
    @Test
    @Timeout(30)
    void serverAndForwarderAcceptTwentyMiBMessages() throws Exception {
        var request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(918L)
                .setGenerateInput(ByteString.copyFrom(new byte[20 * 1024 * 1024]))
                .build();
        // A large response also exercises the production forwarder's receive limit.
        String responsePayload = "x".repeat(20 * 1024 * 1024);
        AtomicReference<FlexlbScheduleProtocol.FlexlbScheduleRequestPB> received = new AtomicReference<>();
        FlexlbServiceImpl service = mock(FlexlbServiceImpl.class, CALLS_REAL_METHODS);
        doAnswer(call -> {
            received.set(call.getArgument(0));
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = call.getArgument(1);
            observer.onNext(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                    .setSuccess(true).setErrorMessage(responsePayload).build());
            observer.onCompleted();
            return null;
        }).when(service).schedule(any(), any());
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(TestFlexlbConfigs.create());
        Environment environment = mock(Environment.class);
        // HTTP + 2 = 0 lets the operating system select a free gRPC port.
        when(environment.getProperty("server.port")).thenReturn("-2");
        NioEventLoopGroup serverGroup = new NioEventLoopGroup(1);
        NioEventLoopGroup clientGroup = new NioEventLoopGroup(1);
        FlexlbGrpcServer server = new FlexlbGrpcServer(service, config, environment, serverGroup,
                null, new GrpcServerTimingInterceptor(), new GrpcQosHeaderInterceptor());
        LBStatusConsistencyService election = mock(LBStatusConsistencyService.class);
        FlexlbGrpcForwarder forwarder = new FlexlbGrpcForwarder(election, config,
                mock(EngineHealthReporter.class), clientGroup, Runnable::run);
        try {
            server.start();
            int port = ((Server) ReflectionTestUtils.getField(server, "server")).getPort();
            when(election.getMasterHostIpPort()).thenReturn("127.0.0.1:" + (port - 2));
            when(election.getLocalHostIp()).thenReturn("127.0.0.2");

            var result = forwarder.forwardScheduleToMaster(request).toCompletableFuture()
                    .get(20, TimeUnit.SECONDS);

            assertNotNull(result.response(), result.failure());
            assertTrue(result.response().getSuccess());
            assertEquals(responsePayload, result.response().getErrorMessage());
            assertEquals(request.getGenerateInput(), received.get().getGenerateInput());
            assertEquals(1, received.get().getForwardHop());
        } finally {
            forwarder.shutdown();
            server.shutdown();
            clientGroup.shutdownGracefully().sync();
            serverGroup.terminationFuture().sync();
        }
    }
}

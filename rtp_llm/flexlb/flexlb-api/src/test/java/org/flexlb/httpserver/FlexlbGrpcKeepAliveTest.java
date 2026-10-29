package org.flexlb.httpserver;

import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.RouteService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.mock.env.MockEnvironment;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * grpc-java answers the third HTTP/2 PING that arrives sooner than the permitted interval with
 * GOAWAY(ENHANCE_YOUR_CALM, "too_many_pings") and closes the connection, failing every call on it.
 * The tests send raw PING frames 20 ms apart to the real server, because a grpc-java client never
 * pings more often than every 10 seconds.
 */
@Timeout(30)
class FlexlbGrpcKeepAliveTest {

    private static final int NO_GOAWAY = -1;
    private static final int ENHANCE_YOUR_CALM = 11;
    private static final int FRAME_GOAWAY = 7;

    @Test
    void pingsSoonerThanPermitKeepAliveTimeCloseTheConnection() throws Exception {
        try (var server = new RunningServer("""
                {"permitKeepAliveTimeMs":60000,"permitKeepAliveWithoutCalls":true}""")) {
            assertEquals(ENHANCE_YOUR_CALM, server.pingRepeatedly(10));
        }
    }

    @Test
    void zeroPermitKeepAliveTimeAcceptsPingsAtAnyRate() throws Exception {
        try (var server = new RunningServer("""
                {"permitKeepAliveTimeMs":0,"permitKeepAliveWithoutCalls":true}""")) {
            assertEquals(NO_GOAWAY, server.pingRepeatedly(10));
        }
    }

    @Test
    void pingsOnAnIdleConnectionCloseItUnlessPermittedWithoutCalls() throws Exception {
        try (var server = new RunningServer("""
                {"permitKeepAliveTimeMs":0,"permitKeepAliveWithoutCalls":false}""")) {
            assertEquals(ENHANCE_YOUR_CALM, server.pingRepeatedly(10));
        }
    }

    @Test
    void defaultsRejectPingsFasterThanTheTenSecondMinimumOfAGrpcJavaClient() throws Exception {
        try (var server = new RunningServer("{}")) {
            assertNotEquals(NO_GOAWAY, server.pingRepeatedly(10));
        }
    }

    private static final class RunningServer implements AutoCloseable {
        private final FlexlbGrpcServer grpc;
        private final int port;

        RunningServer(String grpcServerJson) throws Exception {
            try (var probe = new ServerSocket(0)) {
                port = probe.getLocalPort();
            }
            var configService = mock(ConfigService.class);
            var merged = grpcServerJson.equals("{}")
                    ? "{\"shutdownQuietPeriodMs\":1}"
                    : grpcServerJson.replaceFirst("\\{", "{\"shutdownQuietPeriodMs\":1,");
            when(configService.loadBalanceConfig()).thenReturn(ConfigService.parse("""
                    {"requestLifecycle":{"request":{"timeoutMs":60000}},"grpcServer":%s}
                    """.formatted(merged)));
            var service = new FlexlbServiceImpl(mock(RouteService.class), mock(LBStatusConsistencyService.class),
                    mock(EngineHealthReporter.class), mock(FlexlbGrpcForwarder.class), configService,
                    mock(BatchSchedulerReporter.class), mock(ServerScheduleLatencyRecorder.class),
                    mock(RequestSchedulerReporter.class));
            // The gRPC port is the HTTP port plus FLEXLB_GRPC_PORT_OFFSET.
            var environment = new MockEnvironment().withProperty("server.port",
                    String.valueOf(port - FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET));
            grpc = new FlexlbGrpcServer(service, configService, environment, new NioEventLoopGroup(1),
                    NoOpFlexMonitor.getInstance(), new GrpcServerTimingInterceptor(),
                    new GrpcQosHeaderInterceptor());
            grpc.start();
        }

        /** Sends {@code count} PINGs 20 ms apart; returns the GOAWAY error code the server sent, if any. */
        int pingRepeatedly(int count) throws Exception {
            var goAwayCode = new AtomicInteger(NO_GOAWAY);
            try (var socket = new Socket("127.0.0.1", port)) {
                OutputStream out = socket.getOutputStream();
                out.write("PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".getBytes(StandardCharsets.US_ASCII));
                out.write(frame(4, new byte[0]));
                var reader = new Thread(() -> readFrames(socket, goAwayCode), "raw-http2-reader");
                reader.setDaemon(true);
                reader.start();
                for (int i = 0; i < count && goAwayCode.get() == NO_GOAWAY; i++) {
                    try {
                        out.write(frame(6, new byte[8]));
                        out.flush();
                    } catch (IOException connectionClosedByServer) {
                        break;
                    }
                    Thread.sleep(20);
                }
                reader.join(500);
            }
            return goAwayCode.get();
        }

        private static byte[] frame(int type, byte[] payload) {
            var frame = ByteBuffer.allocate(9 + payload.length);
            frame.put((byte) (payload.length >> 16)).put((byte) (payload.length >> 8)).put((byte) payload.length);
            frame.put((byte) type).put((byte) 0).putInt(0).put(payload);
            return frame.array();
        }

        private static void readFrames(Socket socket, AtomicInteger goAwayCode) {
            try {
                var in = new DataInputStream(socket.getInputStream());
                while (true) {
                    var header = new byte[9];
                    in.readFully(header);
                    var payload = new byte[((header[0] & 0xff) << 16) | ((header[1] & 0xff) << 8)
                            | (header[2] & 0xff)];
                    in.readFully(payload);
                    if (header[3] == FRAME_GOAWAY) {
                        goAwayCode.set(ByteBuffer.wrap(payload).getInt(4));
                    }
                }
            } catch (IOException connectionClosed) {
                // The server closed the connection, or the test closed the socket.
            }
        }

        @Override
        public void close() {
            grpc.shutdown();
        }
    }
}

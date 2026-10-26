package org.flexlb.engine.grpc;

import io.grpc.Attributes;
import io.grpc.Grpc;
import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.EOFException;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GrpcTransportObserverTest {
    private final GrpcTransportObserver observer = new GrpcTransportObserver();
    private final AtomicReference<Throwable> propagatedFailure = new AtomicReference<>();
    private NioEventLoopGroup group;
    private NioSocketChannel channel;
    private ServerSocket server;
    private Socket peer;

    @AfterEach
    void closeSockets() throws IOException {
        try {
            if (channel != null) {
                channel.close().syncUninterruptibly();
            }
            if (peer != null) {
                peer.close();
            }
            if (server != null) {
                server.close();
            }
        } finally {
            if (group != null) {
                group.shutdownGracefully(0, 5, TimeUnit.SECONDS).syncUninterruptibly();
            }
        }
    }

    @Test
    void peer_eof_survives_socket_removal_for_streams_holding_the_state() throws Exception {
        Attributes attributes = connect();
        GrpcTransportObserver.SocketState state = observer.find(attributes);
        assertNotNull(state);
        assertNull(state.observation());

        long before = System.nanoTime();
        peer.close();
        assertTrue(channel.closeFuture().await(5, TimeUnit.SECONDS));

        assertNull(observer.find(attributes));
        assertNotNull(state.observation());
        assertEquals(GrpcTransportObserver.Event.PEER_EOF, state.observation().event());
        assertEquals(GrpcFailure.Kind.UNEXPECTED_EOF, state.observation().causeKind());
        assertTrue(state.observation().nanoTime() - before >= 0);
    }

    @Test
    void local_input_shutdown_returning_eof_is_not_peer_eof() throws Exception {
        Attributes attributes = connect();
        GrpcTransportObserver.SocketState state = observer.find(attributes);
        assertNotNull(state);
        channel.config().setAutoRead(false);
        channel.shutdownInput().syncUninterruptibly();

        int read =
                channel.eventLoop()
                        .submit(
                                () -> {
                                    ByteBuf buffer = Unpooled.buffer(8);
                                    try {
                                        return ((GrpcTransportObserver.ObservedChannel) channel)
                                                .doReadBytes(buffer);
                                    } finally {
                                        buffer.release();
                                    }
                                })
                        .get(5, TimeUnit.SECONDS);

        assertEquals(-1, read);
        assertNull(state.observation());
        assertSame(state, observer.find(attributes));
        channel.close().syncUninterruptibly();
        assertNull(observer.find(attributes));
        assertNull(state.observation());
    }

    @Test
    void lookup_requires_both_exact_socket_addresses() throws Exception {
        Attributes attributes = connect();
        assertNotNull(observer.find(attributes));
        assertNull(observer.find(null));
        assertNull(observer.find(Attributes.EMPTY));
        assertNull(
                observer.find(
                        Attributes.newBuilder()
                                .set(Grpc.TRANSPORT_ATTR_LOCAL_ADDR, channel.localAddress())
                                .build()));
        assertNull(
                observer.find(
                        attributes.toBuilder()
                                .set(
                                        Grpc.TRANSPORT_ATTR_REMOTE_ADDR,
                                        new InetSocketAddress("127.0.0.1", 1))
                                .build()));
        assertNull(
                observer.find(
                        attributes.toBuilder()
                                .set(
                                        Grpc.TRANSPORT_ATTR_LOCAL_ADDR,
                                        new InetSocketAddress("127.0.0.1", 1))
                                .build()));
    }

    @Test
    void pipeline_read_allocation_failure_is_observed_and_propagated_unchanged() throws Exception {
        Attributes attributes = connect();
        GrpcTransportObserver.SocketState state = observer.find(attributes);
        assertNotNull(state);
        OutOfMemoryError failure = new OutOfMemoryError("Cannot reserve direct buffer memory");

        channel.eventLoop()
                .submit(() -> channel.pipeline().fireExceptionCaught(failure))
                .get(5, TimeUnit.SECONDS);

        assertSame(failure, propagatedFailure.get());
        assertEquals(GrpcTransportObserver.Event.READ_FAILURE, state.observation().event());
        assertEquals(GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED, state.observation().causeKind());
        assertNull(observer.find(attributes));
    }

    @Test
    void pipeline_diagnostic_failure_cannot_replace_or_stop_the_original_exception()
            throws Exception {
        Attributes attributes = connect();
        GrpcTransportObserver.SocketState state = observer.find(attributes);
        assertNotNull(state);
        Throwable failure =
                new RuntimeException() {
                    @Override
                    public synchronized Throwable getCause() {
                        throw new OutOfMemoryError("diagnostic failure");
                    }
                };

        channel.eventLoop()
                .submit(() -> channel.pipeline().fireExceptionCaught(failure))
                .get(5, TimeUnit.SECONDS);

        assertSame(failure, propagatedFailure.get());
        assertNull(state.observation());
        assertNull(observer.find(attributes));
    }

    @ParameterizedTest
    @MethodSource("failures")
    void socket_failure_preserves_direction_and_specific_cause(
            boolean read,
            Throwable failure,
            GrpcTransportObserver.Event event,
            GrpcFailure.Kind kind) {
        var state = new GrpcTransportObserver.SocketState();
        state.recordFailure(read, failure);
        assertNotNull(state.observation());
        assertEquals(event, state.observation().event());
        assertEquals(kind, state.observation().causeKind());
    }

    static Stream<Arguments> failures() {
        return Stream.of(
                Arguments.of(
                        true,
                        new SocketException("Connection reset"),
                        GrpcTransportObserver.Event.READ_RESET,
                        GrpcFailure.Kind.CONNECTION_RESET),
                Arguments.of(
                        false,
                        new SocketException("Connection reset by peer"),
                        GrpcTransportObserver.Event.WRITE_RESET,
                        GrpcFailure.Kind.CONNECTION_RESET),
                Arguments.of(
                        false,
                        new SocketException("Broken pipe"),
                        GrpcTransportObserver.Event.WRITE_BROKEN_PIPE,
                        GrpcFailure.Kind.BROKEN_PIPE),
                Arguments.of(
                        true,
                        new SocketException("Broken pipe"),
                        GrpcTransportObserver.Event.READ_FAILURE,
                        GrpcFailure.Kind.BROKEN_PIPE),
                Arguments.of(
                        true,
                        new IOException("read failed"),
                        GrpcTransportObserver.Event.READ_FAILURE,
                        GrpcFailure.Kind.IO_FAILURE),
                Arguments.of(
                        false,
                        new OutOfMemoryError("direct memory"),
                        GrpcTransportObserver.Event.WRITE_FAILURE,
                        GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED),
                Arguments.of(
                        true,
                        new EOFException(),
                        GrpcTransportObserver.Event.READ_FAILURE,
                        GrpcFailure.Kind.UNEXPECTED_EOF));
    }

    @Test
    void later_failures_cannot_replace_the_first_socket_observation() {
        var state = new GrpcTransportObserver.SocketState();
        state.recordFailure(true, new SocketException("Connection reset"));
        var first = state.observation();

        state.recordFailure(false, new SocketException("Broken pipe"));

        assertSame(first, state.observation());
    }

    @Test
    void failed_cause_inspection_is_ignored_without_preventing_later_observation() {
        var state = new GrpcTransportObserver.SocketState();
        Throwable brokenDiagnostic =
                new RuntimeException() {
                    @Override
                    public synchronized Throwable getCause() {
                        throw new OutOfMemoryError("diagnostic failure");
                    }
                };

        assertDoesNotThrow(() -> state.recordFailure(true, brokenDiagnostic));
        assertNull(state.observation());
        state.recordFailure(false, new SocketException("Broken pipe"));
        assertEquals(GrpcTransportObserver.Event.WRITE_BROKEN_PIPE, state.observation().event());
    }

    private Attributes connect() throws IOException {
        server = new ServerSocket(0, 1, InetAddress.getLoopbackAddress());
        server.setSoTimeout(5000);
        group = new NioEventLoopGroup(1);
        channel =
                (NioSocketChannel)
                        new Bootstrap()
                                .group(group)
                                .channelFactory(observer::newChannel)
                                .handler(
                                        new ChannelInboundHandlerAdapter() {
                                            @Override
                                            public void exceptionCaught(
                                                    ChannelHandlerContext context,
                                                    Throwable cause) {
                                                propagatedFailure.compareAndSet(null, cause);
                                                context.close();
                                            }
                                        })
                                .connect(server.getLocalSocketAddress())
                                .syncUninterruptibly()
                                .channel();
        peer = server.accept();
        return Attributes.newBuilder()
                .set(Grpc.TRANSPORT_ATTR_LOCAL_ADDR, channel.localAddress())
                .set(Grpc.TRANSPORT_ATTR_REMOTE_ADDR, channel.remoteAddress())
                .build();
    }
}

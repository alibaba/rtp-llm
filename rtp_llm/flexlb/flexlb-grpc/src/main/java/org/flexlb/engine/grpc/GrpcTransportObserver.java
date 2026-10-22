package org.flexlb.engine.grpc;

import io.grpc.Attributes;
import io.grpc.Grpc;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.channel.ChannelOutboundBuffer;
import io.netty.channel.socket.nio.NioSocketChannel;

import java.net.SocketAddress;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/** Keeps the first socket failure for streams already using that physical connection. */
public final class GrpcTransportObserver {
    private final ConcurrentMap<SocketKey, SocketState> sockets = new ConcurrentHashMap<>();

    public enum Event {
        PEER_EOF,
        READ_RESET,
        WRITE_RESET,
        WRITE_BROKEN_PIPE,
        READ_FAILURE,
        WRITE_FAILURE
    }

    public record Observation(Event event, long nanoTime, GrpcFailure.Kind causeKind) {}

    private record SocketKey(SocketAddress local, SocketAddress remote) {}

    public NioSocketChannel newChannel() {
        return new ObservedChannel();
    }

    public SocketState find(Attributes attributes) {
        try {
            if (attributes == null) {
                return null;
            }
            SocketAddress local = attributes.get(Grpc.TRANSPORT_ATTR_LOCAL_ADDR);
            SocketAddress remote = attributes.get(Grpc.TRANSPORT_ATTR_REMOTE_ADDR);
            return local == null || remote == null
                    ? null
                    : sockets.get(new SocketKey(local, remote));
        } catch (RuntimeException | Error ignored) {
            return null;
        }
    }

    public static final class SocketState {
        // Written only by the socket's EventLoop; a stream callback may read it on another thread.
        private volatile Observation observation;

        public Observation observation() {
            return observation;
        }

        void recordFailure(boolean read, Throwable failure) {
            try {
                if (observation != null) {
                    return;
                }
                GrpcFailure.Kind kind = GrpcFailure.classify(failure);
                Event event = read ? Event.READ_FAILURE : Event.WRITE_FAILURE;
                if (kind == GrpcFailure.Kind.CONNECTION_RESET) {
                    event = read ? Event.READ_RESET : Event.WRITE_RESET;
                } else if (!read && kind == GrpcFailure.Kind.BROKEN_PIPE) {
                    event = Event.WRITE_BROKEN_PIPE;
                }
                record(event, kind);
            } catch (RuntimeException | Error ignored) {
                // Diagnostics must never replace the transport's original failure.
            }
        }

        private void record(Event event, GrpcFailure.Kind kind) {
            try {
                if (observation == null) {
                    observation = new Observation(event, System.nanoTime(), kind);
                }
            } catch (RuntimeException | Error ignored) {
                // In particular, do not make memory exhaustion worse by logging here.
            }
        }
    }

    final class ObservedChannel extends NioSocketChannel {
        private SocketKey key;
        private SocketState state;

        ObservedChannel() {
            try {
                pipeline()
                        .addLast(
                                new ChannelInboundHandlerAdapter() {
                                    @Override
                                    public void exceptionCaught(
                                            ChannelHandlerContext context, Throwable cause) {
                                        // Read-buffer allocation can fail before doReadBytes is
                                        // called.
                                        if (state != null) {
                                            state.recordFailure(true, cause);
                                        }
                                        context.fireExceptionCaught(cause);
                                    }
                                });
            } catch (RuntimeException | Error ignored) {
                // Installing diagnostics must not prevent channel construction.
            }
        }

        @Override
        protected boolean doConnect(SocketAddress remote, SocketAddress local) throws Exception {
            boolean connected = super.doConnect(remote, local);
            if (connected) {
                registerSocket();
            }
            return connected;
        }

        @Override
        protected void doFinishConnect() throws Exception {
            super.doFinishConnect();
            registerSocket();
        }

        private void registerSocket() {
            try {
                key = new SocketKey(localAddress(), remoteAddress());
                state = new SocketState();
                sockets.put(key, state);
            } catch (RuntimeException | Error ignored) {
                // A failed observation must not fail an otherwise connected socket.
            }
        }

        @Override
        protected int doReadBytes(ByteBuf buffer) throws Exception {
            boolean inputOpen = inputOpen();
            try {
                int read = super.doReadBytes(buffer);
                if (read < 0 && inputOpen && inputOpen() && state != null) {
                    state.record(Event.PEER_EOF, GrpcFailure.Kind.UNEXPECTED_EOF);
                }
                return read;
            } catch (Exception | Error failure) {
                if (state != null) {
                    state.recordFailure(true, failure);
                }
                throw failure;
            }
        }

        private boolean inputOpen() {
            try {
                return !isInputShutdown();
            } catch (RuntimeException | Error ignored) {
                return false;
            }
        }

        @Override
        protected void doWrite(ChannelOutboundBuffer buffer) throws Exception {
            try {
                // This also covers gathering writes, which bypass doWriteBytes.
                super.doWrite(buffer);
            } catch (Exception | Error failure) {
                if (state != null) {
                    state.recordFailure(false, failure);
                }
                throw failure;
            }
        }

        @Override
        protected void doClose() throws Exception {
            try {
                super.doClose();
            } finally {
                try {
                    if (key != null && state != null) {
                        sockets.remove(key, state);
                    }
                } catch (RuntimeException | Error ignored) {
                    // Closing the socket must preserve its original result.
                }
            }
        }
    }
}

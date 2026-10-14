package org.flexlb.engine.grpc;

import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.KnownLength;
import io.grpc.MethodDescriptor;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.CompletableFuture;

/** Response byte metadata travels with the future, never with a payload map. */
public final class RpcResponseFuture<T> extends CompletableFuture<T> {
    private volatile long responseBytes = -1;

    public long responseBytes() { return responseBytes; }

    public static long responseBytes(CompletableFuture<?> future) {
        return future instanceof RpcResponseFuture<?> observed ? observed.responseBytes() : -1;
    }

    public Channel observe(Channel delegate) {
        return new Channel() {
            @Override public String authority() { return delegate.authority(); }
            @Override public <ReqT, RespT> ClientCall<ReqT, RespT> newCall(
                    MethodDescriptor<ReqT, RespT> method, CallOptions options) {
                var observed = method.toBuilder(method.getRequestMarshaller(),
                        measuring(method.getResponseMarshaller())).build();
                return delegate.newCall(observed, options);
            }
        };
    }

    <V> MethodDescriptor.Marshaller<V> measuring(MethodDescriptor.Marshaller<V> delegate) {
        return new MethodDescriptor.Marshaller<>() {
            @Override public InputStream stream(V value) { return delegate.stream(value); }
            @Override public V parse(InputStream stream) {
                // gRPC's deframed message has an exact uncompressed length. Keep the
                // original stream so protobuf retains its KnownLength fast path.
                if (stream instanceof KnownLength) {
                    try {
                        long length = stream.available();
                        V result = delegate.parse(stream);
                        responseBytes = length;
                        return result;
                    } catch (IOException ignored) { /* Count reads on unusual streams. */ }
                }
                var counted = new CountedStream(stream);
                V result = delegate.parse(counted);
                responseBytes = counted.count;
                return result;
            }
        };
    }

    private static final class CountedStream extends FilterInputStream {
        long count;
        CountedStream(InputStream stream) { super(stream); }
        @Override public int read() throws IOException {
            int value = in.read(); if (value >= 0) { count++; } return value;
        }
        @Override public int read(byte[] bytes, int offset, int length) throws IOException {
            int read = in.read(bytes, offset, length); if (read > 0) { count += read; } return read;
        }
        @Override public long skip(long bytes) throws IOException {
            long skipped = in.skip(bytes); count += skipped; return skipped;
        }
    }
}

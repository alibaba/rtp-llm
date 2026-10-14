package org.flexlb.mockengine;

import io.grpc.Drainable;
import io.grpc.KnownLength;
import io.grpc.MethodDescriptor;
import io.grpc.ServerMethodDefinition;
import io.grpc.ServerServiceDefinition;
import io.grpc.ServiceDescriptor;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;

import java.io.InputStream;
import java.io.ByteArrayInputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.Set;
import java.util.ArrayList;

/** Reuse immutable full snapshots and their wire encoding, never omit keys by cursor. */
final class CacheStatusResponseCache {
    private record Entry(long available, long total, long blockSize, long version,
                         Set<Long> device, Collection<Long> memory, EngineRpcService.CacheStatusPB response) {}
    private Entry withKeys;
    private Entry withoutKeys;

    synchronized EngineRpcService.CacheStatusPB response(
            long available, long total, long blockSize, long version,
            Set<Long> device, Collection<Long> memory) {
        boolean keys = device != null;
        Entry previous = keys ? withKeys : withoutKeys;
        if (previous != null && previous.available == available && previous.total == total
                && previous.blockSize == blockSize && previous.version == version
                && previous.device == device && previous.memory == memory) {
            return previous.response;
        }
        var builder = EngineRpcService.CacheStatusPB.newBuilder()
                .setAvailableKvCache(available).setTotalKvCache(total)
                .setBlockSize(blockSize).setVersion(version);
        if (keys) {
            for (Long key : device) builder.putCacheKeys(key, true);
            for (Long key : memory) builder.putCacheKeys(key, true);
        }
        var response = builder.build();
        var entry = new Entry(available, total, blockSize, version, device, memory, response);
        if (keys) withKeys = entry; else withoutKeys = entry;
        return response;
    }

    static ServerServiceDefinition bind(RpcServiceGrpc.RpcServiceImplBase service) {
        var original = service.bindService();
        var descriptor = ServiceDescriptor.newBuilder(original.getServiceDescriptor().getName())
                .setSchemaDescriptor(original.getServiceDescriptor().getSchemaDescriptor());
        var methods = new ArrayList<ServerMethodDefinition<?, ?>>();
        for (var method : original.getMethods()) {
            if (method.getMethodDescriptor().getFullMethodName()
                    .equals(RpcServiceGrpc.getGetCacheStatusMethod().getFullMethodName())) {
                methods.add(cacheMethod(method));
            } else {
                methods.add(method);
            }
        }
        methods.forEach(method -> descriptor.addMethod(method.getMethodDescriptor()));
        var result = ServerServiceDefinition.builder(descriptor.build());
        methods.forEach(result::addMethod);
        return result.build();
    }

    @SuppressWarnings("unchecked") // Selected by the generated CacheStatus method descriptor.
    private static ServerMethodDefinition<EngineRpcService.CacheVersionPB, EngineRpcService.CacheStatusPB>
            cacheMethod(ServerMethodDefinition<?, ?> method) {
        var typed = (ServerMethodDefinition<EngineRpcService.CacheVersionPB, EngineRpcService.CacheStatusPB>) method;
        var descriptor = typed.getMethodDescriptor().toBuilder()
                .setResponseMarshaller(new Marshaller(typed.getMethodDescriptor().getResponseMarshaller()))
                .build();
        return ServerMethodDefinition.create(descriptor, typed.getServerCallHandler());
    }

    /** One bounded encoded response per engine; streams have independent cursors. */
    static final class Marshaller implements MethodDescriptor.Marshaller<EngineRpcService.CacheStatusPB> {
        private final MethodDescriptor.Marshaller<EngineRpcService.CacheStatusPB> delegate;
        private EngineRpcService.CacheStatusPB last;
        private byte[] bytes;
        Marshaller(MethodDescriptor.Marshaller<EngineRpcService.CacheStatusPB> delegate) {
            this.delegate = delegate;
        }
        @Override
        public synchronized InputStream stream(EngineRpcService.CacheStatusPB value) {
            if (last != value) {
                bytes = value.toByteArray();
                last = value;
            }
            return new EncodedStream(bytes);
        }
        @Override
        public EngineRpcService.CacheStatusPB parse(InputStream stream) {
            return delegate.parse(stream);
        }
    }

    /** Preserve the protobuf marshaller's known-length/drainable framing path. */
    private static final class EncodedStream extends ByteArrayInputStream implements KnownLength, Drainable {
        EncodedStream(byte[] bytes) { super(bytes); }

        @Override
        public int drainTo(OutputStream target) throws IOException {
            int remaining = count - pos;
            target.write(buf, pos, remaining);
            pos = count;
            return remaining;
        }
    }
}

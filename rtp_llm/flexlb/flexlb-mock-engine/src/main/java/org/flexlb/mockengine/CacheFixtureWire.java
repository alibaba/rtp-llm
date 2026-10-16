package org.flexlb.mockengine;

import io.grpc.Drainable;
import io.grpc.KnownLength;
import io.grpc.MethodDescriptor;
import io.grpc.protobuf.ProtoUtils;
import org.flexlb.engine.grpc.EngineRpcService;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** Test fixture encoding only. Wire fields follow model_rpc_service.proto CacheStatusPB. */
final class CacheFixtureWire {
    static io.grpc.ServerServiceDefinition bind(JavaMockEngineCluster.FastRpcService service) {
        var original = service.bindService();
        var cache = org.flexlb.engine.grpc.RpcServiceGrpc.getGetCacheStatusMethod();
        var wire = cache.toBuilder(cache.getRequestMarshaller(), MARSHALLER).build();
        var result = io.grpc.ServerServiceDefinition.builder(original.getServiceDescriptor().getName());
        for (var method : original.getMethods()) {
            if (method.getMethodDescriptor().getFullMethodName().equals(cache.getFullMethodName())) {
                result.addMethod(wire, io.grpc.stub.ServerCalls.asyncUnaryCall(service::getCacheStatusWire));
            } else {
                copyMethod(result, method);
            }
        }
        return result.build();
    }

    private static <Req, Resp> void copyMethod(io.grpc.ServerServiceDefinition.Builder target,
                                               io.grpc.ServerMethodDefinition<Req, Resp> method) {
        target.addMethod(method);
    }

    static final MethodDescriptor.Marshaller<Frame> MARSHALLER = new MethodDescriptor.Marshaller<>() {
        public InputStream stream(Frame frame) { return frame.stream(); }
        public Frame parse(InputStream input) {
            return new Frame(ProtoUtils.marshaller(EngineRpcService.CacheStatusPB.getDefaultInstance()).parse(input), null);
        }
    };

    record Keys(int worker, int workers, int unique, int shared, long offset, long version,
                int encodedSize, long sum, long xor) {
        int count() { return unique + 2 * shared; }
        long key(int index) {
            if (index < unique) return hash(((long) worker << 40) + offset + index);
            int n = index - unique;
            int segment = n < shared ? worker : (worker + workers - 1) % workers;
            return hash(Long.MIN_VALUE + ((long) segment << 40) + n % shared);
        }
        static Keys create(int worker, int workers, int unique, int shared, long offset, long version) {
            var shape = new Keys(worker, workers, unique, shared, offset, version, 0, 0, 0);
            int bytes = 0; long sum = 0, xor = 0;
            for (int i = 0; i < shape.count(); i++) {
                long key = shape.key(i); sum += key; xor ^= key;
                bytes = Math.addExact(bytes, 5 + varintSize(key));
            }
            return new Keys(worker, workers, unique, shared, offset, version, bytes, sum, xor);
        }
        static long hash(long x) {
            x = (x ^ (x >>> 30)) * 0xbf58476d1ce4e5b9L;
            x = (x ^ (x >>> 27)) * 0x94d049bb133111ebL;
            return x ^ (x >>> 31);
        }
    }

    record Frame(EngineRpcService.CacheStatusPB header, Keys keys) {
        InputStream stream() {
            if (keys == null) return ProtoUtils.marshaller(EngineRpcService.CacheStatusPB.getDefaultInstance()).stream(header);
            return new WireStream(header.toByteArray(), keys);
        }
        EngineRpcService.CacheStatusPB toProto() {
            try (var stream = stream()) { return EngineRpcService.CacheStatusPB.parseFrom(stream); }
            catch (IOException e) { throw new java.io.UncheckedIOException(e); }
        }
    }

    static int varintSize(long value) {
        int n = 1;
        while ((value & ~0x7fL) != 0) { value >>>= 7; n++; }
        return n;
    }

    /** Constant scratch space per RPC; no per-key objects or full response byte array. */
    private static final class WireStream extends InputStream implements KnownLength, Drainable {
        private final byte[] header;
        private final Keys keys;
        private final byte[] entry = new byte[15];
        private int headerPosition, keyPosition, entryPosition, entryLength, remaining;
        WireStream(byte[] header, Keys keys) {
            this.header = header; this.keys = keys;
            remaining = Math.addExact(header.length, keys.encodedSize());
        }
        private boolean nextEntry() {
            if (entryPosition < entryLength) return true;
            if (keyPosition >= keys.count()) return false;
            long key = keys.key(keyPosition++);
            int n = varintSize(key), p = 0;
            entry[p++] = 42; // field 5, length-delimited map entry
            entry[p++] = (byte) (3 + n);
            entry[p++] = 8; // map key: int64 field 1 (negative values use 10-byte varints)
            while ((key & ~0x7fL) != 0) { entry[p++] = (byte) ((key & 0x7f) | 0x80); key >>>= 7; }
            entry[p++] = (byte) key;
            entry[p++] = 16; entry[p++] = 1; // map value: bool field 2, true
            entryPosition = 0; entryLength = p;
            return true;
        }
        @Override public int available() { return remaining; }
        @Override public int read() {
            if (headerPosition < header.length) { remaining--; return header[headerPosition++] & 255; }
            if (!nextEntry()) return -1;
            remaining--; return entry[entryPosition++] & 255;
        }
        @Override public int read(byte[] target, int offset, int length) {
            java.util.Objects.checkFromIndexSize(offset, length, target.length);
            if (length == 0) return 0;
            if (remaining == 0) return -1;
            int written = 0;
            if (headerPosition < header.length) {
                int n = Math.min(length, header.length - headerPosition);
                System.arraycopy(header, headerPosition, target, offset, n);
                headerPosition += n; written += n;
            }
            while (written < length && nextEntry()) {
                int n = Math.min(length - written, entryLength - entryPosition);
                System.arraycopy(entry, entryPosition, target, offset + written, n);
                entryPosition += n; written += n;
            }
            remaining -= written;
            return written;
        }
        @Override public int drainTo(OutputStream target) throws IOException {
            byte[] buffer = new byte[16384];
            int total = 0, n;
            while ((n = read(buffer)) != -1) { target.write(buffer, 0, n); total += n; }
            return total;
        }
    }
}

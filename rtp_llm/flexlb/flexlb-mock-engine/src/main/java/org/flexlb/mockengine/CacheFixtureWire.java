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

    interface KeySequence {
        int count();
        long key(int index);
        int encodedSize();
    }

    record Keys(int worker, int workers, int unique, int shared, long offset, long version,
                int encodedSize, long sum, long xor) implements KeySequence {
        public int count() { return unique + shared; }
        public long key(int index) {
            if (index < unique) return hash(((long) worker << 40) + offset + index);
            // One global namespace: every P owns the same shared keys.
            return hash(Long.MIN_VALUE + index - unique);
        }
        /** Invert the bijective hash to test membership without materializing synthetic keys. */
        boolean contains(long key) {
            long x = key ^ (key >>> 31) ^ (key >>> 62);
            x *= 0x319642b2d24d8ec3L;
            x ^= (x >>> 27) ^ (x >>> 54);
            x *= 0x96de1b173f119089L;
            x ^= (x >>> 30) ^ (x >>> 60);
            long first = ((long) worker << 40) + offset;
            return (x >= first && x - first < unique)
                    || (x < Long.MIN_VALUE + shared);
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

    /** Mock-only compact representation. Arrays are private, owned snapshots, never live cache views. */
    static final class NaturalKeys implements KeySequence {
        private final long[] values;
        private final int count, encodedSize;
        private NaturalKeys(long[] values, int count, int encodedSize) {
            this.values = values; this.count = count; this.encodedSize = encodedSize;
        }
        static NaturalKeys takeSnapshots(long[] device, long[] host, Keys external) {
            long[] values = device;
            if (host.length != 0) {
                values = java.util.Arrays.copyOf(device, Math.addExact(device.length, host.length));
                System.arraycopy(host, 0, values, device.length, host.length);
                java.util.Arrays.sort(values);
            }
            int count = 0, bytes = 0;
            long previous = 0;
            for (int i = 0; i < values.length; i++) {
                long key = values[i];
                boolean duplicate = host.length != 0 && i > 0 && key == previous;
                previous = key;
                if (duplicate || (external != null && external.contains(key))) continue;
                bytes = Math.addExact(bytes, 5 + varintSize(key));
                values[count++] = key;
            }
            return new NaturalKeys(values, count, bytes);
        }
        public int count() { return count; }
        public long key(int index) { return values[index]; }
        public int encodedSize() { return encodedSize; }
    }

    private record JoinedKeys(KeySequence first, KeySequence second) implements KeySequence {
        public int count() { return Math.addExact(first.count(), second.count()); }
        public long key(int index) {
            return index < first.count() ? first.key(index) : second.key(index - first.count());
        }
        public int encodedSize() { return Math.addExact(first.encodedSize(), second.encodedSize()); }
    }

    record Frame(EngineRpcService.CacheStatusPB header, Keys keys, NaturalKeys natural) {
        Frame(EngineRpcService.CacheStatusPB header, Keys keys) { this(header, keys, null); }
        InputStream stream() {
            KeySequence sequence = natural == null ? keys : keys == null ? natural : new JoinedKeys(natural, keys);
            if (sequence == null) return ProtoUtils.marshaller(EngineRpcService.CacheStatusPB.getDefaultInstance()).stream(header);
            return new WireStream(header.toByteArray(), sequence);
        }
        EngineRpcService.CacheStatusPB toProto() {
            if (keys == null && natural == null) return header;
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
        private final KeySequence keys;
        private final byte[] entry = new byte[15];
        private int headerPosition, keyPosition, entryPosition, entryLength, remaining;
        WireStream(byte[] header, KeySequence keys) {
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

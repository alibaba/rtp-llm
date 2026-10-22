package org.flexlb.mockengine;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import io.grpc.CallOptions;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.MethodDescriptor;
import io.grpc.stub.ClientCalls;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/** Port of 969cb846e KVCMEndpointResolver; only the published GetClusterInfo wire subset is needed. */
final class KvcmEndpointResolver implements AutoCloseable {
    @FunctionalInterface interface Addresses { List<String> get() throws Exception; }
    @FunctionalInterface interface ClusterInfo { URI resolve(String target, String instance, int timeoutMs) throws Exception; }
    private final KvcmEventPublisher.Config config;
    private final String instance;
    private final Addresses addresses;
    private final ClusterInfo clusterInfo;
    private URI cached;
    private boolean closed;

    KvcmEndpointResolver(KvcmEventPublisher.Config config, String instance) {
        this(config, instance, () -> addresses(config.discovery()), KvcmEndpointResolver::queryLeader);
    }
    KvcmEndpointResolver(KvcmEventPublisher.Config config, String instance, Addresses addresses, ClusterInfo clusterInfo) {
        this.config = config; this.instance = instance; this.addresses = addresses; this.clusterInfo = clusterInfo;
        cached = config.endpoint();
    }
    synchronized URI endpoint() throws Exception {
        if (closed) throw new IllegalStateException("KVCM resolver closed");
        if (cached != null) return cached;
        Exception failure = null;
        for (String target : addresses.get()) {
            try {
                URI leader = clusterInfo.resolve(target, instance, config.timeoutMs());
                if (leader == null || !"http".equals(leader.getScheme()) || leader.getHost() == null || leader.getPort() <= 0)
                    throw new IllegalArgumentException("invalid KVCM Meta HTTP leader");
                return cached = leader;
            } catch (Exception error) { failure = error; }
        }
        throw new IllegalStateException("KVCM discovery returned no usable leader", failure);
    }
    synchronized void invalidate(URI failed) {
        if (config.endpoint() == null && java.util.Objects.equals(cached, failed)) cached = null;
    }
    @Override public synchronized void close() { closed = true; cached = null; }

    private static List<String> addresses(KvcmEventPublisher.Discovery discovery) throws Exception {
        if (!discovery.vipserver()) return List.of(discovery.address());
        // Same internal VIPClient API as VipServerDiscovery; keep private jars optional in opensource builds.
        Class<?> client = Class.forName("com.taobao.vipserver.client.core.VIPClient");
        Object result = client.getMethod("srvHosts", String.class).invoke(null, discovery.address());
        List<String> targets = new ArrayList<>();
        for (Object host : (Iterable<?>) result) {
            Class<?> type = host.getClass();
            String ip = (String) type.getMethod("getIp").invoke(host);
            int port = ((Number) type.getMethod("getPort").invoke(host)).intValue();
            targets.add((ip.contains(":") ? "[" + ip + "]" : ip) + ":" + port);
        }
        return targets;
    }
    static final MethodDescriptor.Marshaller<byte[]> BYTES = new MethodDescriptor.Marshaller<>() {
        public InputStream stream(byte[] value) { return new ByteArrayInputStream(value); }
        public byte[] parse(InputStream input) {
            try { return input.readAllBytes(); } catch (java.io.IOException error) { throw new IllegalArgumentException(error); }
        }
    };
    static final MethodDescriptor<byte[], byte[]> METHOD = MethodDescriptor.<byte[], byte[]>newBuilder()
            .setType(MethodDescriptor.MethodType.UNARY)
            .setFullMethodName("kv_cache_manager.proto.meta.MetaService/GetClusterInfo")
            .setRequestMarshaller(BYTES).setResponseMarshaller(BYTES).build();
    static URI queryLeader(String target, String instance, int timeoutMs) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        CodedOutputStream out = CodedOutputStream.newInstance(bytes);
        out.writeString(1, "mock-kv-event-discovery"); out.writeString(2, instance); out.flush();
        ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        try {
            byte[] response = ClientCalls.blockingUnaryCall(channel, METHOD,
                    CallOptions.DEFAULT.withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS), bytes.toByteArray());
            return parseLeader(response);
        } finally { channel.shutdownNow(); }
    }
    // Field numbers come from the real engine's events/kvcm_discovery.proto, not guessed JSON conversion.
    static URI parseLeader(byte[] response) throws Exception {
        CodedInputStream input = CodedInputStream.newInstance(response);
        boolean ok = false; String host = null; int port = 0;
        for (int tag; (tag = input.readTag()) != 0;) {
            if (tag == 10) {
                CodedInputStream header = CodedInputStream.newInstance(input.readByteArray());
                for (int h; (h = header.readTag()) != 0;) {
                    if (h == 10) {
                        CodedInputStream status = CodedInputStream.newInstance(header.readByteArray());
                        for (int c; (c = status.readTag()) != 0;) {
                            if (c == 8) ok = status.readInt32() == 1; else status.skipField(c);
                        }
                    } else header.skipField(h);
                }
            } else if (tag == 34) {
                CodedInputStream leader = CodedInputStream.newInstance(input.readByteArray());
                for (int t; (t = leader.readTag()) != 0;) {
                    if (t == 18) host = leader.readStringRequireUtf8();
                    else if (t == 32) port = leader.readInt32();
                    else leader.skipField(t);
                }
            } else input.skipField(tag);
        }
        if (!ok || host == null || host.isBlank() || port <= 0 || port > 65535)
            throw new IllegalArgumentException("KVCM GetClusterInfo did not return an OK Meta HTTP leader");
        return new URI("http", null, host, port, null, null, null);
    }
}

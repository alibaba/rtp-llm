package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.JsonNode;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import java.net.URI;
import java.util.ServiceLoader;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.kvcm.grpc.*;

/** Native bootstrap discovery followed by GetClusterInfo; only the reporting thread calls get(). */
final class KvcmDiscovery implements Supplier<URI>, AutoCloseable {
    private final URI direct;
    private final String bootstrap, instance;
    private final ServiceDiscovery discovery;
    private ManagedChannel channel;
    private String target;
    private URI leader;
    private long refreshAt;
    private boolean closed;

    static KvcmDiscovery from(JsonNode settings, String instance) {
        String override = settings.path("kv_cache_event_manager_endpoint").asText();
        if (!override.isBlank()) return new KvcmDiscovery(URI.create(override), "", instance, null);
        boolean vip = RtpConfig.nativeBool(settings, "kvcm_enable_vipserver", false);
        String bootstrap = settings.path(vip ? "kvcm_vipserver_domain" : "kvcm_server_address").asText();
        if (bootstrap.isBlank()) throw new IllegalArgumentException("KVCM requires a native bootstrap address");
        ServiceDiscovery discovery = null;
        if (vip) {
            var providers = ServiceLoader.load(ServiceDiscovery.class).iterator();
            if (!providers.hasNext()) throw new IllegalArgumentException("VIPServer discovery requires the internal Engine package");
            discovery = providers.next();
            if (providers.hasNext()) {
                discovery.shutdown();
                throw new IllegalArgumentException("Ambiguous Engine service discovery providers");
            }
        }
        return new KvcmDiscovery(null, bootstrap, instance, discovery);
    }

    KvcmDiscovery(URI direct, String bootstrap, String instance, ServiceDiscovery discovery) {
        this.direct = direct;
        this.bootstrap = bootstrap;
        this.instance = instance;
        this.discovery = discovery;
        if (direct != null) org.flexlb.mock.cache.KvcmReporter.Configuration.validateEndpoint(direct);
    }

    @Override public synchronized URI get() {
        if (closed) throw new IllegalStateException("KVCM discovery closed");
        if (direct != null) return direct;
        if (leader != null && System.nanoTime() < refreshAt) return leader;
        leader = null;
        String address = bootstrap;
        if (discovery != null) {
            var hosts = discovery.getHosts(bootstrap);
            if (hosts.isEmpty()) throw new IllegalStateException("No healthy KVCM discovery hosts");
            var host = hosts.getFirst();
            address = authority(host.getIp(), host.getPort());
        }
        if (!address.equals(target)) {
            if (channel != null) channel.shutdownNow();
            channel = ManagedChannelBuilder.forTarget(address).usePlaintext().build();
            target = address;
        }
        var response = MetaServiceGrpc.newBlockingStub(channel).withDeadlineAfter(1500, TimeUnit.MILLISECONDS)
            .getClusterInfo(GetClusterInfoRequest.newBuilder().setInstanceId(instance).build());
        var endpoint = response.getLeaderEndpoint();
        if (response.getHeader().getStatus().getCode() != ErrorCode.OK
                || response.getLeaderNodeId().isBlank()
                || !response.getLeaderNodeId().equals(endpoint.getNodeId()))
            throw new IllegalStateException("KVCM returned no authoritative leader");
        leader = URI.create("http://" + authority(endpoint.getHost(), endpoint.getMetaHttpPort()));
        // Bound staleness after HTTP/leader failures without adding a watcher or a second reporting loop.
        refreshAt = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
        return leader;
    }

    private static String authority(String host, int port) {
        if (host == null || host.isBlank() || port < 1 || port > 65535)
            throw new IllegalStateException("Invalid KVCM endpoint");
        try { return new URI(null, null, host, port, null, null, null).getRawAuthority(); }
        catch (java.net.URISyntaxException error) { throw new IllegalStateException("Invalid KVCM host", error); }
    }

    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        if (channel != null) channel.shutdownNow();
        if (discovery != null) discovery.shutdown();
    }
}

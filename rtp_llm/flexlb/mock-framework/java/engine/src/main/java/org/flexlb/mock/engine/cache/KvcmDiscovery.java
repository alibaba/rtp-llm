package org.flexlb.mock.engine.cache;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import java.net.URI;
import java.util.List;
import java.util.ArrayList;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.kvcm.grpc.*;

/** Engine cache reporting endpoint. Reporting owns lookup; close can cancel outstanding discovery. */
public final class KvcmDiscovery implements AutoCloseable {
    private final URI direct;
    private final String bootstrap, instance;
    private final ServiceDiscovery discovery;
    private URI leader;
    private ManagedChannel activeChannel;
    private FutureTask<List<WorkerHost>> activeLookup;
    private boolean closed;

    public static KvcmDiscovery direct(URI endpoint) {
        return new KvcmDiscovery(KvcmReporter.Configuration.validateEndpoint(endpoint), "", "", null);
    }

    public static KvcmDiscovery discover(String bootstrap, String instance, boolean vip) {
        if (bootstrap == null || bootstrap.isBlank() || instance == null || instance.isBlank())
            throw new IllegalArgumentException("KVCM requires a bootstrap address and instance");
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
    }

    /** Last usable endpoint without performing I/O, also used for best-effort shutdown reporting. */
    public synchronized URI current() { return closed ? null : direct != null ? direct : leader; }

    public URI get() {
        synchronized (this) {
            requireOpen();
            URI cached = current();
            if (cached != null) return cached;
        }
        var failure = new IllegalStateException("No reachable authoritative KVCM leader");
        List<String> addresses = new ArrayList<>();
        if (discovery == null) addresses.add(bootstrap);
        else for (var host : hosts()) {
            try { addresses.add(authority(host.getIp(), host.getPort())); }
            catch (RuntimeException error) { failure.addSuppressed(error); }
        }
        for (String address : addresses) {
            try {
                URI found = resolve(address);
                synchronized (this) { requireOpen(); leader = found; return found; }
            } catch (RuntimeException error) {
                if (Thread.currentThread().isInterrupted()) throw error;
                synchronized (this) { requireOpen(); }
                failure.addSuppressed(error);
            }
        }
        throw failure;
    }

    /** An old failed request must not invalidate a newer endpoint. Explicit direct endpoints stay fixed. */
    public synchronized void invalidate(URI failed) {
        if (direct == null && Objects.equals(leader, failed)) leader = null;
    }

    private List<WorkerHost> hosts() {
        var lookup = new FutureTask<>(() -> discovery.getHosts(bootstrap));
        synchronized (this) { requireOpen(); activeLookup = lookup; }
        try {
            Thread.ofVirtual().name("kvcm-bootstrap").start(lookup);
            return lookup.get(1500, TimeUnit.MILLISECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("KVCM discovery interrupted", error);
        } catch (java.util.concurrent.ExecutionException | java.util.concurrent.TimeoutException error) {
            throw new IllegalStateException("KVCM bootstrap discovery failed", error);
        } finally {
            lookup.cancel(true);
            synchronized (this) { if (activeLookup == lookup) activeLookup = null; }
        }
    }

    private URI resolve(String address) {
        var channel = ManagedChannelBuilder.forTarget(address).usePlaintext().build();
        try {
            synchronized (this) { requireOpen(); activeChannel = channel; }
            var response = MetaServiceGrpc.newBlockingStub(channel).withDeadlineAfter(1500, TimeUnit.MILLISECONDS)
                .getClusterInfo(GetClusterInfoRequest.newBuilder().setInstanceId(instance).build());
            var endpoint = response.getLeaderEndpoint();
            if (response.getHeader().getStatus().getCode() != ErrorCode.OK
                    || response.getLeaderNodeId().isBlank()
                    || !response.getLeaderNodeId().equals(endpoint.getNodeId()))
                throw new IllegalStateException("KVCM returned no authoritative leader");
            return URI.create("http://" + authority(endpoint.getHost(), endpoint.getMetaHttpPort()));
        } finally {
            channel.shutdownNow();
            synchronized (this) { if (activeChannel == channel) activeChannel = null; }
        }
    }

    private void requireOpen() {
        if (closed) throw new IllegalStateException("KVCM discovery closed");
    }

    private static String authority(String host, int port) {
        if (host == null || host.isBlank() || port < 1 || port > 65535)
            throw new IllegalStateException("Invalid KVCM endpoint");
        try { return new URI(null, null, host, port, null, null, null).getRawAuthority(); }
        catch (java.net.URISyntaxException error) { throw new IllegalStateException("Invalid KVCM host", error); }
    }

    @Override public void close() {
        synchronized (this) {
            if (closed) return;
            closed = true;
            if (activeChannel != null) activeChannel.shutdownNow();
            if (activeLookup != null) activeLookup.cancel(true);
        }
        if (discovery != null) discovery.shutdown();
    }
}

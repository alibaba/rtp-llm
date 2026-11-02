package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.kvcm.grpc.*;
import org.flexlb.mock.api.model.Types.Role;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KvcmDiscoveryTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    static final class Meta implements AutoCloseable {
        final Server rpc;
        volatile boolean valid = true;
        volatile String instance;
        Meta(int httpPort) throws Exception {
            rpc = ServerBuilder.forPort(0).addService(new MetaServiceGrpc.MetaServiceImplBase() {
                @Override public void getClusterInfo(GetClusterInfoRequest request, StreamObserver<GetClusterInfoResponse> out) {
                    instance = request.getInstanceId();
                    out.onNext(GetClusterInfoResponse.newBuilder()
                        .setHeader(CommonResponseHeader.newBuilder().setStatus(Status.newBuilder().setCode(ErrorCode.OK)))
                        .setLeaderNodeId("leader")
                        .setLeaderEndpoint(MetaNodeEndpoint.newBuilder().setNodeId(valid ? "leader" : "old-leader")
                            .setHost("127.0.0.1").setMetaHttpPort(httpPort)).build());
                    out.onCompleted();
                }
            }).build().start();
        }
        @Override public void close() throws Exception { rpc.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

    @Test void vipMembershipChangeReplacesBootstrapAndLeader() throws Exception {
        try (var first = new Meta(18001); var second = new Meta(18002)) {
            var hosts = new AtomicReference<>(List.of(WorkerHost.of("127.0.0.1", first.rpc.getPort(), "")));
            var stopped = new java.util.concurrent.atomic.AtomicBoolean();
            var vip = new ServiceDiscovery() {
                public List<WorkerHost> getHosts(String domain) { assertEquals("kvcm-domain", domain); return hosts.get(); }
                public void listen(String domain, ServiceHostListener listener) { fail("Reporting needs no additional watcher"); }
                public void shutdown() { stopped.set(true); }
            };
            var resolver = new KvcmDiscovery(null, "kvcm-domain", "shared_128", vip);
            try (resolver) {
                assertEquals(URI.create("http://127.0.0.1:18001"), resolver.get());
                assertEquals("shared_128", first.instance);
                hosts.set(List.of(WorkerHost.of("127.0.0.1", second.rpc.getPort(), "")));
                Thread.sleep(1100); // Exercise the bound on discovery staleness.
                assertEquals(URI.create("http://127.0.0.1:18002"), resolver.get());
                assertEquals("shared_128", second.instance);
            }
            assertTrue(stopped.get());
            assertThrows(IllegalStateException.class, resolver::get);
        }
    }

    @Test void invalidLeaderIsNotCachedAndCanRecover() throws Exception {
        try (var meta = new Meta(18003);
             var resolver = new KvcmDiscovery(null, "127.0.0.1:" + meta.rpc.getPort(), "deployment_128", null)) {
            meta.valid = false;
            assertThrows(IllegalStateException.class, resolver::get);
            meta.valid = true;
            assertEquals(URI.create("http://127.0.0.1:18003"), resolver.get());
        }
    }

    @Test void directOverrideTakesPrecedenceWithoutLoadingInternalDiscovery() {
        var settings = JSON.createObjectNode().put("kv_cache_event_manager_endpoint", "http://127.0.0.1:6382")
            .put("kvcm_enable_vipserver", true).put("kvcm_vipserver_domain", "unused");
        try (var resolver = KvcmDiscovery.from(settings, "shared_128")) {
            assertEquals(URI.create("http://127.0.0.1:6382"), resolver.get());
        }
        settings.put("kv_cache_event_manager_endpoint", "file:///tmp/not-http");
        assertThrows(IllegalArgumentException.class, () -> KvcmDiscovery.from(settings, "shared_128"));
        assertThrows(IllegalArgumentException.class, () -> KvcmDiscovery.from(JSON.createObjectNode(), "shared_128"));
    }

    @Test void environmentUsesTheActualRealRecoBindings() {
        var settings = RtpSettings.environment(Map.of("RECO_ENABLE_VIPSERVER", "true", "RECO_VIPSERVER_DOMAIN", "kvcm-domain",
            "RECO_SERVER_ADDRESS", "bootstrap:6381", "KVCM_SERVER_ADDRESS", "ignored"), Role.prefill);
        assertEquals("true", settings.path("kvcm_enable_vipserver").asText());
        assertEquals("kvcm-domain", settings.path("kvcm_vipserver_domain").asText());
        assertEquals("bootstrap:6381", settings.path("kvcm_server_address").asText());
    }

    @Test void reporterRegistersAndSendsSnapshotToDiscoveredMetaHttpPort() throws Exception {
        try (var http = new KvcmProtocolTest.Meta(); var rpc = new Meta(http.server.getAddress().getPort())) {
            var root = JSON.valueToTree(RtpNodeTest.configuration("prefill"));
            var settings = (com.fasterxml.jackson.databind.node.ObjectNode) root.path("component").path("config");
            settings.put("kv_cache_event_publisher_type", "kvcm").put("kvcm_server_address", "127.0.0.1:" + rpc.rpc.getPort())
                .put("kv_cache_event_instance_id", "shared_128").put("reuse_cache", true);
            try (var node = RtpNode.start(RtpConfig.load(root))) {
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(8);
                while (!http.has("EVENT_BLOCK_SNAPSHOT")) {
                    assertTrue(System.nanoTime() < deadline, "Discovered reporter did not publish its snapshot"); Thread.sleep(10);
                }
                assertTrue(http.registrations.get() > 0);
                assertEquals("shared_128", rpc.instance);
            }
        }
    }
}

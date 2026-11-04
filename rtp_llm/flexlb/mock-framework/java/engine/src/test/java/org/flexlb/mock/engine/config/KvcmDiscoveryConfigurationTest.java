package org.flexlb.mock.engine.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.flexlb.kvcm.grpc.*;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.execution.RtpNode;
import org.flexlb.mock.engine.scheduler.RtpNodeTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class KvcmDiscoveryConfigurationTest {
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

    @Test void environmentUsesTheActualRealRecoBindings() {
        var settings = RtpConfig.environment(Map.of("RECO_ENABLE_VIPSERVER", "true", "RECO_VIPSERVER_DOMAIN", "kvcm-domain",
            "RECO_SERVER_ADDRESS", "bootstrap:6381", "KVCM_SERVER_ADDRESS", "ignored"), Role.prefill);
        assertEquals("true", settings.path("kvcm_enable_vipserver").asText());
        assertEquals("kvcm-domain", settings.path("kvcm_vipserver_domain").asText());
        assertEquals("bootstrap:6381", settings.path("kvcm_server_address").asText());
    }

    @ParameterizedTest @ValueSource(booleans = {false, true})
    void reporterUsesNativeDiscoveryOrAnExplicitOverrideWithoutLoadingVip(boolean direct) throws Exception {
        try (var http = new KvcmProtocolTest.Meta(); var rpc = new Meta(http.server.getAddress().getPort())) {
            var root = JSON.valueToTree(RtpNodeTest.configuration("prefill"));
            var settings = (com.fasterxml.jackson.databind.node.ObjectNode) root.path("component").path("config");
            settings.put("kv_cache_event_publisher_type", "kvcm").put("kvcm_server_address", "127.0.0.1:" + rpc.rpc.getPort())
                .put("kv_cache_event_instance_id", "shared_128").put("reuse_cache", true);
            if (direct) settings.put("kv_cache_event_manager_endpoint", http.endpoint())
                .put("kvcm_enable_vipserver", true).put("kvcm_vipserver_domain", "unused");
            try (var node = RtpNode.start(RtpConfig.load(root))) {
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(8);
                while (!http.has("EVENT_BLOCK_SNAPSHOT")) {
                    assertTrue(System.nanoTime() < deadline, "Discovered reporter did not publish its snapshot"); Thread.sleep(10);
                }
                assertTrue(http.registrations.get() > 0);
                if (direct) assertNull(rpc.instance); else assertEquals("shared_128", rpc.instance);
            }
        }
    }
}

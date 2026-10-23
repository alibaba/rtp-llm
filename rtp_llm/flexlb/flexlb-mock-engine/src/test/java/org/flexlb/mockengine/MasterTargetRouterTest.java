package org.flexlb.mockengine;

import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import com.sun.net.httpserver.HttpServer;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

/** ZK discovery routing and the terminal error boundary. */
class MasterTargetRouterTest {

    private static final String TARGET_A = "127.0.0.1:18082";
    private static final String TARGET_B = "127.0.0.1:18085";

    private static FlexlbServiceGrpc.FlexlbServiceBlockingStub stub() {
        return mock(FlexlbServiceGrpc.FlexlbServiceBlockingStub.class,
                withSettings().defaultAnswer(org.mockito.Answers.RETURNS_SELF));
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleRequestPB request() {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(42L).build();
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleResponsePB response(int code) {
        return FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setCode(code)
                .setSuccess(code == 200)
                .setEnqueuedByMaster(true)
                .build();
    }

    private static MasterTargetRouter router(
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA,
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB) {
        var a = new MasterRouteDiscovery.Host("a:7001", TARGET_A);
        var b = new MasterRouteDiscovery.Host("b:7001", TARGET_B);
        var discovery = new MasterRouteDiscovery(() -> List.of(a, b),
                host -> new MasterRouteDiscovery.Info(a.http(), true));
        discovery.refresh();
        return new MasterTargetRouter(discovery,
                Map.of(TARGET_A, new FlexlbServiceGrpc.FlexlbServiceBlockingStub[]{stubA},
                        TARGET_B, new FlexlbServiceGrpc.FlexlbServiceBlockingStub[]{stubB}));
    }

    @Test
    void discoveredLeaderServesRequestWithoutFailover() throws Exception {
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any())).thenReturn(response(200));
        MasterTargetRouter router = router(stubA, stubB);

        MasterTargetRouter.ScheduleOutcome outcome = router.schedule(request(), 1000L);

        assertEquals(TARGET_A, outcome.lastTarget);
        assertFalse(outcome.failover);
        assertEquals(MasterTargetRouter.ErrorKind.NONE, outcome.errorKind);
        assertEquals(200, outcome.response.getCode());
        assertEquals(TARGET_A, router.selectedTarget());
        verify(stubB, never()).schedule(any());
    }

    @Test
    void transportFailureRetriesSameRequestWithoutChangingDiscoveredLeader() throws Exception {
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any()))
                .thenThrow(new StatusRuntimeException(Status.UNAVAILABLE));
        when(stubB.schedule(any())).thenReturn(response(200));
        MasterTargetRouter router = router(stubA, stubB);

        MasterTargetRouter.ScheduleOutcome outcome = router.schedule(request(), 1000L);

        // Same-request retry on B succeeded: served by B, failover=true.
        assertEquals(TARGET_B, outcome.lastTarget);
        assertTrue(outcome.failover);
        assertEquals(MasterTargetRouter.ErrorKind.NONE, outcome.errorKind);
        assertEquals(200, outcome.response.getCode());
        assertEquals(2, outcome.attempts.size());
        assertEquals(TARGET_A, outcome.attempts.get(0).target);
        assertEquals("UNAVAILABLE", outcome.attempts.get(0).status);
        assertEquals(TARGET_B, outcome.attempts.get(1).target);
        assertEquals("OK", outcome.attempts.get(1).status);
        assertEquals(Integer.valueOf(200), outcome.attempts.get(1).responseCode);
        assertTrue(outcome.attempts.get(0).endedEpochMs >= outcome.attempts.get(0).startedEpochMs);
        // Only discovery owns the leader, not a successful follower retry.
        assertEquals(TARGET_A, router.selectedTarget());
        verify(stubA).schedule(any());
        verify(stubB).schedule(any());
    }

    @Test
    void followerRetryDoesNotOverrideLeaderForNextRequest() {
        var a = stub();
        var b = stub();
        when(a.schedule(any())).thenThrow(new StatusRuntimeException(Status.UNAVAILABLE))
                .thenReturn(response(200));
        when(b.schedule(any())).thenReturn(response(200));
        var router = router(a, b);
        assertTrue(router.schedule(request(), 1000).failover);
        var second = router.schedule(request(), 1000);
        assertFalse(second.failover);
        assertEquals(TARGET_A, second.lastTarget);
        verify(b).schedule(any());
    }

    @Test
    void allTargetsUnavailableYieldsTransportOutcome() throws Exception {
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any()))
                .thenThrow(new StatusRuntimeException(Status.UNAVAILABLE));
        when(stubB.schedule(any()))
                .thenThrow(new StatusRuntimeException(Status.UNAVAILABLE));
        MasterTargetRouter router = router(stubA, stubB);

        MasterTargetRouter.ScheduleOutcome outcome = router.schedule(request(), 1000L);

        assertNull(outcome.response);
        assertEquals(MasterTargetRouter.ErrorKind.TRANSPORT, outcome.errorKind);
        assertTrue(outcome.failover);
        // Last target attempted in this discovery snapshot.
        assertEquals(TARGET_B, outcome.lastTarget);
        assertTrue(outcome.failure instanceof StatusRuntimeException);
        // Leader remains owned by discovery.
        assertEquals(TARGET_A, router.selectedTarget());
    }

    @Test
    void deadlineExceededIsTerminalNoRetryNoSwitch() throws Exception {
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any()))
                .thenThrow(new StatusRuntimeException(Status.DEADLINE_EXCEEDED));
        MasterTargetRouter router = router(stubA, stubB);

        MasterTargetRouter.ScheduleOutcome outcome = router.schedule(request(), 1000L);

        assertNull(outcome.response);
        assertEquals(MasterTargetRouter.ErrorKind.DEADLINE, outcome.errorKind);
        assertFalse(outcome.failover);
        assertEquals(TARGET_A, outcome.lastTarget);
        assertEquals(TARGET_A, router.selectedTarget());
        verify(stubB, never()).schedule(any());
    }

    @Test
    void businessErrorResponseNeverRetriesOrSwitches() throws Exception {
        // 8431 (admission rejection) and 8511 (forwarding terminal code) both
        // arrive as ordinary Schedule responses: the router hands them back
        // untouched — classification into error_kind=business happens in the
        // caller (handleRequest), retry/switch/direct-fallback never happen.
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any())).thenReturn(response(8431));
        MasterTargetRouter router = router(stubA, stubB);

        MasterTargetRouter.ScheduleOutcome outcome = router.schedule(request(), 1000L);

        assertEquals(8431, outcome.response.getCode());
        assertEquals(MasterTargetRouter.ErrorKind.NONE, outcome.errorKind);
        assertFalse(outcome.failover);
        assertEquals(TARGET_A, router.selectedTarget());
        verify(stubB, never()).schedule(any());
    }

    @Test
    void otherGrpcStatusIsBusinessNoRetry() throws Exception {
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any()))
                .thenThrow(new StatusRuntimeException(Status.INTERNAL));
        MasterTargetRouter router = router(stubA, stubB);

        MasterTargetRouter.ScheduleOutcome outcome = router.schedule(request(), 1000L);

        assertNull(outcome.response);
        assertEquals(MasterTargetRouter.ErrorKind.BUSINESS, outcome.errorKind);
        assertFalse(outcome.failover);
        verify(stubB, never()).schedule(any());
    }

    @Test
    void deadlineAfterTransportFailoverIsStillFailover() throws Exception {
        // A UNAVAILABLE -> retry on B -> B DEADLINE: the request DID switch
        // targets mid-flight (failover=true) even though its terminal error
        // kind is deadline.
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any()))
                .thenThrow(new StatusRuntimeException(Status.UNAVAILABLE));
        when(stubB.schedule(any()))
                .thenThrow(new StatusRuntimeException(Status.DEADLINE_EXCEEDED));
        MasterTargetRouter router = router(stubA, stubB);

        MasterTargetRouter.ScheduleOutcome outcome = router.schedule(request(), 1000L);

        assertNull(outcome.response);
        assertEquals(MasterTargetRouter.ErrorKind.DEADLINE, outcome.errorKind);
        assertTrue(outcome.failover);
        assertEquals(TARGET_B, outcome.lastTarget);
    }

    // ---- error_kind taxonomy (shared with the legacy single-target path) ----

    @Test
    void classifyThrowableMapsGrpcCodes() {
        assertEquals("transport", MasterTargetRouter.classifyThrowable(
                new StatusRuntimeException(Status.UNAVAILABLE)));
        assertEquals("deadline", MasterTargetRouter.classifyThrowable(
                new StatusRuntimeException(Status.DEADLINE_EXCEEDED)));
        assertEquals("business", MasterTargetRouter.classifyThrowable(
                new StatusRuntimeException(Status.INTERNAL)));
        assertEquals("business", MasterTargetRouter.classifyThrowable(
                new RuntimeException("non-grpc")));
    }

    @Test
    void discoveryFollowsLeaderRoleAndRetainsHealthyHostsWhenVipIsEmpty() {
        MasterRouteDiscovery.Host a = new MasterRouteDiscovery.Host("127.0.0.1:18080", TARGET_A);
        MasterRouteDiscovery.Host b = new MasterRouteDiscovery.Host("127.0.0.1:18083", TARGET_B);
        AtomicReference<List<MasterRouteDiscovery.Host>> candidates =
                new AtomicReference<>(List.of(a));
        Map<String, String> leader = new HashMap<>();
        Set<String> down = new HashSet<>();
        leader.put(a.http(), a.http());
        leader.put(b.http(), a.http());
        MasterRouteDiscovery discovery = new MasterRouteDiscovery(candidates::get, host -> {
            if (down.contains(host.http())) throw new IllegalStateException("down");
            return new MasterRouteDiscovery.Info(leader.get(host.http()), true);
        });

        discovery.refresh();
        assertEquals(TARGET_A, discovery.route().master());
        assertNull(discovery.route().slave());

        candidates.set(List.of(a, b));
        leader.replaceAll((host, ignored) -> b.http());
        discovery.refresh();
        assertEquals(TARGET_B, discovery.route().master());
        assertEquals(TARGET_A, discovery.route().slave());

        candidates.set(List.of());
        discovery.refresh();
        assertEquals(TARGET_B, discovery.route().master());
        down.add(b.http());
        discovery.refresh();
        assertEquals(TARGET_B, discovery.route().master()); // one missed probe is tolerated
        discovery.refresh();
        assertNull(discovery.route().master()); // healthy follower cannot promote itself
        assertNull(discovery.route().slave());
    }

    @Test
    void discoveredRouteRetriesUnavailableButNotBusinessResponse() {
        MasterRouteDiscovery.Host a = new MasterRouteDiscovery.Host("127.0.0.1:18080", TARGET_A);
        MasterRouteDiscovery.Host b = new MasterRouteDiscovery.Host("127.0.0.1:18083", TARGET_B);
        MasterRouteDiscovery discovery = new MasterRouteDiscovery(() -> List.of(a, b),
                host -> new MasterRouteDiscovery.Info(a.http(), true));
        discovery.refresh();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any())).thenThrow(new StatusRuntimeException(Status.UNAVAILABLE))
                .thenReturn(response(8431));
        when(stubB.schedule(any())).thenReturn(response(200));
        MasterTargetRouter router = new MasterTargetRouter(discovery,
                Map.of(TARGET_A, new FlexlbServiceGrpc.FlexlbServiceBlockingStub[]{stubA},
                        TARGET_B, new FlexlbServiceGrpc.FlexlbServiceBlockingStub[]{stubB}));

        MasterTargetRouter.ScheduleOutcome retry = router.schedule(request(), 1000L);
        assertEquals(TARGET_B, retry.lastTarget);
        assertTrue(retry.failover);
        assertEquals(200, retry.response.getCode());
        assertEquals("UNAVAILABLE", retry.attempts.get(0).status);

        MasterTargetRouter.ScheduleOutcome business = router.schedule(request(), 1000L);
        assertEquals(TARGET_A, business.lastTarget);
        assertFalse(business.failover);
        assertEquals(8431, business.response.getCode());
        verify(stubB).schedule(any());
    }

    @Test
    void fileDiscoveryUsesHttpMasterInfoAndRefreshesRole(@TempDir Path dir) throws Exception {
        HttpServer a = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        HttpServer b = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        String httpA = "127.0.0.1:" + a.getAddress().getPort();
        String httpB = "127.0.0.1:" + b.getAddress().getPort();
        AtomicReference<String> leader = new AtomicReference<>(httpA);
        AtomicReference<String> election = new AtomicReference<>("\"consistency_enabled\":true,");
        for (HttpServer server : List.of(a, b)) {
            server.createContext("/rtp_llm/master/info", exchange -> {
                assertEquals("POST", exchange.getRequestMethod());
                byte[] body = ("{" + election.get() + "\"real_master_host\":\"" + leader.get() + "\"}")
                        .getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(200, body.length);
                try (var output = exchange.getResponseBody()) { output.write(body); }
            });
            server.start();
        }
        try {
            Path file = dir.resolve("master-discovery.json");
            Files.writeString(file, "{\"hosts\":[{\"http\":\"" + httpA
                    + "\",\"grpc\":\"" + TARGET_A + "\"},{\"http\":\"" + httpB
                    + "\",\"grpc\":\"" + TARGET_B + "\"}]}");
            MasterRouteDiscovery discovery = MasterRouteDiscovery.fromFile(file);
            discovery.refresh();
            assertEquals(TARGET_A, discovery.route().master());
            assertEquals(TARGET_B, discovery.route().slave());
            leader.set(httpB);
            discovery.refresh();
            assertEquals(TARGET_B, discovery.route().master());
            assertEquals(TARGET_A, discovery.route().slave());
            // A file with one candidate must independently activate discovery.
            Files.writeString(file, "{\"hosts\":[{\"http\":\"" + httpB
                    + "\",\"grpc\":\"" + TARGET_B + "\"}]}");
            var config = JavaLoadClient.Config.fromEnvironment(
                    Map.of("MASTER_DISCOVERY_FILE", file.toString()));
            try (var client = new JavaLoadClient(config, false)) {
                var field = JavaLoadClient.class.getDeclaredField("router");
                field.setAccessible(true);
                var router = (MasterTargetRouter) field.get(client);
                assertEquals(TARGET_B, router.selectedTarget());
            }
            election.set("\"consistency_enabled\":false,");
            discovery.refresh();
            assertNull(discovery.route().master());
            election.set(""); // Older endpoints cannot silently opt into discovery.
            discovery.refresh();
            assertNull(discovery.route().master());
        } finally {
            a.stop(0);
            b.stop(0);
        }
    }

    @Test
    void discoveryRequiresElectionAndAnIdentifiedLeader() {
        var host = new MasterRouteDiscovery.Host("a:7001", TARGET_A);
        var info = new AtomicReference<>(new MasterRouteDiscovery.Info(host.http(), false));
        var discovery = new MasterRouteDiscovery(() -> List.of(host), ignored -> info.get());
        discovery.refresh();
        assertNull(discovery.route().master());
        info.set(new MasterRouteDiscovery.Info("", true));
        discovery.refresh();
        assertNull(discovery.route().master());
        info.set(new MasterRouteDiscovery.Info(host.http(), true));
        discovery.refresh();
        assertEquals(TARGET_A, discovery.route().master());
        assertNull(discovery.route().slave());
        info.set(new MasterRouteDiscovery.Info("other:7001", true));
        discovery.refresh();
        assertNull(discovery.route().master());
        info.set(new MasterRouteDiscovery.Info(host.http(), false));
        discovery.refresh();
        assertNull(discovery.route().master());
        var stub = stub();
        var router = new MasterTargetRouter(discovery,
                Map.of(TARGET_A, new FlexlbServiceGrpc.FlexlbServiceBlockingStub[]{stub}));
        var outcome = router.schedule(request(), 1000);
        assertNull(outcome.response);
        assertTrue(outcome.attempts.isEmpty());
        verify(stub, never()).schedule(any());
    }

    @Test
    void routingConfigurationIsExclusiveAndDiscoveryNeedsNoStaticTargets() {
        var discovery = JavaLoadClient.Config.fromEnvironment(Map.of("MASTER_DISCOVERY_FILE", "/test/masters.json"));
        assertEquals("/test/masters.json", discovery.masterDiscoveryFile);
        assertEquals("", discovery.grpcTarget);
        var fixed = JavaLoadClient.Config.fromEnvironment(Map.of("GRPC_TARGET", TARGET_A));
        assertEquals(TARGET_A, fixed.grpcTarget);
        assertEquals("", fixed.masterDiscoveryFile);
        assertThrows(IllegalArgumentException.class, () -> JavaLoadClient.Config.fromEnvironment(
                Map.of("GRPC_TARGET", TARGET_A, "MASTER_DISCOVERY_FILE", "/test/masters.json")));
        assertThrows(IllegalArgumentException.class, () -> JavaLoadClient.Config.fromEnvironment(
                Map.of("GRPC_TARGETS", TARGET_A + "," + TARGET_B)));
    }

    // ---- per-request row schema (additive fields) ----

    @Test
    void perRequestNodeCarriesHaObservabilityFields() {
        JavaLoadClient.RequestResult result = new JavaLoadClient.RequestResult();
        result.rid = "rid-1";
        result.routePath = "failed";
        result.masterTarget = TARGET_B;
        result.failover = true;
        result.errorKind = "transport";

        com.fasterxml.jackson.databind.node.ObjectNode node =
                JavaLoadClient.perRequestNode(result);

        assertEquals("failed", node.get("route_path").asText());
        assertEquals(TARGET_B, node.get("master_target").asText());
        assertTrue(node.get("failover").asBoolean());
        assertEquals("transport", node.get("error_kind").asText());
        // Additive-only: the pre-existing fields are all still present.
        for (String key : List.of("rid", "trace_id", "request_id", "ts", "input_len",
                "output_len", "status", "schedule_ms", "sched_done_epoch_ms", "ttft_ms",
                "total_ms", "enqueued_by_master", "prefill", "decode", "error",
                "route_path", "wall_clock_ts", "send_due_epoch_ms", "send_start_epoch_ms",
                "pacing_lag_ms")) {
            assertTrue(node.has(key), "missing legacy field: " + key);
        }
    }

    @Test
    void perRequestNodeDefaultsAreBackwardCompatible() {
        // A synthetic row (collector timeout) never touched the router: the
        // new fields serialize with neutral defaults instead of nulls.
        JavaLoadClient.RequestResult result = new JavaLoadClient.RequestResult();
        com.fasterxml.jackson.databind.node.ObjectNode node =
                JavaLoadClient.perRequestNode(result);
        assertEquals("", node.get("master_target").asText());
        assertFalse(node.get("failover").asBoolean());
        assertEquals("none", node.get("error_kind").asText());
        assertEquals("master", node.get("route_path").asText());
    }
}

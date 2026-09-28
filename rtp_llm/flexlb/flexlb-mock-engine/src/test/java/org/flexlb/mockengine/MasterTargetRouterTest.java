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

/**
 * Unit tests for the HA multi-target Schedule routing (MasterTargetRouter)
 * and the GRPC_TARGETS config contract — the pure decision core of the HA
 * case-test client retrofit: sticky target, transport-failure same-request
 * retry, symmetric wrap-around failback, and the error-code boundary (only
 * UNAVAILABLE retries; DEADLINE_EXCEEDED / business codes / other gRPC
 * statuses are terminal for the request).
 *
 * <p>Stubs are Mockito mocks with RETURNS_SELF so withDeadlineAfter chains
 * back to the mock; no real channel is ever built.
 */
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
        return new MasterTargetRouter(List.of(TARGET_A, TARGET_B),
                List.of(new FlexlbServiceGrpc.FlexlbServiceBlockingStub[]{stubA},
                        new FlexlbServiceGrpc.FlexlbServiceBlockingStub[]{stubB}));
    }

    @Test
    void stickyTargetServesRequestWithoutFailover() throws Exception {
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any())).thenReturn(response(200));
        MasterTargetRouter router = router(stubA, stubB);

        MasterTargetRouter.ScheduleOutcome outcome = router.schedule(request(), 1000L);

        assertEquals(TARGET_A, outcome.lastTarget);
        assertFalse(outcome.failover);
        assertEquals(MasterTargetRouter.ErrorKind.NONE, outcome.errorKind);
        assertEquals(200, outcome.response.getCode());
        assertEquals(TARGET_A, router.stickyTarget());
        verify(stubB, never()).schedule(any());
    }

    @Test
    void transportFailureRetriesSameRequestOnBackupAndSwitchesSticky() throws Exception {
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
        // Sticky pointer moved to B: the NEXT request goes straight to B.
        assertEquals(TARGET_B, router.stickyTarget());
        verify(stubA).schedule(any());
        verify(stubB).schedule(any());
    }

    @Test
    void stickySwitchSurvivesSubsequentRequests() throws Exception {
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any()))
                .thenThrow(new StatusRuntimeException(Status.UNAVAILABLE));
        when(stubB.schedule(any())).thenReturn(response(200));
        MasterTargetRouter router = router(stubA, stubB);

        router.schedule(request(), 1000L); // switch A -> B
        MasterTargetRouter.ScheduleOutcome second = router.schedule(request(), 1000L);

        assertFalse(second.failover);
        assertEquals(TARGET_B, second.lastTarget);
        // A was attempted exactly once (only during the failover request).
        verify(stubA).schedule(any());
        verify(stubB, org.mockito.Mockito.times(2)).schedule(any());
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
        // lastTarget = the last attempted target in the wrap-around chain.
        assertEquals(TARGET_B, outcome.lastTarget);
        assertTrue(outcome.failure instanceof StatusRuntimeException);
        // Sticky stays on A: the next request repeats the chain (no probing).
        assertEquals(TARGET_A, router.stickyTarget());
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
        assertEquals(TARGET_A, router.stickyTarget());
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
        assertEquals(TARGET_A, router.stickyTarget());
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
    void failbackWrapAroundIsSymmetric() throws Exception {
        // Scene 4 (failback_wraparound): sticky=B after A died; B then dies
        // while A recovered — the retry chain wraps back to A and the sticky
        // pointer follows. Switching has no direction preference.
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any()))
                .thenThrow(new StatusRuntimeException(Status.UNAVAILABLE))
                .thenReturn(response(200)); // recovered on the second call
        when(stubB.schedule(any()))
                .thenReturn(response(200))
                .thenThrow(new StatusRuntimeException(Status.UNAVAILABLE));
        MasterTargetRouter router = router(stubA, stubB);

        MasterTargetRouter.ScheduleOutcome failoverOut = router.schedule(request(), 1000L);
        assertEquals(TARGET_B, failoverOut.lastTarget);
        assertTrue(failoverOut.failover);
        assertEquals(TARGET_B, router.stickyTarget());

        MasterTargetRouter.ScheduleOutcome failbackOut = router.schedule(request(), 1000L);
        assertEquals(TARGET_A, failbackOut.lastTarget);
        assertTrue(failbackOut.failover);
        assertEquals(200, failbackOut.response.getCode());
        assertEquals(TARGET_A, router.stickyTarget());
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
            return new MasterRouteDiscovery.Info(leader.get(host.http()));
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
        assertEquals(TARGET_A, discovery.route().master());
        assertNull(discovery.route().slave());
    }

    @Test
    void discoveredRouteRetriesNonDeadlineGrpcFailureButNotBusinessResponse() {
        MasterRouteDiscovery.Host a = new MasterRouteDiscovery.Host("127.0.0.1:18080", TARGET_A);
        MasterRouteDiscovery.Host b = new MasterRouteDiscovery.Host("127.0.0.1:18083", TARGET_B);
        MasterRouteDiscovery discovery = new MasterRouteDiscovery(() -> List.of(a, b),
                host -> new MasterRouteDiscovery.Info(a.http()));
        discovery.refresh();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubA = stub();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stubB = stub();
        when(stubA.schedule(any())).thenThrow(new StatusRuntimeException(Status.INTERNAL))
                .thenReturn(response(8431));
        when(stubB.schedule(any())).thenReturn(response(200));
        MasterTargetRouter router = new MasterTargetRouter(discovery,
                Map.of(TARGET_A, new FlexlbServiceGrpc.FlexlbServiceBlockingStub[]{stubA},
                        TARGET_B, new FlexlbServiceGrpc.FlexlbServiceBlockingStub[]{stubB}));

        MasterTargetRouter.ScheduleOutcome retry = router.schedule(request(), 1000L);
        assertEquals(TARGET_B, retry.lastTarget);
        assertTrue(retry.failover);
        assertEquals(200, retry.response.getCode());
        assertEquals("INTERNAL", retry.attempts.get(0).status);

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
        for (HttpServer server : List.of(a, b)) {
            server.createContext("/rtp_llm/master/info", exchange -> {
                assertEquals("POST", exchange.getRequestMethod());
                byte[] body = ("{\"real_master_host\":\"" + leader.get() + "\"}")
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
        } finally {
            a.stop(0);
            b.stop(0);
        }
    }

    // ---- GRPC_TARGETS config contract ----

    @Test
    void parseGrpcTargetsSplitsTrimsAndDedups() {
        assertEquals(List.of("127.0.0.1:18082", "127.0.0.2:18085"),
                JavaLoadClient.Config.parseGrpcTargets(
                        " 127.0.0.1:18082 , 127.0.0.2:18085 "));
        assertEquals(List.of("127.0.0.1:18082"),
                JavaLoadClient.Config.parseGrpcTargets("127.0.0.1:18082,127.0.0.1:18082"));
        assertEquals(List.of("a:1", "b:2"),
                JavaLoadClient.Config.parseGrpcTargets("a:1,,b:2,"));
    }

    @Test
    void parseGrpcTargetsFailsFastOnGarbage() {
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.Config.parseGrpcTargets("no-port"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.Config.parseGrpcTargets("host:abc"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.Config.parseGrpcTargets("host:0"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.Config.parseGrpcTargets("host:70000"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.Config.parseGrpcTargets(",:80"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.Config.parseGrpcTargets(",,"));
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

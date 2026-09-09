package org.flexlb.mockengine;

import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.junit.jupiter.api.Test;

import java.util.List;

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

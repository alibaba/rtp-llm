package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.EngineCancelChannel.CancelAck;
import org.flexlb.balance.eviction.EngineCancelChannel.CancelOutcome;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * {@link MockEngineCancelChannel} contract tests against the in-process mock
 * engine cluster (test wiring only):
 * <ul>
 *   <li>accepted: a live request on the addressed worker is cancelled and its
 *       CANCELLED completion surfaces in WorkerStatus;</li>
 *   <li>idempotent: an accepted priority-cancel tombstone stays ACCEPTED;</li>
 *   <li>not found: a completed-before-cancel request answers NOT_FOUND
 *       (production seen-but-terminal branch) and does not scan another
 *       Prefill for a match;</li>
 *   <li>tombstoned: a never-seen rid answers TOMBSTONED, installs the
 *       ABSENT_FENCE tombstone, and a racing later Enqueue of that rid is
 *       rejected pre-admission with the typed 8429 error;</li>
 *   <li>failed: Decode rejects the Prefill-owned Cancel RPC;</li>
 *   <li>unsupported: endpoint whose port maps to no mock engine;</li>
 *   <li>fault injections: an armed cancel_no_respond / cancel_error /
 *       cancel_unexpected_status is an RPC-LAYER failure — the future hangs /
 *       fails, the engine cancel state machine is never touched (no fences,
 *       no tombstones, no census branch), and clearing the injection restores
 *       the normal path.</li>
 * </ul>
 */
class MockEngineCancelChannelTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 62500;

    @TempDir
    Path tempDir;

    private MockEngineTestCluster cluster;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @AfterEach
    void tearDown() {
        if (cluster != null) {
            cluster.close();
        }
    }

    // ---- accepted: mid-flight cancel drives the mock ----

    @Test
    void cancelMidFlightAcceptedAndCancelledSurfacesInWorkerStatus() throws Exception {
        startCluster(model("500"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        int n = 4;
        EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[n];
        for (int i = 0; i < n; i++) {
            inputs[i] = inputWithDecode(i + 1, 10, decodeServices.get(0).getGrpcPort());
        }
        EngineRpcService.EnqueueBatchResponsePB response =
                enqueue(prefill, batch(9000, slot(0, inputs)));
        assertEquals(n, response.getSuccessesCount());
        awaitInflight(prefill, 1, 1_000);

        CancelOutcome outcome = channel
                .cancel(target(prefill.getGrpcPort()), 1L, 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, outcome.ack(),
                "mid-flight cancel must register the intent");

        // The addressed Prefill is the authoritative typed CANCELED producer.
        EngineRpcService.WorkerStatusPB status = workerStatus(prefill, 0);
        boolean cancelledReported = status.getFinishedTaskListList().stream()
                .anyMatch(task -> task.getRequestId() == 1L
                        && task.getErrorInfo().getErrorCode() == 8429L
                        && task.getPriorityPreemptionProgress()
                        == EngineRpcService.PriorityPreemptionProgressPB
                        .PRIORITY_PREEMPTION_CANCELED);
        assertTrue(cancelledReported,
                "typed CANCELED+8429 for request 1 must appear in Prefill WorkerStatus");

        awaitAllInflightZero(10_000);
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertFalse(service.isLeakDetected(),
                    "no leak on port " + service.getGrpcPort());
        }
    }

    @Test
    void cancelStage4RoutesThroughOriginalPrefillToOwnedDecode() throws Exception {
        startCluster(model("10", 10_000.0), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        enqueue(prefill, batch(9050, slot(0,
                inputWithDecode(51, 10, decode.getGrpcPort()))));
        awaitInflight(decode, 1, 1_000);
        awaitNoInflight(prefill, 1_000);
        assertEquals(0, prefill.getInflightCount(),
                "stage 4 begins only after Prefill handed the request to Decode");
        assertTrue(prefill.hasDownstreamOwnership(51L));

        CancelOutcome outcome = channel.cancel(target(prefill.getGrpcPort()), 51L, 2_000)
                .get(2, TimeUnit.SECONDS);

        assertEquals(CancelAck.ACCEPTED, outcome.ack());
        assertEquals(0, decode.getInflightCount());
        assertFalse(prefill.hasDownstreamOwnership(51L));
        assertFalse(decode.hasUpstreamOwnership(51L));
        boolean cancelledReported = workerStatus(decode, 0).getFinishedTaskListList().stream()
                .anyMatch(task -> task.getRequestId() == 51L
                        && task.getErrorInfo().getErrorCode()
                        == EngineRpcService.ErrorCodePB.CANCELLED.getNumber());
        assertTrue(cancelledReported,
                "Decode must retain its ordinary CANCELLED terminal");
        boolean typedCanceledReported = workerStatus(prefill, 0).getFinishedTaskListList().stream()
                .anyMatch(task -> task.getRequestId() == 51L
                        && task.getErrorInfo().getErrorCode() == 8429L
                        && task.getPriorityPreemptionProgress()
                        == EngineRpcService.PriorityPreemptionProgressPB
                        .PRIORITY_PREEMPTION_CANCELED);
        assertTrue(typedCanceledReported,
                "original Prefill must report authoritative typed CANCELED+8429");
    }

    // ---- not found after natural completion / idempotent cancel tombstone ----

    @Test
    void cancelAfterCompletionIsNotFound() throws Exception {
        startCluster(model("10"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        enqueue(prefill, batch(9100, slot(0,
                inputWithDecode(11, 10, decodeServices.get(0).getGrpcPort()))));
        awaitAllInflightZero(5_000);

        CancelOutcome outcome = channel
                .cancel(target(prefill.getGrpcPort()), 11L, 2_000)
                .get(2, TimeUnit.SECONDS);
        // Production-faithful branch (C++ Cancel handler):
        // seen-but-terminal answers NOT_FOUND — the completion record
        // stays in the retain-window backlog for GetWorkerStatus delivery
        // (TOMBSTONED is reserved for never-seen rids).
        assertEquals(CancelAck.NOT_FOUND, outcome.ack());
        // Behavior: the request had already finished; nothing is re-inflight.
        assertEquals(0, prefill.getInflightCount());
        assertEquals(0, prefill.getDownstreamOwnershipCount());
        assertEquals(0, decodeServices.get(0).getUpstreamOwnershipCount());
    }

    @Test
    void repeatedPriorityCancelStaysAcceptedAndPublishesOneTerminal() throws Exception {
        startCluster(model("500"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        enqueue(prefill, batch(9200, slot(0,
                inputWithDecode(21, 10, decodeServices.get(0).getGrpcPort()))));
        awaitInflight(prefill, 1, 1_000);

        CancelOutcome first = channel
                .cancel(target(prefill.getGrpcPort()), 21L, 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, first.ack());

        CancelOutcome second = channel
                .cancel(target(prefill.getGrpcPort()), 21L, 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, second.ack(),
                "accepted priority-cancel tombstones are idempotent");
        long terminalCount = workerStatus(prefill, -1).getFinishedTaskListList().stream()
                .filter(task -> task.getRequestId() == 21L
                        && task.getErrorInfo().getErrorCode() == 8429L
                        && task.getPriorityPreemptionProgress()
                        == EngineRpcService.PriorityPreemptionProgressPB
                        .PRIORITY_PREEMPTION_CANCELED)
                .count();
        assertEquals(1L, terminalCount,
                "a retry must not publish a second CANCELED+8429 terminal");
        awaitAllInflightZero(10_000);
    }

    // ---- not found: unknown request id / wrong worker ----

    @Test
    void cancelUnknownRequestIsTombstonedAndFencesLaterEnqueue() throws Exception {
        startCluster(model("10"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        CancelOutcome outcome = channel
                .cancel(target(prefill.getGrpcPort()), 424242L, 2_000)
                .get(2, TimeUnit.SECONDS);
        // Never-seen rid: TOMBSTONED — the ABSENT_FENCE tombstone is
        // installed (production Prefill contract).
        assertEquals(CancelAck.TOMBSTONED, outcome.ack());

        // Fence idempotence: a retried cancel still reads TOMBSTONED (it
        // must NOT flip onto the ACCEPTED ACTIVE_CANCEL tombstone branch).
        CancelOutcome retry = channel
                .cancel(target(prefill.getGrpcPort()), 424242L, 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.TOMBSTONED, retry.ack());

        // The absent fence rejects a racing later Enqueue of the same rid
        // with the typed 8429 error, pre-admission: no success ack, no
        // engine state, no inflight residue.
        EngineRpcService.EnqueueBatchResponsePB response = enqueue(prefill,
                batch(9101, slot(0,
                        inputWithDecode(424242L, 10, decodeServices.get(0).getGrpcPort()))));
        assertEquals(0, response.getSuccessesCount(),
                "a fenced rid must not be acked as admitted");
        assertEquals(1, response.getErrorsCount(),
                "the fenced rid carries exactly one ack error");
        assertEquals(424242L, response.getErrors(0).getRequestId());
        assertEquals(8429L, response.getErrors(0).getErrorInfo().getErrorCode(),
                "absent-fence rejection carries the typed 8429 (PRIORITY_PREEMPTED)");
        assertEquals(0, prefill.getInflightCount(),
                "a fenced rid must leave no engine-side inflight residue");
    }

    @Test
    void decodeTargetFailsWithoutScanningOrCancellingPrefill() throws Exception {
        startCluster(model("500"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        enqueue(prefill, batch(9300, slot(0,
                inputWithDecode(31, 10, decodeServices.get(0).getGrpcPort()))));
        awaitInflight(prefill, 1, 1_000);

        CancelOutcome outcome = channel
                .cancel(target(decodeServices.get(0).getGrpcPort()), 31L, 2_000)
                .get(2, TimeUnit.SECONDS);

        assertEquals(CancelAck.FAILED, outcome.ack(),
                "Decode must model the real UNIMPLEMENTED Cancel contract");
        assertTrue(prefill.getInflightCount() > 0,
                "a wrong target must not cancel the request on another worker");
    }

    // ---- unsupported branch + isSupported gate ----

    @Test
    void unknownPortIsUnsupported() throws Exception {
        startCluster(model("10"), 1, 1);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        assertTrue(channel.isSupported(endpoint(prefillServices.get(0).getGrpcPort())));
        assertTrue(channel.isSupported(endpoint(decodeServices.get(0).getGrpcPort())));
        assertFalse(channel.isSupported(endpoint(59999)));

        CancelOutcome outcome = channel
                .cancel(target(59999), 1L, 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.UNSUPPORTED, outcome.ack());
    }

    // ---- cancel fault injections: RPC-layer failures, engine state untouched ----

    @Test
    void cancelNoRespondInjectionHangsTheFutureAndLeavesEngineStateUntouched() throws Exception {
        // Prefill expression 3000ms — a wide-enough execution window that
        // the request stays tracked through the 500ms hang assertion below,
        // yet short enough that the post-clear cancel settles inside the
        // 10s awaitAllInflightZero window.
        startCluster(model("3000"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        enqueue(prefill, batch(9400, slot(0,
                inputWithDecode(41, 10, decodeServices.get(0).getGrpcPort()))));
        awaitInflight(prefill, 1, 1_000);

        prefill.setFaultConfig(prefill.getFaultConfig().toBuilder()
                .cancelNoRespond(true)
                .build());
        try {
            CompletableFuture<CancelOutcome> future = channel
                    .cancel(target(prefill.getGrpcPort()), 41L, 2_000);
            assertThrows(TimeoutException.class,
                    () -> future.get(500, TimeUnit.MILLISECONDS),
                    "cancel_no_respond must leave the cancel future pending");
            assertFalse(future.isDone(),
                    "the injected cancel future must stay incomplete (hanging RPC)");
            assertEquals(1, prefill.getInflightCount(),
                    "an injected cancel must not touch engine state (still in flight)");
            assertEquals(1L, cluster.stats().cancelCensusInjected.sum(),
                    "the injected arrival must be censused");
            assertEquals(0L, cluster.stats().cancelCensusTracked.sum(),
                    "the engine cancel state machine was never entered");
        } finally {
            prefill.clearFaultConfig();
        }

        // After the injection clears, the same rid cancels normally — proof
        // the fault path installed no tombstone and no fence.
        CancelOutcome outcome = channel
                .cancel(target(prefill.getGrpcPort()), 41L, 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelAck.ACCEPTED, outcome.ack());
        awaitAllInflightZero(10_000);
    }

    @Test
    void cancelErrorInjectionFailsTheFutureAndInstallsNoFence() throws Exception {
        startCluster(model("3000"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        // A never-seen rid: a REAL cancel would install the ABSENT_FENCE
        // tombstone and answer TOMBSTONED — the injected transport failure
        // must short-circuit before any of that.
        prefill.setFaultConfig(prefill.getFaultConfig().toBuilder()
                .cancelError(true)
                .build());
        try {
            CompletableFuture<CancelOutcome> future = channel
                    .cancel(target(prefill.getGrpcPort()), 424243L, 2_000);
            ExecutionException failure = assertThrows(ExecutionException.class,
                    () -> future.get(2, TimeUnit.SECONDS),
                    "cancel_error must surface as a failed future");
            assertTrue(failure.getCause() instanceof IllegalStateException,
                    "the transport-layer failure surfaces as IllegalStateException");
            assertEquals(1L, cluster.stats().cancelCensusInjected.sum(),
                    "the injected arrival must be censused");
            assertEquals(0L, cluster.stats().cancelCensusTombstone.sum(),
                    "an injected cancel must NOT install the absent-fence tombstone");
            assertEquals(0L, cluster.stats().cancelCensusUnknown.sum(),
                    "the engine cancel state machine was never entered");
        } finally {
            prefill.clearFaultConfig();
        }

        // No fence was installed: the same never-seen rid enqueues cleanly
        // (an ABSENT_FENCE would have rejected it with the typed 8429).
        EngineRpcService.EnqueueBatchResponsePB response = enqueue(prefill,
                batch(9401, slot(0,
                        inputWithDecode(424243L, 10, decodeServices.get(0).getGrpcPort()))));
        assertEquals(1, response.getSuccessesCount(),
                "an injected cancel failure must not fence later enqueues");
        awaitAllInflightZero(10_000);
    }

    @Test
    void cancelUnexpectedStatusInjectionFailsTheFutureWithoutTerminalJudgment() throws Exception {
        startCluster(model("3000"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineCancelChannel channel = new MockEngineCancelChannel(services);

        prefill.setFaultConfig(prefill.getFaultConfig().toBuilder()
                .cancelUnexpectedStatus(true)
                .build());
        try {
            CompletableFuture<CancelOutcome> future = channel
                    .cancel(target(prefill.getGrpcPort()), 424244L, 2_000);
            ExecutionException failure = assertThrows(ExecutionException.class,
                    () -> future.get(2, TimeUnit.SECONDS),
                    "an out-of-contract ack status must fail the future");
            assertTrue(failure.getCause() instanceof IllegalStateException);
            assertTrue(failure.getCause().getMessage().contains("unexpected cancel ack status"),
                    "the failure must name the out-of-contract status mapping");
            assertEquals(1L, cluster.stats().cancelCensusInjected.sum(),
                    "the injected arrival must be censused");
            assertEquals(0L, cluster.stats().cancelCensusTombstone.sum(),
                    "an injected cancel must NOT install the absent-fence tombstone");
        } finally {
            prefill.clearFaultConfig();
        }

        // Engine state untouched: the never-seen rid enqueues cleanly.
        EngineRpcService.EnqueueBatchResponsePB response = enqueue(prefill,
                batch(9402, slot(0,
                        inputWithDecode(424244L, 10, decodeServices.get(0).getGrpcPort()))));
        assertEquals(1, response.getSuccessesCount(),
                "an unexpected-status cancel must not fence later enqueues");
        awaitAllInflightZero(10_000);
    }

    // ---- helpers ----

    private static CancelTarget target(int grpcPort) {
        return new CancelTarget("127.0.0.1", grpcPort);
    }

    private static DecodeEndpoint endpoint(int grpcPort) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.DECODE, "test", "127.0.0.1",
                grpcPort - 2, grpcPort, null);
        return new DecodeEndpoint(status, mock(EndpointEventSink.class));
    }

    private void startCluster(MockPerformanceModel model, int nPrefill, int nDecode)
            throws IOException {
        cluster = MockEngineTestCluster.create(model, BASE_PORT, nPrefill, nDecode);
        services = cluster.services();
        prefillServices = cluster.prefills();
        decodeServices = cluster.decodes();
    }

    private MockPerformanceModel model(String formula) throws IOException {
        return model(formula, 1.0);
    }

    private MockPerformanceModel model(String formula, double decodeStepMs) throws IOException {
        return MockEngineTestSupport.performanceModel(tempDir, formula, 1.0, decodeStepMs);
    }

    private void awaitInflight(JavaMockEngineCluster.FastRpcService service, int min, long timeoutMs)
            throws InterruptedException {
        cluster.awaitInflight(service, min, timeoutMs);
    }

    private void awaitNoInflight(JavaMockEngineCluster.FastRpcService service, long timeoutMs)
            throws InterruptedException {
        cluster.awaitNoInflight(service, timeoutMs);
    }

    private void awaitAllInflightZero(long timeoutMs) throws InterruptedException {
        cluster.awaitAllInflightZero(timeoutMs);
    }

}

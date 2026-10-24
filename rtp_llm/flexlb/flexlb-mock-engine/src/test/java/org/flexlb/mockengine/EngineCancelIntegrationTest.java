package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.engine.grpc.EngineGrpcClient.CancelReply;
import org.flexlb.engine.grpc.EngineGrpcClient;
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
 * {@link MockEngineClients} contract tests against the in-process mock
 * engine cluster (test wiring only):
 * <ul>
 *   <li>accepted: a live request on the addressed worker is cancelled and its
 *       CANCELLED completion surfaces in WorkerStatus;</li>
 *   <li>idempotent: an accepted priority-cancel terminal record stays ACCEPTED;</li>
 *   <li>not found: a completed-before-cancel request answers NOT_FOUND
 *       (production seen-but-terminal branch) and does not scan another
 *       Prefill for a match;</li>
 *   <li>request_fenced: a never-seen rid answers REQUEST_FENCED, installs the
 *       ABSENT_FENCE record, and a racing later Enqueue of that rid is
 *       rejected pre-admission with the typed 8429 error;</li>
 *   <li>failed: Decode rejects the Prefill-owned Cancel RPC;</li>
 *   <li>unsupported: endpoint whose port maps to no mock engine;</li>
 *   <li>fault injections: an armed cancel_no_respond / cancel_error /
 *       cancel_unexpected_status is an RPC-LAYER failure — the future hangs /
 *       fails, the engine cancel state machine is never touched (no fences,
 *       no terminal records, no census branch), and clearing the injection restores
 *       the normal path.</li>
 * </ul>
 */
class EngineCancelIntegrationTest {

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

    @Test
    void ordinaryCancelBeforeEnqueueFencesAdmissionWithoutPriorityError() throws Exception {
        startCluster(model("10"), 1, 1);
        var prefill = prefillServices.getFirst();
        var channel = MockEngineClients.inProcess(services);
        for (var reason : List.of(CancelReason.CLIENT_CANCELLED, CancelReason.DEADLINE_EXCEEDED, CancelReason.SHUTDOWN)) {
            long id = 801L + reason.ordinal();
            assertEquals(CancelReply.REQUEST_CLEANED, channel.cancelAsync("127.0.0.1", prefill.getGrpcPort(),
                    reason.toRequest(id), 500L).get(1, TimeUnit.SECONDS));
            channel.cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(id), 500L).get();
            var response = enqueue(prefill, batch(9801L + id, slot(0,
                    inputWithDecode(id, 10, decodeServices.getFirst().getGrpcPort()))));
            assertEquals(0, response.getSuccessesCount());
            assertEquals(reason == CancelReason.DEADLINE_EXCEEDED ? 603 : EngineRpcService.ErrorCodePB.CANCELLED.getNumber(),
                    response.getErrors(0).getErrorInfo().getErrorCode());
            assertEquals(0, prefill.getInflightCount());
            assertEquals(0, decodeServices.getFirst().getInflightCount());
        }
    }

    @Test
    void ordinaryNoFetchCleanupReleasesTheOriginalDecodeAndKeepsItsReason() throws Exception {
        startCluster(model("10", 10_000.0), 1, 1);
        var prefill = prefillServices.getFirst();
        var decode = decodeServices.getFirst();
        var channel = MockEngineClients.inProcess(services);
        for (var reason : List.of(CancelReason.CLIENT_CANCELLED, CancelReason.DEADLINE_EXCEEDED, CancelReason.SHUTDOWN)) {
            long id = 802L + reason.ordinal();
            enqueue(prefill, batch(9802L + id, slot(0, inputWithDecode(id, 10, decode.getGrpcPort()))));
            awaitInflight(decode, 1, 1_000);
            assertEquals(CancelReply.REQUEST_CLEANED, channel.cancelAsync("127.0.0.1", prefill.getGrpcPort(),
                    reason.toRequest(id), 500L).get(1, TimeUnit.SECONDS));
            assertEquals(0, decode.getInflightCount());
            assertEquals(0, prefill.getInflightCount());
            assertFalse(MockEngineTestSupport.hasDownstreamOwnership(prefill, id));
            assertFalse(MockEngineTestSupport.hasUpstreamOwnership(decode, id));
            var terminal = workerStatus(prefill, 0).getFinishedTaskListList().stream()
                    .filter(task -> task.getRequestId() == id && task.hasErrorInfo()).findFirst().orElseThrow();
            assertEquals(reason == CancelReason.DEADLINE_EXCEEDED ? 603 : EngineRpcService.ErrorCodePB.CANCELLED.getNumber(),
                    terminal.getErrorInfo().getErrorCode());
            assertEquals(CancelReply.REQUEST_CLEANED, channel.cancelAsync("127.0.0.1", prefill.getGrpcPort(),
                    CancelReason.CLIENT_CANCELLED.toRequest(id), 500L).get(1, TimeUnit.SECONDS));
            assertEquals(terminal, workerStatus(prefill, 0).getFinishedTaskListList().stream()
                    .filter(task -> task.getRequestId() == id && task.hasErrorInfo()).findFirst().orElseThrow());
            assertFalse(MockEngineTestSupport.isLeakDetected(prefill));
            assertFalse(MockEngineTestSupport.isLeakDetected(decode));
        }
    }

    // ---- accepted: mid-flight cancel drives the mock ----

    @Test
    void cancelMidFlightAcceptedAndCancelledSurfacesInWorkerStatus() throws Exception {
        startCluster(model("500"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineGrpcClient channel = MockEngineClients.inProcess(services);

        int n = 4;
        EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[n];
        for (int i = 0; i < n; i++) {
            inputs[i] = inputWithDecode(i + 1, 10, decodeServices.get(0).getGrpcPort());
        }
        EngineRpcService.EnqueueBatchResponsePB response =
                enqueue(prefill, batch(9000, slot(0, inputs)));
        assertEquals(n, response.getSuccessesCount());
        awaitInflight(prefill, 1, 1_000);

        CancelReply outcome = channel
                .cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(1L), 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelReply.ACCEPTED, outcome,
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
            assertFalse(MockEngineTestSupport.isLeakDetected(service),
                    "no leak on port " + service.getGrpcPort());
        }
    }

    @Test
    void cancelStage4RoutesThroughOriginalPrefillToOwnedDecode() throws Exception {
        startCluster(model("10", 10_000.0), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);
        EngineGrpcClient channel = MockEngineClients.inProcess(services);

        enqueue(prefill, batch(9050, slot(0,
                inputWithDecode(51, 10, decode.getGrpcPort()))));
        awaitInflight(decode, 1, 1_000);
        awaitNoInflight(prefill, 1_000);
        assertEquals(0, prefill.getInflightCount(),
                "stage 4 begins only after Prefill handed the request to Decode");
        assertTrue(MockEngineTestSupport.hasDownstreamOwnership(prefill, 51L));

        CancelReply outcome = channel.cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(51L), 2_000)
                .get(2, TimeUnit.SECONDS);

        assertEquals(CancelReply.ACCEPTED, outcome);
        assertEquals(0, decode.getInflightCount());
        assertFalse(MockEngineTestSupport.hasDownstreamOwnership(prefill, 51L));
        assertFalse(MockEngineTestSupport.hasUpstreamOwnership(decode, 51L));
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

    // ---- not found after natural completion / idempotent cancel terminal record ----

    @Test
    void cancelAfterCompletionIsNotFound() throws Exception {
        startCluster(model("10"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineGrpcClient channel = MockEngineClients.inProcess(services);

        enqueue(prefill, batch(9100, slot(0,
                inputWithDecode(11, 10, decodeServices.get(0).getGrpcPort()))));
        awaitAllInflightZero(5_000);

        CancelReply outcome = channel
                .cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(11L), 2_000)
                .get(2, TimeUnit.SECONDS);
        // Production-faithful branch (C++ Cancel handler):
        // seen-but-terminal answers NOT_FOUND — the completion record
        // stays in the retain-window backlog for GetWorkerStatus delivery
        // (REQUEST_FENCED is reserved for never-seen rids).
        assertEquals(CancelReply.NOT_FOUND, outcome);
        // Behavior: the request had already finished; nothing is re-inflight.
        assertEquals(0, prefill.getInflightCount());
        assertEquals(0, MockEngineTestSupport.getDownstreamOwnershipCount(prefill));
        assertEquals(0, MockEngineTestSupport.getUpstreamOwnershipCount(decodeServices.get(0)));
    }

    @Test
    void repeatedPriorityCancelStaysAcceptedAndPublishesOneTerminal() throws Exception {
        startCluster(model("500"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineGrpcClient channel = MockEngineClients.inProcess(services);

        enqueue(prefill, batch(9200, slot(0,
                inputWithDecode(21, 10, decodeServices.get(0).getGrpcPort()))));
        awaitInflight(prefill, 1, 1_000);

        CancelReply first = channel
                .cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(21L), 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelReply.ACCEPTED, first);

        CancelReply second = channel
                .cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(21L), 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelReply.ACCEPTED, second,
                "accepted priority-cancel terminal records are idempotent");
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
    void cancelUnknownRequestIsFencedAndRejectsLaterEnqueue() throws Exception {
        startCluster(model("10"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineGrpcClient channel = MockEngineClients.inProcess(services);

        CancelReply outcome = channel
                .cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(424242L), 2_000)
                .get(2, TimeUnit.SECONDS);
        // Never-seen rid: REQUEST_FENCED — the ABSENT_FENCE record is
        // installed (production Prefill contract).
        assertEquals(CancelReply.REQUEST_FENCED, outcome);

        // Fence idempotence: a retried cancel still reads REQUEST_FENCED (it
        // must NOT flip onto the ACCEPTED ACTIVE_CANCEL terminal record branch).
        CancelReply retry = channel
                .cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(424242L), 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelReply.REQUEST_FENCED, retry);

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
        EngineGrpcClient channel = MockEngineClients.inProcess(services);

        enqueue(prefill, batch(9300, slot(0,
                inputWithDecode(31, 10, decodeServices.get(0).getGrpcPort()))));
        awaitInflight(prefill, 1, 1_000);

        var outcome = channel.cancelAsync("127.0.0.1", decodeServices.get(0).getGrpcPort(),
                CancelReason.PRIORITY_PREEMPTED.toRequest(31L), 2_000);
        assertThrows(ExecutionException.class, () -> outcome.get(2, TimeUnit.SECONDS),
                "Decode must model the real UNIMPLEMENTED Cancel contract");
        assertTrue(prefill.getInflightCount() > 0,
                "a wrong target must not cancel the request on another worker");
    }

    // ---- unsupported branch ----

    @Test
    void unknownPortIsUnsupported() throws Exception {
        startCluster(model("10"), 1, 1);
        EngineGrpcClient channel = MockEngineClients.inProcess(services);


        CancelReply outcome = channel
                .cancelAsync("127.0.0.1", 59999, CancelReason.PRIORITY_PREEMPTED.toRequest(1L), 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelReply.UNSUPPORTED, outcome);
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
        EngineGrpcClient channel = MockEngineClients.inProcess(services);

        enqueue(prefill, batch(9400, slot(0,
                inputWithDecode(41, 10, decodeServices.get(0).getGrpcPort()))));
        awaitInflight(prefill, 1, 1_000);

        prefill.setFaultConfig(prefill.getFaultConfig().toBuilder()
                .cancelNoRespond(true)
                .build());
        try {
            CompletableFuture<CancelReply> future = channel
                    .cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(41L), 2_000);
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
        // the fault path installed no terminal record and no fence.
        CancelReply outcome = channel
                .cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(41L), 2_000)
                .get(2, TimeUnit.SECONDS);
        assertEquals(CancelReply.ACCEPTED, outcome);
        awaitAllInflightZero(10_000);
    }

    @Test
    void cancelErrorInjectionFailsTheFutureAndInstallsNoFence() throws Exception {
        startCluster(model("3000"), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        EngineGrpcClient channel = MockEngineClients.inProcess(services);

        // A never-seen rid: a REAL cancel would install the ABSENT_FENCE
        // terminal record and answer REQUEST_FENCED — the injected transport failure
        // must short-circuit before any of that.
        prefill.setFaultConfig(prefill.getFaultConfig().toBuilder()
                .cancelError(true)
                .build());
        try {
            CompletableFuture<CancelReply> future = channel
                    .cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(424243L), 2_000);
            ExecutionException failure = assertThrows(ExecutionException.class,
                    () -> future.get(2, TimeUnit.SECONDS),
                    "cancel_error must surface as a failed future");
            assertTrue(failure.getCause() instanceof IllegalStateException,
                    "the transport-layer failure surfaces as IllegalStateException");
            assertEquals(1L, cluster.stats().cancelCensusInjected.sum(),
                    "the injected arrival must be censused");
            assertEquals(0L, cluster.stats().cancelCensusAlreadyCancelled.sum(),
                    "an injected cancel must NOT install the request fence");
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
        EngineGrpcClient channel = MockEngineClients.inProcess(services);

        prefill.setFaultConfig(prefill.getFaultConfig().toBuilder()
                .cancelUnexpectedStatus(true)
                .build());
        try {
            CompletableFuture<CancelReply> future = channel
                    .cancelAsync("127.0.0.1", prefill.getGrpcPort(), CancelReason.PRIORITY_PREEMPTED.toRequest(424244L), 2_000);
            ExecutionException failure = assertThrows(ExecutionException.class,
                    () -> future.get(2, TimeUnit.SECONDS),
                    "an out-of-contract ack status must fail the future");
            assertTrue(failure.getCause() instanceof IllegalStateException);
            assertTrue(failure.getCause().getMessage().contains("unexpected cancel ack status"),
                    "the failure must name the out-of-contract status mapping");
            assertEquals(1L, cluster.stats().cancelCensusInjected.sum(),
                    "the injected arrival must be censused");
            assertEquals(0L, cluster.stats().cancelCensusAlreadyCancelled.sum(),
                    "an injected cancel must NOT install the request fence");
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

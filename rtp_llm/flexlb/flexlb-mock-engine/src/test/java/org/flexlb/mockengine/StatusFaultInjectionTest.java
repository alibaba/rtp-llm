package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.httpGet;
import static org.flexlb.mockengine.MockEngineTestSupport.httpPost;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the status-report fault-injection family (12 new types):
 * 9 getWorkerStatus output-layer faults (suppress_finished / suppress_running /
 * suppress_rids / no_respond / fake_task / duplicate_finished / cursor_regress /
 * version_regress / zombie_running) and 3 EnqueueBatch ack faults
 * (enqueue_ack_partial_fail / enqueue_ack_error_code / enqueue_ack_drop),
 * plus the execution-phase partial-failure injection
 * (prefill_async_partial_fail: batch members that fail AT the prefill
 * completion callback and surface as typed terminals on the status channel —
 * production "stream->reportError → dequeue fills task_info.error_code /
 * error_message → finished_task_list" semantics).
 *
 * <p>All injections go through the real HTTP /inject endpoint of
 * {@link MockControlServer} (Java type format) and are asserted against the
 * gRPC service directly, mirroring {@link ComprehensiveFaultInjectionTest}.
 */
class StatusFaultInjectionTest {

    private static final AtomicInteger PORT_ALLOCATOR = new AtomicInteger(63100);

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private MockControlServer controlServer;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @AfterEach
    void tearDown() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
            controlServer = null;
        }
        if (services != null) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
            services = null;
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
            scheduler = null;
        }
        prefillServices = null;
        decodeServices = null;
    }

    // ════════════════════════════════════════════════════════════════
    //  status_suppress_finished
    // ════════════════════════════════════════════════════════════════

    @Test
    void statusSuppressFinishedHidesCompletionsPermanently() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        // Baseline: rid 1 flows normally at cursor 0.
        enqueue(prefill, batch(1001, slot(0, input(1, 10))));
        EngineRpcService.WorkerStatusPB first = awaitFinished(prefill, 0, 1, 5_000);
        assertEquals(1, first.getFinishedTaskListCount());
        assertEquals(1, first.getLatestFinishedVersion());

        // A second completion (v2) becomes visible at cursor 1.
        enqueue(prefill, batch(1002, slot(0, input(2, 10))));
        EngineRpcService.WorkerStatusPB second = awaitFinished(prefill, 1, 1, 5_000);
        assertEquals(1, second.getFinishedTaskListCount());
        assertEquals(2, second.getFinishedTaskList(0).getRequestId());
        assertEquals(2, second.getLatestFinishedVersion());

        // Suppress: the SAME query now hides the completion, but the cursor
        // (latestFinishedVersion) still reports the REAL value — the master
        // advances past a completion it never received.
        inject(basePort, "status_suppress_finished", null);
        EngineRpcService.WorkerStatusPB suppressed = workerStatus(prefill, 1);
        assertEquals(0, suppressed.getFinishedTaskListCount(),
                "suppress_finished must empty finishedTaskList");
        assertEquals(2, suppressed.getLatestFinishedVersion(),
                "latestFinishedVersion stays REAL under suppression");

        // Master advances to the real cursor. Under the retain-window
        // protocol the read no longer destroys the record — but for THIS
        // master (cursor now at 2) the suppressed completion is permanently
        // lost: no future poll at its advanced cursor ever returns it.
        workerStatus(prefill, 2);
        clearInject(basePort);
        EngineRpcService.WorkerStatusPB afterClear = workerStatus(prefill, 2);
        assertEquals(0, afterClear.getFinishedTaskListCount(),
                "suppressed completion stays lost for the master whose cursor advanced");
        assertEquals(2, afterClear.getLatestFinishedVersion());

        // A consumer whose cursor has NOT advanced past the suppressed
        // version (e.g. the standby master in dual-flexlb HA, polling the
        // same engine with an independent cursor) still receives it — the
        // shared backlog survives reads and is bounded only by the retain
        // window trimmed in periodicCleanup.
        EngineRpcService.WorkerStatusPB laggingConsumer = workerStatus(prefill, 1);
        assertEquals(1, laggingConsumer.getFinishedTaskListCount(),
                "an un-advanced cursor still sees the suppressed completion");
        assertEquals(2, laggingConsumer.getFinishedTaskList(0).getRequestId());
        assertEquals(2, laggingConsumer.getLatestFinishedVersion());
    }

    // ════════════════════════════════════════════════════════════════
    //  status_suppress_running
    // ════════════════════════════════════════════════════════════════

    @Test
    void statusSuppressRunningHidesRunningSnapshot() throws Exception {
        MockPerformanceModel model = model("300");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        enqueue(prefill, batch(2001, slot(0, input(21, 10))));
        EngineRpcService.WorkerStatusPB normal = workerStatus(prefill, 0);
        assertEquals(1, normal.getRunningTaskInfoCount(), "running snapshot real without fault");

        inject(basePort, "status_suppress_running", null);
        EngineRpcService.WorkerStatusPB suppressed = workerStatus(prefill, 0);
        assertEquals(0, suppressed.getRunningTaskInfoCount(),
                "suppress_running must empty runningTaskInfo");
        assertEquals(1, suppressed.getRunningQueryLen(),
                "runningQueryLen stays REAL (self-inconsistent report)");

        // finished is NOT suppressed by the running fault.
        EngineRpcService.WorkerStatusPB done = awaitFinished(prefill, 0, 1, 5_000);
        assertEquals(1, done.getFinishedTaskListCount(),
                "suppress_running must not touch finishedTaskList");
    }

    // ════════════════════════════════════════════════════════════════
    //  status_suppress_rids
    // ════════════════════════════════════════════════════════════════

    @Test
    void statusSuppressRidsHidesSpecificRequestsFromBothLists() throws Exception {
        MockPerformanceModel model = model("300");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "status_suppress_rids", "\"rids\":[7007]");
        enqueue(prefill, batch(2002, slot(0, input(7007, 10), input(7008, 10))));

        // Running snapshot: rid 7007 swallowed, rid 7008 reported.
        EngineRpcService.WorkerStatusPB running = workerStatus(prefill, 0);
        assertEquals(1, running.getRunningTaskInfoCount());
        assertEquals(7008, running.getRunningTaskInfo(0).getRequestId());
        assertEquals(2, running.getRunningQueryLen(), "runningQueryLen stays REAL");

        // Finished list: rid 7007 swallowed too (double swallow).
        EngineRpcService.WorkerStatusPB done = awaitFinished(prefill, 0, 1, 5_000);
        assertEquals(1, done.getFinishedTaskListCount());
        assertEquals(7008, done.getFinishedTaskList(0).getRequestId());
        assertEquals(2, done.getLatestFinishedVersion(),
                "latestFinishedVersion counts BOTH completions (7007 is only hidden)");
    }

    // ════════════════════════════════════════════════════════════════
    //  status_no_respond
    // ════════════════════════════════════════════════════════════════

    @Test
    void statusNoRespondHangsWorkerStatusRpc() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "status_no_respond", null);

        CountDownLatch latch = new CountDownLatch(1);
        prefill.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder().setLatestFinishedVersion(0).build(),
                new StreamObserver<>() {
                    @Override
                    public void onNext(EngineRpcService.WorkerStatusPB value) {
                        latch.countDown();
                    }

                    @Override
                    public void onError(Throwable throwable) {
                        latch.countDown();
                    }

                    @Override
                    public void onCompleted() {
                        latch.countDown();
                    }
                });
        assertFalse(latch.await(500, TimeUnit.MILLISECONDS),
                "getWorkerStatus must hang under status_no_respond");

        clearInject(basePort);
        EngineRpcService.WorkerStatusPB recovered = workerStatus(prefill, 0);
        assertNotNull(recovered, "getWorkerStatus recovers after clear");
    }

    // ════════════════════════════════════════════════════════════════
    //  status_fake_task (running + finished forms)
    // ════════════════════════════════════════════════════════════════

    @Test
    void statusFakeTaskRunningFormIsContinuouslyReported() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "status_fake_task", "\"rid\":9001,\"batch_id\":42,\"phase\":\"RUNNING\"");
        EngineRpcService.WorkerStatusPB first = workerStatus(prefill, 0);
        assertEquals(1, first.getRunningTaskInfoCount());
        EngineRpcService.TaskInfoPB fake = first.getRunningTaskInfo(0);
        assertEquals(9001, fake.getRequestId());
        assertEquals(42, fake.getBatchId());
        assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING, fake.getPhase());
        assertEquals(0, first.getRunningQueryLen(),
                "runningQueryLen stays REAL (fakes do not touch it)");

        // Second inject accumulates; KV_ALLOCATED maps to the enum phase.
        inject(basePort, "status_fake_task", "\"rid\":9002,\"phase\":\"KV_ALLOCATED\"");
        EngineRpcService.WorkerStatusPB second = workerStatus(prefill, 0);
        assertEquals(2, second.getRunningTaskInfoCount());
        assertEquals(9001, second.getRunningTaskInfo(0).getRequestId());
        assertEquals(9002, second.getRunningTaskInfo(1).getRequestId());
        assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED,
                second.getRunningTaskInfo(1).getPhase());
        assertTrue(second.getRunningTaskInfo(1).getIsWaiting(),
                "withLegacyTaskState marks non-RUNNING fakes as waiting");

        // Disabling clears the whole synthetic set.
        disable(basePort, "status_fake_task");
        EngineRpcService.WorkerStatusPB cleared = workerStatus(prefill, 0);
        assertEquals(0, cleared.getRunningTaskInfoCount());
    }

    @Test
    void statusFakeTaskFinishedFormReportsSyntheticCompletion() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "status_fake_task",
                "\"rid\":8801,\"batch_id\":7,\"phase\":\"finished\",\"error_code\":5");
        EngineRpcService.WorkerStatusPB first = workerStatus(prefill, 0);
        assertEquals(1, first.getFinishedTaskListCount());
        EngineRpcService.TaskInfoPB fake = first.getFinishedTaskList(0);
        assertEquals(8801, fake.getRequestId());
        assertEquals(7, fake.getBatchId());
        assertTrue(fake.hasErrorInfo(), "errorCode 5 must surface as error_info");
        assertEquals(5, fake.getErrorInfo().getErrorCode());
        assertEquals(0, first.getLatestFinishedVersion(),
                "fakes never touch the real cursor");

        // Continuous reporting: EVERY poll re-reports it, whatever the cursor.
        EngineRpcService.WorkerStatusPB again = workerStatus(prefill, 999);
        assertEquals(1, again.getFinishedTaskListCount(),
                "finished-form fake is re-reported on every poll");

        // A finished form without errorCode carries no error_info.
        inject(basePort, "status_fake_task", "\"rid\":8802,\"phase\":\"finished\"");
        EngineRpcService.WorkerStatusPB both = workerStatus(prefill, 999);
        assertEquals(2, both.getFinishedTaskListCount());
        assertFalse(both.getFinishedTaskList(1).hasErrorInfo());
        assertEquals(8802, both.getFinishedTaskList(1).getRequestId());
    }

    // ════════════════════════════════════════════════════════════════
    //  status_duplicate_finished
    // ════════════════════════════════════════════════════════════════

    @Test
    void statusDuplicateFinishedDoublesCompletionInOnePoll() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "status_duplicate_finished", null);
        enqueue(prefill, batch(3001, slot(0, input(31, 10))));

        EngineRpcService.WorkerStatusPB done = awaitFinished(prefill, 0, 2, 5_000);
        assertEquals(2, done.getFinishedTaskListCount(),
                "duplicate_finished must enqueue the completion twice");
        assertEquals(31, done.getFinishedTaskList(0).getRequestId());
        assertEquals(31, done.getFinishedTaskList(1).getRequestId());
        assertEquals(1, done.getLatestFinishedVersion(),
                "both copies share ONE version");

        // Advancing the cursor past that version consumes BOTH copies.
        EngineRpcService.WorkerStatusPB next = workerStatus(prefill, 1);
        assertEquals(0, next.getFinishedTaskListCount());
    }

    // ════════════════════════════════════════════════════════════════
    //  status_cursor_regress
    // ════════════════════════════════════════════════════════════════

    @Test
    void statusCursorRegressRewindsLatestFinishedVersion() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        enqueue(prefill, batch(4001, slot(0, input(41, 10))));
        EngineRpcService.WorkerStatusPB done = awaitFinished(prefill, 0, 1, 5_000);
        assertEquals(1, done.getLatestFinishedVersion());

        inject(basePort, "status_cursor_regress", "\"n\":1");
        EngineRpcService.WorkerStatusPB rewound = workerStatus(prefill, 1);
        assertEquals(0, rewound.getLatestFinishedVersion(),
                "cursor_regress must rewind latestFinishedVersion");
        assertEquals(0, rewound.getFinishedTaskListCount(),
                "the replayed interval is empty (already consumed)");
    }

    // ════════════════════════════════════════════════════════════════
    //  status_version_regress
    // ════════════════════════════════════════════════════════════════

    @Test
    void statusVersionRegressDecreasesStatusVersionPerPoll() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "status_version_regress", null);
        long v1 = workerStatus(prefill, 0).getStatusVersion();
        long v2 = workerStatus(prefill, 0).getStatusVersion();
        long v3 = workerStatus(prefill, 0).getStatusVersion();
        assertTrue(v2 < v1, "statusVersion must decrease per poll (" + v1 + " -> " + v2 + ")");
        assertTrue(v3 < v2, "statusVersion must decrease per poll (" + v2 + " -> " + v3 + ")");
        assertEquals(v1 - 2, v3, "exactly one step down per poll");
    }

    // ════════════════════════════════════════════════════════════════
    //  status_zombie_running (prefill + decode completion points)
    // ════════════════════════════════════════════════════════════════

    @Test
    void statusZombieRunningKeepsPrefillRequestRunningForever() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "status_zombie_running", null);
        enqueue(prefill, batch(5001, slot(0, input(51, 10))));

        // Let the (10ms) prefill finish internally.
        Thread.sleep(150);
        assertEquals(1, prefill.getCompletedCount(),
                "the request DID complete internally");

        // The report keeps it RUNNING forever and never publishes finished.
        for (int i = 0; i < 3; i++) {
            EngineRpcService.WorkerStatusPB status = workerStatus(prefill, 0);
            assertEquals(1, status.getRunningTaskInfoCount(),
                    "zombie must stay in runningTaskInfo (poll " + i + ")");
            assertEquals(51, status.getRunningTaskInfo(0).getRequestId());
            assertEquals(0, status.getFinishedTaskListCount(),
                    "zombie must never appear in finishedTaskList (poll " + i + ")");
            assertEquals(0, status.getLatestFinishedVersion(),
                    "no completion version is ever published (poll " + i + ")");
            Thread.sleep(50);
        }

        // Counters released normally: the zombie poisons only the report.
        assertEquals(0, prefill.getInflightCount(),
                "pendingRequests must be released (engine not wedged)");
        assertEquals(1, prefill.getRunningCount(),
                "the runningTasks entry itself stays (that IS the zombie)");
    }

    @Test
    void statusZombieRunningKeepsDecodeRequestRunningForever() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 0, 1);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);
        int decodePort = basePort;

        inject(decodePort, "status_zombie_running", null);
        fireAndForgetGenerate(decode, input(61, 10));

        // One decode step (1ms) finishes internally.
        Thread.sleep(150);
        for (int i = 0; i < 3; i++) {
            EngineRpcService.WorkerStatusPB status = workerStatus(decode, 0);
            assertEquals(1, status.getRunningTaskInfoCount(),
                    "decode zombie must stay in runningTaskInfo (poll " + i + ")");
            assertEquals(61, status.getRunningTaskInfo(0).getRequestId());
            assertEquals(0, status.getFinishedTaskListCount(),
                    "decode zombie must never appear in finishedTaskList (poll " + i + ")");
            assertEquals(0, status.getLatestFinishedVersion(),
                    "no decode completion version is published (poll " + i + ")");
            Thread.sleep(50);
        }
        assertEquals(0, decode.getInflightCount(),
                "decode slot/pending counters must be released");
        assertEquals(1, decode.getRunningCount());
    }

    // ════════════════════════════════════════════════════════════════
    //  enqueue_ack_partial_fail + enqueue_ack_error_code
    // ════════════════════════════════════════════════════════════════

    @Test
    void enqueueAckPartialFailMovesMembersToErrors() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "enqueue_ack_partial_fail", "\"k\":2");
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(7001, slot(0, input(1, 10), input(2, 10), input(3, 10))));
        assertEquals(1, ack.getSuccessesCount(), "k=2 of 3 members stay successes");
        assertEquals(3, ack.getSuccesses(0).getRequestId());
        assertEquals(2, ack.getErrorsCount(), "k=2 members are moved to errors");
        assertEquals(1, ack.getErrors(0).getRequestId());
        assertEquals(2, ack.getErrors(1).getRequestId());
        assertEquals(13L, ack.getErrors(0).getErrorInfo().getErrorCode(),
                "default error code is 13");
        assertEquals(13L, ack.getErrors(1).getErrorInfo().getErrorCode());

        // Pre-admission split (production-calibrated rejection): rejected
        // members never enter the engine, so only the 1 admitted member
        // completes engine-side — no ghost completions can desync the
        // master's decode bookkeeping (run-1788360948 forensics).
        EngineRpcService.WorkerStatusPB done = awaitFinished(prefill, 0, 1, 5_000);
        assertEquals(1, done.getFinishedTaskListCount(),
                "only the admitted member completes; rejected members never ran");

        // Custom error code replaces 13 per-request.
        inject(basePort, "enqueue_ack_error_code", "\"code\":77");
        EngineRpcService.EnqueueBatchResponsePB ack2 =
                enqueue(prefill, batch(7002, slot(0, input(4, 10), input(5, 10), input(6, 10))));
        assertEquals(1, ack2.getSuccessesCount());
        assertEquals(2, ack2.getErrorsCount());
        assertEquals(77L, ack2.getErrors(0).getErrorInfo().getErrorCode());
        assertEquals(77L, ack2.getErrors(1).getErrorInfo().getErrorCode());
        EngineRpcService.WorkerStatusPB done2 = awaitFinished(prefill, 1, 1, 5_000);
        assertEquals(1, done2.getFinishedTaskListCount(),
                "only the admitted member completes under the injected code");
    }

    // ════════════════════════════════════════════════════════════════
    //  enqueue_ack_drop
    // ════════════════════════════════════════════════════════════════

    @Test
    void enqueueAckDropReturnsEmptyAckButProcessesBatch() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "enqueue_ack_drop", null);
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(8001, slot(0, input(81, 10), input(82, 10))));
        assertEquals(8001, ack.getBatchId(), "batchId is still echoed");
        assertEquals(0, ack.getSuccessesCount(), "empty ack: no successes");
        assertEquals(0, ack.getErrorsCount(), "empty ack: no errors");

        // The engine processed the batch normally (unlike crash_after, it is
        // not stopped): completions still surface through getWorkerStatus.
        EngineRpcService.WorkerStatusPB done = awaitFinished(prefill, 0, 2, 5_000);
        assertEquals(2, done.getFinishedTaskListCount(),
                "the silently-acked batch still executes engine-side");

        // Subsequent RPCs behave normally after clearing.
        clearInject(basePort);
        EngineRpcService.EnqueueBatchResponsePB ack2 =
                enqueue(prefill, batch(8002, slot(0, input(83, 10))));
        assertEquals(1, ack2.getSuccessesCount());
        assertEquals(0, ack2.getErrorsCount());
        assertFalse(prefill.isStopped(), "enqueue_ack_drop must never set stopped");
    }

    // ════════════════════════════════════════════════════════════════
    //  prefill_async_partial_fail (execution-phase partial failure)
    // ════════════════════════════════════════════════════════════════

    @Test
    void prefillAsyncPartialFailFailsFirstKAtCompletionCallback() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "prefill_async_partial_fail", "\"k\":1,\"code\":8500");
        // The ack is HONEST for every member (this is the execution-phase
        // counterpart of enqueue_ack_partial_fail): all members acknowledged.
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(7101, slot(0, input(101, 10), input(102, 10), input(103, 10))));
        assertEquals(3, ack.getSuccessesCount(),
                "execution-phase injection must not touch the ack");
        assertEquals(0, ack.getErrorsCount());

        // All 3 members surface in finished_task_list: the failed one as a
        // TYPED terminal (error_code 8500), the survivors as normal terminals
        // — same batch_id throughout (stream group_id → TaskInfo).
        EngineRpcService.WorkerStatusPB done = awaitFinished(prefill, 0, 3, 5_000);
        assertEquals(3, done.getFinishedTaskListCount());
        assertEquals(3, done.getLatestFinishedVersion());
        EngineRpcService.TaskInfoPB failed = findByRid(done, 101);
        assertNotNull(failed, "the batch's first member must be reported finished");
        assertTrue(failed.hasErrorInfo(), "the injected member carries error_info");
        assertEquals(8500L, failed.getErrorInfo().getErrorCode());
        assertEquals("injected prefill_async_partial_fail",
                failed.getErrorInfo().getErrorMessage());
        assertEquals(7101, failed.getBatchId(),
                "the typed failure terminal keeps the batch_id (master reconcile)");
        assertFalse(findByRid(done, 102).hasErrorInfo(), "survivor 102 completes normally");
        assertFalse(findByRid(done, 103).hasErrorInfo(), "survivor 103 completes normally");

        // Prefill-only cluster: survivors take the completed branch, the
        // failed member does NOT count as completed.
        assertEquals(2, prefill.getCompletedCount(),
                "the failed member is not a completed request");

        // k=2 with a custom code: the first TWO members of a fresh batch fail.
        inject(basePort, "prefill_async_partial_fail", "\"k\":2,\"code\":8431");
        EngineRpcService.EnqueueBatchResponsePB ack2 =
                enqueue(prefill, batch(7102, slot(0, input(104, 10), input(105, 10), input(106, 10))));
        assertEquals(3, ack2.getSuccessesCount());
        EngineRpcService.WorkerStatusPB done2 = awaitFinished(prefill, 3, 3, 5_000);
        assertTrue(findByRid(done2, 104).hasErrorInfo());
        assertEquals(8431L, findByRid(done2, 104).getErrorInfo().getErrorCode());
        assertTrue(findByRid(done2, 105).hasErrorInfo());
        assertFalse(findByRid(done2, 106).hasErrorInfo(),
                "k=2: only the first two members fail");
        assertEquals(3, prefill.getCompletedCount(), "2 + 1 survivors are completed");

        // Cumulative counter: 1 + 2 failed members exported via /snapshot.
        assertEquals(3L, snapshotCounter("prefill", "prefill_async_partial_fails"),
                "/snapshot carries prefill_async_partial_fails");
    }

    @Test
    void prefillAsyncPartialFailMemberSkipsDecodeAndReleasesReservations() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);
        int decodePort = basePort + 1;

        inject(basePort, "prefill_async_partial_fail", "\"k\":1,\"code\":8500");
        EngineRpcService.EnqueueBatchResponsePB ack = enqueue(prefill, batch(7201, slot(0,
                inputWithDecode(201, 10, decodePort),
                inputWithDecode(202, 10, decodePort),
                inputWithDecode(203, 10, decodePort))));
        assertEquals(3, ack.getSuccessesCount());

        // Prefill terminals: 1 typed failure (201) + 2 normal (202/203).
        EngineRpcService.WorkerStatusPB done = awaitFinished(prefill, 0, 3, 5_000);
        assertTrue(findByRid(done, 201).hasErrorInfo());
        assertEquals(8500L, findByRid(done, 201).getErrorInfo().getErrorCode());
        assertFalse(findByRid(done, 202).hasErrorInfo());
        assertFalse(findByRid(done, 203).hasErrorInfo());

        // Decode engine: ONLY the survivors hand off — the failed member
        // never starts decode (no ghost decode stream, no D-side lease held).
        EngineRpcService.WorkerStatusPB decodeDone = awaitFinished(decode, 0, 2, 5_000);
        assertEquals(2, decodeDone.getFinishedTaskListCount(),
                "exactly the two survivors complete decode");
        List<Long> decodeRids = new ArrayList<>();
        for (EngineRpcService.TaskInfoPB task : decodeDone.getFinishedTaskListList()) {
            decodeRids.add(task.getRequestId());
        }
        assertTrue(decodeRids.contains(202L) && decodeRids.contains(203L),
                "decode completions are the survivors: " + decodeRids);
        assertFalse(decodeRids.contains(201L),
                "the failed member never reaches decode: " + decodeRids);

        // No leaked slots on either engine (the failed member's P-side lease
        // AND D-side reservation both returned; survivors finished decode).
        Thread.sleep(150);
        assertEquals(0, prefill.getInflightCount(), "prefill slots quiesce");
        assertEquals(0, prefill.getRunningCount(), "no zombie running entry");
        assertEquals(0, decode.getInflightCount(), "decode slots quiesce");
        assertEquals(1L, snapshotCounter("prefill", "prefill_async_partial_fails"));
    }

    @Test
    void prefillAsyncPartialFailOffByDefaultAndDisarmsCleanly() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        // Default (never injected): zero impact.
        EngineRpcService.EnqueueBatchResponsePB cleanAck =
                enqueue(prefill, batch(7301, slot(0, input(301, 10), input(302, 10))));
        assertEquals(2, cleanAck.getSuccessesCount());
        EngineRpcService.WorkerStatusPB cleanDone = awaitFinished(prefill, 0, 2, 5_000);
        assertFalse(findByRid(cleanDone, 301).hasErrorInfo());
        assertFalse(findByRid(cleanDone, 302).hasErrorInfo());
        assertEquals(2, prefill.getCompletedCount());

        // Explicit k=0 is equally inert.
        inject(basePort, "prefill_async_partial_fail", "\"k\":0");
        enqueue(prefill, batch(7302, slot(0, input(303, 10))));
        EngineRpcService.WorkerStatusPB k0Done = awaitFinished(prefill, 2, 1, 5_000);
        assertFalse(findByRid(k0Done, 303).hasErrorInfo(), "k=0 must inject nothing");
        assertEquals(3, prefill.getCompletedCount());

        // Armed (k=1, custom code) → exactly one typed failure.
        inject(basePort, "prefill_async_partial_fail", "\"k\":1,\"code\":9001");
        enqueue(prefill, batch(7303, slot(0, input(304, 10), input(305, 10))));
        EngineRpcService.WorkerStatusPB armedDone = awaitFinished(prefill, 3, 2, 5_000);
        assertTrue(findByRid(armedDone, 304).hasErrorInfo());
        assertEquals(9001L, findByRid(armedDone, 304).getErrorInfo().getErrorCode());
        assertFalse(findByRid(armedDone, 305).hasErrorInfo());

        // Disarmed (enabled=false) → the next batch is fully clean.
        disable(basePort, "prefill_async_partial_fail");
        enqueue(prefill, batch(7304, slot(0, input(306, 10), input(307, 10))));
        EngineRpcService.WorkerStatusPB disarmedDone = awaitFinished(prefill, 5, 2, 5_000);
        assertFalse(findByRid(disarmedDone, 306).hasErrorInfo(),
                "disabled injection must not fail members");
        assertFalse(findByRid(disarmedDone, 307).hasErrorInfo());
        assertEquals(6, prefill.getCompletedCount(), "2+1+1+2 completed requests");
    }

    @Test
    void prefillAsyncPartialFailTerminatesFailedMemberStreamWithTypedErrorFrame() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 0);
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        inject(basePort, "prefill_async_partial_fail", "\"k\":1,\"code\":8500");

        // Open BOTH client-side FetchResponse streams BEFORE enqueuing (the
        // real client's call order: the stream is issued right after the
        // schedule ack, well before the 10 ms prefill executes; the queue is
        // shared with the EnqueueBatch Phase-1 admission either way).
        StreamCollector failedStream = new StreamCollector();
        StreamCollector survivorStream = new StreamCollector();
        prefill.fetchResponse(EngineRpcService.FetchRequestPB.newBuilder()
                .setRequestId(401).build(), failedStream.observer());
        prefill.fetchResponse(EngineRpcService.FetchRequestPB.newBuilder()
                .setRequestId(402).build(), survivorStream.observer());

        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(7401, slot(0, input(401, 10), input(402, 10))));
        assertEquals(2, ack.getSuccessesCount());

        failedStream.await(10_000);
        survivorStream.await(10_000);

        // Failed member: exactly ONE frame — the in-band typed error terminal
        // (production stream->reportError's client-visible half). The code
        // rides the RAW enum value (8500 sits outside ErrorCodePB; proto3 open
        // enums keep it on the wire) and the message text carries the numeric
        // too; the stream COMPLETES right after the frame (the pump treats
        // hasErrorInfo frames as terminal), it does not error.
        assertNull(failedStream.error.get(),
                "the error frame must complete the stream, not error it");
        assertEquals(1, failedStream.frames.size(),
                "the failed member delivers exactly the error frame");
        EngineRpcService.GenerateOutputsPB frame = failedStream.frames.get(0);
        assertEquals(401L, frame.getRequestId());
        assertTrue(frame.hasErrorInfo(), "the terminal frame carries error_info");
        assertEquals(8500, frame.getErrorInfo().getErrorCodeValue(),
                "the injected code rides the raw enum value");
        assertTrue(frame.getErrorInfo().getErrorMessage().contains("8500"),
                "the message text carries the numeric code as well");
        assertTrue(frame.getErrorInfo().getErrorMessage()
                        .contains("injected prefill_async_partial_fail"),
                "the message identifies the injection");

        // Survivor of the SAME batch: the normal single terminal frame
        // (prefill-only cluster), no error_info — members settle
        // independently, no batch-level failure propagation.
        assertNull(survivorStream.error.get());
        assertEquals(1, survivorStream.frames.size());
        assertFalse(survivorStream.frames.get(0).hasErrorInfo());
        assertEquals(1, survivorStream.frames.get(0).getFlattenOutput().getFinishedCount());
        assertTrue(survivorStream.frames.get(0).getFlattenOutput().getFinished(0),
                "the survivor's frame is the normal terminal frame");
    }

    /** Async FetchResponse / GenerateStreamCall collector (MultiFrameStreamTtft
     *  style, adapted so several streams can be opened before the batch runs). */
    private static final class StreamCollector {
        final List<EngineRpcService.GenerateOutputsPB> frames = new CopyOnWriteArrayList<>();
        final AtomicReference<Throwable> error = new AtomicReference<>();
        final CountDownLatch terminal = new CountDownLatch(1);

        StreamObserver<EngineRpcService.GenerateOutputsPB> observer() {
            return new StreamObserver<>() {
                @Override
                public void onNext(EngineRpcService.GenerateOutputsPB value) {
                    frames.add(value);
                }

                @Override
                public void onError(Throwable t) {
                    error.set(t);
                    terminal.countDown();
                }

                @Override
                public void onCompleted() {
                    terminal.countDown();
                }
            };
        }

        void await(long ms) throws InterruptedException {
            assertTrue(terminal.await(ms, TimeUnit.MILLISECONDS),
                    "stream must terminate (completed or error) within " + ms + "ms");
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Cluster scaffolding (direct service calls, MockControlServer on :0)
    // ════════════════════════════════════════════════════════════════

    private int startCluster(MockPerformanceModel model, int nPrefill, int nDecode)
            throws IOException {
        int basePort = PORT_ALLOCATOR.getAndAdd(nPrefill + nDecode + 10);
        scheduler = Executors.newScheduledThreadPool(8, r -> {
            Thread t = new Thread(r, "status-fault-test-scheduler");
            t.setDaemon(true);
            return t;
        });
        services = new ConcurrentHashMap<>();
        prefillServices = new ArrayList<>();
        decodeServices = new ArrayList<>();

        for (int i = 0; i < nPrefill; i++) {
            int port = basePort + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, service);
            prefillServices.add(service);
        }
        for (int i = 0; i < nDecode; i++) {
            int port = basePort + nPrefill + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, service);
            decodeServices.add(service);
        }

        controlServer = new MockControlServer(services, new ConcurrentHashMap<>(), null, null,
                "127.0.0.1", 0);
        controlServer.start();
        return basePort;
    }

    private void inject(int enginePort, String type, String extra) throws Exception {
        httpPost(controlServer.getPort(), "/inject",
                "{\"port\":" + enginePort + ",\"type\":\"" + type + "\",\"enabled\":true"
                        + (extra == null ? "" : "," + extra) + "}");
    }

    private void disable(int enginePort, String type) throws Exception {
        httpPost(controlServer.getPort(), "/inject",
                "{\"port\":" + enginePort + ",\"type\":\"" + type + "\",\"enabled\":false}");
    }

    private void clearInject(int enginePort) throws Exception {
        httpPost(controlServer.getPort(), "/clear_inject",
                "{\"port\":" + enginePort + "}");
    }

    /** Poll until the finished list reaches {@code expectedCount} (or timeout). */
    private static EngineRpcService.WorkerStatusPB awaitFinished(
            JavaMockEngineCluster.FastRpcService service,
            long sinceVersion,
            int expectedCount,
            long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        EngineRpcService.WorkerStatusPB last = workerStatus(service, sinceVersion);
        while (last.getFinishedTaskListCount() < expectedCount
                && System.nanoTime() < deadline) {
            Thread.sleep(10);
            last = workerStatus(service, sinceVersion);
        }
        return last;
    }

    /** First finished-task entry matching {@code rid} (null if absent). */
    private static EngineRpcService.TaskInfoPB findByRid(
            EngineRpcService.WorkerStatusPB status, long rid) {
        for (EngineRpcService.TaskInfoPB task : status.getFinishedTaskListList()) {
            if (task.getRequestId() == rid) {
                return task;
            }
        }
        return null;
    }

    /** Read a per-engine long field from GET /snapshot (counter assertions).
     *  Matches by snapshot "role": the test constructor derives engineName
     *  as "&lt;role&gt;-&lt;port&gt;", so a name match would never hit. */
    private long snapshotCounter(String role, String field) throws Exception {
        JsonNode snapshot = new ObjectMapper()
                .readTree(httpGet(controlServer.getPort(), "/snapshot"));
        for (JsonNode engine : snapshot.path("engines")) {
            if (role.equals(engine.path("role").asText())) {
                return engine.path(field).asLong(-1);
            }
        }
        return -1L;
    }

    /** Drive generateStreamCall without blocking on its stream (decode zombie test). */
    private static void fireAndForgetGenerate(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.GenerateInputPB request) {
        service.generateStreamCall(request, new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.GenerateOutputsPB value) {
            }

            @Override
            public void onError(Throwable throwable) {
            }

            @Override
            public void onCompleted() {
            }
        });
    }

    private MockPerformanceModel model(String formula) throws Exception {
        return MockEngineTestSupport.performanceModel(tempDir, formula);
    }
}

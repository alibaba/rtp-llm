package org.flexlb.mockengine;

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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.httpPost;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the status-report fault-injection family (12 new types):
 * 9 getWorkerStatus output-layer faults (suppress_finished / suppress_running /
 * suppress_rids / no_respond / fake_task / duplicate_finished / cursor_regress /
 * version_regress / zombie_running) and 3 EnqueueBatch ack faults
 * (enqueue_ack_partial_fail / enqueue_ack_error_code / enqueue_ack_drop).
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

        // Master advances to the real cursor → the head-trim consumes the
        // never-delivered completion. After clearing, it is gone forever.
        workerStatus(prefill, 2);
        clearInject(basePort);
        EngineRpcService.WorkerStatusPB afterClear = workerStatus(prefill, 1);
        assertEquals(0, afterClear.getFinishedTaskListCount(),
                "suppressed completion must be permanently lost after the cursor advanced");
        assertEquals(2, afterClear.getLatestFinishedVersion());
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

        // The engine still EXECUTED all three (the ack lies).
        EngineRpcService.WorkerStatusPB done = awaitFinished(prefill, 0, 3, 5_000);
        assertEquals(3, done.getFinishedTaskListCount(),
                "all members complete engine-side despite the lying ack");

        // Custom error code replaces 13 per-request.
        inject(basePort, "enqueue_ack_error_code", "\"code\":77");
        EngineRpcService.EnqueueBatchResponsePB ack2 =
                enqueue(prefill, batch(7002, slot(0, input(4, 10), input(5, 10), input(6, 10))));
        assertEquals(1, ack2.getSuccessesCount());
        assertEquals(2, ack2.getErrorsCount());
        assertEquals(77L, ack2.getErrors(0).getErrorInfo().getErrorCode());
        assertEquals(77L, ack2.getErrors(1).getErrorInfo().getErrorCode());
        EngineRpcService.WorkerStatusPB done2 = awaitFinished(prefill, 3, 3, 5_000);
        assertEquals(3, done2.getFinishedTaskListCount());
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

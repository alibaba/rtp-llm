package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.balance.scheduler.priority.AdmissionLease;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Ack-only release gate (default-on) ordering tests: the frontend-facing
 * fetch release must complete only on the Prefill EnqueueBatch ACK semantic
 * — a direct/late ACK or a Prefill WorkerStatus observation of the exact
 * dispatch generation. Decode observations record ownership but never
 * release. The gate-off (legacy shortcut) semantics are pinned by
 * {@link DecodeAcceptanceLinearizationTest}.
 */
class AckOnlyReleaseGateTest {

    private static final long REQUEST_ID = 902L;
    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";
    private static final String DECODE_IP_PORT = "10.0.0.2:8081";

    private FlexlbConfig config;
    private FlexlbBatchScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private DecodeEndpoint decodeEndpoint;
    private PrefillEndpoint prefillEndpoint;
    private BatchItem item;
    private EngineCancelChannel cancelChannel;
    private long batchId;

    @BeforeEach
    void setUp() {
        ConfigService configService = mock(ConfigService.class);
        Router router = mock(Router.class);
        BatchDispatcher dispatcher = mock(BatchDispatcher.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        config = new FlexlbConfig();
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchFixedWaitMs(3_600_000);
        config.setFlexlbBatchSizeMax(100);
        config.setFlexlbAckOnlyReleaseEnabled(true);
        when(configService.loadBalanceConfig()).thenReturn(config);

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        cancelChannel = mock(EngineCancelChannel.class);
        // A pending cancel future keeps the reconciliation fence armed without
        // spinning the retry chain during the test.
        when(cancelChannel.cancel(any(), anyLong(), anyLong()))
                .thenReturn(new CompletableFuture<>());
        scheduler = new FlexlbBatchScheduler(configService, router, endpointRegistry,
                dispatcher, reporter, null, null, cancelChannel);

        WorkerStatus prefillStatus = new WorkerStatus();
        prefillStatus.setIp("10.0.0.1");
        prefillStatus.setPort(8080);
        prefillStatus.setGrpcPort(8081);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, prefillStatus);
        prefillEndpoint = endpointRegistry.getPrefill(PREFILL_IP_PORT);

        WorkerStatus decodeStatus = new WorkerStatus();
        decodeStatus.setIp("10.0.0.2");
        decodeStatus.setPort(8081);
        decodeStatus.setGrpcPort(8082);
        decodeStatus.setAvailableKvCacheTokens(new AtomicLong(1_000_000));
        decodeStatus.setTotalKvCacheTokens(new AtomicLong(2_000_000));
        endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeStatus);
        decodeEndpoint = endpointRegistry.getDecode(DECODE_IP_PORT);

        item = item();
        assertTrue(scheduler.registerInflight(item));
        AdmissionLease lease = new AdmissionLease(item, decodeEndpoint,
                prefillEndpoint.getBatcher().queueManager(), scheduler, 0, null);
        assertTrue(scheduler.attachAdmissionLease(item, lease));
        lease.bindTo(item.future());
        decodeEndpoint.reserve(REQUEST_ID, 128, 136, 50,
                System.currentTimeMillis() + 30_000);
        decodeEndpoint.markQueuedPhase(REQUEST_ID);
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        RequestLifecycleSnapshot dispatched = scheduler.getRequestState(REQUEST_ID, 0);
        assertEquals(RequestLifecycleState.DISPATCHING, dispatched.state());
        batchId = dispatched.batchId();
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    /** Timeline 1: uncertain dispatch, then the late ACK arrives → release. */
    @Test
    void lateAckDuringReconciliationReleasesFetch() {
        scheduler.onDispatchUncertain(item, batchId, new TimeoutException("lost Enqueue ACK"));
        assertFalse(item.future().isDone());

        scheduler.onSuccess(item, batchId);

        assertReleasedByPrefillEvidence();
    }

    /** Timeline 2: ACK never fires; the Prefill running observation releases. */
    @Test
    void prefillRunningObservationReleasesUncertainDispatch() {
        scheduler.onDispatchUncertain(item, batchId, new TimeoutException("lost Enqueue ACK"));
        assertFalse(item.future().isDone());

        reportPrefillRunning(batchId);

        assertReleasedByPrefillEvidence();
    }

    /** Timeline 2b: running window skipped by polling; finished(success) releases. */
    @Test
    void prefillFinishedSuccessReleasesUncertainDispatch() {
        scheduler.onDispatchUncertain(item, batchId, new TimeoutException("lost Enqueue ACK"));
        assertFalse(item.future().isDone());

        reportPrefillFinished(batchId, 0);

        assertReleasedByPrefillEvidence();
    }

    /** Timeline 3: Decode arrives first — it must not release; Prefill does. */
    @Test
    void decodeAcceptanceAloneDoesNotRelease() {
        reportDecode(TaskPhase.KV_ALLOCATED);
        scheduler.onDispatchUncertain(item, batchId, new TimeoutException("lost Enqueue ACK"));
        assertFalse(item.future().isDone());

        reportDecode(TaskPhase.RUNNING);
        assertFalse(item.future().isDone());

        reportPrefillRunning(batchId);

        assertReleasedByPrefillEvidence();
    }

    /** A stale dispatch generation must not release the current entry. */
    @Test
    void prefillObservationWithMismatchedBatchIdDoesNotRelease() {
        scheduler.onDispatchUncertain(item, batchId, new TimeoutException("lost Enqueue ACK"));

        reportPrefillRunning(batchId + 1);
        assertFalse(item.future().isDone());

        reportPrefillFinished(batchId + 1, 0);
        assertFalse(item.future().isDone());
    }

    /** PAIR variant: an explicit engine rejection after Decode acceptance
     *  must surface a typed failure instead of a false schedule success. */
    @Test
    void engineRejectionWithDecodeOwnershipFailsInsteadOfFalseSuccess() {
        reportDecode(TaskPhase.KV_ALLOCATED);

        scheduler.onFailure(item, new RuntimeException("engine rejected EnqueueBatch"));

        assertTrue(item.future().isDone());
        assertFalse(item.future().join().isSuccess());
        assertFalse(decodeEndpoint.reservedView().containsKey(REQUEST_ID));
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(REQUEST_ID, 0).state());
    }

    /** Gate off restores the legacy Decode shortcut for uncertain dispatch. */
    @Test
    void gateOffRestoresDecodeShortcutRelease() {
        config.setFlexlbAckOnlyReleaseEnabled(false);
        reportDecode(TaskPhase.KV_ALLOCATED);

        scheduler.onDispatchUncertain(item, batchId, new TimeoutException("lost Enqueue ACK"));

        Response response = item.future().join();
        assertTrue(response.isSuccess(), response.getErrorMessage());
        assertTrue(response.isEnqueuedByMaster());
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(REQUEST_ID, batchId).state());
    }

    private void assertReleasedByPrefillEvidence() {
        Response response = item.future().join();
        assertTrue(response.isSuccess(), response.getErrorMessage());
        assertTrue(response.isEnqueuedByMaster());
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(REQUEST_ID, batchId).state());
    }

    private void reportPrefillRunning(long taskBatchId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(REQUEST_ID);
        task.setBatchId(taskBatchId);
        task.setPhase(TaskPhase.RUNNING);
        task.setInputLength(128);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setRunningTaskInfo(Map.of(String.valueOf(REQUEST_ID), task));
        scheduler.onWorkerStatusUpdate(response);
    }

    private void reportPrefillFinished(long taskBatchId, long errorCode) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(REQUEST_ID);
        task.setBatchId(taskBatchId);
        task.setPhase(TaskPhase.RUNNING);
        task.setErrorCode(errorCode);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(Map.of(String.valueOf(REQUEST_ID), task));
        scheduler.onWorkerStatusUpdate(response);
    }

    private void reportDecode(TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(REQUEST_ID);
        task.setPhase(phase);
        task.setInputLength(128);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setRunningTaskInfo(Map.of(String.valueOf(REQUEST_ID), task));
        endpointRegistry.getDecode(DECODE_IP_PORT)
                .onWorkerStatusUpdate(new WorkerStatus(), response);
        scheduler.onWorkerStatusUpdate(response);
    }

    private BatchItem item() {
        Request request = new Request();
        request.setRequestId(REQUEST_ID);
        request.setPriority(50);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setBudget(ScheduleBudget.forDeadline(
                50, System.currentTimeMillis(), System.currentTimeMillis() + 30_000));

        ServerStatus prefill = server(RoleType.PREFILL, "10.0.0.1", 8080, 8081);
        ServerStatus decode = server(RoleType.DECODE, "10.0.0.2", 8081, 8082);
        Response route = new Response();
        route.setSuccess(true);
        route.setServerStatus(List.of(prefill, decode));
        return new BatchItem(context, new CompletableFuture<>(), route,
                prefill, decode, prefillEndpoint, decodeEndpoint,
                System.currentTimeMillis());
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setRequestId(REQUEST_ID);
        return status;
    }
}

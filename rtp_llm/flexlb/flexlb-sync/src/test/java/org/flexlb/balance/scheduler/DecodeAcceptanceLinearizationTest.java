package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.priority.AdmissionLease;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentMatchers;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Deterministic ordering tests for Decode ownership versus Enqueue outcome. */
class DecodeAcceptanceLinearizationTest {

    private static final long REQUEST_ID = 901L;
    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";
    private static final String DECODE_IP_PORT = "10.0.0.2:8081";

    private PriorityScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private DecodeEndpoint decodeEndpoint;
    private PrefillEndpoint prefillEndpoint;
    private BatchItem item;
    private AdmissionLease lease;
    private EngineCancelChannel cancelChannel;
    private long batchId;

    @BeforeEach
    void setUp() {
        ConfigService configService = mock(ConfigService.class);
        Router router = mock(Router.class);
        BatchDispatcher dispatcher = mock(BatchDispatcher.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(3_600_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(100);
        when(configService.loadBalanceConfig()).thenReturn(config);

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        cancelChannel = mock(EngineCancelChannel.class);
        scheduler = new PriorityScheduler(configService, router, endpointRegistry,
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

        item = item(config);
        assertTrue(scheduler.registerInflight(item));
        lease = new AdmissionLease(item, decodeEndpoint,
                prefillEndpoint.getBatcher().queueManager(), scheduler,
                0, null, null);
        assertTrue(scheduler.attachAdmissionLease(item, lease));
        lease.bindTo(item.future());
        decodeEndpoint.reserve(String.valueOf(REQUEST_ID), 128, 136, 50);
        decodeEndpoint.markQueuedPhase(String.valueOf(REQUEST_ID));
        scheduler.onDecisionGroupReady(List.of(item), new DecisionGroupMetadata("test", 0));
        RequestLifecycleSnapshot dispatched = scheduler.getRequestState(String.valueOf(REQUEST_ID), 0);
        assertEquals(RequestLifecycleState.DISPATCHING, dispatched.state());
        batchId = dispatched.batchId();
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void decodeAcceptanceBeforeFailedAckRetainsEngineOwnedResources() {
        reportDecode(TaskPhase.KV_ALLOCATED);

        scheduler.onFailure(item, new RuntimeException("failed Enqueue ACK"));

        assertEngineOwnedScheduleSucceeded();
    }

    @Test
    void decodeAcceptanceBeforeAckTimeoutRetainsEngineOwnedResources() {
        reportDecode(TaskPhase.RUNNING);

        scheduler.onTimeout(item, new TimeoutException("Enqueue ACK timeout"));

        assertEngineOwnedScheduleSucceeded();
    }

    @Test
    void decodeAcceptanceBeforeUncertainAckSkipsCancelAndCompletesSuccess() {
        reportDecode(TaskPhase.KV_ALLOCATED);

        scheduler.onUncertain(item, new TimeoutException("lost Enqueue ACK"));

        assertEngineOwnedScheduleSucceeded();
        verify(cancelChannel, never()).cancel(any(), ArgumentMatchers.anyString(), anyLong());
    }

    @Test
    void decodeAcceptanceAfterCancelInvocationRetainsFenceUntilTombstoned()
            throws Exception {
        CompletableFuture<EngineCancelChannel.CancelOutcome> cancelResult =
                new CompletableFuture<>();
        when(cancelChannel.cancel(any(), ArgumentMatchers.anyString(), anyLong())).thenReturn(cancelResult);

        scheduler.onUncertain(item, new TimeoutException("lost Enqueue ACK"));
        verify(cancelChannel, times(1)).cancel(any(), ArgumentMatchers.anyString(), anyLong());
        assertFalse(item.future().isDone());

        reportDecode(TaskPhase.KV_ALLOCATED);

        // The active Decode sample raced after the Cancel invocation boundary.
        // It proves that Decode owned the request at the sample instant, but it
        // cannot prove that Prefill did not already install a cancel intent.
        // Keep both the public result and accounting behind the Engine fence.
        assertFalse(item.future().isDone());
        assertTrue(decodeEndpoint.isConfirmedTracked(String.valueOf(REQUEST_ID)));

        cancelResult.complete(EngineCancelChannel.CancelOutcome.tombstoned());

        Response terminal = item.future().get(5, TimeUnit.SECONDS);
        assertFalse(terminal.isSuccess());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(String.valueOf(REQUEST_ID), batchId).state());
        verify(cancelChannel, times(1)).cancel(any(), ArgumentMatchers.anyString(), anyLong());
    }

    @Test
    void failedAckBeforeDecodeAcceptanceWinsAndLateStatusCannotRevive() {
        scheduler.onFailure(item, new RuntimeException("failed Enqueue ACK"));

        assertFalse(awaitTerminal().isSuccess());
        assertFalse(decodeEndpoint.reservedView().containsKey(String.valueOf(REQUEST_ID)));
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(String.valueOf(REQUEST_ID), 0).state());

        reportDecode(TaskPhase.KV_ALLOCATED);

        assertFalse(awaitTerminal().isSuccess());
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(String.valueOf(REQUEST_ID), 0).state());
    }

    @Test
    void ackTimeoutBeforeDecodeAcceptanceWinsAndLateStatusCannotRevive() {
        scheduler.onTimeout(item, new TimeoutException("Enqueue ACK timeout"));

        assertFalse(awaitTerminal().isSuccess());
        assertFalse(decodeEndpoint.reservedView().containsKey(String.valueOf(REQUEST_ID)));
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(String.valueOf(REQUEST_ID), 0).state());

        reportDecode(TaskPhase.RUNNING);

        assertFalse(awaitTerminal().isSuccess());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(String.valueOf(REQUEST_ID), 0).state());
    }

    @Test
    void decodeAcceptanceBeforeAdmissionDeadlineRetainsEngineOwnership() throws Exception {
        reportDecode(TaskPhase.KV_ALLOCATED);

        Method deadline = PriorityScheduler.class.getDeclaredMethod(
                "onRequestExpired", String.class, CompletableFuture.class);
        deadline.setAccessible(true);
        deadline.invoke(scheduler, String.valueOf(REQUEST_ID), item.future());

        assertEngineOwnedScheduleSucceeded();
    }

    @Test
    void nonDecodeRunningPhaseDoesNotTransferOwnership() {
        reportRunning(RoleType.FRONTEND, TaskPhase.RUNNING);

        scheduler.onFailure(item, new RuntimeException("failed Enqueue ACK"));

        assertFalse(awaitTerminal().isSuccess());
        assertFalse(decodeEndpoint.reservedView().containsKey(String.valueOf(REQUEST_ID)));
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(String.valueOf(REQUEST_ID), 0).state());
    }

    @Test
    void decodeFinishedErrorBeforeFailedAckCannotPublishScheduleSuccess() {
        reportDecodeFinishedError();

        scheduler.onFailure(item, new RuntimeException("late failed Enqueue ACK"));

        assertDecodeWorkerFailure();
    }

    @Test
    void failedAckBeforeDecodeFinishedErrorCannotPublishScheduleSuccess() {
        scheduler.onFailure(item, new RuntimeException("failed Enqueue ACK"));

        reportDecodeFinishedError();

        Response terminal = awaitTerminal();
        assertFalse(terminal.isSuccess());
        assertFalse(decodeEndpoint.reservedView().containsKey(String.valueOf(REQUEST_ID)));
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(String.valueOf(REQUEST_ID), 0).state());
    }

    @Test
    void racingDecodeFinishedErrorAndFailedAckCannotPublishScheduleSuccess()
            throws Exception {
        WorkerStatusResponse finished = decodeFinishedError();
        endpointRegistry.getDecode(DECODE_IP_PORT)
                .onWorkerStatusUpdate(new WorkerStatus(), finished);
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<?> worker = executor.submit(() -> {
                await(start);
                scheduler.onWorkerStatusUpdate(finished);
            });
            Future<?> ack = executor.submit(() -> {
                await(start);
                scheduler.onFailure(item,
                        new RuntimeException("racing failed Enqueue ACK"));
            });

            start.countDown();
            worker.get(5, TimeUnit.SECONDS);
            ack.get(5, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }

        assertFalse(awaitTerminal().isSuccess());
        assertFalse(decodeEndpoint.reservedView().containsKey(String.valueOf(REQUEST_ID)));
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(String.valueOf(REQUEST_ID), 0).state());
    }

    private void reportDecodeFinishedError() {
        WorkerStatusResponse response = decodeFinishedError();
        endpointRegistry.getDecode(DECODE_IP_PORT)
                .onWorkerStatusUpdate(new WorkerStatus(), response);
        scheduler.onWorkerStatusUpdate(response);
    }

    private WorkerStatusResponse decodeFinishedError() {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(REQUEST_ID));
        task.setPhase(TaskPhase.RUNNING);
        task.setErrorCode(1234);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setFinishedTaskInfo(Map.of(String.valueOf(REQUEST_ID), task));
        return response;
    }

    private void assertDecodeWorkerFailure() {
        Response terminal = awaitTerminal();
        assertFalse(terminal.isSuccess());
        assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(),
                terminal.getCode());
        assertFalse(decodeEndpoint.reservedView().containsKey(String.valueOf(REQUEST_ID)));
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(String.valueOf(REQUEST_ID), 0).state());
    }

    private void assertEngineOwnedScheduleSucceeded() {
        Response response = awaitTerminal();
        assertTrue(response.isSuccess(), response.getErrorMessage());
        assertTrue(response.isEnqueuedByMaster());
        assertTrue(decodeEndpoint.isConfirmedTracked(String.valueOf(REQUEST_ID)));
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(String.valueOf(REQUEST_ID), batchId).state());
    }

    private void reportDecode(TaskPhase phase) {
        reportRunning(RoleType.DECODE, phase);
    }

    private void reportRunning(RoleType role, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(REQUEST_ID));
        task.setPhase(phase);
        task.setInputLength(128);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setRunningTaskInfo(Map.of(String.valueOf(REQUEST_ID), task));
        if (role == RoleType.DECODE) {
            endpointRegistry.getDecode(DECODE_IP_PORT)
                    .onWorkerStatusUpdate(new WorkerStatus(), response);
        }
        scheduler.onWorkerStatusUpdate(response);
    }

    private BatchItem item(FlexlbConfig config) {
        Request request = new Request();
        request.setRequestId(String.valueOf(REQUEST_ID));
        request.setPriority(50);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 30_000));

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
        status.setRequestId(String.valueOf(REQUEST_ID));
        return status;
    }

    private Response awaitTerminal() {
        try {
            return item.future().get(5, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new AssertionError("interrupted while awaiting frontend publication", error);
        } catch (ExecutionException | TimeoutException error) {
            throw new AssertionError("frontend publication did not complete normally", error);
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError(e);
        }
    }
}

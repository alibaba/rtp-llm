package org.flexlb.autotpm;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.InflightState;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Terminal attribution of preempted victims (blueprint §1.10, decision D7):
 *
 * <ul>
 *   <li>{@link CancelReasonMapper#isAutoTpmPreempted} is the only sanctioned
 *       predicate — errorCode=CANCELLED(8100) AND cancelReason=2</li>
 *   <li>{@code DecodeEndpoint#processFinishedTasks} settles the preempted
 *       victim's {@link InflightItem} as AUTO_TPM_PREEMPTED (4290) with a
 *       meaningful error message</li>
 *   <li>ordinary client cancels (cancelReason=1/0) keep the existing
 *       CANCELLED path — never 4290</li>
 *   <li>settlement is idempotent (CAS terminal — an already-terminal item
 *       is untouched)</li>
 * </ul>
 */
class VictimTerminalAttributionTest {

    private static final long VICTIM_ID = 100L;
    private static final long ENGINE_CANCELLED = 8100L;
    private static final int REASON_PRIORITY_PREEMPTED = 2;
    private static final int REASON_USER_CANCELLED = 1;

    private WorkerStatus status;
    private InflightStore inflightStore;
    private DecodeEndpoint endpoint;

    private CompletableFuture<Response> victimFuture;
    private InflightItem victimItem;

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        inflightStore = mock(InflightStore.class);
        endpoint = new DecodeEndpoint(status, inflightStore);

        Request request = new Request();
        request.setRequestId(VICTIM_ID);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        victimFuture = new CompletableFuture<>();
        victimItem = new InflightItem(ctx, victimFuture, null);
        when(inflightStore.get(String.valueOf(VICTIM_ID))).thenReturn(victimItem);
    }

    // ---- a) attribution predicate: 8100 + cancelReason=2, nothing else ----

    @Test
    void isAutoTpmPreempted_cancelledWithPriorityPreemptedReason_true() {
        assertTrue(CancelReasonMapper.isAutoTpmPreempted(
                finishedTask(VICTIM_ID, ENGINE_CANCELLED, REASON_PRIORITY_PREEMPTED)));
    }

    @Test
    void isAutoTpmPreempted_anyOtherCombination_false() {
        // right error code, wrong reason
        assertFalse(CancelReasonMapper.isAutoTpmPreempted(
                finishedTask(VICTIM_ID, ENGINE_CANCELLED, REASON_USER_CANCELLED)));
        assertFalse(CancelReasonMapper.isAutoTpmPreempted(
                finishedTask(VICTIM_ID, ENGINE_CANCELLED, 0)));
        // right reason, wrong error code (success / other failure)
        assertFalse(CancelReasonMapper.isAutoTpmPreempted(
                finishedTask(VICTIM_ID, 0, REASON_PRIORITY_PREEMPTED)));
        assertFalse(CancelReasonMapper.isAutoTpmPreempted(
                finishedTask(VICTIM_ID, 4513, REASON_PRIORITY_PREEMPTED)));
        // null task
        assertFalse(CancelReasonMapper.isAutoTpmPreempted(null));
    }

    // ---- b) preempted finished task settles the item as 4290 ----

    @Test
    void processFinishedTasks_preemptedTask_settlesItemAs4290WithMessage() {
        endpoint.reserve(VICTIM_ID, 500, 500);

        reportFinished(finishedTask(VICTIM_ID, ENGINE_CANCELLED, REASON_PRIORITY_PREEMPTED));

        assertTrue(victimItem.isTerminated());
        assertEquals(InflightState.FAILED, victimItem.state());
        Response settled = victimFuture.join();
        assertFalse(settled.isSuccess());
        assertEquals(4290, settled.getCode());
        assertEquals("AUTO_TPM_PREEMPTED", settled.getErrorMessage());
        // the finished task also left both tracking layers
        assertEquals(0, endpoint.decodeTotalLoad());
    }

    @Test
    void processFinishedTasks_preemptedTaskNeverTracked_stillSettlesItem() {
        // wait-timeout close-out: the controller path already gave up, the
        // engine's finished report is the authoritative settle signal even
        // when the endpoint no longer tracks the request
        reportFinished(finishedTask(VICTIM_ID, ENGINE_CANCELLED, REASON_PRIORITY_PREEMPTED));

        assertTrue(victimItem.isTerminated());
        assertEquals(4290, victimFuture.join().getCode());
    }

    // ---- c) ordinary client cancel keeps the existing CANCELLED path ----

    @Test
    void processFinishedTasks_userCancelled_noAutoTpmSettle() {
        endpoint.reserve(VICTIM_ID, 500, 500);

        reportFinished(finishedTask(VICTIM_ID, ENGINE_CANCELLED, REASON_USER_CANCELLED));

        // not attributed → the endpoint must not settle it as preempted;
        // the existing RouteService cancel path owns this terminal
        assertFalse(victimItem.isTerminated());
        assertFalse(victimFuture.isDone());
        verify(inflightStore, never()).get(Mockito.anyString());
        assertEquals(0, endpoint.decodeTotalLoad());
    }

    @Test
    void processFinishedTasks_cancelReasonUnset_noAutoTpmSettle() {
        endpoint.reserve(VICTIM_ID, 500, 500);

        reportFinished(finishedTask(VICTIM_ID, ENGINE_CANCELLED, 0));

        assertFalse(victimItem.isTerminated());
        verify(inflightStore, never()).get(Mockito.anyString());
    }

    // ---- d) settle idempotency (CAS terminal) ----

    @Test
    void processFinishedTasks_itemAlreadyCancelled_settleIsNoOp() {
        endpoint.reserve(VICTIM_ID, 500, 500);
        // client cancel raced ahead: the item is already terminal
        assertTrue(victimItem.cancel());
        assertEquals(InflightState.CANCELLED, victimItem.state());

        reportFinished(finishedTask(VICTIM_ID, ENGINE_CANCELLED, REASON_PRIORITY_PREEMPTED));

        // terminal state untouched — no re-settle to FAILED/4290
        assertEquals(InflightState.CANCELLED, victimItem.state());
        assertTrue(victimFuture.isCompletedExceptionally());
    }

    @Test
    void processFinishedTasks_duplicatePreemptedReports_settleOnce() {
        endpoint.reserve(VICTIM_ID, 500, 500);

        reportFinished(finishedTask(VICTIM_ID, ENGINE_CANCELLED, REASON_PRIORITY_PREEMPTED));
        assertEquals(InflightState.FAILED, victimItem.state());
        long firstTerminalTime = victimItem.getTerminalTime();

        // duplicate finished report (e.g. controller settle raced calibrate)
        reportFinished(finishedTask(VICTIM_ID, ENGINE_CANCELLED, REASON_PRIORITY_PREEMPTED));

        assertEquals(InflightState.FAILED, victimItem.state());
        assertEquals(firstTerminalTime, victimItem.getTerminalTime(), "no second terminal transition");
        assertEquals(4290, victimFuture.join().getCode());
    }

    // ==================== fixtures ====================

    private static TaskInfo finishedTask(long requestId, long errorCode, int cancelReason) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setErrorCode(errorCode);
        task.setCancelReason(cancelReason);
        if (errorCode != 0) {
            task.setErrorMessage("cancelled");
        }
        return task;
    }

    /** Drive processFinishedTasks through the public worker-status update path. */
    private void reportFinished(TaskInfo task) {
        status.getAvailableKvCacheTokens().set(10_000);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of(String.valueOf(task.getRequestId()), task));
        endpoint.onWorkerStatusUpdate(status, response);
    }
}

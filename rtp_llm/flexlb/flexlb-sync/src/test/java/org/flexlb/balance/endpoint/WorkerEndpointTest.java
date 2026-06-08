package org.flexlb.balance.endpoint;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.TaskStateEnum;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class WorkerEndpointTest {

    private WorkerStatus status;
    private PrefillTimePredictor predictor;
    private WorkerEndpoint endpoint;

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        predictor = new PrefillTimePredictor(0, 1, 0, 0, 0, 0);
        endpoint = new WorkerEndpoint("10.0.0.1", 8080, 8081, status, predictor);
    }

    @Test
    void commitBatch_incrementsEstimate() {
        endpoint.commitBatch(1L, 500);
        assertEquals(500, endpoint.getEstimatedWaitingTimeMs());

        endpoint.commitBatch(2L, 300);
        assertEquals(800, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void releaseBatch_decrementsEstimate() {
        endpoint.commitBatch(1L, 500);
        endpoint.commitBatch(2L, 300);

        endpoint.releaseBatch(1L);
        assertEquals(300, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void releaseBatch_unknownBatchId_noEffect() {
        endpoint.commitBatch(1L, 500);
        endpoint.releaseBatch(999L);
        assertEquals(500, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void releaseBatch_neverGoesNegative() {
        endpoint.commitBatch(1L, 100);
        endpoint.releaseBatch(1L);
        endpoint.releaseBatch(1L);
        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void calibrate_emptyLocalTaskMap_resetsToZero() {
        endpoint.commitBatch(1L, 500);
        endpoint.calibrateWaitingTime();

        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
        assertTrue(endpoint.getInflightBatches().isEmpty());
    }

    @Test
    void calibrate_confirmedBatchTasks_usesPredictor() {
        TaskInfo t1 = task(100L, 1000, 0, 5L);
        TaskInfo t2 = task(101L, 2000, 0, 5L);
        status.putLocalTask(100L, t1);
        status.putLocalTask(101L, t2);
        t1.updateTaskState(TaskStateEnum.CONFIRMED);
        t2.updateTaskState(TaskStateEnum.CONFIRMED);

        endpoint.commitBatch(5L, 9999);

        endpoint.calibrateWaitingTime();

        long expected = predictor.predictBatchMs(java.util.List.of(
                new org.flexlb.balance.strategy.RequestProfile(1000, 0),
                new org.flexlb.balance.strategy.RequestProfile(2000, 0)));
        assertEquals(expected, endpoint.getEstimatedWaitingTimeMs());
        assertFalse(endpoint.getInflightBatches().containsKey(5L));
    }

    @Test
    void calibrate_unbatchedTasks_sumsIndividually() {
        TaskInfo t1 = task(100L, 500, 100, -1L);
        TaskInfo t2 = task(101L, 1000, 200, -1L);
        status.putLocalTask(100L, t1);
        status.putLocalTask(101L, t2);
        t1.updateTaskState(TaskStateEnum.CONFIRMED);
        t2.updateTaskState(TaskStateEnum.CONFIRMED);

        endpoint.calibrateWaitingTime();

        long expected = predictor.estimateMs(500, 100) + predictor.estimateMs(1000, 200);
        assertEquals(expected, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void calibrate_mixedBatchAndUnbatched() {
        TaskInfo batched1 = task(100L, 1000, 0, 5L);
        TaskInfo batched2 = task(101L, 2000, 0, 5L);
        TaskInfo unbatched = task(102L, 500, 100, -1L);
        status.putLocalTask(100L, batched1);
        status.putLocalTask(101L, batched2);
        status.putLocalTask(102L, unbatched);
        batched1.updateTaskState(TaskStateEnum.CONFIRMED);
        batched2.updateTaskState(TaskStateEnum.CONFIRMED);
        unbatched.updateTaskState(TaskStateEnum.CONFIRMED);

        endpoint.commitBatch(5L, 9999);

        endpoint.calibrateWaitingTime();

        long batchEstimate = predictor.predictBatchMs(java.util.List.of(
                new org.flexlb.balance.strategy.RequestProfile(1000, 0),
                new org.flexlb.balance.strategy.RequestProfile(2000, 0)));
        long unbatchedEstimate = predictor.estimateMs(500, 100);
        assertEquals(batchEstimate + unbatchedEstimate, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void calibrate_partialBatchFailure() {
        TaskInfo survived1 = task(100L, 1000, 0, 5L);
        TaskInfo survived2 = task(101L, 2000, 0, 5L);
        status.putLocalTask(100L, survived1);
        status.putLocalTask(101L, survived2);
        survived1.updateTaskState(TaskStateEnum.CONFIRMED);
        survived2.updateTaskState(TaskStateEnum.CONFIRMED);

        endpoint.commitBatch(5L, 9999);

        endpoint.calibrateWaitingTime();

        long expected = predictor.predictBatchMs(java.util.List.of(
                new org.flexlb.balance.strategy.RequestProfile(1000, 0),
                new org.flexlb.balance.strategy.RequestProfile(2000, 0)));
        assertEquals(expected, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void calibrate_inflightUnconfirmedBatchesSurvive() {
        TaskInfo confirmed = task(100L, 500, 0, 5L);
        status.putLocalTask(100L, confirmed);
        confirmed.updateTaskState(TaskStateEnum.CONFIRMED);

        endpoint.commitBatch(5L, 1000);
        endpoint.commitBatch(7L, 2000);

        endpoint.calibrateWaitingTime();

        long confirmedEstimate = predictor.predictBatchMs(java.util.List.of(
                new org.flexlb.balance.strategy.RequestProfile(500, 0)));
        assertEquals(confirmedEstimate + 2000, endpoint.getEstimatedWaitingTimeMs());
        assertFalse(endpoint.getInflightBatches().containsKey(5L));
        assertTrue(endpoint.getInflightBatches().containsKey(7L));
    }

    @Test
    void calibrate_skipsNonConfirmedTasks() {
        TaskInfo inTransit = task(100L, 1000, 0, 5L);
        status.putLocalTask(100L, inTransit);

        endpoint.commitBatch(5L, 500);
        endpoint.calibrateWaitingTime();

        assertEquals(500, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void calibrate_nullPredictor_noOp() {
        endpoint.setPredictor(null);
        endpoint.commitBatch(1L, 500);

        TaskInfo t = task(100L, 1000, 0, 1L);
        status.putLocalTask(100L, t);
        t.updateTaskState(TaskStateEnum.CONFIRMED);

        endpoint.calibrateWaitingTime();
        assertEquals(500, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080", endpoint.ipPort());
    }

    private TaskInfo task(long requestId, long inputLength, long prefixLength, long batchId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(inputLength);
        task.setPrefixLength(prefixLength);
        task.setBatchId(batchId);
        return task;
    }
}

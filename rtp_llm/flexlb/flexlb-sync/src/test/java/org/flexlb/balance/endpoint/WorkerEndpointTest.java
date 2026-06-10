package org.flexlb.balance.endpoint;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.balance.strategy.RequestProfile;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class WorkerEndpointTest {

    private WorkerStatus status;
    private PrefillTimePredictor predictor;
    private PrefillEndpoint endpoint;

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        predictor = new PrefillTimePredictor(0, 1, 0, 0, 0, 0);
        endpoint = new PrefillEndpoint("10.0.0.1", 8080, 8081, status, predictor);
    }

    @Test
    void commitBatch_incrementsEstimate() {
        endpoint.commitBatch(1L, 500, List.of(100L), List.of(new RequestProfile(1000, 0)));
        assertEquals(500, endpoint.getEstimatedWaitingTimeMs());

        endpoint.commitBatch(2L, 300, List.of(101L), List.of(new RequestProfile(500, 0)));
        assertEquals(800, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void releaseBatch_decrementsEstimate() {
        endpoint.commitBatch(1L, 500, List.of(100L), List.of(new RequestProfile(1000, 0)));
        endpoint.commitBatch(2L, 300, List.of(101L), List.of(new RequestProfile(500, 0)));

        endpoint.releaseBatch(1L);
        assertEquals(300, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void releaseBatch_unknownBatchId_noEffect() {
        endpoint.commitBatch(1L, 500, List.of(100L), List.of(new RequestProfile(1000, 0)));
        endpoint.releaseBatch(999L);
        assertEquals(500, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void releaseBatch_neverGoesNegative() {
        endpoint.commitBatch(1L, 100, List.of(100L), List.of(new RequestProfile(1000, 0)));
        endpoint.releaseBatch(1L);
        endpoint.releaseBatch(1L);
        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void calibrate_noInflight_resetsToZero() {
        endpoint.commitBatch(1L, 500, List.of(100L), List.of(new RequestProfile(1000, 0)));

        TaskInfo finished = task(100L, 1000, 0, 1L);
        finished.setErrorCode(0);
        endpoint.calibrate(Map.of("100", finished), null);

        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
        assertTrue(endpoint.getInflightBatches().isEmpty());
    }

    @Test
    void calibrate_finishedBatch_removedFromInflight() {
        endpoint.commitBatch(5L, 9999, List.of(100L, 101L),
                List.of(new RequestProfile(1000, 0), new RequestProfile(2000, 0)));

        TaskInfo t1 = task(100L, 1000, 0, 5L);
        t1.setErrorCode(0);
        TaskInfo t2 = task(101L, 2000, 0, 5L);
        t2.setErrorCode(0);
        endpoint.calibrate(Map.of("100", t1, "101", t2), null);

        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
        assertFalse(endpoint.getInflightBatches().containsKey(5L));
    }

    @Test
    void calibrate_partialBatchFailure_repacks() {
        endpoint.commitBatch(5L, 9999, List.of(100L, 101L),
                List.of(new RequestProfile(1000, 0), new RequestProfile(2000, 0)));

        TaskInfo failed = task(100L, 1000, 0, 5L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        TaskInfo success = task(101L, 2000, 0, 5L);
        success.setErrorCode(0);
        endpoint.calibrate(Map.of("100", failed, "101", success), null);

        assertFalse(endpoint.getInflightBatches().containsKey(5L));
        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void calibrate_inflightUnconfirmedBatchesSurvive() {
        endpoint.commitBatch(5L, 1000, List.of(100L), List.of(new RequestProfile(500, 0)));
        endpoint.commitBatch(7L, 2000, List.of(200L), List.of(new RequestProfile(1000, 0)));

        TaskInfo finished = task(100L, 500, 0, 5L);
        finished.setErrorCode(0);
        endpoint.calibrate(Map.of("100", finished), null);

        assertFalse(endpoint.getInflightBatches().containsKey(5L));
        assertTrue(endpoint.getInflightBatches().containsKey(7L));
        assertEquals(2000, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void calibrate_nullPredictor_noOp() {
        endpoint.setPredictor(null);
        endpoint.commitBatch(1L, 500, List.of(100L), List.of(new RequestProfile(1000, 0)));

        TaskInfo t = task(100L, 1000, 0, 1L);
        t.setErrorCode(0);
        endpoint.calibrate(Map.of("100", t), null);

        assertEquals(500, endpoint.getEstimatedWaitingTimeMs());
    }

    @Test
    void repackBatch_removesFailedRequests() {
        endpoint.commitBatch(5L, 9999, List.of(100L, 101L, 102L),
                List.of(new RequestProfile(1000, 0), new RequestProfile(2000, 0), new RequestProfile(3000, 0)));

        BatchInflight repacked = endpoint.repackBatch(5L, java.util.Set.of(101L));

        assertNotNull(repacked);
        assertEquals(2, repacked.requestIds().size());
        assertTrue(repacked.requestIds().contains(100L));
        assertTrue(repacked.requestIds().contains(102L));
        assertFalse(repacked.requestIds().contains(101L));
    }

    @Test
    void repackBatch_allFailed_removesBatch() {
        endpoint.commitBatch(5L, 500, List.of(100L), List.of(new RequestProfile(1000, 0)));

        BatchInflight repacked = endpoint.repackBatch(5L, java.util.Set.of(100L));

        assertNull(repacked);
        assertFalse(endpoint.getInflightBatches().containsKey(5L));
        assertEquals(0, endpoint.getEstimatedWaitingTimeMs());
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

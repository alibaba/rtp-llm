package org.flexlb.dao.pv;

import org.flexlb.dao.BatchScheduleContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@DisplayName("BatchPvLogData constructor mapping")
class BatchPvLogDataTest {

    @Test
    void successResponse_populatesAllFields() {
        BatchScheduleRequest req = new BatchScheduleRequest();
        req.setBatchCount(3);

        List<BatchScheduleTarget> targets = List.of(
                new BatchScheduleTarget("10.1.2.1", 28100, 28101),
                new BatchScheduleTarget("10.1.2.2", 28100, 28101),
                new BatchScheduleTarget("10.1.2.1", 28100, 28101));
        BatchScheduleResponse resp = BatchScheduleResponse.success(targets);

        BatchScheduleContext bctx = new BatchScheduleContext();
        bctx.setBatchRequest(req);
        bctx.setBatchResponse(resp);

        BatchPvLogData data = new BatchPvLogData(bctx);

        assertEquals("batch_schedule", data.getType());
        assertEquals(3, data.getBatchCount());
        assertEquals(3, data.getTargetCount());
        assertTrue(data.isSuccess());
        assertEquals(200, data.getCode());
        assertNull(data.getError());
        assertEquals(bctx.getStartTime(), data.getStartTimeMs());
        assertTrue(data.getCostMs() >= 0);
    }

    @Test
    void failureResponse_capturesCodeAndError() {
        BatchScheduleRequest req = new BatchScheduleRequest();
        req.setBatchCount(5);

        BatchScheduleResponse resp = BatchScheduleResponse.error(
                StrategyErrorType.INVALID_REQUEST, "batch_count must be in [1, 1000]");

        BatchScheduleContext bctx = new BatchScheduleContext();
        bctx.setBatchRequest(req);
        bctx.setBatchResponse(resp);
        bctx.setSuccess(false);
        bctx.setErrorMessage("batch_count must be in [1, 1000]");

        BatchPvLogData data = new BatchPvLogData(bctx);

        assertEquals(5, data.getBatchCount());
        assertEquals(0, data.getTargetCount());
        assertFalse(data.isSuccess());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), data.getCode());
        assertEquals("batch_count must be in [1, 1000]", data.getError());
    }

    @Test
    void nullRequest_defaultsBatchCountToZero() {
        BatchScheduleContext bctx = new BatchScheduleContext();
        bctx.setSuccess(false);
        bctx.setErrorMessage("body parse error");

        BatchPvLogData data = new BatchPvLogData(bctx);

        assertEquals(0, data.getBatchCount());
        assertEquals(0, data.getTargetCount());
        assertFalse(data.isSuccess());
        assertEquals(0, data.getCode());
        assertEquals("body parse error", data.getError());
    }

    @Test
    void nullResponse_leavesResponseDerivedFieldsAtDefault() {
        BatchScheduleRequest req = new BatchScheduleRequest();
        req.setBatchCount(7);

        BatchScheduleContext bctx = new BatchScheduleContext();
        bctx.setBatchRequest(req);
        bctx.setSuccess(false);
        bctx.setErrorMessage("downstream failure");

        BatchPvLogData data = new BatchPvLogData(bctx);

        assertEquals(7, data.getBatchCount());
        assertEquals(0, data.getTargetCount());
        assertEquals(0, data.getCode());
        assertFalse(data.isSuccess());
        assertEquals("downstream failure", data.getError());
    }

    @Test
    void responseWithNullServerStatus_targetCountIsZero() {
        BatchScheduleRequest req = new BatchScheduleRequest();
        req.setBatchCount(2);

        BatchScheduleResponse resp = BatchScheduleResponse.error(
                StrategyErrorType.NO_AVAILABLE_WORKER, "no worker");

        BatchScheduleContext bctx = new BatchScheduleContext();
        bctx.setBatchRequest(req);
        bctx.setBatchResponse(resp);
        bctx.setSuccess(false);

        BatchPvLogData data = new BatchPvLogData(bctx);

        assertEquals(0, data.getTargetCount());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), data.getCode());
    }
}

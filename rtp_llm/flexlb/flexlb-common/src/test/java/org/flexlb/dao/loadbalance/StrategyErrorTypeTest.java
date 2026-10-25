package org.flexlb.dao.loadbalance;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class StrategyErrorTypeTest {

    @Test
    void encoderWorkerAndStaleVitRouteHaveDistinctCodes() {
        assertEquals(8409, StrategyErrorType.NO_ENCODER_WORKER.getErrorCode());
        assertEquals(8408, StrategyErrorType.VIT_ROUTE_STALE.getErrorCode());
    }

    @Test
    void responseFactoryEnforcesAllCodeReasonPairs() {
        for (StrategyErrorType type : StrategyErrorType.values()) {
            for (RejectReason reason : RejectReason.values()) {
                boolean valid = switch (type) {
                    case PRIORITY_ADMISSION_REJECTED ->
                        reason == RejectReason.HIGHER_PRIORITY_AHEAD || reason == RejectReason.SAME_PRIORITY_AHEAD;
                    case RESOURCE_EXHAUSTED -> reason == RejectReason.RESOURCE_EXHAUSTED;
                    default -> reason == RejectReason.UNSPECIFIED;
                };
                assertEquals(valid, type.acceptsRejectReason(reason), type + "/" + reason);
                if (valid) {
                    Response response = Response.error(type, reason);
                    assertFalse(response.isSuccess());
                    assertEquals(type.getErrorCode(), response.getCode());
                    assertEquals(reason, response.getAdmissionRejectReason());
                } else {
                    assertThrows(IllegalArgumentException.class, () -> Response.error(type, reason));
                }
            }
        }
    }

    @Test
    void priorityMessagesCannotBeOverwrittenByRetryDiagnostics() {
        for (RejectReason reason : new RejectReason[]{RejectReason.HIGHER_PRIORITY_AHEAD,
                RejectReason.SAME_PRIORITY_AHEAD}) {
            Response response = Response.error(StrategyErrorType.PRIORITY_ADMISSION_REJECTED, reason, "retry detail");
            assertEquals(reason == RejectReason.HIGHER_PRIORITY_AHEAD
                    ? "higher-priority requests are ahead"
                    : "same-priority requests are ahead", response.getErrorMessage());
            assertEquals(reason, response.getAdmissionRejectReason());
        }
        assertEquals("admission unavailable; blocker priority attribution is unavailable",
                Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE, RejectReason.UNSPECIFIED, "retry detail")
                        .getErrorMessage());
    }

    @Test
    void capacityDiagnosticIsNormalizedOnce() {
        Response failure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED, RejectReason.RESOURCE_EXHAUSTED,
                "decode engine slots exhausted");
        String message = "admission capacity is temporarily exhausted; trigger=decode engine slots exhausted";
        assertEquals(message, failure.getErrorMessage());
        assertEquals(failure,
                Response.error(StrategyErrorType.RESOURCE_EXHAUSTED, RejectReason.RESOURCE_EXHAUSTED, message));
    }

    @Test
    void schedulingRejectionsKeepInternalErrorCodesAndDetail() {
        String detail = "preempted by higher-priority request 62";
        Response response = Response.buildErrorResponse(StrategyErrorType.PRIORITY_PREEMPTED, detail);
        assertEquals(8429, response.getCode());
        assertEquals(detail, response.getErrorMessage());
        assertEquals(8502, Response.buildErrorResponse(StrategyErrorType.QUEUE_FULL, "queue full").getCode());
        assertEquals(8433, StrategyErrorType.QUEUE_FULL_LOW_PRIORITY_REJECTED.getErrorCode());
        assertEquals(8434, StrategyErrorType.QUEUE_FULL_HIGH_PRIORITY_REJECTED.getErrorCode());
    }

    @Test
    void queueRejectionIsChosenByPriorityThreshold() {
        assertEquals(StrategyErrorType.QUEUE_FULL_LOW_PRIORITY_REJECTED, StrategyErrorType.queueRejectionFor(40, 50));
        assertEquals(StrategyErrorType.QUEUE_FULL_HIGH_PRIORITY_REJECTED, StrategyErrorType.queueRejectionFor(50, 50));
        assertEquals(StrategyErrorType.QUEUE_FULL_HIGH_PRIORITY_REJECTED, StrategyErrorType.queueRejectionFor(60, 50));
    }
}

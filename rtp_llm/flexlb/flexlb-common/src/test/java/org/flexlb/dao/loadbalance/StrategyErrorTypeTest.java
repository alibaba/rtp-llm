package org.flexlb.dao.loadbalance;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class StrategyErrorTypeTest {

    @Test
    void responseFactoryEnforcesAllCodeReasonPairs() {
        for (StrategyErrorType type : StrategyErrorType.values()) {
            for (AdmissionRejectReason reason : AdmissionRejectReason.values()) {
                boolean valid = switch (type) {
                    case PRIORITY_ADMISSION_REJECTED -> reason == AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                            || reason == AdmissionRejectReason.SAME_PRIORITY_AHEAD;
                    case RESOURCE_EXHAUSTED -> reason == AdmissionRejectReason.RESOURCE_EXHAUSTED;
                    default -> reason == AdmissionRejectReason.UNSPECIFIED;
                };
                assertEquals(valid, type.acceptsAdmissionRejectReason(reason), type + "/" + reason);
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
    void admissionUnavailableIsRetainedForLegacyErrorCodeDecoding() {
        assertEquals(StrategyErrorType.ADMISSION_UNAVAILABLE,
                StrategyErrorType.fromErrorCode(8432));
    }

    @Test
    void priorityMessagesCannotBeOverwrittenByRetryDiagnostics() {
        for (AdmissionRejectReason reason : new AdmissionRejectReason[]{
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD, AdmissionRejectReason.SAME_PRIORITY_AHEAD}) {
            Response response = Response.error(StrategyErrorType.PRIORITY_ADMISSION_REJECTED, reason, "retry detail");
            assertEquals(reason == AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                    ? "higher-priority requests are ahead" : "same-priority requests are ahead",
                    response.getErrorMessage());
            assertEquals(reason, response.getAdmissionRejectReason());
        }
        assertEquals("admission unavailable; blocker priority attribution is unavailable",
                Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE,
                        AdmissionRejectReason.UNSPECIFIED, "retry detail").getErrorMessage());
    }

    @Test
    void capacityDiagnosticIsNormalizedOnce() {
        Response failure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED, "decode engine slots exhausted");
        String message = "admission capacity is temporarily exhausted; trigger=decode engine slots exhausted";
        assertEquals(message, failure.getErrorMessage());
        assertEquals(failure, Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED, message));
    }
}

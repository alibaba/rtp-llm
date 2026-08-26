package org.flexlb.balance.admission;

import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.StrategyErrorType;

/** Typed incoming-admission failure produced at the Master decision point. */
public record AdmissionFailure(StrategyErrorType errorType,
                               AdmissionRejectReason reason,
                               String message) {

    public AdmissionFailure {
        if (errorType == null || reason == null) {
            throw new IllegalArgumentException("error type and reason are required");
        }
    }

    public static AdmissionFailure higherPriorityAhead() {
        return new AdmissionFailure(StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
                "higher-priority requests are ahead");
    }

    public static AdmissionFailure samePriorityAhead() {
        return new AdmissionFailure(StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                "same-priority requests are ahead");
    }

    public static AdmissionFailure resourceExhausted() {
        return new AdmissionFailure(StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
                "admission capacity is temporarily exhausted");
    }

    /**
     * Admission is blocked, but at least one causally relevant occupant does
     * not carry trustworthy priority provenance.  This is deliberately not a
     * resource-exhaustion result: without that provenance the Master cannot
     * prove higher/same-priority blocking or a pure capacity shortage.
     */
    public static AdmissionFailure priorityAttributionUnavailable() {
        return new AdmissionFailure(StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED,
                "admission unavailable; blocker priority attribution is unavailable");
    }

}

package org.flexlb.balance.admission;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.PriorityNormalizer;

import java.util.List;

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

    /** Classify one timed-out priority request from the exact queue prefix ahead of it. */
    public static AdmissionFailure classifyQueuedTimeout(
            int incomingPriority,
            List<ScheduledRequest> itemsAhead) {
        boolean higher = false;
        boolean same = false;
        boolean hasUnattributedOccupant = false;
        if (itemsAhead != null) {
            for (ScheduledRequest occupant : itemsAhead) {
                if (!PriorityNormalizer.hasPriority(occupant.priority())) {
                    hasUnattributedOccupant = true;
                } else if (occupant.priority() > incomingPriority) {
                    higher = true;
                } else if (occupant.priority() == incomingPriority) {
                    same = true;
                }
            }
        }
        if (higher) {
            return higherPriorityAhead();
        }
        if (same) {
            return samePriorityAhead();
        }
        if (hasUnattributedOccupant) {
            return priorityAttributionUnavailable();
        }
        return resourceExhausted();
    }

}

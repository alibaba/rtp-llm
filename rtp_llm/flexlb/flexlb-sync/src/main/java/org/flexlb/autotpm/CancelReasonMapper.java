package org.flexlb.autotpm;

import org.flexlb.dao.master.TaskInfo;

/**
 * Structured cancel attribution mapper (D7).
 * <p>
 * A finished engine task is attributed to auto-tpm priority preemption iff
 * error_code == CANCELLED (engine ErrorCode, 8100) AND
 * cancel_reason == ENGINE_CANCEL_REASON_PRIORITY_PREEMPTED (2).
 * <p>
 * This is the ONLY sanctioned attribution predicate: matching on
 * errorMessage content is forbidden project-wide.
 */
public final class CancelReasonMapper {

    /** Engine-side ErrorCode::CANCELLED numeric value carried in TaskInfoPB.error_info.error_code. */
    public static final long ENGINE_ERROR_CODE_CANCELLED = 8100L;

    /** EngineCancelReasonPB.ENGINE_CANCEL_REASON_PRIORITY_PREEMPTED numeric value. */
    public static final int CANCEL_REASON_PRIORITY_PREEMPTED = 2;

    private CancelReasonMapper() {
    }

    /**
     * @return true iff the finished task was cancelled by auto-tpm priority preemption.
     */
    public static boolean isAutoTpmPreempted(TaskInfo taskInfo) {
        return taskInfo != null
                && taskInfo.getErrorCode() == ENGINE_ERROR_CODE_CANCELLED
                && taskInfo.getCancelReason() == CANCEL_REASON_PRIORITY_PREEMPTED;
    }
}

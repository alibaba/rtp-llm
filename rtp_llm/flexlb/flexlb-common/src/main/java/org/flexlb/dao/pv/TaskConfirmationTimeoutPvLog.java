package org.flexlb.dao.pv;

/**
 * A locally tracked task was not confirmed by WorkerStatus before the configured timeout.
 */
public record TaskConfirmationTimeoutPvLog(
        String eventType,
        long requestId,
        String role,
        String workerIp,
        int workerPort,
        String taskState,
        long ageMs,
        long confirmationTimeoutMs,
        long inputTokens,
        long predictedHitTokens,
        String cacheMatchSource,
        long estimatedPrefillTime) {

    public static final String EVENT_TYPE = "task_confirmation_timeout";
}

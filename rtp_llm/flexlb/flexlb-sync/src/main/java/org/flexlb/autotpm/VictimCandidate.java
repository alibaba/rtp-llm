package org.flexlb.autotpm;

/**
 * Snapshot of a RUNNING decode engine task considered for Auto-TPM
 * preemption. Produced by {@code DecodeEndpoint#snapshotRunningCandidates}
 * and consumed by {@link InflightVictimSelector}.
 *
 * <p>Phase is not carried here on purpose: the snapshot only emits
 * RUNNING-phase tasks, so every candidate is RUNNING by construction.
 *
 * @param requestId      engine request id
 * @param priority       Auto-TPM priority registered at dispatch time
 *                       (lower value = lower priority)
 * @param iterateCount   engine-reported generated-token count (progress depth)
 * @param kvTokens       KV reservation carried by the task's inflight entry
 *                       (0 when the request was never reserved locally)
 * @param runningSinceMs epoch millis when the task was first observed RUNNING
 * @param endpoint       owning decode endpoint identity ({@code ip:port})
 */
public record VictimCandidate(
        long requestId,
        int priority,
        long iterateCount,
        long kvTokens,
        long runningSinceMs,
        String endpoint) {
}

package org.flexlb.sync.runner;

import java.util.concurrent.ScheduledExecutorService;

/**
 * Effective getWorkerStatus long-poll settings (scheduler upgrade A).
 *
 * <p>When enabled, status requests carry {@code wait_timeout_ms} so the engine
 * holds the poll until a new completion event, and {@link GrpcWorkerStatusRunner}
 * re-arms the next poll as soon as a response arrives instead of waiting for the
 * fixed SYNC_STATUS_INTERVAL tick. {@code rearmScheduler} applies a minimal
 * re-arm delay so a misbehaving engine that answers instantly cannot spin the
 * status loop into a hot busy-poll.
 *
 * @param enabled        master-side long-poll switch (FLEXLB_STATUS_LONG_POLL_ENABLED)
 * @param timeoutMs      wait_timeout_ms sent to the engine (FLEXLB_STATUS_LONG_POLL_TIMEOUT_MS)
 * @param rearmScheduler scheduler used to delay-launch the next poll of the chain
 */
public record StatusLongPollConfig(boolean enabled,
                                   long timeoutMs,
                                   ScheduledExecutorService rearmScheduler) {
}

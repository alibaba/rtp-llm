package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Second-layer inflight entry: a task the engine has acknowledged in its
 * status report, tracked with a phase ({@link EngineTaskPhase}) and the last
 * calibrate round it was observed in.
 *
 * <p>Kept deliberately slim (payload + phase + lastSeenRound + timestamps).
 * The payload type is generic so prefill attaches its
 * {@link PrefillInflightEntry} (batch members are needed for completion /
 * repack decisions) while decode can attach its own entry type.
 *
 * <p>Eviction paths:
 * <ul>
 *   <li>normal: removed when the engine reports the task finished</li>
 *   <li>stale: absent from engine reports for N consecutive calibrate rounds</li>
 *   <li>backstop: wall-clock TTL via {@link InflightEvictor} (covers a worker
 *       that stops reporting entirely, so calibrate rounds no longer advance)</li>
 * </ul>
 *
 * @param <E> payload entry type
 */
public class EngineTask<E> implements InflightEvictor.TtlTracked {

    private volatile E entry;
    private volatile EngineTaskPhase phase;
    private volatile long lastSeenRound;
    private final long acceptedAtMs;

    /**
     * Progress anchor for wait-time estimation, same semantics as the legacy
     * BatchInflight: while WAITING the anchor keeps advancing to the latest
     * status time (queued work spends no predicted time); on the first
     * RUNNING observation it freezes so elapsed time starts counting.
     */
    private final AtomicLong progressBaseMs;
    private volatile boolean running;

    public EngineTask(E entry, EngineTaskPhase phase, long round, long acceptedAtMs) {
        this.entry = entry;
        this.phase = phase;
        this.lastSeenRound = round;
        this.acceptedAtMs = acceptedAtMs;
        this.progressBaseMs = new AtomicLong(acceptedAtMs);
        this.running = false;
        if (phase == EngineTaskPhase.RUNNING) {
            this.running = true;
        }
    }

    public E entry() {
        return entry;
    }

    /** Replace the payload (e.g. batch repack shrinks the member list). */
    public void updateEntry(E newEntry) {
        this.entry = newEntry;
    }

    public EngineTaskPhase phase() {
        return phase;
    }

    public long lastSeenRound() {
        return lastSeenRound;
    }

    /** @return epoch millis when the engine first acknowledged this task. */
    @Override
    public long createdAtMs() {
        return acceptedAtMs;
    }

    public long progressBaseMs() {
        return progressBaseMs.get();
    }

    public boolean running() {
        return running;
    }

    /**
     * Record an observation from the current calibrate round: refresh the
     * phase and lastSeenRound, and update the progress anchor.
     */
    public void observe(EngineTaskPhase newPhase, long round, long statusMs) {
        this.phase = newPhase;
        this.lastSeenRound = round;
        if (!running) {
            progressBaseMs.updateAndGet(base -> Math.max(base, statusMs));
            if (newPhase == EngineTaskPhase.RUNNING) {
                running = true;
            }
        }
    }
}

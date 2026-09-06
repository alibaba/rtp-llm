package org.flexlb.balance.scheduler.priority;

import org.flexlb.enums.DecodeTaskPhase;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One decode preemption transaction and its resource-ownership ledger.
 *
 * <p>The object owns only state; RPC orchestration lives in
 * {@link DecodePreemptionCoordinator}.  Every mutation is serialized here so
 * a Cancel callback and a WorkerStatus callback can converge without applying
 * a victim settlement twice.</p>
 */
public final class PreemptionAttempt {

    public enum State {
        PLANNED,
        CLAIMED,
        CANCEL_IN_FLIGHT,
        WAITING_CANCELED,
        READY_COMMIT,
        COMMITTED,
        ABORTED,
        CONTROL_FAILED
    }

    public enum VictimState {
        PLANNED,
        CLAIMED,
        CANCEL_IN_FLIGHT,
        CANCEL_REQUESTED,
        CANCELED_SETTLED,
        NOT_FOUND_STALE,
        CANCEL_UNKNOWN
    }

    public record Victim(String requestId,
                         int priority,
                         long kvTokens,
                         DecodeTaskPhase phase,
                         EngineCancelChannel.CancelTarget target) {
        public Victim {
            if (requestId == null || requestId.isBlank()) {
                throw new IllegalArgumentException("victim requestId must not be blank");
            }
            if (phase == null || !phase.requiresEngineCancel()) {
                throw new IllegalArgumentException("victim must require Engine Cancel");
            }
        }
    }

    private final long token;
    private final String incomingRequestId;
    private final long snapshotVersion;
    private final Map<String, Victim> victims;
    private final Map<String, VictimState> victimStates;
    private State state = State.PLANNED;

    public PreemptionAttempt(long token,
                             String incomingRequestId,
                             long snapshotVersion,
                             List<Victim> victims) {
        if (token <= 0 || (incomingRequestId == null || incomingRequestId.isBlank()) || victims == null || victims.isEmpty()) {
            throw new IllegalArgumentException("token, incoming request and victims are required");
        }
        this.token = token;
        this.incomingRequestId = incomingRequestId;
        this.snapshotVersion = snapshotVersion;
        this.victims = new LinkedHashMap<>();
        this.victimStates = new LinkedHashMap<>();
        for (Victim victim : victims) {
            if (this.victims.putIfAbsent(victim.requestId(), victim) != null) {
                throw new IllegalArgumentException("duplicate victim " + victim.requestId());
            }
            this.victimStates.put(victim.requestId(), VictimState.PLANNED);
        }
    }

    public long token() { return token; }
    public String incomingRequestId() { return incomingRequestId; }
    public long snapshotVersion() { return snapshotVersion; }
    public List<Victim> victims() { return List.copyOf(victims.values()); }

    public synchronized State state() { return state; }

    public synchronized VictimState victimState(String requestId) {
        return requireVictim(requestId);
    }

    public synchronized boolean claimAll() {
        if (state != State.PLANNED) {
            return false;
        }
        victimStates.replaceAll((ignored, old) -> VictimState.CLAIMED);
        state = State.CLAIMED;
        return true;
    }

    /** Linearization immediately before the first outbound Cancel RPC. */
    public synchronized boolean markCancelInFlight() {
        if (state != State.CLAIMED) {
            return false;
        }
        victimStates.replaceAll((ignored, old) -> VictimState.CANCEL_IN_FLIGHT);
        state = State.CANCEL_IN_FLIGHT;
        return true;
    }

    public synchronized boolean recordAccepted(String requestId) {
        if (requireVictim(requestId) != VictimState.CANCEL_IN_FLIGHT) {
            return false;
        }
        victimStates.put(requestId, VictimState.CANCEL_REQUESTED);
        return true;
    }

    public synchronized boolean recordNotFound(String requestId) {
        VictimState current = requireVictim(requestId);
        if (current != VictimState.CANCEL_IN_FLIGHT) {
            return false;
        }
        victimStates.put(requestId, VictimState.NOT_FOUND_STALE);
        return true;
    }

    public synchronized boolean recordUnknown(String requestId) {
        VictimState current = requireVictim(requestId);
        if (current != VictimState.CANCEL_IN_FLIGHT
                && current != VictimState.CANCEL_REQUESTED) {
            return false;
        }
        victimStates.put(requestId, VictimState.CANCEL_UNKNOWN);
        return true;
    }

    /** Engine absence plus an atomic late-enqueue fence is terminal proof. */
    public synchronized boolean recordTombstoned(String requestId) {
        if (requireVictim(requestId) != VictimState.CANCEL_IN_FLIGHT) {
            return false;
        }
        victimStates.put(requestId, VictimState.CANCELED_SETTLED);
        return true;
    }

    public synchronized void beginCanceledWait() {
        if (state != State.CANCEL_IN_FLIGHT) {
            throw new IllegalStateException("cannot wait for CANCELED from " + state);
        }
        state = State.WAITING_CANCELED;
        advanceReadyIfSettled();
    }

    /** Exactly-once typed-CANCELED settlement for one victim. */
    public synchronized boolean recordCanceled(String requestId) {
        VictimState current = requireVictim(requestId);
        if (current != VictimState.CANCEL_REQUESTED
                && current != VictimState.CANCEL_UNKNOWN) {
            return false;
        }
        victimStates.put(requestId, VictimState.CANCELED_SETTLED);
        advanceReadyIfSettled();
        return true;
    }

    public synchronized boolean hasNonCommittedOutcome() {
        return victimStates.values().stream().anyMatch(value ->
                value == VictimState.NOT_FOUND_STALE
                        || value == VictimState.CANCEL_UNKNOWN);
    }

    public synchronized boolean markCommitted() {
        if (state != State.READY_COMMIT) {
            return false;
        }
        state = State.COMMITTED;
        return true;
    }

    public synchronized void markAborted(boolean controlFailed) {
        if (state != State.COMMITTED) {
            state = controlFailed ? State.CONTROL_FAILED : State.ABORTED;
        }
    }

    private VictimState requireVictim(String requestId) {
        VictimState state = victimStates.get(requestId);
        if (state == null) {
            throw new IllegalArgumentException("request is not a victim: " + requestId);
        }
        return state;
    }

    private void advanceReadyIfSettled() {
        if (state == State.WAITING_CANCELED
                && victimStates.values().stream()
                    .allMatch(value -> value == VictimState.CANCELED_SETTLED)) {
            state = State.READY_COMMIT;
        }
    }
}

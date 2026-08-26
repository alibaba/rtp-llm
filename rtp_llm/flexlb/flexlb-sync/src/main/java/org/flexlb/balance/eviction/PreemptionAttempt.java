package org.flexlb.balance.eviction;

import org.flexlb.balance.preemption.CancelTarget;
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
        WAITING_TERMINAL,
        READY_COMMIT,
        COMMITTED,
        ABORTED
    }

    public enum VictimState {
        PLANNED,
        CLAIMED,
        CANCEL_IN_FLIGHT,
        CANCEL_REQUESTED,
        TERMINAL,
        NOT_FOUND_STALE,
        CANCEL_UNKNOWN
    }

    public record Victim(long requestId,
                         int priority,
                         long kvTokens,
                         DecodeTaskPhase phase,
                         long reservationToken,
                         CancelTarget target) {
        public Victim {
            if (requestId <= 0 || reservationToken <= 0) {
                throw new IllegalArgumentException(
                        "victim requestId and reservation token must be positive");
            }
            if (phase == null || !phase.requiresEngineCancel()) {
                throw new IllegalArgumentException("victim must require Engine Cancel");
            }
        }
    }

    private final long token;
    private final List<Victim> victims;
    private final Map<Long, VictimState> victimStates;
    private State state = State.PLANNED;

    public PreemptionAttempt(long token, List<Victim> victims) {
        if (token <= 0 || victims == null || victims.isEmpty()) {
            throw new IllegalArgumentException("token and victims are required");
        }
        this.token = token;
        this.victims = List.copyOf(victims);
        this.victimStates = new LinkedHashMap<>();
        for (Victim victim : this.victims) {
            if (this.victimStates.putIfAbsent(
                    victim.requestId(), VictimState.PLANNED) != null) {
                throw new IllegalArgumentException("duplicate victim " + victim.requestId());
            }
        }
    }

    public long token() { return token; }
    public List<Victim> victims() { return victims; }

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

    public synchronized boolean recordAccepted(long requestId) {
        VictimState current = requireVictim(requestId);
        if (current == VictimState.TERMINAL) {
            return true;
        }
        if (current != VictimState.CANCEL_IN_FLIGHT) {
            return false;
        }
        victimStates.put(requestId, VictimState.CANCEL_REQUESTED);
        return true;
    }

    public synchronized boolean recordNotFound(long requestId) {
        VictimState current = requireVictim(requestId);
        if (current == VictimState.TERMINAL) {
            return true;
        }
        if (current != VictimState.CANCEL_IN_FLIGHT) {
            return false;
        }
        victimStates.put(requestId, VictimState.NOT_FOUND_STALE);
        return true;
    }

    public synchronized boolean recordUnknown(long requestId) {
        VictimState current = requireVictim(requestId);
        if (current == VictimState.TERMINAL) {
            return true;
        }
        if (current != VictimState.CANCEL_IN_FLIGHT
                && current != VictimState.CANCEL_REQUESTED) {
            return false;
        }
        victimStates.put(requestId, VictimState.CANCEL_UNKNOWN);
        return true;
    }

    public synchronized void beginTerminalWait() {
        if (state != State.CANCEL_IN_FLIGHT) {
            throw new IllegalStateException("cannot wait for CANCELED from " + state);
        }
        state = State.WAITING_TERMINAL;
        advanceReadyIfSettled();
    }

    /** Exactly-once canonical terminal settlement for one victim. */
    public synchronized boolean recordTerminal(long requestId) {
        VictimState current = requireVictim(requestId);
        if (current == VictimState.TERMINAL) {
            return true;
        }
        if (current != VictimState.CANCEL_IN_FLIGHT
                && current != VictimState.CANCEL_REQUESTED
                && current != VictimState.CANCEL_UNKNOWN
                && current != VictimState.NOT_FOUND_STALE) {
            return false;
        }
        victimStates.put(requestId, VictimState.TERMINAL);
        advanceReadyIfSettled();
        return true;
    }

    public synchronized boolean markCommitted() {
        if (state != State.READY_COMMIT) {
            return false;
        }
        state = State.COMMITTED;
        return true;
    }

    public synchronized boolean isTerminal(long requestId) {
        return requireVictim(requestId) == VictimState.TERMINAL;
    }

    public synchronized boolean allVictimsTerminal() {
        return victimStates.values().stream()
                .allMatch(value -> value == VictimState.TERMINAL);
    }

    public synchronized void markAborted() {
        if (state != State.COMMITTED) {
            state = State.ABORTED;
        }
    }

    private VictimState requireVictim(long requestId) {
        VictimState state = victimStates.get(requestId);
        if (state == null) {
            throw new IllegalArgumentException("request is not a victim: " + requestId);
        }
        return state;
    }

    private void advanceReadyIfSettled() {
        if (state == State.WAITING_TERMINAL
                && victimStates.values().stream()
                    .allMatch(value -> value == VictimState.TERMINAL)) {
            state = State.READY_COMMIT;
        }
    }
}

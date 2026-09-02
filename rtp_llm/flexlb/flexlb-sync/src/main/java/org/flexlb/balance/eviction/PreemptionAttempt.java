package org.flexlb.balance.eviction;

import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.enums.DecodeTaskPhase;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * One decode preemption transaction and its resource-ownership ledger.
 *
 * <p>The object owns only state; RPC orchestration lives in
 * {@link DecodePreemptionCoordinator}.  Every mutation is serialized here so
 * a Cancel callback and a WorkerStatus callback can converge without applying
 * a victim settlement twice.</p>
 */
public final class PreemptionAttempt {

    private enum AttemptPhase {
        PLANNED,
        CLAIMED,
        CANCEL_IN_FLIGHT,
        WAITING_TERMINAL,
        READY_COMMIT,
        COMMITTED,
        ABORTED
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
    private final Set<Long> victimIds;
    /** Victim outcomes are canonical elsewhere; this attempt needs only convergence. */
    private final Set<Long> terminalVictims = new LinkedHashSet<>();
    private AttemptPhase phase = AttemptPhase.PLANNED;
    /** Historical outbound boundary retained even if the aggregate later aborts. */
    private boolean cancelStarted;

    public PreemptionAttempt(long token, List<Victim> victims) {
        if (token <= 0 || victims == null || victims.isEmpty()) {
            throw new IllegalArgumentException("token and victims are required");
        }
        this.token = token;
        this.victims = List.copyOf(victims);
        this.victimIds = new LinkedHashSet<>();
        for (Victim victim : this.victims) {
            if (!this.victimIds.add(victim.requestId())) {
                throw new IllegalArgumentException("duplicate victim " + victim.requestId());
            }
        }
    }

    public long token() { return token; }
    public List<Victim> victims() { return victims; }

    public synchronized boolean claimAll() {
        if (phase != AttemptPhase.PLANNED) {
            return false;
        }
        phase = AttemptPhase.CLAIMED;
        return true;
    }

    /** Linearization immediately before the first outbound Cancel RPC. */
    public synchronized boolean markCancelInFlight() {
        if (phase != AttemptPhase.CLAIMED) {
            return false;
        }
        cancelStarted = true;
        phase = AttemptPhase.CANCEL_IN_FLIGHT;
        return true;
    }

    public synchronized void beginTerminalWait() {
        if (phase != AttemptPhase.CANCEL_IN_FLIGHT) {
            throw new IllegalStateException(
                    "cannot wait for CANCELED from " + phase);
        }
        phase = AttemptPhase.WAITING_TERMINAL;
        advanceReadyIfSettled();
    }

    /** Exactly-once terminal convergence after Cancel may have gone outbound. */
    public synchronized boolean recordTerminal(long requestId) {
        requireVictim(requestId);
        if (terminalVictims.contains(requestId)) {
            return true;
        }
        if (!cancelStarted) {
            return false;
        }
        terminalVictims.add(requestId);
        advanceReadyIfSettled();
        return true;
    }

    public synchronized boolean markCommitted() {
        if (phase != AttemptPhase.READY_COMMIT) {
            return false;
        }
        phase = AttemptPhase.COMMITTED;
        return true;
    }

    public synchronized boolean isTerminal(long requestId) {
        requireVictim(requestId);
        return terminalVictims.contains(requestId);
    }

    public synchronized boolean allVictimsTerminal() {
        return terminalVictims.size() == victimIds.size();
    }

    public synchronized void markAborted() {
        if (phase != AttemptPhase.COMMITTED) {
            phase = AttemptPhase.ABORTED;
        }
    }

    private void requireVictim(long requestId) {
        if (!victimIds.contains(requestId)) {
            throw new IllegalArgumentException("request is not a victim: " + requestId);
        }
    }

    private void advanceReadyIfSettled() {
        if (phase == AttemptPhase.WAITING_TERMINAL
                && terminalVictims.size() == victimIds.size()) {
            phase = AttemptPhase.READY_COMMIT;
        }
    }
}

package org.flexlb.balance.scheduler;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Partial-order adjudication layer for refined slot phases
 * (plan 3.1 item 3, v2 design S4 / section 2.2 ruling matrix).
 *
 * <p>This is the single ruling entrance that the ~56 scattered per-domain
 * enums will converge onto in later stages.  The layer itself is a pure,
 * lock-free, side-effect-free function: it never mutates a
 * {@link RequestSlot}, never touches endpoint queues, and is safe to call
 * under or outside the slot monitor.  Adapters that feed domain facts in
 * (and apply the resulting verdicts) arrive with the enum migration; this
 * stage delivers the ruling core plus its contract tests.</p>
 *
 * <h2>Refined phase lattice (v2 2.1)</h2>
 *
 * <pre>
 *   INIT → ROUTED → QUEUED → DISPATCHING → DISPATCHED → P_RECEIVED
 *        → P_WAITING_UNLOADED → P_WAITING_LOADED → P_RUNNING
 *        → PREFILL_DONE → D_LOADING → D_RUNNING
 *        ┆ (all intermediates imply) ┆
 *   COMPLETED | CANCELLED | SLO_TIMEOUT | FAILED   (pairwise-incomparable
 *                                                    lattice maxima)
 * </pre>
 *
 * <p>{@code KV_TRANSFERRING} is deliberately absent: v2 S9 declares it a
 * master-side <em>derived view</em> (P running ∧ D reported), derived views
 * never take part in event ruling, so the adjudicator must never see it.</p>
 *
 * <h2>Ruling matrix (v2 2.2)</h2>
 *
 * <ol>
 *   <li><b>Version LWW</b> — an event whose version is below the last seen
 *       version (or a same-tick duplicate) is discarded
 *       ({@code DiscardStaleVersion}, dedup counter bucket).</li>
 *   <li><b>Monotonic drop-late</b> — an event indicating a phase the state
 *       already implies is a late intermediate and is discarded
 *       ({@code DiscardLateEvent}, late-event counter bucket).</li>
 *   <li><b>Implication-closure advance</b> — an event indicating a higher
 *       phase advances the state directly, recording the skipped waypoints
 *       (their {@code enteredAt} stamps are back-filled by the adapter);
 *       e.g. a D event arriving while parked on DISPATCHED jumps straight
 *       to {@code D_LOADING} (v2 appendix L9).</li>
 *   <li><b>Same-tick finish precedence</b> — when a finish collides with an
 *       already-applied intermediate of the <em>same</em> version, finish
 *       wins ({@code Advance#finishWins()} == true; safety net + WARN per
 *       v2 S4 — with the engine single-lock snapshot contract E6 this never
 *       fires in a healthy engine).</li>
 * </ol>
 *
 * <p>Two orthogonal rulings complete the matrix: a terminal state absorbs
 * every later event ({@code TombstoneAbsorb}, v2 I3 — never resurrect, never
 * settle twice) and an event from another generation is rejected before any
 * ordering judgement ({@code RejectCrossGeneration}, v2 S8 — generation
 * checks are orthogonal to out-of-order delivery).</p>
 *
 * <h2>Invariants</h2>
 *
 * <ul>
 *   <li>I1 — the ruling result only ever moves the phase forward
 *       (retries are new identities, v2 S6).</li>
 *   <li>I2 — implication consistency: any {@code D_*} ruling carries
 *       {@code PREFILL_DONE} and {@code DISPATCHED} in its skipped-waypoint
 *       closure; a violation is ledger corruption, reported loudly rather
 *       than silently repaired.</li>
 *   <li>I3 — terminal closure: once terminal, only
 *       {@code TombstoneAbsorb} can be ruled.</li>
 * </ul>
 *
 * <h2>Attachment to the coarse SlotPhase{3} track</h2>
 *
 * <p>{@link CoarsePhase} is the projection of the package-visible
 * {@code RequestSlot.SlotPhase} three-state track
 * (ACTIVE / TERMINALIZING / TOMBSTONE) — see
 * {@link #coarsePhaseOf(RequestSlot.SlotPhase)}.  Intermediate refined
 * phases live in {@code ACTIVE}; a refined terminal places the slot onto
 * the {@code TERMINALIZING} track (plan 3.2 two-phase death), and TOMBSTONE
 * itself is owned by the tombstone installation channel (v2 S7 retention),
 * never by event ruling.  The projection is consumed for real by the M1
 * ledger-reconciliation harness: every slot audit captures its phase via
 * this layer, so the coarse track is not a parallel enum but the ruling
 * entrance the reconciliation surface sees.</p>
 */
public final class SlotPhaseAdjudicator {

    private SlotPhaseAdjudicator() {
    }

    /**
     * Rules one incoming protocol fact against the adjudication state.
     *
     * <p>Ruling precedence (v2 2.2 order): cross-generation reject →
     * terminal closure → finish handling (same-tick precedence) → version
     * LWW → monotonic drop-late → implication-closure advance.</p>
     *
     * @param current adjudication state (seen version + current phase)
     * @param event   incoming protocol fact
     * @return the ruling; never null
     */
    public static Verdict adjudicate(State current, Event event) {
        Objects.requireNonNull(current, "current");
        Objects.requireNonNull(event, "event");
        if (event.generationId() != current.generationId()) {
            return new Verdict.RejectCrossGeneration(
                    current.generationId(), event.generationId());
        }
        if (current.phase().isTerminal()) {
            return new Verdict.TombstoneAbsorb(
                    current.phase(), event.indicatedPhase());
        }
        if (event.finish()) {
            if (event.version() < current.seenVersion()) {
                return new Verdict.DiscardStaleVersion(
                        event.version(), current.seenVersion());
            }
            boolean finishWinsSameTick = event.version() == current.seenVersion();
            return advance(current, event, finishWinsSameTick);
        }
        if (event.version() <= current.seenVersion()) {
            return new Verdict.DiscardStaleVersion(
                    event.version(), current.seenVersion());
        }
        if (!event.indicatedPhase().exceeds(current.phase())) {
            return new Verdict.DiscardLateEvent(
                    current.phase(), event.indicatedPhase());
        }
        return advance(current, event, false);
    }

    private static Verdict advance(State current, Event event, boolean finishWins) {
        State next = new State(
                current.generationId(), event.version(), event.indicatedPhase());
        return new Verdict.Advance(
                current.phase(),
                next,
                skippedWaypoints(current.phase(), event.indicatedPhase()),
                finishWins);
    }

    /**
     * Chain-open-interval waypoints between {@code from} (exclusive) and
     * {@code to} (exclusive).  Terminal phases are never waypoints: a jump
     * straight to a lattice maximum skips every remaining intermediate.
     */
    private static List<RefinedPhase> skippedWaypoints(
            RefinedPhase from, RefinedPhase to) {
        List<RefinedPhase> skipped = new ArrayList<>();
        for (RefinedPhase waypoint : RefinedPhase.values()) {
            if (waypoint.isTerminal()) {
                continue;
            }
            if (waypoint.exceeds(from) && to.exceeds(waypoint)) {
                skipped.add(waypoint);
            }
        }
        return List.copyOf(skipped);
    }

    /**
     * Refined protocol phases of the v2 2.1 lattice.  Declaration order is
     * the intermediate chain order; the four terminals are pairwise
     * incomparable maxima.
     */
    public enum RefinedPhase {
        INIT,
        ROUTED,
        QUEUED,
        DISPATCHING,
        DISPATCHED,
        P_RECEIVED,
        P_WAITING_UNLOADED,
        P_WAITING_LOADED,
        P_RUNNING,
        PREFILL_DONE,
        D_LOADING,
        D_RUNNING,
        COMPLETED,
        CANCELLED,
        SLO_TIMEOUT,
        FAILED;

        /**
         * v2 implication {@code this ⇒ other}: reaching {@code this}
         * phase implies {@code other} has already been passed.  A later
         * intermediate implies every strictly earlier intermediate; a
         * terminal implies every intermediate (v2 I2: "D_* ⇒ PREFILL_DONE
         * ⇒ DISPATCHED" and "任一终态 ⇒ 全部关联资源统一释放"); an
         * intermediate implies no terminal (reaching PREFILL_DONE does not
         * imply COMPLETED); terminals are pairwise-incomparable (lattice
         * maxima).
         */
        public boolean implies(RefinedPhase other) {
            Objects.requireNonNull(other, "other");
            if (this == other) {
                return true;
            }
            if (this.isTerminal()) {
                return !other.isTerminal();
            }
            return !other.isTerminal()
                    && other.ordinal() < this.ordinal();
        }

        /** Strict lattice order {@code this > other}: exactly one
         *  advancement step (adjacent or closure jump) or a finish. */
        public boolean exceeds(RefinedPhase other) {
            Objects.requireNonNull(other, "other");
            return this.implies(other) && this != other;
        }

        /** Lattice maxima: COMPLETED / CANCELLED / SLO_TIMEOUT / FAILED. */
        public boolean isTerminal() {
            return this == COMPLETED || this == CANCELLED
                    || this == SLO_TIMEOUT || this == FAILED;
        }
    }

    /**
     * Coarse three-state projection of the package-visible
     * {@code RequestSlot.SlotPhase} track — the attachment point of the
     * refined lattice onto the slot lifecycle.
     */
    public enum CoarsePhase {
        ACTIVE,
        TERMINALIZING,
        TOMBSTONE
    }

    /**
     * Coarse-track projection of one refined phase.  A refined terminal
     * places the slot onto the TERMINALIZING track; TOMBSTONE is reached
     * only through the tombstone installation channel (v2 S7), which the
     * adjudicator never rules on.
     */
    public static CoarsePhase coarsePhaseOf(RefinedPhase refined) {
        Objects.requireNonNull(refined, "refined");
        return refined.isTerminal()
                ? CoarsePhase.TERMINALIZING
                : CoarsePhase.ACTIVE;
    }

    /**
     * Coarse-track projection of the live {@code RequestSlot.SlotPhase}
     * storage/cleanup track.  The mapping is the identity on the three
     * states; routing it through this layer (rather than letting consumers
     * switch on the slot enum directly) keeps the adjudicator the single
     * ruling entrance the plan requires, and makes the refined-lattice
     * and the storage-track projections provably agree (a refined
     * terminal maps to TERMINALIZING exactly as the storage track does
     * after {@code beginTerminalizing}).
     */
    public static CoarsePhase coarsePhaseOf(RequestSlot.SlotPhase slotPhase) {
        Objects.requireNonNull(slotPhase, "slotPhase");
        return switch (slotPhase) {
            case ACTIVE -> CoarsePhase.ACTIVE;
            case TERMINALIZING -> CoarsePhase.TERMINALIZING;
            case TOMBSTONE -> CoarsePhase.TOMBSTONE;
        };
    }

    /**
     * Adjudication state: generation identity, last seen event version and
     * the current refined phase.  The seen version only ever grows
     * (advances adopt the event version; discards keep the state).
     */
    public record State(long generationId, long seenVersion, RefinedPhase phase) {

        public static State initial(long generationId) {
            return new State(generationId, 0L, RefinedPhase.INIT);
        }

        public State {
            Objects.requireNonNull(phase, "phase");
        }
    }

    /**
     * One protocol fact offered for adjudication.  A finish fact must carry
     * a terminal phase and an intermediate fact must carry a non-terminal
     * phase; the generation identity is the cross-generation guard input
     * (v2 S8 — reservationToken + endpointGenerationId + attemptToken in
     * the full wiring; a plain long here keeps the ruling core testable in
     * isolation).
     */
    public record Event(
            long generationId,
            long version,
            RefinedPhase indicatedPhase,
            boolean finish) {

        public Event {
            Objects.requireNonNull(indicatedPhase, "indicatedPhase");
            if (indicatedPhase.isTerminal() != finish) {
                throw new IllegalArgumentException(
                        "finish facts must indicate a terminal refined phase and"
                                + " intermediate facts a non-terminal one, got "
                                + indicatedPhase + " with finish=" + finish);
            }
        }
    }

    /** Immutable ruling over one event. */
    public sealed interface Verdict permits
            Verdict.Advance,
            Verdict.DiscardStaleVersion,
            Verdict.DiscardLateEvent,
            Verdict.RejectCrossGeneration,
            Verdict.TombstoneAbsorb {

        /** Whether the ruling moves the adjudication state forward. */
        boolean advancesState();

        /**
         * Rule 3 (+ rule 4): implication-closure advance to a higher phase.
         * {@code skippedWaypoints} lists the chain intermediates crossed by
         * the jump (for back-filled enteredAt stamps); {@code finishWins}
         * marks a same-tick finish overriding an already-applied
         * intermediate of the same version (rule 4, WARN-worthy safety
         * net).
         */
        record Advance(
                RefinedPhase from,
                State nextState,
                List<RefinedPhase> skippedWaypoints,
                boolean finishWins) implements Verdict {

            public Advance {
                Objects.requireNonNull(from, "from");
                Objects.requireNonNull(nextState, "nextState");
                Objects.requireNonNull(skippedWaypoints, "skippedWaypoints");
            }

            @Override
            public boolean advancesState() {
                return true;
            }

            /** Target refined phase of this advance. */
            public RefinedPhase to() {
                return nextState.phase();
            }
        }

        /**
         * Rule 1: version LWW — an older or same-version duplicate event
         * is discarded (dedup counter bucket).
         */
        record DiscardStaleVersion(long eventVersion, long seenVersion)
                implements Verdict {

            @Override
            public boolean advancesState() {
                return false;
            }
        }

        /**
         * Rule 2: monotonic drop-late — the event indicates a phase the
         * state already implies (late intermediate, late-event counter
         * bucket).
         */
        record DiscardLateEvent(
                RefinedPhase currentPhase,
                RefinedPhase indicatedPhase) implements Verdict {

            public DiscardLateEvent {
                Objects.requireNonNull(currentPhase, "currentPhase");
                Objects.requireNonNull(indicatedPhase, "indicatedPhase");
            }

            @Override
            public boolean advancesState() {
                return false;
            }
        }

        /**
         * Orthogonal guard (v2 S8): the event belongs to another
         * generation — rejected before any ordering judgement.
         */
        record RejectCrossGeneration(long stateGeneration, long eventGeneration)
                implements Verdict {

            @Override
            public boolean advancesState() {
                return false;
            }
        }

        /**
         * Terminal closure (v2 I3): a terminal state absorbs every later
         * event — no resurrection, no second settlement; late facts go to
         * tombstone audit only.
         */
        record TombstoneAbsorb(
                RefinedPhase terminalPhase,
                RefinedPhase indicatedPhase) implements Verdict {

            public TombstoneAbsorb {
                Objects.requireNonNull(terminalPhase, "terminalPhase");
                Objects.requireNonNull(indicatedPhase, "indicatedPhase");
            }

            @Override
            public boolean advancesState() {
                return false;
            }
        }
    }
}

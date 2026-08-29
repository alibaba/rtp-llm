package org.flexlb.balance.endpoint;

/**
 * Stage-2 T7 S3 port: the slot-side host of the decode-admission
 * authority (placement-domain migration).
 *
 * <p>The decode admission sub-state — the reservation fence
 * (endpoint + generation + token), the preloaded numeric row and the
 * queued / dispatch-permit / engine-lifecycle bits — moves its
 * <em>authority</em> onto the {@code RequestSlot} (slot monitor).  The
 * layer-1 {@code RequestInflight} entry flags and the endpoint's O(1)
 * aggregate counters remain as mirrors that production readers keep
 * consuming until the S2 read-source switch; this port is the channel
 * through which every admission flip updates the authority.
 *
 * <h2>Lock discipline (ruling 3 / S3 approval point 1)</h2>
 *
 * <p>When the two monitors nest, the lock order is strictly
 * <b>slot monitor &#8594; endpoint admissionLock</b>, one-way.  No path
 * may take a slot monitor while holding the admissionLock.  The flip
 * order inside {@link #executeUnderDecodeAdmission} is anchored as:
 * <b>the slot monitor flips the authority first (projection), then the
 * admission body flips the layer-1 mirror plus the counters.</b>  The
 * wrapper holds the slot monitor across the admission body precisely so
 * projections for one request serialize with their commits; the body
 * itself must not notify capacity listeners or take any other slot
 * monitor (both belong strictly outside this critical section).
 *
 * <h2>Removal-side clears (projection-lag)</h2>
 *
 * <p>Layer-1 <em>removal</em> sites (rollback, local-shadow release,
 * settle, abort, calibration, TTL eviction, local eviction victims) run
 * inside admissionLock transactions and cannot take a slot monitor
 * there.  They collect the removed reservation fences during the
 * transaction and deliver {@link #clearDecodeAdmission} after the
 * transaction commits — the authority lags the mirror by that &#181;s
 * window, which the mirror-consistency reconciliation rule classifies
 * through its confirm window.  Slot-side death paths (decode
 * acceptance, terminalizing, tombstone) clear the authority
 * unconditionally inside their own monitor tick, so a stale authority
 * can never outlive its slot.
 */
public interface DecodePlacementAuthorityPort {

    /**
     * Run one admission flip under the slot monitor's authority
     * projection.
     *
     * <p>Implementation contract: look up the request slot; when no
     * current ACTIVE slot hosts the request, run {@code body} bare (the
     * layer-1 flip keeps its exact legacy semantics).  Otherwise, inside
     * {@code synchronized (slot)}: stage the projection (fence-guarded,
     * snapshot the prior authority), run {@code body}, and on success
     * refresh the authority from the layer-1 entry state the
     * {@code entryReader} captures (absent entry &#8658; clear; foreign
     * fence &#8658; restore the prior snapshot).  On any {@code body}
     * failure the prior authority snapshot is restored before the
     * failure propagates.
     *
     * @param <T>        flip result type
     * @param requestId  exact request identity
     * @param projection target authority state (fence + sub-state)
     * @param body       the admission transaction (takes the admissionLock
     *                   itself; must not take slot monitors)
     * @param entryReader lock-free reader of the current layer-1 entry
     *                   sub-state (null entry when absent)
     * @return the body result
     */
    <T> T executeUnderDecodeAdmission(
            long requestId,
            Projection projection,
            AdmissionFlipBody<T> body,
            EntryReader entryReader);

    /**
     * Clear the slot-side decode-admission authority of one exact
     * reservation fence.  Fence-guarded and idempotent: a no-op when the
     * slot hosts no authority, a different fence (a newer reservation
     * already replaced it), or no current slot at all.  Called only
     * after the layer-1 removal transaction committed (projection-lag
     * delivery).
     *
     * @param requestId         exact request identity
     * @param endpoint          endpoint identity of the fence
     * @param endpointGeneration endpoint generation of the fence
     * @param reservationToken  reservation token of the fence
     */
    void clearDecodeAdmission(
            long requestId,
            DecodeEndpoint endpoint,
            long endpointGeneration,
            long reservationToken);

    /**
     * Post-commit projection delivery for admission transactions whose
     * full fence (or flip outcome) is only known inside the admissionLock
     * transaction — the dispatch-permit acquisition (the reservation
     * fence is the entry's, read under the lock) and the embedded
     * reservation installs of priority-preemption begin and local
     * eviction-and-reserve. The layer-1 mutation already committed when
     * this is called, strictly after the admissionLock released; the µs
     * projection-lag window is the same class the mirror-consistency
     * reconciliation rule absorbs through its confirm window.
     *
     * <p>Fence-guarded on the slot side: an install projection overwrites
     * (a newer reservation is the newer fact), a flip projection installs
     * only on a fence match. No-op when no current ACTIVE slot hosts the
     * request.
     *
     * @param requestId  exact request identity
     * @param projection the committed projection (install or flip)
     */
    void deliverDecodeAdmissionAfterCommit(
            long requestId,
            Projection projection);

    /** One admission transaction; runs inside the wrapper's slot tick. */
    @FunctionalInterface
    interface AdmissionFlipBody<T> {
        T run();
    }

    /** Lock-free reader of the current layer-1 entry sub-state. */
    @FunctionalInterface
    interface EntryReader {
        DecodeAdmissionEntry read();
    }

    /**
     * Snapshot of one layer-1 inflight entry's fence and sub-state
     * (the reader returns {@code null} when the entry is absent).
     */
    record DecodeAdmissionEntry(
            long reservationToken,
            boolean masterQueued,
            long dispatchPermitToken,
            boolean engineLifecycleOwned) {
    }

    /**
     * Target slot-side authority state for one admission flip.
     *
     * <p>The fence triple (endpoint identity, endpoint generation,
     * reservation token) names one exact layer-1 reservation: tokens are
     * endpoint-generation-local and monotonic, so two endpoints may mint
     * the same token value — the endpoint reference disambiguates.
     * Tokens are pre-allocated before the wrapper opens (AtomicLong) so
     * the projection can carry the full fence ahead of the admission
     * body; a flip that never commits merely burns a gap.
     *
     * <p>{@code install} carries the preloaded numeric row
     * (kvTokens / expectedKvTokens / priority — ruling 2(a): an
     * independent authority field, never the slot's pRow, so the
     * {@code item == null &#8658; rows null} invariant stays intact).
     * The preload row lives only until the publication bind replaces it
     * with the real pRow.  Aspect {@code flip} projections carry only
     * the sub-state bits.
     */
    record Projection(
            DecodeEndpoint endpoint,
            long endpointGeneration,
            long reservationToken,
            boolean install,
            long kvTokens,
            long expectedKvTokens,
            int priority,
            boolean masterQueued,
            long dispatchPermitToken,
            boolean engineLifecycleOwned) {

        /** Reserve install: full fence + preload numeric row + initial bits. */
        public static Projection install(
                DecodeEndpoint endpoint,
                long endpointGeneration,
                long reservationToken,
                long kvTokens,
                long expectedKvTokens,
                int priority,
                boolean masterQueued) {
            return new Projection(
                    endpoint, endpointGeneration, reservationToken,
                    true, kvTokens, expectedKvTokens, priority,
                    masterQueued, 0L, false);
        }

        /** Aspect flip: fence + target sub-state bits (no numerics). */
        public static Projection flip(
                DecodeEndpoint endpoint,
                long endpointGeneration,
                long reservationToken,
                boolean masterQueued,
                long dispatchPermitToken,
                boolean engineLifecycleOwned) {
            return new Projection(
                    endpoint, endpointGeneration, reservationToken,
                    false, 0L, 0L, 0,
                    masterQueued, dispatchPermitToken, engineLifecycleOwned);
        }
    }
}

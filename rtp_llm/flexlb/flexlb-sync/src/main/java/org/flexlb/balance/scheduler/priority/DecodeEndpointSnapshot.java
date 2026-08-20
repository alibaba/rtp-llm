package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.enums.DecodeTaskPhase;

import java.util.ArrayList;
import java.util.List;

/**
 * Read-only view of a decode endpoint at snapshot time (design doc 10.2).
 *
 * <p><b>Consistency:</b> the ({@link #admissionVersion}, {@link #reserved},
 * {@link #accepted}, {@link #running}) tuple is captured atomically under the
 * endpoint admission lock ({@code DecodeEndpoint.layeredAdmissionView()}), so
 * "version unchanged at commit time" implies the layered view is still
 * current. The remaining KV / load fields are weakly-consistent reads; any
 * interleaved admission mutation bumps {@code admissionVersion} and the
 * optimistic commit-time re-check (design doc 11.5/17.2) rejects the plan, so
 * staleness only causes a retry, never a wrong commit.
 *
 * <p><b>Layered view (Phase 5):</b> {@link #reserved} holds only
 * Master-queued and Engine-may-have-seen shadow entries; {@link #accepted} holds
 * engine-confirmed {@code ACCEPTED_NOT_RUNNING} entries without a pending
 * cancel; {@link #running} holds engine-confirmed {@code RUNNING} entries
 * without a pending cancel. Both lists are eviction candidates only behind
 * the accepted-evict gate. {@code totalLoad} aggregates
 * confirmed running + reserved inflight counts.
 *
 * @param endpoint           the live endpoint (used at commit time only)
 * @param endpointId         endpoint key ("ip:httpPort")
 * @param admissionVersion   admission version at snapshot time
 * @param realKvAvailable    engine-reported available KV minus local hard reservations
 * @param realKvTotal        engine-reported total KV capacity
 * @param totalLoad          confirmed running + local inflight request count
 * @param engineLoad         engine-facing load: confirmed running + non-queued
 *                           inflight (N2/P1-3 — slot-deficit planning must use
 *                           the same measure as the concurrency gate)
 * @param concurrencyLimit   configured decode concurrency limit (0 = unlimited)
 * @param hardKvReserved     sum of reserved hard KV tokens (shadow, 10.2)
 * @param expectedKvReserved sum of reserved expected KV tokens (shadow, 10.2)
 * @param reserved           reserved (engine-unconfirmed) entry details for
 *                           eviction planning; confirmed requests never appear
 * @param accepted           engine-confirmed accepted-not-running entries with
 *                           no pending cancel (Phase 5 candidates behind gate)
 * @param running            engine-confirmed running entries with no pending
 *                           cancel (Phase 5 candidates behind the same gate)
 */
public record DecodeEndpointSnapshot(
        DecodeEndpoint endpoint,
        String endpointId,
        long admissionVersion,
        long realKvAvailable,
        long realKvTotal,
        int totalLoad,
        int engineLoad,
        long concurrencyLimit,
        long hardKvReserved,
        long expectedKvReserved,
        List<DecodeRequestSnapshot> reserved,
        List<DecodeRequestSnapshot> accepted,
        List<DecodeRequestSnapshot> running) {

    public static DecodeEndpointSnapshot capture(DecodeEndpoint endpoint, long concurrencyLimit) {
        // Atomic (version, layered views) tuple: capturing the views first and
        // the version afterwards could pair a stale view with a fresher
        // version and falsely pass the commit-time re-check.
        DecodeEndpoint.LayeredAdmissionView view = endpoint.layeredAdmissionView();
        List<DecodeRequestSnapshot> reserved = new ArrayList<>();
        view.reserved().forEach((requestId, entry) -> {
            if (view.claimed().contains(requestId)) {
                return;
            }
            boolean masterQueued = view.queued().contains(requestId);
            DecodeTaskPhase phase = masterQueued
                    ? DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED
                    : DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN;
            reserved.add(new DecodeRequestSnapshot(requestId, entry.priority(), phase,
                    entry.releasableKvTokens(), entry.expectedKvTokens(), true, masterQueued));
        });
        List<DecodeRequestSnapshot> accepted = new ArrayList<>();
        List<DecodeRequestSnapshot> running = new ArrayList<>();
        for (DecodeEndpoint.ConfirmedTaskView task : view.confirmed()) {
            // A cancel-requested confirmed entry is already claimed by an
            // in-flight eviction, regardless of whether it has started running.
            if (task.claimedForPreemption()) {
                continue;
            }
            if (task.phase() == DecodeTaskPhase.ACCEPTED_NOT_RUNNING) {
                accepted.add(toSnapshot(task));
            } else if (task.phase() == DecodeTaskPhase.RUNNING) {
                running.add(toSnapshot(task));
            }
        }
        return new DecodeEndpointSnapshot(
                endpoint,
                endpoint.ipPort(),
                view.admissionVersion(),
                endpoint.realKvAvailable(),
                endpoint.realKvTotal(),
                endpoint.getTotalLoad(),
                endpoint.getEngineLoad(),
                concurrencyLimit,
                endpoint.inflightHardKvReserved(),
                endpoint.inflightExpectedKvReserved(),
                List.copyOf(reserved),
                List.copyOf(accepted),
                List.copyOf(running));
    }

    private static DecodeRequestSnapshot toSnapshot(DecodeEndpoint.ConfirmedTaskView task) {
        // Confirmed KV is engine-owned; the tracked inputLength estimate serves
        // as both releasable and expected KV (no generation-growth estimate).
        return new DecodeRequestSnapshot(task.requestId(), task.priority(), task.phase(),
                task.kvTokens(), task.kvTokens(), task.priorityKnown(), false);
    }
}

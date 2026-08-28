package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.enums.DecodeTaskPhase;

import java.util.ArrayList;
import java.util.List;

/**
 * Read-only view of a decode endpoint at snapshot time (design doc 10.2).
 *
 * <p><b>Consistency:</b> victims and capacity values are captured under the
 * endpoint admission lock. Commit later presents the exact reservation
 * capabilities and capacity policy to the endpoint, which revalidates the
 * current canonical owners under that same lock before any mutation.
 *
 * <p><b>Layered view:</b> {@link #reserved} holds only
 * Master-queued and Engine-may-have-seen shadow entries; {@link #accepted} holds
 * engine-confirmed {@code ACCEPTED_NOT_RUNNING} entries without a pending
 * cancel; {@link #running} holds engine-confirmed {@code RUNNING} entries
 * without a pending cancel. Both lists are eviction candidates only behind
 * the accepted-evict gate.
 *
 * @param endpoint           the live endpoint (used at commit time only)
 * @param endpointId         endpoint key ("ip:httpPort")
 * @param realKvAvailable    engine-reported available KV minus local hard reservations
 * @param realKvTotal        engine-reported total KV capacity
 * @param engineLoad         engine-facing load: confirmed running + non-queued
 *                           inflight; slot-deficit planning uses the same
 *                           measure as the concurrency gate
 * @param concurrencyLimit   configured decode concurrency limit (0 = unlimited)
 * @param reserved           reserved (engine-unconfirmed) entry details for
 *                           eviction planning; confirmed requests never appear
 * @param accepted           engine-confirmed accepted-not-running entries with
 *                           no pending cancel when engine-owned eviction is enabled
 * @param running            engine-confirmed running entries with no pending
 *                           cancel when engine-owned eviction is enabled
 */
public record DecodeEndpointSnapshot(
        DecodeEndpoint endpoint,
        String endpointId,
        long realKvAvailable,
        long realKvTotal,
        int engineLoad,
        long concurrencyLimit,
        List<DecodeRequestView> reserved,
        List<DecodeRequestView> accepted,
        List<DecodeRequestView> running) {

    public static DecodeEndpointSnapshot capture(DecodeEndpoint endpoint, long concurrencyLimit) {
        DecodeEndpoint.LayeredAdmissionView view = endpoint.layeredAdmissionView();
        DecodeEndpoint.DecodeRoutingView routing = view.routing();
        List<DecodeRequestView> reserved = new ArrayList<>();
        view.reserved().forEach((requestId, entry) -> {
            if (entry.claimedForPreemption()) {
                return;
            }
            reserved.add(entry);
        });
        List<DecodeRequestView> accepted = new ArrayList<>();
        List<DecodeRequestView> running = new ArrayList<>();
        for (DecodeRequestView task : view.confirmed()) {
            // A cancel-requested confirmed entry is already claimed by an
            // in-flight eviction, regardless of whether it has started running.
            if (task.claimedForPreemption()) {
                continue;
            }
            if (task.phase() == DecodeTaskPhase.ACCEPTED_NOT_RUNNING) {
                accepted.add(task);
            } else if (task.phase() == DecodeTaskPhase.RUNNING) {
                running.add(task);
            }
        }
        return new DecodeEndpointSnapshot(
                endpoint,
                endpoint.ipPort(),
                routing.realKvAvailable(),
                routing.totalKv(),
                routing.engineLoad(),
                concurrencyLimit,
                List.copyOf(reserved),
                List.copyOf(accepted),
                List.copyOf(running));
    }

}

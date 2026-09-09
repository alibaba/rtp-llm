package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.enums.DecodeTaskPhase;

import java.util.ArrayList;
import java.util.List;

/** Victims and placement occupancy captured under one endpoint admission lock. */
public record DecodeEndpointSnapshot(
        DecodeEndpoint endpoint,
        String endpointId,
        DecodeEndpoint.AdmissionCapacity capacity,
        DecodeEndpoint.CapacityUsage usage,
        List<DecodeRequestView> reserved,
        List<DecodeRequestView> accepted,
        List<DecodeRequestView> running) {

    public static DecodeEndpointSnapshot capture(DecodeEndpoint endpoint, DecodeEndpoint.AdmissionCapacity policy) {
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
                policy,
                routing.placementUsage(),
                List.copyOf(reserved),
                List.copyOf(accepted),
                List.copyOf(running));
    }

}

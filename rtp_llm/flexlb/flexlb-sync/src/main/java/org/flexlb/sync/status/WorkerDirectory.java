package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Process-local owner of discovered worker generations and routable endpoint
 * generations.
 *
 * <p>A status entry describes the generation currently being polled. Routing
 * authority always comes from the matching endpoint generation in
 * {@link EndpointRegistry}; callers must capture a pin before using it. Keeping
 * both views behind this directory makes disagreement visible instead of
 * hiding it behind a derived capacity value.</p>
 */
@Component
public final class WorkerDirectory implements WorkerStatusProvider {

    private final Map<RoleType, Map<String, WorkerStatus>> statusesByRole;
    private final EndpointRegistry endpointRegistry;

    public WorkerDirectory(EndpointRegistry endpointRegistry) {
        this.endpointRegistry = endpointRegistry;
        EnumMap<RoleType, Map<String, WorkerStatus>> statuses =
                new EnumMap<>(RoleType.class);
        for (RoleType role : RoleType.values()) {
            statuses.put(role, new ConcurrentHashMap<>());
        }
        this.statusesByRole = Collections.unmodifiableMap(statuses);
    }

    /**
     * Mutable generation table used by the worker-sync transaction.
     *
     * <p>Mutation is intentionally limited to generation-aware sync code. A
     * routing or monitoring caller should use the snapshot/capture methods
     * below.</p>
     */
    public Map<String, WorkerStatus> statusMap(RoleType role) {
        if (role == null) {
            return Map.of();
        }
        return statusesByRole.get(role);
    }

    public int discoveredCount(RoleType role) {
        return statusMap(role).size();
    }

    public int discoveredCount() {
        int total = 0;
        for (Map<String, WorkerStatus> statuses : statusesByRole.values()) {
            total += statuses.size();
        }
        return total;
    }

    /** Capacity means currently published routing capacity, not discovery. */
    public int routingCapacity(RoleType role) {
        return role == null ? 0 : endpointRegistry.getEndpointCount(role);
    }

    /** Immutable, non-owning address snapshot for lazy selection. */
    public List<String> endpointAddressSnapshot(RoleType role) {
        return endpointRegistry.endpointAddressSnapshot(role);
    }

    /** Capture one exact currently published endpoint generation by address. */
    public WorkerEndpoint.GenerationPin captureEndpoint(
            RoleType role, String ipPort) {
        return endpointRegistry.capture(role, ipPort);
    }

    /** Immutable Decode routing values for one group. */
    public List<EndpointRegistry.DecodeRoutingSnapshot> decodeRoutingSnapshot(
            String group) {
        List<EndpointRegistry.DecodeRoutingSnapshot> snapshots =
                endpointRegistry.decodeRoutingSnapshot();
        if (group == null || snapshots.isEmpty()) {
            return snapshots;
        }
        ArrayList<EndpointRegistry.DecodeRoutingSnapshot> matching = null;
        for (int index = 0; index < snapshots.size(); index++) {
            EndpointRegistry.DecodeRoutingSnapshot snapshot = snapshots.get(index);
            if (!group.equals(snapshot.topology().group())) {
                if (matching == null) {
                    matching = new ArrayList<>(snapshots.size());
                    matching.addAll(snapshots.subList(0, index));
                }
            } else if (matching != null) {
                matching.add(snapshot);
            }
        }
        return matching == null
                ? snapshots : Collections.unmodifiableList(matching);
    }

    /** Pin the exact generation represented by a Decode routing snapshot. */
    public WorkerEndpoint.GenerationPin captureDecodeGeneration(
            EndpointRegistry.DecodeRoutingSnapshot expected) {
        return endpointRegistry.captureDecodeGeneration(expected);
    }

    /** Capture exact endpoint generations for one routing decision. */
    public List<WorkerEndpoint.GenerationPin> captureEndpoints(
            RoleType role, String group) {
        List<WorkerEndpoint.GenerationPin> captured = endpointRegistry.capture(role);
        List<WorkerEndpoint.GenerationPin> matching =
                new ArrayList<>(captured.size());
        try {
            for (int index = 0; index < captured.size(); index++) {
                WorkerEndpoint.GenerationPin pin = captured.get(index);
                WorkerStatus.TopologySnapshot topology =
                        pin.endpoint().getStatus().topologySnapshot();
                if (group != null && !group.equals(topology.group())) {
                    pin.close();
                    captured.set(index, null);
                    continue;
                }
                matching.add(pin);
                captured.set(index, null);
            }
            return matching;
        } catch (Throwable failure) {
            closePins(captured);
            closePins(matching);
            throw failure;
        }
    }

    /** Non-owning monitoring projection; no live endpoint map escapes. */
    public List<String> endpointAddresses(RoleType role, String group) {
        List<WorkerEndpoint.GenerationPin> captured = captureEndpoints(role, group);
        try {
            List<String> addresses = new ArrayList<>(captured.size());
            for (WorkerEndpoint.GenerationPin pin : captured) {
                addresses.add(pin.endpoint().ipPort());
            }
            return addresses;
        } finally {
            closePins(captured);
        }
    }

    @Override
    public List<String> getWorkerIpPorts(RoleType role, String group) {
        return endpointAddresses(role, group);
    }

    private static void closePins(List<WorkerEndpoint.GenerationPin> pins) {
        for (WorkerEndpoint.GenerationPin pin : pins) {
            if (pin != null) {
                pin.close();
            }
        }
    }
}

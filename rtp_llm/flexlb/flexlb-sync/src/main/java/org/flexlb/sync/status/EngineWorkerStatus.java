package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Component
public class EngineWorkerStatus {

    public static final ModelWorkerStatus MODEL_ROLE_WORKER_STATUS = new ModelWorkerStatus();

    private final EndpointRegistry endpointRegistry;

    public EngineWorkerStatus(EndpointRegistry endpointRegistry) {
        this.endpointRegistry = endpointRegistry;
    }

    /**
     * Return an immutable, non-owning address snapshot for lazy selection.
     *
     * <p>The addresses carry no generation authority.  A selector must obtain
     * an exact pin with {@link #captureModelWorkerEndpoint(RoleType, String)}
     * before reading generation-local routing state.</p>
     */
    public List<String> modelWorkerAddressSnapshot(RoleType roleType) {
        return endpointRegistry.endpointAddressSnapshot(roleType);
    }

    /** Capture one exact currently published endpoint generation by address. */
    public WorkerEndpoint.GenerationPin captureModelWorkerEndpoint(
            RoleType roleType, String ipPort) {
        return endpointRegistry.capture(roleType, ipPort);
    }

    /**
     * Immutable Decode routing values for one group. No endpoint or live map
     * reference escapes the registry.
     */
    public List<EndpointRegistry.DecodeRoutingSnapshot>
            decodeWorkerRoutingSnapshot(String group) {
        List<EndpointRegistry.DecodeRoutingSnapshot> snapshots =
                endpointRegistry.decodeRoutingSnapshot();
        if (group == null || snapshots.isEmpty()) {
            return snapshots;
        }
        ArrayList<EndpointRegistry.DecodeRoutingSnapshot> matching = null;
        for (int index = 0; index < snapshots.size(); index++) {
            EndpointRegistry.DecodeRoutingSnapshot snapshot =
                    snapshots.get(index);
            if (!group.equals(snapshot.topology().group())) {
                if (matching == null) {
                    matching = new ArrayList<>(snapshots.size());
                    matching.addAll(snapshots.subList(0, index));
                }
                continue;
            }
            if (matching != null) {
                matching.add(snapshot);
            }
        }
        if (matching == null) {
            return snapshots;
        }
        return Collections.unmodifiableList(matching);
    }

    /** Exact-capture and revalidate one previously observed Decode winner. */
    public WorkerEndpoint.GenerationPin captureCurrentDecodeWorker(
            EndpointRegistry.DecodeRoutingSnapshot expected) {
        return endpointRegistry.captureCurrentDecode(expected);
    }

    /**
     * Capture exact endpoint generations for one routing decision.
     *
     * <p>EndpointRegistry linearizes each capture with publication at the
     * corresponding address.  Group filtering happens while the returned pins
     * are already owned, so a same-address replacement can never be mistaken
     * for the selected generation.</p>
     */
    public List<WorkerEndpoint.GenerationPin> captureModelWorkerEndpoints(
            RoleType roleType, String group) {
        List<WorkerEndpoint.GenerationPin> captured =
                endpointRegistry.capture(roleType);
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
            for (WorkerEndpoint.GenerationPin pin : captured) {
                if (pin != null) {
                    pin.close();
                }
            }
            for (WorkerEndpoint.GenerationPin pin : matching) {
                pin.close();
            }
            throw failure;
        }
    }

    /** Non-owning monitoring projection; no live endpoint map escapes. */
    public List<String> modelWorkerAddresses(RoleType roleType, String group) {
        List<WorkerEndpoint.GenerationPin> captured =
                captureModelWorkerEndpoints(roleType, group);
        try {
            List<String> addresses = new ArrayList<>(captured.size());
            for (WorkerEndpoint.GenerationPin pin : captured) {
                addresses.add(pin.endpoint().ipPort());
            }
            return addresses;
        } finally {
            for (WorkerEndpoint.GenerationPin pin : captured) {
                pin.close();
            }
        }
    }

    public int getModelWorkerCapacity(RoleType roleType) {
        Map<String, WorkerStatus> roleStatusMap = MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(roleType);
        int statusCount = roleStatusMap == null ? 0 : roleStatusMap.size();
        return Math.max(statusCount, endpointRegistry.getEndpointCount(roleType));
    }

}

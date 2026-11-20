package org.flexlb.service.lease;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.PrefillLeaseConfig;
import org.flexlb.dao.master.PrefillChunkLease;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Changes chunk permission under the same worker lock used by chunk reservation.
 */
@Component
public final class PrefillLeaseManager {
    private final ConfigService configService;
    private final WorkerDirectory workers;

    public PrefillLeaseManager(ConfigService configService, WorkerDirectory workers) {
        this.configService = configService;
        this.workers = workers;
    }

    public record WorkerView(String address,
                             long generation,
                             long version,
                             boolean enabled,
                             boolean confirmed,
                             boolean healthy,
                             int pendingRequests,
                             String state) {
    }

    public List<WorkerView> status(String group) {
        List<WorkerView> result = new ArrayList<>();
        for (WorkerStatus worker : workers.getWorkerStatuses(RoleType.PREFILL, group)) {
            worker.lock.lock();
            try {
                var endpoint = workers.exactEndpoint(RoleType.PREFILL, worker.getLogicalIpPort(), worker);
                int pending = Math.max(worker.committedEngineObservation().runningTaskList().size(),
                        endpoint instanceof PrefillEndpoint prefill ? prefill.getCommittedRequestCount() : 0);
                PrefillChunkLease lease = worker.prefillChunkLease();
                PrefillChunkLease.Command command = lease.command();
                boolean enabled = command != null && command.enabled();
                boolean healthy = worker.isActiveGeneration() && worker.isAlive() && endpoint != null;
                String state = command == null
                        ? "UNMANAGED"
                        : enabled
                                ? lease.confirmed() ? "SERVING" : "ACTIVATING"
                                : lease.confirmed() && pending == 0 ? "STANDBY" : "DRAINING";
                result.add(new WorkerView(worker.getLogicalIpPort(), worker.getGenerationId(),
                        command == null ? 0 : command.version(), enabled, lease.confirmed(), healthy, pending, state));
            } finally {
                worker.lock.unlock();
            }
        }
        result.sort(Comparator.comparing(WorkerView::address));
        return List.copyOf(result);
    }

    /**
     * Serializes manual and automatic changes; stops the batch if its worker snapshot becomes stale.
     * Already issued leases remain valid and must be confirmed before the next decision.
     */
    public synchronized List<String> changeIfCurrent(String group,
                                                     String expectedMembers,
                                                     List<String> addresses,
                                                     boolean enabled,
                                                     PrefillLeaseConfig config) {
        List<String> changed = new ArrayList<>();
        for (String address : addresses) {
            List<WorkerView> current = status(group);
            if (configService.loadBalanceConfig().getRouter().getRoles().getPrefill().getChunkLease() != config
                    || !PrefillTtftLeaseController.members(current).equals(expectedMembers)) {
                break;
            }
            WorkerView updated = change(group, address, enabled);
            changed.add(address);
            // Advance only our own command; unrelated changes must invalidate the remaining decision.
            List<WorkerView> expected =
                    current.stream().map(worker -> worker.address().equals(address) ? updated : worker).toList();
            expectedMembers = PrefillTtftLeaseController.members(expected);
        }
        return List.copyOf(changed);
    }

    public synchronized WorkerView change(String group, String address, boolean enabled) {
        PrefillLeaseConfig config =
                configService.loadBalanceConfig().getRouter().getRoles().getPrefill().getChunkLease();
        if (!config.isEnabled() || !config.getGroups().contains(group)) {
            throw new IllegalArgumentException("chunk lease control is not enabled for group " + group);
        }
        WorkerStatus worker = workers.statusSnapshot(RoleType.PREFILL).get(address);
        if (worker == null || !group.equals(worker.getGroup())) {
            throw new IllegalArgumentException("unknown Prefill worker in group");
        }
        worker.lock.lock();
        try {
            if (!worker.isActiveGeneration() || !workers.isCurrentStatus(RoleType.PREFILL, address, worker)) {
                throw new IllegalArgumentException("worker generation is no longer current");
            }
            if (enabled && (!worker.isAlive() || workers.exactEndpoint(RoleType.PREFILL, address, worker) == null)) {
                throw new IllegalArgumentException("worker must be healthy before receiving a lease");
            }
            if (worker.prefillChunkLease().change(enabled)) {
                worker.prefillChunkCapacity().freeze();
            }
        } finally {
            worker.lock.unlock();
        }
        return status(group).stream().filter(view -> view.address().equals(address)).findFirst().orElseThrow();
    }
}

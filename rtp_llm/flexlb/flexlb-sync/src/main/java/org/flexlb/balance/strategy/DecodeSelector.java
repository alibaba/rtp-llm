package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRoutingView;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeMode;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class DecodeSelector {

    private static final int MAX_SELECTION_ATTEMPTS = 2;
    private enum Availability { READY, BUSY, IMPOSSIBLE }
    private final WorkerDirectory workerDirectory;
    private final EndpointRoundRobin rotation = new EndpointRoundRobin();

    public DecodeSelector(WorkerDirectory workerDirectory) {
        this.workerDirectory = workerDirectory;
    }

    public PlacementResult<SelectedRole, RoleType> select(
            DecodeBinding request, String group) {
        for (int attempt = 0; attempt < MAX_SELECTION_ATTEMPTS; attempt++) {
            List<DecodeRoutingView> snapshots = workerDirectory.decodeRoutingSnapshot(group);
            if (snapshots.isEmpty()) {
                return PlacementResult.blocked(RoleType.DECODE);
            }
            Availability[] availabilityByWorker = new Availability[snapshots.size()];
            Availability preferredAvailability = Availability.IMPOSSIBLE;
            boolean allWorkersTooSmall = true;
            long largestKvBudget = 0L;
            for (int index = 0; index < snapshots.size(); index++) {
                DecodeRoutingView view = snapshots.get(index);
                Availability availability = availability(request, view);
                availabilityByWorker[index] = availability;
                allWorkersTooSmall &= availability == Availability.IMPOSSIBLE;
                largestKvBudget = Math.max(largestKvBudget, request.capacity().kvBudget(view.totalKv()));
                if (availability == Availability.READY
                        || availability == Availability.BUSY
                            && preferredAvailability == Availability.IMPOSSIBLE
                            && request.mode() != DecodeMode.IMMEDIATE) {
                    preferredAvailability = availability;
                }
            }
            if (allWorkersTooSmall) {
                return PlacementResult.rejected(staticCapacityFailure(
                        request.expectedKvTokens(), largestKvBudget));
            }
            if (preferredAvailability == Availability.IMPOSSIBLE) {
                return PlacementResult.blocked(RoleType.DECODE);
            }
            Availability selectedAvailability = preferredAvailability;
            int selectedIndex = rotation.next(RoleType.DECODE, group, snapshots.size(),
                    i -> availabilityByWorker[i] == selectedAvailability, i -> snapshots.get(i).address());
            if (selectedIndex < 0) { throw new IllegalStateException("Decode snapshot candidate disappeared"); }
            DecodeRoutingView selected = snapshots.get(selectedIndex);
            WorkerEndpoint.GenerationPin pin = workerDirectory.captureDecodeGeneration(selected);
            if (pin != null) {
                return PlacementResult.success(buildSelectedRole(
                        selected, pin, request.requestId()));
            }
        }
        return PlacementResult.blocked(RoleType.DECODE);
    }

    private static Availability availability(DecodeBinding request, DecodeRoutingView view) {
        if (view.totalKv() > 0L && request.expectedKvTokens() > request.capacity().kvBudget(view.totalKv())) {
            return Availability.IMPOSSIBLE;
        }
        var usage = switch (request.mode()) {
            case IMMEDIATE, WAIT_AT_DISPATCH -> view.dispatchUsage();
            case PREEMPT_AT_PLACEMENT -> view.placementUsage();
        };
        return request.capacity().evaluate(usage, request.hardKvTokens(), request.expectedKvTokens()).fits()
                ? Availability.READY : Availability.BUSY;
    }

    private static Response staticCapacityFailure(long required, long maximum) {
        Response response = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        response.setErrorMessage(StrategyErrorType.RESOURCE_EXHAUSTED.buildErrorMessage(
                "Decode prompt plus output demand=" + required
                        + " exceeds every worker KV admission budget; maximum=" + maximum));
        return response;
    }

    private SelectedRole buildSelectedRole(
            DecodeRoutingView selected,
            WorkerEndpoint.GenerationPin selectedPin,
            long requestId) {
        try {
            if (selectedPin.generationId() != selected.generationId()
                    || !(selectedPin.endpoint() instanceof DecodeEndpoint)) {
                throw new IllegalStateException(
                        "Decode snapshot pin changed before selection handoff");
            }
            WorkerStatus.TopologySnapshot topology = selected.topology();
            WorkerStatus.EngineObservation status =
                    selected.workerStatus().fields();
            ServerStatus result = new ServerStatus();
            result.setSuccess(true);
            result.setRole(RoleType.DECODE);
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(status.dpRank());
            result.setGroup(topology.group());
            result.setRequestId(requestId);

            // SelectedRole consumes the pin even if its validation rejects.
            WorkerEndpoint.GenerationPin factoryPin = selectedPin;
            selectedPin = null;
            return SelectedRole.decode(
                    factoryPin, result, selected.admissionVersion());
        } finally {
            if (selectedPin != null) {
                selectedPin.close();
            }
        }
    }

}

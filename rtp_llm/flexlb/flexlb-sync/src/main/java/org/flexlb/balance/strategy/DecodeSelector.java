package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.RequestRequirements;
import org.flexlb.balance.scheduler.RequestRequirements.DecodeMode;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.google.common.base.Preconditions.checkState;

@Component
public class DecodeSelector {

    private static final int MAX_SELECTION_ATTEMPTS = 2;
    private enum Availability { READY, BUSY, IMPOSSIBLE }
    private final EndpointRegistry endpointRegistry;
    private final EndpointRoundRobin rotation = new EndpointRoundRobin();

    public DecodeSelector(EndpointRegistry endpointRegistry) {
        this.endpointRegistry = endpointRegistry;
    }

    public PlacementResult<WorkerAssignment, RoleType> select(
            RequestRequirements request, String group) {
        for (int attempt = 0; attempt < MAX_SELECTION_ATTEMPTS; attempt++) {
            List<DecodeRoutingView> snapshots = endpointRegistry.decodeRoutingSnapshot(group);
            if (snapshots.isEmpty()) {
                return PlacementResult.blocked(RoleType.DECODE);
            }
            Availability[] availabilityByWorker = new Availability[snapshots.size()];
            Availability preferredAvailability = Availability.IMPOSSIBLE;
            long largestKvBudget = 0L;
            for (int index = 0; index < snapshots.size(); index++) {
                DecodeRoutingView view = snapshots.get(index);
                Availability availability = availability(request, view);
                availabilityByWorker[index] = availability;
                largestKvBudget = Math.max(largestKvBudget, request.capacity().kvBudget(view.totalKv()));
                if (availability == Availability.READY
                        || availability == Availability.BUSY
                            && preferredAvailability == Availability.IMPOSSIBLE) {
                    preferredAvailability = availability;
                }
            }
            if (preferredAvailability == Availability.IMPOSSIBLE) {
                return PlacementResult.rejected(oversizedRequestFailure(
                        request.expectedKvTokens(), largestKvBudget));
            }
            if (preferredAvailability == Availability.BUSY && request.mode() == DecodeMode.IMMEDIATE) {
                return classifyCapacityFailure(request, snapshots);
            }
            Availability selectedAvailability = preferredAvailability;
            double[] costByWorker = new double[snapshots.size()];
            double minimumCost = Double.POSITIVE_INFINITY;
            for (int index = 0; index < snapshots.size(); index++) {
                if (availabilityByWorker[index] == selectedAvailability) {
                    DecodeRoutingView view = snapshots.get(index);
                    double cost = request.costFormula().evaluate(view.totalLoad(),
                            request.capacity().maxEngineRequests(), view.realKvUsed(), view.totalKv());
                    costByWorker[index] = cost;
                    if (Double.isFinite(cost)) {
                        minimumCost = Math.min(minimumCost, cost);
                    }
                }
            }
            if (!Double.isFinite(minimumCost)) {
                throw new IllegalStateException("Decode cost formula produced no finite score: "
                        + request.costFormula().expression());
            }
            double selectedCost = minimumCost;
            int selectedIndex = rotation.next(RoleType.DECODE, group, snapshots.size(),
                    i -> availabilityByWorker[i] == selectedAvailability && costByWorker[i] == selectedCost,
                    i -> snapshots.get(i).address());
            checkState(selectedIndex >= 0, "Decode snapshot candidate disappeared");
            DecodeRoutingView selected = snapshots.get(selectedIndex);
            WorkerEndpoint.GenerationPin pin = endpointRegistry.captureDecodeGeneration(selected);
            if (pin != null) {
                return PlacementResult.success(buildWorkerAssignment(
                        selected, pin, request.requestId()));
            }
        }
        return PlacementResult.blocked(RoleType.DECODE);
    }

    private PlacementResult<WorkerAssignment, RoleType> classifyCapacityFailure(
            RequestRequirements request, List<DecodeRoutingView> views) {
        Response failure = null;
        boolean uniformFailure = true;
        List<Map<String, Object>> evidence = new ArrayList<>();
        for (DecodeRoutingView view : views) {
            Response workerFailure;
            try (WorkerEndpoint.GenerationPin pin = endpointRegistry.captureDecodeGeneration(view)) {
                if (pin == null) {
                    workerFailure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
                } else {
                    var snapshot = ((DecodeEndpoint) pin.endpoint()).admissionSummary();
                    evidence.add(Map.of("endpoint", view.address(), "version", snapshot.routing().admissionVersion(),
                            "engineLoad", snapshot.routing().engineLoad(), "totalLoad", snapshot.routing().totalLoad(),
                            "kvTotal", snapshot.routing().totalKv(), "kvAvailable", snapshot.routing().placementUsage().hardKvAvailable()));
                    workerFailure = classifyCapacityFailure(request, snapshot);
                }
            }
            if (failure == null) {
                failure = workerFailure;
            } else if (failure.getCode() != workerFailure.getCode()
                    || failure.getAdmissionRejectReason() != workerFailure.getAdmissionRejectReason()) {
                uniformFailure = false;
            }
            if (workerFailure.getCode() == StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode()) {
                failure = workerFailure;
            }
        }
        if (failure == null || !uniformFailure && failure.getCode() != StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode()) {
            failure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        }
        return PlacementResult.blocked(RoleType.DECODE, failure, Map.of("cause", "Decode capacity exhausted",
                "capturedAtMs", System.currentTimeMillis(), "decode", List.copyOf(evidence)));
    }

    /** Classify only the dimensions that prevent this request from fitting. */
    static Response classifyCapacityFailure(RequestRequirements request, DecodeResources.AdmissionSummary snapshot) {
        boolean dispatch = request.mode() == DecodeMode.IMMEDIATE;
        var routing = snapshot.routing();
        var usage = dispatch ? routing.dispatchUsage() : routing.placementUsage();
        if (usage.totalKvTokens() > 0
                && request.expectedKvTokens() > request.capacity().kvBudget(usage.totalKvTokens())) {
            return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        }
        CapacityRelease lower = CapacityRelease.NONE;
        CapacityRelease higher = CapacityRelease.NONE;
        CapacityRelease same = CapacityRelease.NONE;
        for (int priority = 1; priority <= PriorityNormalizer.MAX_PRIORITY; priority++) {
            CapacityRelease occupied = dispatch ? snapshot.engineOccupancy(priority) : snapshot.placementOccupancy(priority);
            if (occupied == CapacityRelease.NONE) { continue; }
            if (priority < request.priority()) {
                lower = lower.plus(occupied);
            } else if (priority == request.priority()) {
                same = occupied;
            } else {
                higher = higher.plus(occupied);
            }
        }
        // Removing lower-priority occupancy is a counterfactual for attribution,
        // not an authorization to preempt its owners.
        var residual = request.capacity().evaluate(usage, request.hardKvTokens(), request.expectedKvTokens(), lower);
        if (residual.fits()) {
            return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        }
        CapacityRelease attributed = higher.plus(same);
        if (residual.requests() > attributed.requests()
                || residual.hardKvTokens() > attributed.hardKvTokens()
                || residual.expectedKvTokens() > attributed.expectedKvTokens()) {
            return Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE);
        }
        boolean higherBlocks = residual.requests() > 0 && higher.requests() > 0
                || residual.hardKvTokens() > 0 && higher.hardKvTokens() > 0
                || residual.expectedKvTokens() > 0 && higher.expectedKvTokens() > 0;
        return Response.error(StrategyErrorType.PRIORITY_ADMISSION_REJECTED, higherBlocks
                ? AdmissionRejectReason.HIGHER_PRIORITY_AHEAD : AdmissionRejectReason.SAME_PRIORITY_AHEAD);
    }

    private static Availability availability(RequestRequirements request, DecodeRoutingView view) {
        if (view.totalKv() > 0L && request.expectedKvTokens() > request.capacity().kvBudget(view.totalKv())) {
            return Availability.IMPOSSIBLE;
        }
        var usage = switch (request.mode()) {
            case IMMEDIATE -> view.dispatchUsage();
            case WAIT_AT_PLACEMENT, PREEMPT_AT_PLACEMENT -> view.placementUsage();
        };
        return request.capacity().evaluate(usage, request.hardKvTokens(), request.expectedKvTokens(), CapacityRelease.NONE).fits()
                ? Availability.READY : Availability.BUSY;
    }

    private static Response oversizedRequestFailure(long required, long maximum) {
        return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
                "Decode prompt plus output demand=" + required
                        + " exceeds every worker KV admission budget; maximum=" + maximum);
    }

    private WorkerAssignment buildWorkerAssignment(
            DecodeRoutingView selected,
            WorkerEndpoint.GenerationPin selectedPin,
            long requestId) {
        try {
            checkState(selectedPin.generationId() == selected.generationId()
                    && (selectedPin.endpoint() instanceof DecodeEndpoint),
                    "Decode snapshot pin changed before selection handoff");
            WorkerStatus.TopologySnapshot topology = selected.topology();
            WorkerStatus.EngineObservation status =
                    selected.workerStatus().fields();
            ServerStatus result = WorkerAssignment.workerMetadata(RoleType.DECODE, requestId, topology, status);

            // WorkerAssignment consumes the pin even if its validation rejects.
            WorkerEndpoint.GenerationPin factoryPin = selectedPin;
            selectedPin = null;
            return WorkerAssignment.decode(
                    factoryPin, result, selected.admissionVersion());
        } finally {
            if (selectedPin != null) {
                selectedPin.close();
            }
        }
    }

}

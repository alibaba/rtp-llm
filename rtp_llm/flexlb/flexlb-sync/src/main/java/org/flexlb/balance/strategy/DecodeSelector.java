package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.CapacityRelease;
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
import org.flexlb.util.PriorityNormalizer;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

@Component
public class DecodeSelector {

    private static final int MAX_SELECTION_ATTEMPTS = 2;
    private enum Availability { READY, BUSY, IMPOSSIBLE }
    private final WorkerDirectory workerDirectory;
    private final EndpointRoundRobin rotation = new EndpointRoundRobin();

    public DecodeSelector(WorkerDirectory workerDirectory) {
        this.workerDirectory = workerDirectory;
    }

    /** At queue placement only permanent per-request incompatibility can reject a request. */
    public Response validateDemand(DecodeBinding request, String group) {
        return validateDemand(request, workerDirectory.decodeRoutingSnapshot(group));
    }

    private static Response validateDemand(DecodeBinding request, List<DecodeRoutingView> workers) {
        long largestBudget = 0L;
        for (DecodeRoutingView worker : workers) {
            if (worker.totalKv() <= 0L) { return null; }
            long budget = request.capacity().kvBudget(worker.totalKv());
            if (request.expectedKvTokens() <= budget) { return null; }
            largestBudget = Math.max(largestBudget, budget);
        }
        return workers.isEmpty() ? null : oversizedRequestFailure(request.expectedKvTokens(), largestBudget);
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
                return PlacementResult.rejected(oversizedRequestFailure(
                        request.expectedKvTokens(), largestKvBudget));
            }
            if (preferredAvailability == Availability.IMPOSSIBLE) {
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

    /** A decision-time claim. Both the reservation and permit belong to the accepting transaction. */
    public record Dispatch(ServerStatus status, DecodeEndpoint endpoint,
                           DecodeEndpoint.ReservationHandle reservation, DecodeEndpoint.EngineDispatchPermit permit) { }

    /** Try every eligible candidate at most once; a lost race on one worker cannot strand an idle peer. */
    public CapacityBoundary.Attempt<Dispatch> tryDispatch(DecodeBinding request, String group, boolean allowsEnginePreemption) {
        // Capture before the snapshot. Capacity may return between the failed attempt and listener installation.
        int preemptionPriority = allowsEnginePreemption ? request.priority() : PriorityNormalizer.NO_PRIORITY;
        var changes = workerDirectory.observeDecodeCapacity(group, preemptionPriority);
        List<DecodeRoutingView> workers = workerDirectory.decodeRoutingSnapshot(group);
        Response incompatible = validateDemand(request, workers);
        if (incompatible != null) {
            return CapacityBoundary.Attempt.rejected(CapacityBoundary.failed(
                    new IllegalStateException(incompatible.getErrorMessage())));
        }
        double[] costs = new double[workers.size()];
        Arrays.fill(costs, Double.POSITIVE_INFINITY);
        boolean hasReady = false;
        for (int i = 0; i < workers.size(); i++) {
            DecodeRoutingView worker = workers.get(i);
            if (availability(request, worker) == Availability.READY) {
                hasReady = true;
                double cost = dispatchCost(request, worker);
                if (Double.isFinite(cost)) { costs[i] = cost; }
            }
        }
        double minimum = Arrays.stream(costs).min().orElse(Double.POSITIVE_INFINITY);
        if (hasReady && !Double.isFinite(minimum)) {
            throw new IllegalStateException("Decode cost formula produced no finite score: "
                    + request.costFormula().expression());
        }
        while (Double.isFinite(minimum)) {
            double candidateCost = minimum;
            int index = rotation.next(RoleType.DECODE, group, workers.size(), i -> costs[i] == candidateCost,
                    i -> workers.get(i).address());
            DecodeRoutingView selected = workers.get(index);
            try (var pin = workerDirectory.captureDecodeGeneration(selected)) {
                if (pin != null) {
                    DecodeEndpoint endpoint = (DecodeEndpoint) pin.endpoint();
                    ServerStatus status = serverStatus(selected, request.requestId());
                    var permit = endpoint.reserveForDispatch(pin, request.requestId(), request.hardKvTokens(),
                            request.expectedKvTokens(), request.priority(), request.capacity());
                    if (permit != null) {
                        return CapacityBoundary.Attempt.accepted(new Dispatch(status, endpoint, permit.reservation(), permit));
                    }
                }
            }
            costs[index] = Double.POSITIVE_INFINITY;
            minimum = Arrays.stream(costs).min().orElse(Double.POSITIVE_INFINITY);
        }
        return CapacityBoundary.Attempt.rejected(CapacityBoundary.deliveryUnavailable(changes));
    }

    public boolean hasDispatchCapacity(DecodeBinding request, String group) {
        return workerDirectory.decodeRoutingSnapshot(group).stream()
                .anyMatch(worker -> availability(request, worker) == Availability.READY);
    }

    /** The eviction transaction revalidates the exact returned generation before touching victims. */
    public List<DecodeEndpoint> preemptionCandidates(DecodeBinding request, String group) {
        var busy = workerDirectory.decodeRoutingSnapshot(group).stream()
                .filter(worker -> availability(request, worker) == Availability.BUSY)
                .sorted(Comparator.comparingDouble((DecodeRoutingView worker) -> dispatchCost(request, worker))
                        .thenComparing(DecodeRoutingView::address))
                .toList();
        List<DecodeEndpoint> candidates = new ArrayList<>(busy.size());
        for (var worker : busy) {
            try (var pin = workerDirectory.captureDecodeGeneration(worker)) {
                if (pin != null) { candidates.add((DecodeEndpoint) pin.endpoint()); }
            }
        }
        return candidates;
    }

    private static double dispatchCost(DecodeBinding request, DecodeRoutingView worker) {
        return request.costFormula().evaluate(worker.engineCapacityUsed(),
                request.capacity().maxEngineRequests(), worker.engineFacingKvUsed(), worker.totalKv());
    }

    private PlacementResult<SelectedRole, RoleType> classifyCapacityFailure(
            DecodeBinding request, List<DecodeRoutingView> views) {
        Response failure = null;
        boolean uniformFailure = true;
        List<Map<String, Object>> evidence = new ArrayList<>();
        for (DecodeRoutingView view : views) {
            Response workerFailure;
            try (WorkerEndpoint.GenerationPin pin = workerDirectory.captureDecodeGeneration(view)) {
                if (pin == null) {
                    workerFailure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
                } else {
                    var snapshot = ((DecodeEndpoint) pin.endpoint()).admissionSummary();
                    evidence.add(Map.of("endpoint", view.address(), "version", snapshot.routing().admissionVersion(),
                            "engineLoad", snapshot.routing().engineLoad(), "totalLoad", snapshot.routing().totalLoad(),
                            "kvTotal", snapshot.routing().totalKv(), "kvAvailable", snapshot.routing().dispatchUsage().hardKvAvailable()));
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
    static Response classifyCapacityFailure(DecodeBinding request, DecodeEndpoint.AdmissionSummary snapshot) {
        var routing = snapshot.routing();
        var usage = routing.dispatchUsage();
        if (usage.totalKvTokens() > 0
                && request.expectedKvTokens() > request.capacity().kvBudget(usage.totalKvTokens())) {
            return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        }
        CapacityRelease lower = CapacityRelease.NONE;
        CapacityRelease higher = CapacityRelease.NONE;
        CapacityRelease same = CapacityRelease.NONE;
        for (int priority = 1; priority <= PriorityNormalizer.MAX_PRIORITY; priority++) {
            CapacityRelease occupied = snapshot.engineOccupancy(priority);
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

    private static Availability availability(DecodeBinding request, DecodeRoutingView view) {
        if (view.totalKv() > 0L && request.expectedKvTokens() > request.capacity().kvBudget(view.totalKv())) {
            return Availability.IMPOSSIBLE;
        }
        return request.capacity().evaluate(view.dispatchUsage(), request.hardKvTokens(), request.expectedKvTokens()).fits()
                ? Availability.READY : Availability.BUSY;
    }

    private static Response oversizedRequestFailure(long required, long maximum) {
        return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
                "Decode prompt plus output demand=" + required
                        + " exceeds every worker KV admission budget; maximum=" + maximum);
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
            ServerStatus result = serverStatus(selected, requestId);
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

    private static ServerStatus serverStatus(DecodeRoutingView selected, long requestId) {
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
        return result;
    }
}

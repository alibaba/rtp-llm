package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.DecodeCostFormula;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkState;

@Component
public class DecodeSelector {

    private static final int MAX_SELECTION_ATTEMPTS = 2;
    private static final int MAX_CACHED_WORKERS = 4096;
    private enum Availability { READY, BUSY, IMPOSSIBLE } // Ordered from most to least preferred.
    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);
    private static final class Scratch {
        private final ArrayList<DecodeRoutingView> views = new ArrayList<>();
        private final ArrayList<DecodeRoutingView> candidates = new ArrayList<>();
        private boolean inUse;
    }

    private final EndpointRegistry endpointRegistry;
    private final EndpointRoundRobin rotation = new EndpointRoundRobin();

    private final AdmissionCapacity capacity;
    private final DecodeCostFormula costFormula;
    private final boolean direct;

    public DecodeSelector(EndpointRegistry endpointRegistry, ConfigService configService,
                          AdmissionCapacity capacity) {
        this.endpointRegistry = Objects.requireNonNull(endpointRegistry, "endpointRegistry");
        this.capacity = Objects.requireNonNull(capacity, "capacity");
        var config = configService.loadBalanceConfig();
        this.direct = config.isDirect();
        this.costFormula = config.getRouter().getRoles().getDecode().getCostEstimator().compiledFormula();
    }

    public PlacementResult<WorkerAssignment, RoleType> select(
            Request request, SchedulingMetadata metadata, String group) {
        Scratch cached = SCRATCH.get();
        Scratch scratch = cached.inUse ? new Scratch() : cached;
        scratch.inUse = true;
        int captured = 0;
        try {
            for (int attempt = 0; attempt < MAX_SELECTION_ATTEMPTS; attempt++) {
                var snapshots = scratch.views;
                var candidates = scratch.candidates;
                snapshots.clear();
                candidates.clear();
                endpointRegistry.decodeRoutingSnapshot(group, snapshots);
                captured = Math.max(captured, snapshots.size());
                if (snapshots.isEmpty()) { return PlacementResult.blocked(RoleType.DECODE); }
                Availability preferred = Availability.IMPOSSIBLE;
                long largestKvBudget = 0L;
                for (DecodeRoutingView view : snapshots) {
                    Availability available = availability(request, view);
                    largestKvBudget = Math.max(largestKvBudget, capacity.kvBudget(view.totalKv()));
                    if (available.ordinal() < preferred.ordinal()) {
                        preferred = available;
                        candidates.clear();
                    }
                    if (available == preferred) { candidates.add(view); }
                }
                if (preferred == Availability.IMPOSSIBLE) {
                    return PlacementResult.rejected(oversizedRequestFailure(
                            request.getExpectedKvTokens(), largestKvBudget));
                }
                if (preferred == Availability.BUSY && direct) {
                    return classifyCapacityFailure(request, metadata, snapshots);
                }
                double minimumCost = Double.POSITIVE_INFINITY;
                int winners = 0;
                for (int index = 0; index < candidates.size(); index++) {
                    DecodeRoutingView view = candidates.get(index);
                    double cost = costFormula.evaluate(view.totalLoad(),
                            capacity.maxEngineRequests(), view.realKvUsed(), view.totalKv());
                    if (!Double.isFinite(cost)) { continue; }
                    if (cost < minimumCost) {
                        minimumCost = cost;
                        winners = 0;
                    }
                    // Compact ties in place; the write index never overtakes the read index.
                    if (cost == minimumCost) { candidates.set(winners++, view); }
                }
                if (winners == 0) {
                    throw new IllegalStateException("Decode cost formula produced no finite score: "
                            + costFormula.expression());
                }
                int selectedIndex = rotation.next(RoleType.DECODE, group, winners,
                        i -> true, i -> candidates.get(i).address());
                checkState(selectedIndex >= 0, "Decode snapshot candidate disappeared");
                DecodeRoutingView selected = candidates.get(selectedIndex);
                WorkerEndpoint.GenerationPin pin = endpointRegistry.captureDecodeGeneration(selected);
                if (pin != null) {
                    return PlacementResult.success(buildWorkerAssignment(selected, pin, request.getRequestId()));
                }
            }
            return PlacementResult.blocked(RoleType.DECODE);
        } finally {
            // Clear every capture, including losers and partially captured exceptional traversals.
            captured = Math.max(captured, scratch.views.size());
            scratch.views.clear();
            scratch.candidates.clear();
            scratch.inUse = false;
            if (scratch == cached && captured > MAX_CACHED_WORKERS) { SCRATCH.remove(); }
        }
    }

    private PlacementResult<WorkerAssignment, RoleType> classifyCapacityFailure(
            Request request, SchedulingMetadata metadata, List<DecodeRoutingView> views) {
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
                    workerFailure = classifyCapacityFailure(request, metadata, snapshot);
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
    private Response classifyCapacityFailure(Request request, SchedulingMetadata metadata, DecodeResources.AdmissionSummary snapshot) {
        var routing = snapshot.routing();
        var usage = direct ? routing.dispatchUsage() : routing.placementUsage();
        if (usage.totalKvTokens() > 0
                && request.getExpectedKvTokens() > capacity.kvBudget(usage.totalKvTokens())) {
            return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        }
        CapacityRelease lower = CapacityRelease.NONE;
        CapacityRelease higher = CapacityRelease.NONE;
        CapacityRelease same = CapacityRelease.NONE;
        for (int priority = 1; priority <= PriorityNormalizer.MAX_PRIORITY; priority++) {
            CapacityRelease occupied = direct ? snapshot.engineOccupancy(priority) : snapshot.placementOccupancy(priority);
            if (occupied == CapacityRelease.NONE) { continue; }
            if (priority < metadata.priority()) {
                lower = lower.plus(occupied);
            } else if (priority == metadata.priority()) {
                same = occupied;
            } else {
                higher = higher.plus(occupied);
            }
        }
        // Removing lower-priority occupancy is a counterfactual for attribution,
        // not an authorization to preempt its owners.
        var residual = capacity.evaluate(usage, request.getHardKvTokens(), request.getExpectedKvTokens(), lower);
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

    private Availability availability(Request request, DecodeRoutingView view) {
        if (view.totalKv() > 0L && request.getExpectedKvTokens() > capacity.kvBudget(view.totalKv())) {
            return Availability.IMPOSSIBLE;
        }
        var usage = direct ? view.dispatchUsage() : view.placementUsage();
        return capacity.evaluate(usage, request.getHardKvTokens(), request.getExpectedKvTokens(), CapacityRelease.NONE).fits()
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

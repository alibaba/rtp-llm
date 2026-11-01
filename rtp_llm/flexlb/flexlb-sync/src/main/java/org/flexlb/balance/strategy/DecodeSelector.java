package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.prediction.DecodeCostFormula;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkState;

public class DecodeSelector implements DecodeSelectionStrategy {

    private static final int MAX_CACHED_WORKERS = 4096;

    private enum Availability {
        READY,
        BUSY,
        IMPOSSIBLE
    } // Ordered from most to least preferred.

    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

    private static final class Scratch {
        private String[] addresses = new String[0];
        private int[] candidates = new int[0];
        private boolean inUse;

        private void prepare(int size) {
            if (candidates.length < size) {
                addresses = new String[size];
                candidates = new int[size];
            }
        }
    }

    private final EndpointRoundRobin rotation = new EndpointRoundRobin();

    private final AdmissionCapacity capacity;
    private final DecodeCostFormula costFormula;
    private final boolean direct;

    public DecodeSelector(ConfigService configService, AdmissionCapacity capacity) {
        this.capacity = Objects.requireNonNull(capacity, "capacity");
        var config = configService.loadBalanceConfig();
        this.direct = config.isDirect();
        this.costFormula =
                config.getRouter().getRoles().getDecode().getCostEstimator().compiledFormula();
    }

    @Override
    public PlacementResult<DecodeRoutingView, PlacementKey> select(
            Request request,
            SchedulingMetadata metadata,
            String group,
            List<DecodeRoutingView> snapshots,
            CapacityReader reader) {
        if (snapshots.isEmpty()) {
            return PlacementResult.blocked(new PlacementKey(RoleType.DECODE, group, null));
        }
        Scratch cached = SCRATCH.get();
        Scratch scratch = cached.inUse ? new Scratch() : cached;
        scratch.inUse = true;
        try {
            scratch.prepare(snapshots.size());
            var candidates = scratch.candidates;
            Availability preferred = Availability.IMPOSSIBLE;
            long largestKvBudget = 0L;
            int candidateCount = 0;
            for (int index = 0; index < snapshots.size(); index++) {
                DecodeRoutingView view = snapshots.get(index);
                scratch.addresses[index] = view.address();
                Availability available = availability(request, view);
                largestKvBudget = Math.max(largestKvBudget, capacity.kvBudget(view.totalKv()));
                if (available.ordinal() < preferred.ordinal()) {
                    preferred = available;
                    candidateCount = 0;
                }
                if (available == preferred) {
                    candidates[candidateCount++] = index;
                }
            }
            if (preferred == Availability.IMPOSSIBLE) {
                return PlacementResult.rejected(
                        oversizedRequestFailure(request.getExpectedKvTokens(), largestKvBudget));
            }
            if (preferred == Availability.BUSY && direct) {
                return classifyCapacityFailure(request, metadata, group, snapshots, reader);
            }
            double minimumCost = Double.POSITIVE_INFINITY;
            int winners = 0;
            var evaluator = costFormula.evaluator();
            for (int index = 0; index < candidateCount; index++) {
                DecodeRoutingView view = snapshots.get(candidates[index]);
                double cost =
                        evaluator.evaluate(
                                view.totalLoad(),
                                capacity.maxEngineRequests(),
                                view.realKvUsed(),
                                view.totalKv());
                if (!Double.isFinite(cost)) {
                    continue;
                }
                if (cost < minimumCost) {
                    minimumCost = cost;
                    winners = 0;
                }
                if (cost == minimumCost) {
                    candidates[winners++] = candidates[index];
                }
            }
            checkState(
                    winners > 0,
                    "Decode cost formula produced no finite score: %s",
                    costFormula.expression());
            int selected =
                    rotation.next(RoleType.DECODE, group, scratch.addresses, candidates, winners);
            checkState(selected >= 0, "Decode snapshot candidate disappeared");
            return PlacementResult.success(snapshots.get(selected));
        } finally {
            Arrays.fill(
                    scratch.addresses,
                    0,
                    Math.min(snapshots.size(), scratch.addresses.length),
                    null);
            scratch.inUse = false;
            if (scratch == cached && scratch.addresses.length > MAX_CACHED_WORKERS) {
                SCRATCH.remove();
            }
        }
    }

    private PlacementResult<DecodeRoutingView, PlacementKey> classifyCapacityFailure(
            Request request,
            SchedulingMetadata metadata,
            String group,
            List<DecodeRoutingView> views,
            CapacityReader reader) {
        Response failure = null;
        boolean uniformFailure = true;
        List<Map<String, Object>> evidence = new ArrayList<>();
        for (DecodeRoutingView view : views) {
            Response workerFailure;
            var snapshot = reader.read(view);
            if (snapshot == null) {
                workerFailure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
            } else {
                var routing = snapshot.routing();
                evidence.add(
                        Map.of(
                                "endpoint",
                                view.address(),
                                "version",
                                routing.admissionVersion(),
                                "engineLoad",
                                routing.engineLoad(),
                                "totalLoad",
                                routing.totalLoad(),
                                "kvTotal",
                                routing.totalKv(),
                                "kvAvailable",
                                routing.placementUsage().hardKvAvailable()));
                workerFailure = classifyCapacityFailure(request, metadata, snapshot);
            }
            if (failure == null) {
                failure = workerFailure;
            } else if (failure.getCode() != workerFailure.getCode()
                    || failure.getAdmissionRejectReason()
                            != workerFailure.getAdmissionRejectReason()) {
                uniformFailure = false;
            }
            if (workerFailure.getCode() == StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode()) {
                failure = workerFailure;
            }
        }
        if (failure == null
                || !uniformFailure
                        && failure.getCode()
                                != StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode()) {
            failure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        }
        return PlacementResult.blocked(
                new PlacementKey(RoleType.DECODE, group, null),
                failure,
                Map.of(
                        "cause",
                        "Decode capacity exhausted",
                        "capturedAtMs",
                        System.currentTimeMillis(),
                        "decode",
                        List.copyOf(evidence)));
    }

    /** Classify only the dimensions that prevent this request from fitting. */
    private Response classifyCapacityFailure(
            Request request,
            SchedulingMetadata metadata,
            DecodeResources.AdmissionSummary snapshot) {
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
            CapacityRelease occupied =
                    direct
                            ? snapshot.engineOccupancy(priority)
                            : snapshot.placementOccupancy(priority);
            if (occupied == CapacityRelease.NONE) {
                continue;
            }
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
        var residual =
                capacity.evaluate(
                        usage, request.getHardKvTokens(), request.getExpectedKvTokens(), lower);
        if (residual.fits()) {
            return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        }
        CapacityRelease attributed = higher.plus(same);
        if (residual.requests() > attributed.requests()
                || residual.hardKvTokens() > attributed.hardKvTokens()
                || residual.expectedKvTokens() > attributed.expectedKvTokens()) {
            return Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE);
        }
        boolean higherBlocks =
                residual.requests() > 0 && higher.requests() > 0
                        || residual.hardKvTokens() > 0 && higher.hardKvTokens() > 0
                        || residual.expectedKvTokens() > 0 && higher.expectedKvTokens() > 0;
        return Response.error(
                StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                higherBlocks
                        ? AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                        : AdmissionRejectReason.SAME_PRIORITY_AHEAD);
    }

    private Availability availability(Request request, DecodeRoutingView view) {
        if (view.totalKv() > 0L
                && request.getExpectedKvTokens() > capacity.kvBudget(view.totalKv())) {
            return Availability.IMPOSSIBLE;
        }
        var usage = direct ? view.dispatchUsage() : view.placementUsage();
        return capacity.evaluate(
                                usage,
                                request.getHardKvTokens(),
                                request.getExpectedKvTokens(),
                                CapacityRelease.NONE)
                        .fits()
                ? Availability.READY
                : Availability.BUSY;
    }

    private static Response oversizedRequestFailure(long required, long maximum) {
        return Response.error(
                StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
                "Decode prompt plus output demand="
                        + required
                        + " exceeds every worker KV admission budget; maximum="
                        + maximum);
    }
}

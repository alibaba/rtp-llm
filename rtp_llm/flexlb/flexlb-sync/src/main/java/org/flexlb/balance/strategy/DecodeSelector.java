package org.flexlb.balance.strategy;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.prediction.DecodeCostFormula;
import org.flexlb.balance.scheduler.CapacityKey;
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

/**
 * Ranks Decode candidates by availability and cost, refreshing admission information when registration conflicts.
 */
public class DecodeSelector implements DecodeSelectionStrategy {
    private static final int MAX_CACHED_WORKERS = 4096;

    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

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
    public RoutingResult<DecodeRoutingView, CapacityKey> selectAndRegister(
            Request request,
            SchedulingMetadata metadata,
            String group,
            List<DecodeRoutingView> snapshots,
            CapacityReader reader,
            java.util.function.UnaryOperator<DecodeRoutingView> refresh,
            java.util.function.Function<DecodeRoutingView, RoutingResult.Registration> register) {
        if (snapshots.isEmpty()) return noDecodeCandidates(group);
        Scratch cached = SCRATCH.get();
        Scratch scratch = cached.inUse ? new Scratch() : cached;
        scratch.inUse = true;
        try {
            scratch.prepare(snapshots.size());
            for (var snapshot : snapshots) scratch.views.add(snapshot);
            for (int i = 0; i < snapshots.size(); i++) scratch.costs[i] = cost(snapshots.get(i));
            for (int attempt = 0; attempt < 8; attempt++) {
                var result =
                        selectCandidates(request, metadata, group, scratch.views, reader, scratch);
                if (result.status() != RoutingResult.Status.SUCCESS) return result;
                var selected = result.value();
                var registration = register.apply(selected);
                if (registration == RoutingResult.Registration.REGISTERED) return result;
                if (registration == RoutingResult.Registration.BLOCKED)
                    return RoutingResult.blocked(
                            CapacityKey.exact(
                                    RoleType.DECODE,
                                    selected.topology().group(),
                                    selected.address()),
                            Response.buildErrorResponse(
                                    StrategyErrorType.RESOURCE_EXHAUSTED,
                                    "Decode capacity rejected registration at "
                                            + selected.address()
                                            + "; deficit="
                                            + capacity.evaluate(
                                                    selected.placementUsage(),
                                                    request.getHardKvTokens(),
                                                    request.getExpectedKvTokens(),
                                                    CapacityRelease.NONE)),
                            Map.of(
                                    "cause",
                                    "Decode capacity rejected registration",
                                    "decodeCandidate",
                                    selected,
                                    "capturedAtMs",
                                    System.currentTimeMillis()));
                int index = scratch.views.indexOf(selected);
                var current =
                        registration == RoutingResult.Registration.UNAVAILABLE
                                ? null
                                : refresh.apply(selected);
                scratch.views.set(index, current);
                scratch.costs[index] = current == null ? Double.POSITIVE_INFINITY : cost(current);
            }
            return RoutingResult.retry(
                    Response.buildErrorResponse(
                            StrategyErrorType.SCHEDULER_PLAN_CONFLICT,
                            "Decode selection could not register a current candidate after 8 attempts"));
        } finally {
            scratch.clear(snapshots.size());
            if (scratch == cached && scratch.addresses.length > MAX_CACHED_WORKERS) {
                SCRATCH.remove();
            }
        }
    }

    private RoutingResult<DecodeRoutingView, CapacityKey> selectCandidates(
            Request request,
            SchedulingMetadata metadata,
            String group,
            List<DecodeRoutingView> snapshots,
            CapacityReader reader,
            Scratch scratch) {
        var candidates = scratch.candidates;
        Availability preferred = Availability.IMPOSSIBLE;
        int candidateCount = 0;
        for (int index = 0; index < snapshots.size(); index++) {
            DecodeRoutingView view = snapshots.get(index);
            if (view == null) continue;
            scratch.addresses[index] = view.address();
            Availability available = availability(request, view);
            if (available.ordinal() < preferred.ordinal()) {
                preferred = available;
                candidateCount = 0;
            }
            if (available == preferred) {
                candidates[candidateCount++] = index;
            }
        }
        if (candidateCount == 0) return noDecodeCandidates(group);
        if (preferred == Availability.IMPOSSIBLE) {
            return RoutingResult.rejected(
                    oversizedRequestFailure(
                            request.getExpectedKvTokens(),
                            snapshots.stream()
                                    .filter(Objects::nonNull)
                                    .mapToLong(view -> capacity.kvBudget(view.totalKv()))
                                    .max()
                                    .orElse(0L)));
        }
        if (preferred == Availability.BUSY && direct) {
            return classifyCapacityFailure(request, metadata, group, snapshots, reader);
        }
        int winners = selectLowestCost(scratch, candidateCount);
        int selected =
                rotation.next(
                        RoleType.DECODE, group, scratch.addresses, scratch.candidates, winners);
        checkState(selected >= 0, "Decode snapshot candidate disappeared");
        return RoutingResult.success(snapshots.get(selected));
    }

    private Availability availability(Request request, DecodeRoutingView view) {
        if (view.totalKv() > 0L
                && request.getExpectedKvTokens() > capacity.kvBudget(view.totalKv())) {
            return Availability.IMPOSSIBLE;
        }
        var usage = view.placementUsage();
        return capacity.evaluate(
                                usage,
                                request.getHardKvTokens(),
                                request.getExpectedKvTokens(),
                                CapacityRelease.NONE)
                        .fits()
                ? Availability.READY
                : Availability.BUSY;
    }

    private RoutingResult<DecodeRoutingView, CapacityKey> noDecodeCandidates(String group) {
        String cause = "No current Decode candidate in routing group " + group;
        return RoutingResult.blocked(
                new CapacityKey(RoleType.DECODE, group, null),
                Response.buildErrorResponse(StrategyErrorType.NO_DECODE_WORKER, cause),
                Map.of("cause", cause, "capturedAtMs", System.currentTimeMillis()));
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

    private RoutingResult<DecodeRoutingView, CapacityKey> classifyCapacityFailure(
            Request request,
            SchedulingMetadata metadata,
            String group,
            List<DecodeRoutingView> views,
            CapacityReader reader) {
        Response failure = null;
        boolean uniformFailure = true;
        List<Map<String, Object>> evidence = new ArrayList<>();
        for (DecodeRoutingView view : views) {
            if (view == null) continue;
            Response workerFailure;
            var snapshot = reader.read(view);
            if (snapshot == null) {
                workerFailure =
                        Response.buildErrorResponse(
                                StrategyErrorType.NO_DECODE_WORKER,
                                "Decode candidate changed before capacity attribution: "
                                        + view.address());
                evidence.add(Map.of("endpoint", view.address(), "reason", "CANDIDATE_CHANGED"));
            } else {
                var deficit =
                        capacity.evaluate(
                                snapshot.routing().placementUsage(),
                                request.getHardKvTokens(),
                                request.getExpectedKvTokens(),
                                CapacityRelease.NONE);
                // A capacity sample may recover after ranking; retry rather than invent a shortage.
                if (deficit.fits())
                    return RoutingResult.retry(
                            Response.buildErrorResponse(
                                    StrategyErrorType.SCHEDULER_PLAN_CONFLICT,
                                    "Decode capacity changed between ranking and attribution; retry selection"));
                workerFailure = classifyCapacityFailure(request, metadata, snapshot);
                workerFailure.setErrorMessage(
                        workerFailure.getErrorMessage()
                                + "; endpoint="
                                + view.address()
                                + "; deficit="
                                + deficit);
                var detail =
                        new java.util.LinkedHashMap<String, Object>(
                                capacityEvidence(view.address(), snapshot.routing()));
                detail.put("deficit", deficit);
                detail.put("reason", workerFailure.getErrorMessage());
                evidence.add(Map.copyOf(detail));
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
        if (!uniformFailure) {
            failure =
                    Response.buildErrorResponse(
                            StrategyErrorType.NO_DECODE_WORKER,
                            "No allowed Decode endpoint admitted this request; candidate rejection reasons differ: "
                                    + evidence);
        }
        return RoutingResult.blocked(
                new CapacityKey(RoleType.DECODE, group, null),
                failure,
                Map.of(
                        "cause",
                        failure.getErrorMessage(),
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
        var usage = routing.placementUsage();
        if (usage.totalKvTokens() > 0
                && request.getExpectedKvTokens() > capacity.kvBudget(usage.totalKvTokens())) {
            return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        }
        CapacityRelease lower = CapacityRelease.NONE;
        CapacityRelease higher = CapacityRelease.NONE;
        CapacityRelease same = CapacityRelease.NONE;
        for (int priority = 1; priority <= PriorityNormalizer.MAX_PRIORITY; priority++) {
            CapacityRelease occupied = snapshot.placementOccupancy(priority);
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
        return classifyProtectedCapacity(residual, higher, same);
    }

    private static Response classifyProtectedCapacity(
            DecodeResources.CapacityDeficit residual,
            CapacityRelease higher,
            CapacityRelease same) {
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

    private static Map<String, Object> capacityEvidence(String address, DecodeRoutingView routing) {
        return Map.of(
                "endpoint", address,
                "version", routing.admissionVersion(),
                "engineLoad", routing.engineLoad(),
                "totalLoad", routing.totalLoad(),
                "kvTotal", routing.totalKv(),
                "kvAvailable", routing.placementUsage().hardKvAvailable());
    }

    private int selectLowestCost(Scratch scratch, int candidateCount) {
        var candidates = scratch.candidates;
        double minimumCost = Double.POSITIVE_INFINITY;
        int winners = 0;
        for (int index = 0; index < candidateCount; index++) {
            double cost = scratch.costs[candidates[index]];
            if (!Double.isFinite(cost)) continue;
            if (cost < minimumCost) {
                minimumCost = cost;
                winners = 0;
            }
            if (cost == minimumCost) candidates[winners++] = candidates[index];
        }
        checkState(
                winners > 0,
                "Decode cost formula produced no finite score: %s",
                costFormula.expression());
        return winners;
    }

    /** Evaluate only the selected capacity class; preserve ties for round-robin selection. */
    private double cost(DecodeRoutingView view) {
        return costFormula
                .evaluator()
                .evaluate(
                        view.totalLoad(),
                        capacity.maxEngineRequests(),
                        view.realKvUsed(),
                        view.totalKv());
    }

    private enum Availability {
        READY,
        BUSY,
        IMPOSSIBLE
    } // Ordered from most to least preferred.

    /**
     * Reuses candidate arrays within one thread and guards against reentrant selection sharing the same buffers.
     */
    private static final class Scratch {
        private String[] addresses = new String[0];
        private int[] candidates = new int[0];
        private double[] costs = new double[0];
        private final java.util.ArrayList<DecodeRoutingView> views = new java.util.ArrayList<>();
        private boolean inUse;

        private void prepare(int size) {
            if (candidates.length < size) {
                addresses = new String[size];
                candidates = new int[size];
                costs = new double[size];
            }
        }

        private void clear(int size) {
            Arrays.fill(addresses, 0, Math.min(size, addresses.length), null);
            views.clear();
            inUse = false;
        }
    }
}

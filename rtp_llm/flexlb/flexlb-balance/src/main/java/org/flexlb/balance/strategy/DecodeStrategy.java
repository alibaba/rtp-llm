package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Request;
import org.flexlb.balance.request.Response;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.SchedulingMetadata;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.formula.DecodeCostFormula;
import org.flexlb.worker.endpoint.AdmissionRejectReason;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerAdmissionCapacity;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

import static com.google.common.base.Preconditions.checkState;

/**
 * Filters Decode capacity, compares formula costs and registers the selected worker.
 * Conflicts use the shared bounded retry loop; capacity failures retain their waiting and preemption target.
 */
public class DecodeStrategy
        extends AbstractWorkerSelectionStrategy<DecodeEndpoint, DecodeStrategy.Selection> {
    private static final ThreadLocal<DecodeInputs> DECODE_INPUTS =
            ThreadLocal.withInitial(DecodeInputs::new);

    private static final int MAX_CACHED_WORKERS = 4096;

    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

    private final EndpointRegistry endpoints;
    private final AdmissionCapacity admissionCapacity;

    private final boolean direct;

    private final EndpointRoundRobin rotation = new EndpointRoundRobin();
    private final DecodeCostFormula costFormula;

    private final long maxRequests;

    public DecodeStrategy(
            ConfigService config, AdmissionCapacity capacity, EndpointRegistry endpoints) {
        this.endpoints = Objects.requireNonNull(endpoints);
        this.admissionCapacity = capacity;
        this.direct = config.loadBalanceConfig().isDirect();
        maxRequests = capacity.maxEngineRequests();
        costFormula =
                config.loadBalanceConfig()
                        .getRouter()
                        .getRoles()
                        .getDecode()
                        .getCostEstimator()
                        .compiledFormula();
    }

    public WorkerAdmissionCapacity requestCapacity(DecodeEndpoint endpoint) {
        var snapshot = endpoint.selectionSnapshot();
        long slots =
                maxRequests == 0L
                        ? Integer.MAX_VALUE
                        : Math.max(0L, maxRequests - snapshot.usage().occupiedRequests());
        return new WorkerAdmissionCapacity(
                CapacityKey.exact(RoleType.DECODE, snapshot.topology().group(), snapshot.address()),
                snapshot.generationId(),
                snapshot.admissionVersion(),
                endpoint.acceptsNewRequests() ? slots : 0L);
    }

    public Optional<DecodeEndpoint.SelectionSnapshot> chooseBest(
            String group, List<DecodeEndpoint.SelectionSnapshot> candidates) {
        if (candidates.isEmpty()) return Optional.empty();
        Scratch cached = SCRATCH.get();
        Scratch scratch = cached.inUse ? new Scratch() : cached;
        scratch.prepare(candidates.size());
        scratch.inUse = true;
        try {
            double minimum = Double.POSITIVE_INFINITY;
            int winners = 0;
            for (int i = 0; i < candidates.size(); i++) {
                var candidate = candidates.get(i);
                scratch.addresses[i] = candidate.address();
                double score =
                        costFormula
                                .evaluator()
                                .evaluate(
                                        candidate.usage().occupiedRequests(),
                                        maxRequests,
                                        candidate.realKvUsed(),
                                        candidate.totalKv());
                if (!Double.isFinite(score)) continue;
                if (score < minimum) {
                    minimum = score;
                    winners = 0;
                }
                if (score == minimum) scratch.candidates[winners++] = i;
            }
            checkState(
                    winners > 0,
                    "Decode cost formula produced no finite score: %s",
                    costFormula.expression());
            int selected =
                    rotation.next(
                            RoleType.DECODE, group, scratch.addresses, scratch.candidates, winners);
            return Optional.of(candidates.get(selected));
        } finally {
            Arrays.fill(scratch.addresses, 0, candidates.size(), null);
            scratch.inUse = false;
            if (scratch == cached && scratch.addresses.length > MAX_CACHED_WORKERS)
                SCRATCH.remove();
        }
    }

    @Override
    public SelectionResult<RoleSelection, CapacityKey> selectAndRegister(
            RequestContext context, String group) {
        DecodeInputs cached = DECODE_INPUTS.get();
        DecodeInputs inputs = cached.inUse ? new DecodeInputs() : cached;
        inputs.inUse = true;
        try {
            var scope = new CapacityKey(RoleType.DECODE, group, null);
            return selectCandidates(
                    context,
                    scope,
                    new RegistrationCandidates<>(
                            endpoints.<DecodeEndpoint>endpoints(RoleType.DECODE, null)),
                    candidates -> chooseDecode(context, group, candidates, inputs));
        } finally {
            int captured = inputs.decode.size();
            inputs.decode.clear();
            inputs.eligible.clear();
            inputs.inUse = false;
            if (inputs == cached && captured > 4096) DECODE_INPUTS.remove();
        }
    }

    @Override
    protected DecodeEndpoint candidate(Selection selected) {
        return selected.endpoint();
    }

    @Override
    protected WorkerEndpoint.Registration<RoleSelection> register(
            RequestContext context, RoleType role, String group, Selection selected) {
        return registerDecode(
                context.getRequest(), context.getSchedulingMetadata(), selected.snapshot());
    }

    @Override
    protected SelectionResult<RoleSelection, CapacityKey> capacityFull(
            Selection selected, CapacityKey scope) {
        return decodeCapacityBlocked(selected.snapshot());
    }

    private SelectionResult<Selection, CapacityKey> chooseDecode(
            RequestContext context,
            String group,
            List<DecodeEndpoint> workers,
            DecodeInputs inputs) {
        var candidates = inputs.decode;
        var eligible = inputs.eligible;
        candidates.clear();
        eligible.clear();
        var request = context.getRequest();
        int possible = 0;
        long maximumBudget = 0L;
        for (var endpoint : workers) {
            if (!endpoint.acceptsNewRequests()) continue;
            var candidate = endpoint.selectionSnapshot();
            if (group != null && !group.equals(candidate.topology().group())) continue;
            candidates.add(candidate);
            long budget = admissionCapacity.kvBudget(candidate.totalKv());
            maximumBudget = Math.max(maximumBudget, budget);
            if (candidate.totalKv() > 0L && request.getExpectedKvTokens() > budget) continue;
            possible++;
            if (admissionCapacity
                    .evaluate(
                            candidate.usage(),
                            request.getHardKvTokens(),
                            request.getExpectedKvTokens(),
                            DecodeResources.Capacity.NONE)
                    .isZero()) eligible.add(candidate);
        }
        if (possible == 0 && !candidates.isEmpty())
            return SelectionResult.rejected(
                    Response.error(
                            StrategyErrorType.RESOURCE_EXHAUSTED,
                            org.flexlb.worker.endpoint.AdmissionRejectReason.RESOURCE_EXHAUSTED,
                            "Decode prompt plus output demand="
                                    + request.getExpectedKvTokens()
                                    + " exceeds every worker KV admission budget; maximum="
                                    + maximumBudget));
        boolean capacityFull = eligible.isEmpty() && possible > 0;
        if (capacityFull) {
            if (direct) return decodeCapacityFailure(context, group, candidates).asFailure();
            for (var candidate : candidates)
                if (candidate != null
                        && (candidate.totalKv() == 0L
                                || request.getExpectedKvTokens()
                                        <= admissionCapacity.kvBudget(candidate.totalKv())))
                    eligible.add(candidate);
        }
        var selected = chooseBest(group, java.util.Collections.unmodifiableList(eligible));
        if (selected.isEmpty())
            return SelectionResult.blocked(
                    new CapacityKey(RoleType.DECODE, group, null),
                    Response.error(StrategyErrorType.NO_DECODE_WORKER),
                    null);
        if (capacityFull) return decodeCapacityBlocked(selected.get()).asFailure();
        var snapshot = selected.get();
        for (var endpoint : workers)
            if (endpoint.ipPort().equals(snapshot.address()))
                return SelectionResult.success(new Selection(endpoint, snapshot));
        throw new IllegalStateException("selected Decode worker is not a candidate");
    }

    private SelectionResult<DecodeEndpoint.SelectionSnapshot, CapacityKey> decodeCapacityFailure(
            RequestContext context,
            String group,
            List<DecodeEndpoint.SelectionSnapshot> candidates) {
        Response failure = null;
        boolean uniform = true;
        Map<String, Object> firstRejection = null;
        for (var candidate : candidates) {
            var endpoint = currentEndpoint(candidate);
            var snapshot = candidate;
            DecodeResources.Capacity deficit = null;
            Response reason;
            if (endpoint == null) {
                reason = Response.error(StrategyErrorType.NO_DECODE_WORKER);
            } else {
                var evaluation =
                        endpoint.evaluateSelection(
                                context.getRequest().getHardKvTokens(),
                                context.getRequest().getExpectedKvTokens(),
                                context.getPriority(),
                                admissionCapacity);
                snapshot = evaluation.snapshot();
                deficit = evaluation.deficit();
                reason =
                        switch (evaluation.failure()) {
                            case NONE -> null;
                            case INSUFFICIENT_CAPACITY ->
                                    Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
                            case UNKNOWN_BLOCKER ->
                                    Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE);
                            case BLOCKED_BY_HIGHER_PRIORITY ->
                                    Response.error(
                                            StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                                            AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
                            case BLOCKED_BY_SAME_PRIORITY ->
                                    Response.error(
                                            StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                                            AdmissionRejectReason.SAME_PRIORITY_AHEAD);
                        };
                if (deficit.isZero()) {
                    snapshot = candidate;
                    deficit =
                            admissionCapacity.evaluate(
                                    candidate.usage(),
                                    context.getRequest().getHardKvTokens(),
                                    context.getRequest().getExpectedKvTokens(),
                                    DecodeResources.Capacity.NONE);
                    reason =
                            Response.buildErrorResponse(
                                    StrategyErrorType.NO_DECODE_WORKER,
                                    "No Decode capacity was available in the selection snapshot");
                }
            }
            if (firstRejection == null) {
                if (deficit == null)
                    deficit =
                            admissionCapacity.evaluate(
                                    snapshot.usage(),
                                    context.getRequest().getHardKvTokens(),
                                    context.getRequest().getExpectedKvTokens(),
                                    DecodeResources.Capacity.NONE);
                firstRejection =
                        Map.of(
                                "endpoint",
                                candidate.address(),
                                "version",
                                snapshot.admissionVersion(),
                                "deficit",
                                deficit,
                                "reason",
                                reason.getErrorMessage());
            }
            if (failure == null) failure = reason;
            else if (failure.getCode() != reason.getCode()
                    || failure.getAdmissionRejectReason() != reason.getAdmissionRejectReason())
                uniform = false;
            if (reason.getCode() == StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode())
                failure = reason;
        }
        if (!uniform)
            failure =
                    Response.buildErrorResponse(
                            StrategyErrorType.NO_DECODE_WORKER,
                            "No allowed Decode endpoint admitted this request; candidate rejection reasons differ");
        return SelectionResult.blocked(
                new CapacityKey(RoleType.DECODE, group, null),
                failure,
                Map.of(
                        "cause",
                        failure.getErrorMessage(),
                        "capturedAtMs",
                        System.currentTimeMillis(),
                        "candidateCount",
                        candidates.size(),
                        "firstRejection",
                        firstRejection));
    }

    private DecodeEndpoint currentEndpoint(DecodeEndpoint.SelectionSnapshot expected) {
        var endpoint = endpoints.getEndpoint(RoleType.DECODE, expected.address());
        return endpoint instanceof DecodeEndpoint decode
                        && decode.acceptsNewRequests()
                        && decode.getStatus().getGenerationId() == expected.generationId()
                ? decode
                : null;
    }

    private SelectionResult<RoleSelection, CapacityKey> decodeCapacityBlocked(
            DecodeEndpoint.SelectionSnapshot selected) {
        return SelectionResult.blocked(
                CapacityKey.exact(RoleType.DECODE, selected.topology().group(), selected.address()),
                Response.error(StrategyErrorType.RESOURCE_EXHAUSTED),
                Map.of(
                        "cause",
                        "Decode capacity is insufficient",
                        "decodeCandidate",
                        selected,
                        "capturedAtMs",
                        System.currentTimeMillis()));
    }

    private WorkerEndpoint.Registration<RoleSelection> registerDecode(
            Request request,
            SchedulingMetadata metadata,
            DecodeEndpoint.SelectionSnapshot selected) {
        var endpoint = currentEndpoint(selected);
        if (endpoint == null)
            return WorkerEndpoint.Registration.rejected(
                    WorkerEndpoint.RegistrationStatus.UNAVAILABLE);
        var accepted =
                endpoint.tryRegisterSelection(
                        request.getRequestId(),
                        selected,
                        request.getHardKvTokens(),
                        request.getExpectedKvTokens(),
                        metadata.priority(),
                        admissionCapacity);
        if (accepted.status() != WorkerEndpoint.RegistrationStatus.REGISTERED)
            return WorkerEndpoint.Registration.rejected(accepted.status());
        try {
            return new WorkerEndpoint.Registration<>(
                    WorkerEndpoint.RegistrationStatus.REGISTERED,
                    RoleSelection.decode(endpoint, selected.topology(), accepted.value()));
        } catch (RuntimeException | Error failure) {
            Failures.append(
                    failure,
                    Failures.run(null, () -> endpoint.rollbackSelection(accepted.value())));
            throw failure;
        }
    }

    /**
     * Keeps the chosen endpoint and the capacity values compared during this selection round.
     */
    record Selection(DecodeEndpoint endpoint, DecodeEndpoint.SelectionSnapshot snapshot) {}

    /**
     * Reuses scoring arrays within one Decode selection call; inUse protects nested calls on the same thread.
     */
    private static final class Scratch {
        private String[] addresses = new String[0];
        private int[] candidates = new int[0];
        private boolean inUse;

        private void prepare(int size) {
            if (addresses.length < size) {
                addresses = new String[size];
                candidates = new int[size];
            }
        }
    }

    /**
     * Reuses captured and eligible Decode candidate lists, clearing request-specific references
     * after selection.
     */
    private static final class DecodeInputs {
        final ArrayList<DecodeEndpoint.SelectionSnapshot> decode = new ArrayList<>();
        final ArrayList<DecodeEndpoint.SelectionSnapshot> eligible = new ArrayList<>();
        boolean inUse;
    }
}

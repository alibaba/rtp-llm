package org.flexlb.balance.strategy;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.cache.VitCacheDirectory;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.WorkerStatus;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Prefers VIT cache matches in hash, embedding, GPU and pending order, choosing uniformly among ties.
 * Registers the chosen endpoint through the shared retry loop and records cold cache keys in the directory.
 * Candidates without cache preferences are selected uniformly.
 */
public class VitStrategy
        extends AbstractWorkerSelectionStrategy<VitStrategy.Candidate, VitStrategy.Candidate> {
    private final VitCacheDirectory directory;
    private final EndpointRegistry workers;

    public VitStrategy(VitCacheDirectory directory, EndpointRegistry workers) {
        this.directory = Objects.requireNonNull(directory);
        this.workers = Objects.requireNonNull(workers);
    }

    public synchronized RoleSelection selectForRoute(RequestContext ctx, String group) {
        ServerStatus result =
                ctx.getRequest().getSelectedVit() == null
                        ? selectCache(ctx, group)
                        : validate(ctx, group);
        if (!result.isSuccess()) {
            return null;
        }
        WorkerEndpoint selectedEndpoint =
                workers.getEndpoint(RoleType.VIT, result.getLogicalIpPort());
        if (selectedEndpoint == null || !selectedEndpoint.acceptsNewRequests()) {
            return null;
        }
        if (selectedEndpoint.getStatus().getGenerationId() != result.getWorkerGeneration()) {
            return null;
        }
        return RoleSelection.auxiliary(selectedEndpoint, result);
    }

    public synchronized ServerStatus selectCache(RequestContext ctx, String group) {
        var keys = ctx.getRequest().getCacheAffinityKeys();
        if (keys == null
                || keys.isEmpty()
                || keys.size() > 256
                || keys.stream().anyMatch(k -> k == null || k.isEmpty() || k.length() > 4096)) {
            return errorStatus(StrategyErrorType.INVALID_REQUEST);
        }
        keys = new ArrayList<>(new LinkedHashSet<>(keys));
        Map<String, VitCacheDirectory.Candidate> candidates = directory.candidates(group, keys);
        long now = System.currentTimeMillis();
        var evaluated = new ArrayList<VitStrategy.Candidate>();
        for (VitCacheDirectory.Candidate candidate : candidates.values()) {
            if (group != null && !Objects.equals(group, candidate.worker().getGroup())) continue;
            var endpoint = workers.getEndpoint(RoleType.VIT, candidate.worker().getLogicalIpPort());
            if (endpoint == null
                    || !endpoint.acceptsNewRequests()
                    || endpoint.getStatus().getGenerationId()
                            != candidate.worker().getGenerationId()) continue;
            evaluated.add(
                    new VitStrategy.Candidate(
                            endpoint,
                            candidate.hashHits(),
                            candidate.embeddingHits(),
                            candidate.gpuHits(),
                            directory.pendingHits(candidate, keys, now),
                            candidate));
        }
        var result =
                selectCandidates(
                        ctx,
                        new CapacityKey(RoleType.VIT, group, null),
                        evaluated,
                        current ->
                                select(current)
                                        .map(
                                                value ->
                                                        RoutingResult
                                                                .<Candidate, CapacityKey>success(
                                                                        value))
                                        .orElseGet(
                                                () ->
                                                        RoutingResult.blocked(
                                                                new CapacityKey(
                                                                        RoleType.VIT,
                                                                        group,
                                                                        null))));
        return result.status() == RoutingResult.Status.SUCCESS
                ? result.value().toServerStatus(ctx.getRequestId())
                : errorStatus(StrategyErrorType.NO_VIT_WORKER);
    }

    public Optional<Candidate> select(List<Candidate> candidates) {
        Candidate best = null;
        int ties = 0;
        for (var candidate : candidates) {
            int comparison = best == null ? 1 : compare(candidate, best);
            if (comparison > 0) {
                best = candidate;
                ties = 1;
            } else if (comparison == 0 && ThreadLocalRandom.current().nextInt(++ties) == 0)
                best = candidate;
        }
        return Optional.ofNullable(best);
    }

    public RoutingResult<RoleSelection, CapacityKey> select(RequestContext context, String group) {
        var candidates = new ArrayList<VitStrategy.Candidate>();
        for (var endpoint : workers.endpoints(RoleType.VIT, group)) {
            if (endpoint != null
                    && endpoint.acceptsNewRequests()
                    && (group == null || group.equals(endpoint.getStatus().getGroup())))
                candidates.add(new VitStrategy.Candidate(endpoint, 0, 0, 0, 0, null));
        }
        var scope = new CapacityKey(RoleType.VIT, group, null);
        return selectCandidates(
                context,
                scope,
                candidates,
                current ->
                        this.select(current)
                                .map(
                                        candidate ->
                                                RoutingResult
                                                        .<VitStrategy.Candidate, CapacityKey>
                                                                success(candidate))
                                .orElseGet(() -> RoutingResult.blocked(scope)));
    }

    public synchronized ServerStatus validate(RequestContext ctx, String group) {
        ServerStatus selected = ctx.getRequest().getSelectedVit();
        if (selected == null) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        var selectedEndpoint = workers.getEndpoint(RoleType.VIT, selected.getLogicalIpPort());
        WorkerStatus worker = selectedEndpoint == null ? null : selectedEndpoint.getStatus();
        if (selected.getRole() != RoleType.VIT
                || worker == null
                || !selectedEndpoint.acceptsNewRequests()
                || (group != null && !Objects.equals(group, worker.getGroup()))
                || selected.getGrpcPort() != worker.getGrpcPort()
                || !Objects.equals(selected.getGroup(), worker.getGroup())
                || selected.getWorkerGeneration()
                        != selectedEndpoint.getStatus().getGenerationId()) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        return status(worker, ctx.getRequestId());
    }

    @Override
    protected Candidate candidate(Candidate selected) {
        return selected;
    }

    @Override
    protected Candidate refresh(Candidate candidate) {
        return candidate;
    }

    @Override
    protected WorkerEndpoint.Registration<RoleSelection> register(
            RequestContext context, RoleType role, String group, Candidate selected) {
        var endpoint = selected.endpoint();
        if (!endpoint.acceptsNewRequests()
                || workers.getEndpoint(RoleType.VIT, endpoint.ipPort()) != endpoint
                || group != null && !group.equals(endpoint.getStatus().getGroup()))
            return WorkerEndpoint.Registration.rejected(
                    WorkerEndpoint.RegistrationStatus.UNAVAILABLE);
        if (selected.cache() != null) {
            var keys =
                    new ArrayList<>(
                            new LinkedHashSet<>(context.getRequest().getCacheAffinityKeys()));
            directory.reserve(
                    keys,
                    selected.cache(),
                    context.getRequest().getGenerateTimeout(),
                    System.currentTimeMillis());
        }
        return new WorkerEndpoint.Registration<>(
                WorkerEndpoint.RegistrationStatus.REGISTERED,
                RoleSelection.auxiliary(
                        endpoint, status(endpoint.getStatus(), context.getRequestId())));
    }

    private static int compare(Candidate left, Candidate right) {
        int comparison = Integer.compare(left.hashHits(), right.hashHits());
        if (comparison == 0)
            comparison = Integer.compare(left.embeddingHits(), right.embeddingHits());
        if (comparison == 0) comparison = Integer.compare(left.gpuHits(), right.gpuHits());
        if (comparison == 0) comparison = Integer.compare(left.pendingHits(), right.pendingHits());
        return comparison;
    }

    private static ServerStatus errorStatus(StrategyErrorType error) {
        ServerStatus status = new ServerStatus();
        status.setCode(error.getErrorCode());
        status.setMessage(error.getErrorMsg());
        return status;
    }

    private static ServerStatus status(WorkerStatus worker, String requestId) {
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.VIT);
        status.setWorkerGeneration(worker.getGenerationId());
        status.setServerIp(worker.getIp());
        status.setHttpPort(worker.getPort());
        status.setSelectedEngineIndex(
                worker.topologySnapshot().engineIndex(),
                worker.topologySnapshot().multiEngineNum());
        status.setGrpcPort(worker.getGrpcPort());
        status.setGroup(worker.getGroup());
        status.setRequestId(requestId);
        status.setSuccess(true);
        return status;
    }

    /**
     * Ordinary routing has zero cache scores; cache routing retains its exact directory observation.
     */
    public record Candidate(
            WorkerEndpoint endpoint,
            int hashHits,
            int embeddingHits,
            int gpuHits,
            int pendingHits,
            VitCacheDirectory.Candidate cache) {}
}

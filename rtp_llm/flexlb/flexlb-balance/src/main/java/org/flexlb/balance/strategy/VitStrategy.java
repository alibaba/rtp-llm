package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Response;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.cache.VitWorkerCacheManager;
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
 * Registers the chosen endpoint through the shared retry loop and records pending image keys in the cache manager.
 * Candidates without cache preferences are selected uniformly.
 */
public class VitStrategy
        extends AbstractWorkerSelectionStrategy<VitStrategy.Candidate, VitStrategy.Candidate> {
    private final VitWorkerCacheManager cacheManager;
    private final EndpointRegistry workers;

    public VitStrategy(VitWorkerCacheManager cacheManager, EndpointRegistry workers) {
        this.cacheManager = Objects.requireNonNull(cacheManager);
        this.workers = Objects.requireNonNull(workers);
    }

    public synchronized RoleSelection selectForRoute(RequestContext ctx, String group) {
        RoleSelection result =
                ctx.getRequest().getSelectedVit() == null
                        ? selectCachedWorker(ctx, group).value()
                        : validateSelection(ctx, group);
        if (result == null
                || !result.endpoint().acceptsNewRequests()
                || workers.getEndpoint(RoleType.VIT, result.capacityKey().endpoint())
                        != result.endpoint()) {
            return null;
        }
        return result;
    }

    public synchronized ServerStatus selectCache(RequestContext ctx, String group) {
        var result = selectCachedWorker(ctx, group);
        if (result.status() == SelectionResult.Status.SUCCESS)
            return result.value().toServerStatus(ctx.getRequestId());
        var response = new ServerStatus();
        response.setCode(result.failure().getCode());
        response.setMessage(result.failure().getErrorMessage());
        return response;
    }

    @Override
    public SelectionResult<RoleSelection, CapacityKey> selectAndRegister(
            RequestContext context, String group) {
        var candidates = new ArrayList<VitStrategy.Candidate>();
        for (var endpoint : workers.endpoints(RoleType.VIT, group)) {
            if (endpoint != null
                    && endpoint.acceptsNewRequests()
                    && (group == null || group.equals(endpoint.getStatus().getGroup())))
                candidates.add(new VitStrategy.Candidate(endpoint, 0, 0, 0, 0, null));
        }
        return registerCandidates(context, group, candidates);
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

    @Override
    protected Candidate candidate(Candidate selected) {
        return selected;
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
            cacheManager.reserve(
                    keys,
                    selected.cache(),
                    context.getRequest().getGenerateTimeout(),
                    System.currentTimeMillis());
        }
        return new WorkerEndpoint.Registration<>(
                WorkerEndpoint.RegistrationStatus.REGISTERED,
                RoleSelection.auxiliary(endpoint, endpoint.getStatus().topologySnapshot()));
    }

    synchronized RoleSelection validateSelection(RequestContext ctx, String group) {
        ServerStatus selected = ctx.getRequest().getSelectedVit();
        if (selected == null) return null;
        var endpoint = workers.getEndpoint(RoleType.VIT, selected.getLogicalIpPort());
        WorkerStatus worker = endpoint == null ? null : endpoint.getStatus();
        if (worker == null || !endpoint.acceptsNewRequests()) return null;
        var topology = worker.topologySnapshot();
        if (selected.getRole() != RoleType.VIT
                || group != null && !Objects.equals(group, topology.group())
                || selected.getGrpcPort() != topology.grpcPort()
                || !Objects.equals(selected.getGroup(), topology.group())
                || selected.getWorkerGeneration() != worker.getGenerationId()) return null;
        return RoleSelection.auxiliary(endpoint, topology);
    }

    private SelectionResult<RoleSelection, CapacityKey> selectCachedWorker(
            RequestContext ctx, String group) {
        var keys = ctx.getRequest().getCacheAffinityKeys();
        if (keys == null
                || keys.isEmpty()
                || keys.size() > 256
                || keys.stream().anyMatch(k -> k == null || k.isEmpty() || k.length() > 4096)) {
            return SelectionResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        keys = new ArrayList<>(new LinkedHashSet<>(keys));
        Map<String, VitWorkerCacheManager.Candidate> candidates =
                cacheManager.candidates(group, keys);
        long now = System.currentTimeMillis();
        var evaluated = new ArrayList<VitStrategy.Candidate>();
        for (VitWorkerCacheManager.Candidate candidate : candidates.values()) {
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
                            cacheManager.pendingHits(candidate, keys, now),
                            candidate));
        }
        var result = registerCandidates(ctx, group, evaluated);
        return result.status() == SelectionResult.Status.SUCCESS
                ? result
                : SelectionResult.rejected(Response.error(StrategyErrorType.NO_VIT_WORKER));
    }

    private SelectionResult<RoleSelection, CapacityKey> registerCandidates(
            RequestContext context, String group, List<Candidate> scored) {
        var scope = new CapacityKey(RoleType.VIT, group, null);
        return selectCandidates(
                context,
                scope,
                new RegistrationCandidates<>(scored),
                remaining -> {
                    var selected = select(remaining);
                    return selected.isPresent()
                            ? SelectionResult.success(selected.get())
                            : SelectionResult.blocked(scope);
                });
    }

    private static int compare(Candidate left, Candidate right) {
        int comparison = Integer.compare(left.hashHits(), right.hashHits());
        if (comparison == 0)
            comparison = Integer.compare(left.embeddingHits(), right.embeddingHits());
        if (comparison == 0) comparison = Integer.compare(left.gpuHits(), right.gpuHits());
        if (comparison == 0) comparison = Integer.compare(left.pendingHits(), right.pendingHits());
        return comparison;
    }

    /**
     * Ordinary routing has zero cache scores; cache routing retains the observed worker cache candidate.
     */
    public record Candidate(
            WorkerEndpoint endpoint,
            int hashHits,
            int embeddingHits,
            int gpuHits,
            int pendingHits,
            VitWorkerCacheManager.Candidate cache) {}
}

package org.flexlb.service;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.strategy.VitSelectionStrategy;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.VitCacheDirectory.Candidate;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Selects cache-affine ViT endpoints and owns temporary cold-key placements. */
@Component
public class VitCacheSelector {
    private final VitSelectionStrategy strategy;
    private final VitCacheDirectory directory;
    private final EndpointRegistry workers;

    private record MatchScore(int hashHits, int embeddingHits, int gpuHits, int pendingHits)
            implements Comparable<MatchScore> {
        @Override
        public int compareTo(MatchScore other) {
            int result = Integer.compare(hashHits, other.hashHits);
            if (result == 0) {
                result = Integer.compare(embeddingHits, other.embeddingHits);
            }
            if (result == 0) {
                result = Integer.compare(gpuHits, other.gpuHits);
            }
            return result == 0 ? Integer.compare(pendingHits, other.pendingHits) : result;
        }
    }

    public VitCacheSelector(
            VitCacheDirectory directory, EndpointRegistry workers, VitSelectionStrategy strategy) {
        this.strategy = Objects.requireNonNull(strategy);
        this.directory = directory;
        this.workers = workers;
    }

    public synchronized ServerStatus select(RequestContext ctx, String group) {
        var keys = ctx.data().getRequest().getCacheAffinityKeys();
        if (keys == null
                || keys.isEmpty()
                || keys.size() > 256
                || keys.stream().anyMatch(k -> k == null || k.isEmpty() || k.length() > 4096)) {
            return errorStatus(StrategyErrorType.INVALID_REQUEST);
        }
        keys = new ArrayList<>(new LinkedHashSet<>(keys));
        Map<String, Candidate> candidates = directory.candidates(group, keys);
        long now = System.currentTimeMillis();
        List<String> best = new ArrayList<>();
        MatchScore bestScore = null;
        for (Candidate candidate : candidates.values()) {
            if (group != null && !Objects.equals(group, candidate.worker().getGroup())) {
                continue;
            }
            MatchScore score =
                    new MatchScore(
                            candidate.hashHits(),
                            candidate.embeddingHits(),
                            candidate.gpuHits(),
                            directory.pendingHits(candidate, keys, now));
            int comparison = bestScore == null ? 1 : score.compareTo(bestScore);
            if (comparison > 0) {
                best.clear();
                bestScore = score;
            }
            if (comparison >= 0) {
                best.add(candidate.worker().getLogicalIpPort());
            }
        }
        if (best.isEmpty()) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        int selectedIndex = strategy.selectIndex(List.copyOf(best));
        com.google.common.base.Preconditions.checkArgument(
                selectedIndex >= 0 && selectedIndex < best.size(),
                "VIT strategy returned an invalid index");
        Candidate selected = candidates.get(best.get(selectedIndex));
        directory.reserve(keys, selected, ctx.data().getRequest().getGenerateTimeout(), now);
        return status(selected.worker(), ctx.data().getRequestId());
    }

    /**
     * Revalidates the first selection against the full route's policy group and
     * current endpoint generation. Address equality alone cannot identify a worker
     * after replacement; generation zero is not a wildcard.
     */
    public synchronized ServerStatus validate(RequestContext ctx, String group) {
        ServerStatus selected = ctx.data().getRequest().getSelectedVit();
        if (selected == null) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        try (var pin = workers.capture(RoleType.VIT, selected.getLogicalIpPort())) {
            WorkerStatus worker = pin == null ? null : pin.endpoint().getStatus();
            if (selected.getRole() != RoleType.VIT
                    || worker == null
                    || (group != null && !Objects.equals(group, worker.getGroup()))
                    || selected.getGrpcPort() != worker.getGrpcPort()
                    || !Objects.equals(selected.getGroup(), worker.getGroup())
                    || selected.getWorkerGeneration() != pin.generationId()) {
                return errorStatus(StrategyErrorType.NO_VIT_WORKER);
            }
            return status(worker, ctx.data().getRequestId());
        }
    }

    /** Holds the selected endpoint generation through the admission handoff. */
    public synchronized ServerStatus selectPinned(
            RequestContext ctx, String group, RequestRoute route) {
        ServerStatus result =
                ctx.data().getRequest().getSelectedVit() == null
                        ? select(ctx, group)
                        : validate(ctx, group);
        if (!result.isSuccess()) {
            return null;
        }
        WorkerEndpoint.GenerationPin pin = workers.capture(RoleType.VIT, result.getLogicalIpPort());
        if (pin == null) {
            return null;
        }
        if (pin.generationId() != result.getWorkerGeneration()) {
            pin.close();
            return null;
        }
        return route.add(pin, result, -1L, 0L);
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
}

package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Response;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Selects and registers the least uncached Encoder work, breaking ties by concurrent requests.
 * Each retry compares fresh candidate loads; registration rechecks capacity and the inputs used for ranking.
 */
@Component
public class EncoderStrategy
        extends AbstractWorkerSelectionStrategy<
                EncoderEndpoint, EncoderEndpoint.SelectionSnapshot> {
    private final EndpointRegistry endpoints;

    private final int maxRequests;

    public EncoderStrategy(ConfigService config, EndpointRegistry endpoints) {
        this.endpoints = Objects.requireNonNull(endpoints);
        this.maxRequests =
                config.loadBalanceConfig().getRouter().getRoles().getEncoder().getMaxRequests();
    }

    public SelectionResult<RoleSelection, CapacityKey> selectAndRegister(
            RequestContext context, String group) {
        var candidates = endpoints.<EncoderEndpoint>endpoints(RoleType.ENCODER, group);
        var scope = new CapacityKey(RoleType.ENCODER, group, null);
        var request = context.getRequest();
        return selectCandidates(
                context,
                scope,
                candidates,
                current -> {
                    var snapshots =
                            new ArrayList<EncoderEndpoint.SelectionSnapshot>(current.size());
                    for (var endpoint : current)
                        snapshots.add(
                                endpoint.selectionSnapshot(
                                        request.getEncoderCacheHitLen() != null));
                    return this.select(
                                    snapshots,
                                    group,
                                    request.getEncoderCacheHitLen() != null,
                                    maxRequests)
                            .map(
                                    candidate ->
                                            SelectionResult
                                                    .<org.flexlb.worker.endpoint.EncoderEndpoint
                                                                    .SelectionSnapshot,
                                                            CapacityKey>
                                                            success(candidate))
                            .orElseGet(
                                    () ->
                                            SelectionResult.blocked(
                                                    scope,
                                                    Response.error(
                                                            org.flexlb.worker.endpoint
                                                                    .StrategyErrorType.forRole(
                                                                    RoleType.ENCODER)),
                                                    null));
                });
    }

    public java.util.Optional<EncoderEndpoint.SelectionSnapshot> select(
            List<EncoderEndpoint.SelectionSnapshot> candidates,
            String group,
            boolean weighted,
            int requestLimit) {
        EncoderEndpoint.SelectionSnapshot winner = null;
        long leastTokens = Long.MAX_VALUE;
        long fewestRequests = Long.MAX_VALUE;
        for (var candidate : candidates) {
            if (!candidate.available() || group != null && !group.equals(candidate.group()))
                continue;
            long requests = candidate.requestCount();
            if (requests >= requestLimit) continue;
            long tokens = weighted ? candidate.uncachedTokens() : 0L;
            if (tokens < leastTokens || tokens == leastTokens && requests < fewestRequests) {
                winner = candidate;
                leastTokens = tokens;
                fewestRequests = requests;
            }
        }
        return java.util.Optional.ofNullable(winner);
    }

    @Override
    protected WorkerEndpoint.Registration<RoleSelection> register(
            RequestContext context,
            RoleType role,
            String group,
            EncoderEndpoint.SelectionSnapshot selected) {
        var request = context.getRequest();
        var endpoint = selected.endpoint();
        if (!endpoint.acceptsNewRequests()
                || endpoints.getEndpoint(RoleType.ENCODER, endpoint.ipPort()) != endpoint)
            return WorkerEndpoint.Registration.rejected(
                    WorkerEndpoint.RegistrationStatus.UNAVAILABLE);
        long hit =
                request.getEncoderCacheHitLen() == null
                        ? 0L
                        : Math.max(0L, request.getEncoderCacheHitLen());
        var accepted =
                endpoint.tryRegisterSelection(
                        request.getRequestId(),
                        selected,
                        request.getEncoderCacheHitLen() != null,
                        group,
                        maxRequests,
                        Math.max(0L, request.getSeqLen() - hit));
        if (accepted.status() != WorkerEndpoint.RegistrationStatus.REGISTERED)
            return WorkerEndpoint.Registration.rejected(accepted.status());
        try {
            return new WorkerEndpoint.Registration<>(
                    WorkerEndpoint.RegistrationStatus.REGISTERED,
                    RoleSelection.encoder(
                            endpoint, endpoint.getStatus().topologySnapshot(), accepted.value()));
        } catch (RuntimeException | Error failure) {
            Failures.append(
                    failure,
                    Failures.run(null, () -> endpoint.rollbackSelection(accepted.value())));
            throw failure;
        }
    }

    @Override
    protected EncoderEndpoint refresh(EncoderEndpoint candidate) {
        return candidate;
    }

    @Override
    protected SelectionResult<RoleSelection, CapacityKey> capacityFull(
            EncoderEndpoint.SelectionSnapshot selected, CapacityKey scope) {
        return SelectionResult.blocked(
                CapacityKey.exact(RoleType.ENCODER, selected.group(), selected.address()));
    }

    @Override
    protected EncoderEndpoint candidate(EncoderEndpoint.SelectionSnapshot selected) {
        return selected.endpoint();
    }
}

package org.flexlb.balance.strategy;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.ServerStatus;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Selects and registers the least uncached Encoder work, breaking ties by concurrent requests.
 * Candidate selection and resource registration use the same configured request limit in every scheduling mode.
 */
@Component
public class EncoderStrategy
        extends AbstractWorkerSelectionStrategy<
                EncoderEndpoint.SelectionSnapshot, EncoderEndpoint.SelectionSnapshot> {
    private final EndpointRegistry endpoints;
    private final int maxRequests;

    public EncoderStrategy(ConfigService config, EndpointRegistry endpoints) {
        this.endpoints = Objects.requireNonNull(endpoints);
        this.maxRequests =
                config.loadBalanceConfig().getRouter().getRoles().getEncoder().getMaxRequests();
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

    public RoutingResult<RoleSelection, CapacityKey> select(RequestContext context, String group) {
        var candidates = new ArrayList<EncoderEndpoint.SelectionSnapshot>();
        for (var endpoint : endpoints.<EncoderEndpoint>endpoints(RoleType.ENCODER, group))
            candidates.add(((EncoderEndpoint) endpoint).selectionSnapshot(context.getRequest()));
        var scope = new CapacityKey(RoleType.ENCODER, group, null);
        var request = context.getRequest();
        return selectCandidates(
                context,
                scope,
                candidates,
                current ->
                        this.select(
                                        current,
                                        group,
                                        request.getEncoderCacheHitLen() != null,
                                        maxRequests)
                                .map(
                                        candidate ->
                                                RoutingResult
                                                        .<org.flexlb.worker.endpoint.EncoderEndpoint
                                                                        .SelectionSnapshot,
                                                                CapacityKey>
                                                                success(candidate))
                                .orElseGet(
                                        () ->
                                                RoutingResult.blocked(
                                                        scope,
                                                        Response.error(
                                                                org.flexlb.worker.endpoint
                                                                        .StrategyErrorType.forRole(
                                                                        RoleType.ENCODER)),
                                                        null)));
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
                        maxRequests,
                        Math.max(0L, request.getSeqLen() - hit));
        if (accepted.status() != WorkerEndpoint.RegistrationStatus.REGISTERED)
            return WorkerEndpoint.Registration.rejected(accepted.status());
        try {
            var status = endpoint.getStatus();
            return new WorkerEndpoint.Registration<>(
                    WorkerEndpoint.RegistrationStatus.REGISTERED,
                    RoleSelection.encoder(
                            endpoint,
                            ServerStatus.forWorker(
                                    RoleType.ENCODER,
                                    request.getRequestId(),
                                    status.topologySnapshot(),
                                    status.committedEngineObservation()),
                            accepted.value()));
        } catch (RuntimeException | Error failure) {
            Failures.append(
                    failure,
                    Failures.run(null, () -> endpoint.rollbackSelection(accepted.value())));
            throw failure;
        }
    }

    @Override
    protected EncoderEndpoint.SelectionSnapshot refresh(
            EncoderEndpoint.SelectionSnapshot candidate) {
        return candidate.endpoint().selectionSnapshot();
    }

    @Override
    protected RoutingResult<RoleSelection, CapacityKey> capacityFull(
            EncoderEndpoint.SelectionSnapshot selected, CapacityKey scope) {
        return RoutingResult.blocked(
                CapacityKey.exact(RoleType.ENCODER, selected.group(), selected.address()));
    }

    @Override
    protected EncoderEndpoint.SelectionSnapshot candidate(
            EncoderEndpoint.SelectionSnapshot selected) {
        return selected;
    }

    @Override
    protected boolean tryOtherWhenFull() {
        return false;
    }
}

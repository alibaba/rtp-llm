package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.util.RateLimitedWarn;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Resolves N pre-assigned BE targets in a single shot for the dispatcher by calling the master's
 * {@link BatchScheduleCoordinator} bean directly — same JVM, no HTTP, no JSON round-trip.
 *
 * <p>The dispatcher and master share the 7001 listener and the master's spring beans, so
 * routing this call through {@code WebClient → localhost:7001} would serialize/deserialize
 * the request twice for nothing.
 *
 * <p>All failure paths collapse to {@link List#of()} — coordinator transport errors,
 * business-level {@code success=false} responses, and unexpected exceptions all return an
 * empty list with a single WARN. Callers degrade silently to the no-pre-assignment fanout
 * path; never block real traffic on a routing optimization.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
@RequiredArgsConstructor
public class BatchScheduleClient {

    /**
     * Whole-call bound on the pre-assign resolution. Pre-assignment is a routing optimization
     * the request degrades away from, so it must never pin a {@code /dispatcher} request behind
     * a hung transport (slave forwarding to a wedged master). Local resolution is sub-ms and a
     * healthy master round-trip is a few ms, so 3s is generous headroom while staying far below
     * the request's own FE budget ({@code batchTimeoutMs}); a fire maps into the same
     * degrade-to-empty path as any other failure.
     */
    static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(3);

    private final BatchScheduleCoordinator coordinator;
    /**
     * Stamps the master's single-cursor {@code fe_url} when this node resolves the batch locally
     * (it is the elected master, or consistency is off) — the in-process path that never passes
     * through {@link org.flexlb.httpserver.HttpLoadBalanceServer}, so without this the master's own
     * dispatcher would fail every chunk with {@code CHUNK_NO_FE}. On a slave the response already
     * carries the master's stamp and the assigner is a guarded no-op (see {@link MasterFeAssigner}).
     */
    private final MasterFeAssigner masterFeAssigner;
    /**
     * A misconfigured deployment (multi-role fleet, master unreachable) fails pre-assignment on
     * every batch request; at dispatcher QPS an unlimited WARN per request is a log flood, so
     * cap it like {@link FanoutService} does its chunk-failure WARNs.
     */
    private final RateLimitedWarn noTargetsWarn = new RateLimitedWarn(1, TimeUnit.SECONDS);

    public Mono<List<BatchScheduleTarget>> requestTargets(int count) {
        BatchScheduleRequest req = new BatchScheduleRequest();
        req.setBatchCount(count);
        return coordinator.schedule(req)
                .timeout(REQUEST_TIMEOUT)
                .map(resp -> {
                    if (!resp.isSuccess() || resp.getServerStatus() == null) {
                        noTargetsWarn.warn("dispatcher batch_schedule returned no targets: count={}, success={}, msg={}",
                                count, resp.isSuccess(), resp.getErrorMessage());
                        return List.<BatchScheduleTarget>of();
                    }
                    List<BatchScheduleTarget> targets = resp.getServerStatus();
                    // Master-local (or consistency-off) resolution bypasses the HTTP handler, so it
                    // must stamp fe_url here; a slave's forwarded response is already stamped and the
                    // assigner leaves it untouched. One master cursor, exactly once per request.
                    masterFeAssigner.assign(targets);
                    return targets;
                })
                .switchIfEmpty(Mono.fromSupplier(() -> {
                    noTargetsWarn.warn("dispatcher batch_schedule returned empty Mono: count={}", count);
                    return List.of();
                }))
                .onErrorResume(e -> {
                    noTargetsWarn.warn("dispatcher batch_schedule call failed: count={}, err={}: {}",
                            count, e.getClass().getSimpleName(), e.getMessage());
                    return Mono.just(List.of());
                });
    }
}

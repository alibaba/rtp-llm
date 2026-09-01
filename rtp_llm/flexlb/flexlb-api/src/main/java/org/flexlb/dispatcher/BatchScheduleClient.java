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
 * Resolves N index-aligned allocation targets in a single shot through the master's
 * {@link BatchScheduleCoordinator}. Request flags independently select BE fields and master FE
 * URLs, so callers never move a cursor for a value they will discard.
 *
 * <p>The dispatcher and master share the 7001 listener and the master's spring beans, so
 * routing this call through {@code WebClient → localhost:7001} would serialize/deserialize
 * the request twice for nothing.
 *
 * <p>All failure paths collapse to {@link List#of()} — coordinator transport errors,
 * business-level {@code success=false} responses, and unexpected exceptions all return an
 * empty list with a rate-limited WARN. In master FE mode this makes uncovered chunks fail with
 * {@code CHUNK_NO_FE}; in local FE mode fanout remains independent and only optional BE stamping
 * is lost.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
@RequiredArgsConstructor
public class BatchScheduleClient {

    /**
     * Whole-call bound on target resolution. It must never pin a {@code /dispatcher} request
     * behind a hung transport (slave forwarding to a wedged master). Local resolution is sub-ms and a
     * healthy master round-trip is a few ms, so 3s is generous headroom while staying far below
     * the request's own FE budget ({@code batchTimeoutMs}); a timeout maps into the same empty-target
     * failure path as any other failure.
     */
    static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(3);

    private final BatchScheduleCoordinator coordinator;
    /**
     * When {@code assign_fe=true}, stamps the master's single-cursor {@code fe_url} if this node
     * resolves locally — the in-process path that never passes through
     * {@link org.flexlb.httpserver.HttpLoadBalanceServer}. On a slave the response is already
     * stamped; with {@code assign_fe=false} this is a no-op (see {@link MasterFeAssigner}).
     */
    private final MasterFeAssigner masterFeAssigner;
    /**
     * An unavailable master or unsupported BE topology can fail allocation at dispatcher QPS;
     * cap repeated diagnostics like {@link FanoutService} does its chunk-failure WARNs.
     */
    private final RateLimitedWarn noTargetsWarn = new RateLimitedWarn(1, TimeUnit.SECONDS);

    /** Original wire contract: request both BE and FE assignments. */
    public Mono<List<BatchScheduleTarget>> requestTargets(int count) {
        return requestTargets(count, true, true);
    }

    /**
     * Request exactly the assignment dimensions the dispatcher will consume. Keeping both flags on
     * the request lets FE-only paths avoid BE cursor movement and local-FE mode avoid master FE
     * cursor movement without adding another endpoint or response schema.
     */
    public Mono<List<BatchScheduleTarget>> requestTargets(
            int count, boolean assignBe, boolean assignFe) {
        BatchScheduleRequest req = new BatchScheduleRequest();
        req.setBatchCount(count);
        req.setAssignBe(assignBe);
        req.setAssignFe(assignFe);
        return coordinator.schedule(req)
                .timeout(REQUEST_TIMEOUT)
                .map(resp -> {
                    if (!resp.isSuccess() || resp.getServerStatus() == null) {
                        noTargetsWarn.warn("dispatcher batch_schedule returned no targets: count={}, "
                                        + "assignBe={}, assignFe={}, success={}, msg={}",
                                count, assignBe, assignFe, resp.isSuccess(), resp.getErrorMessage());
                        return List.<BatchScheduleTarget>of();
                    }
                    List<BatchScheduleTarget> targets = resp.getServerStatus();
                    // Master-local (or consistency-off) resolution bypasses the HTTP handler, so it
                    // stamps fe_url here when requested; a slave's forwarded response is already
                    // stamped and the assigner leaves it untouched. One master cursor, at most once.
                    masterFeAssigner.assign(req, targets);
                    return targets;
                })
                .switchIfEmpty(Mono.fromSupplier(() -> {
                    noTargetsWarn.warn("dispatcher batch_schedule returned empty Mono: count={}, "
                            + "assignBe={}, assignFe={}", count, assignBe, assignFe);
                    return List.of();
                }))
                .onErrorResume(e -> {
                    noTargetsWarn.warn("dispatcher batch_schedule call failed: count={}, assignBe={}, "
                                    + "assignFe={}, err={}: {}",
                            count, assignBe, assignFe, e.getClass().getSimpleName(), e.getMessage());
                    return Mono.just(List.of());
                });
    }
}

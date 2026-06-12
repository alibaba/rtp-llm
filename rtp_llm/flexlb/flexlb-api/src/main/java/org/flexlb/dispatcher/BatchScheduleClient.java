package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.List;

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

    private final BatchScheduleCoordinator coordinator;

    public Mono<List<BatchScheduleTarget>> requestTargets(int count) {
        BatchScheduleRequest req = new BatchScheduleRequest();
        req.setBatchCount(count);
        return coordinator.schedule(req)
                .map(resp -> {
                    if (resp == null || !resp.isSuccess() || resp.getServerStatus() == null) {
                        Logger.warn("dispatcher batch_schedule returned no targets: count={}, success={}, msg={}",
                                count,
                                resp != null && resp.isSuccess(),
                                resp == null ? "null" : resp.getErrorMessage());
                        return List.<BatchScheduleTarget>of();
                    }
                    return resp.getServerStatus();
                })
                .switchIfEmpty(Mono.fromSupplier(() -> {
                    Logger.warn("dispatcher batch_schedule returned empty Mono: count={}", count);
                    return List.of();
                }))
                .onErrorResume(e -> {
                    Logger.warn("dispatcher batch_schedule call failed: count={}, err={}: {}",
                            count, e.getClass().getSimpleName(), e.getMessage());
                    return Mono.just(List.of());
                });
    }
}

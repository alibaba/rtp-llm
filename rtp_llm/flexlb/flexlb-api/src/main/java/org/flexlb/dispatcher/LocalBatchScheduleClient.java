package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.util.Logger;
import reactor.core.publisher.Mono;

import java.util.List;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;

/**
 * In-process {@link BatchScheduleClient} that calls the master's
 * {@link BatchScheduleCoordinator} bean directly — same JVM, no HTTP, no JSON round-trip.
 *
 * <p>The dispatcher and master share the 7001 listener and the master's spring beans, so
 * routing this call through {@code WebClient → localhost:7001} would serialize/deserialize
 * the request twice for nothing. Future cross-process deployments can swap in an HTTP
 * implementation without touching {@link GenericBatchHandler}.
 *
 * <p>All failure paths collapse to {@link List#of()} — coordinator transport errors,
 * business-level {@code success=false} responses, and unexpected exceptions all return an
 * empty list with a single WARN. Callers degrade silently to the no-pre-assignment fanout
 * path; never block real traffic on a routing optimization.
 */
@RequiredArgsConstructor
public class LocalBatchScheduleClient implements BatchScheduleClient {

    private final BatchScheduleCoordinator coordinator;

    @Override
    public Mono<List<BatchScheduleTarget>> requestTargets(int count) {
        BatchScheduleRequest req = new BatchScheduleRequest();
        req.setBatchCount(count);
        return coordinator.schedule(req)
                .map(outcome -> {
                    BatchScheduleResponse resp = outcome.getResponse();
                    if (resp == null || !resp.isSuccess() || resp.getServerStatus() == null) {
                        Logger.warn("dispatcher batch_schedule returned no targets: count={}, success={}, msg={}",
                                count,
                                resp != null && resp.isSuccess(),
                                resp == null ? "null" : resp.getErrorMessage());
                        return List.<BatchScheduleTarget>of();
                    }
                    return resp.getServerStatus();
                })
                .onErrorResume(e -> {
                    Logger.warn("dispatcher batch_schedule call failed: count={}, err={}: {}",
                            count, e.getClass().getSimpleName(), e.getMessage());
                    return Mono.just(List.of());
                });
    }
}

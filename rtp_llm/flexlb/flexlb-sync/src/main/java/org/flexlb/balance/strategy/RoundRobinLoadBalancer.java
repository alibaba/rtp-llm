package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.sync.status.WorkerDirectory;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/** Stateless batch placement for a single-role deployment; no LLM admission or reservations. */
@Component
public final class RoundRobinLoadBalancer {

    private final WorkerDirectory workerDirectory;
    private final ModelMetaConfig modelMetaConfig;
    private final FlexlbConfig config;
    private final AtomicLong cursor = new AtomicLong();

    public RoundRobinLoadBalancer(WorkerDirectory workerDirectory,
                                  ModelMetaConfig modelMetaConfig,
                                  ConfigService configService) {
        this.workerDirectory = workerDirectory;
        this.modelMetaConfig = modelMetaConfig;
        this.config = configService.loadBalanceConfig();
    }

    public Mono<BatchScheduleResponse> schedule(int count) {
        return Mono.fromCallable(() -> scheduleBatch(count))
                .subscribeOn(Schedulers.parallel());
    }

    private BatchScheduleResponse scheduleBatch(int count) {
        TrafficPolicyConfig policy = config.getRouter().getGroupSelector();
        if (policy != null && (!policy.getRules().isEmpty() || !policy.getDefaultTargets().isEmpty())) {
            return BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                    "batch_schedule BE allocation is unavailable while traffic policy routing is "
                            + "active; defer backend placement to request-aware /schedule");
        }
        List<RoleType> roles = modelMetaConfig.requiredRoles();
        if (roles.size() != 1) {
            return BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                    "batch_schedule supports single-role deployments only; use /schedule "
                            + "for multi-role routing. Configured roles: " + roles);
        }
        RoleType role = roles.getFirst();
        EngineType engineType = config.getWorkerRegistry().getEngineType();
        List<WorkerHost> candidates = candidates(role, engineType);
        if (candidates.isEmpty()) {
            return BatchScheduleResponse.error(role.getErrorType());
        }
        // Registry iteration order may change between snapshots. A stable order keeps rotation fair.
        candidates.sort(Comparator.comparing(WorkerHost::getIp)
                .thenComparingInt(WorkerHost::getHttpPort));
        long start = cursor.getAndAdd(count);
        List<BatchScheduleTarget> targets = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            WorkerHost candidate = candidates.get(Math.floorMod(start + i, candidates.size()));
            targets.add(BatchScheduleTarget.of(candidate, role, engineType));
        }
        return BatchScheduleResponse.success(targets);
    }

    private List<WorkerHost> candidates(RoleType role, EngineType engineType) {
        if (engineType == EngineType.EMBEDDING) {
            return workerDirectory.statusSnapshot(role).values().stream()
                    .filter(WorkerStatus::isActiveGeneration)
                    .map(status -> new WorkerHost(status.getIp(), status.getPort()))
                    .collect(Collectors.toCollection(ArrayList::new));
        }
        List<WorkerHost> candidates = new ArrayList<>();
        for (String address : workerDirectory.endpointAddressSnapshot(role)) {
            try (WorkerEndpoint.GenerationPin pin = workerDirectory.captureEndpoint(role, address)) {
                if (pin == null) {
                    continue;
                }
                WorkerStatus status = pin.endpoint().getStatus();
                if (status.pollHealth().reportedAlive()) {
                    candidates.add(new WorkerHost(status.getIp(), status.getPort()));
                }
            }
        }
        return candidates;
    }
}

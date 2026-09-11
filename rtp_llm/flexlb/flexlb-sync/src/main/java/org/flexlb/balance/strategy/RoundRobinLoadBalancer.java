package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.sync.synchronizer.MasterEngineSynchronizer;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/** Stateless batch placement for a single-role deployment; no LLM admission or reservations. */
@Component
public final class RoundRobinLoadBalancer {

    private final WorkerDirectory workerDirectory;
    private final MasterEngineSynchronizer synchronizer;
    private final ModelMetaConfig modelMetaConfig;
    private final FlexlbConfig config;
    private final AtomicLong cursor = new AtomicLong();

    public RoundRobinLoadBalancer(WorkerDirectory workerDirectory,
                                  MasterEngineSynchronizer synchronizer,
                                  ModelMetaConfig modelMetaConfig,
                                  ConfigService configService) {
        this.workerDirectory = workerDirectory;
        this.synchronizer = synchronizer;
        this.modelMetaConfig = modelMetaConfig;
        this.config = configService.loadBalanceConfig();
    }

    public Mono<BatchScheduleResponse> schedule(BatchScheduleRequest request) {
        return Mono.fromCallable(() -> scheduleBatch(request))
                .subscribeOn(Schedulers.parallel());
    }

    private BatchScheduleResponse scheduleBatch(BatchScheduleRequest request) {
        int maxCount = config.getRouter().getBatchScheduleMaxCount();
        if (request == null || request.getBatchCount() < 1 || request.getBatchCount() > maxCount) {
            return BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                    "batch_count must be in [1, " + maxCount + "]");
        }
        if (!request.isAssignBe() && !request.isAssignFe()) {
            return BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                    "batch_schedule must request at least one of assign_be or assign_fe");
        }
        int count = request.getBatchCount();
        if (!request.isAssignBe()) {
            List<BatchScheduleTarget> targets = new ArrayList<>(count);
            for (int i = 0; i < count; i++) {
                targets.add(new BatchScheduleTarget());
            }
            return BatchScheduleResponse.success(targets);
        }
        TrafficPolicyConfig policy = config.getRouter().getGroupSelector();
        if (policy != null && (!policy.getRules().isEmpty() || !policy.getDefaultTargets().isEmpty())) {
            return BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                    "batch_schedule assign_be is unavailable while traffic policy routing is "
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
            return new ArrayList<>(synchronizer.embeddingWorkerSnapshot(role));
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

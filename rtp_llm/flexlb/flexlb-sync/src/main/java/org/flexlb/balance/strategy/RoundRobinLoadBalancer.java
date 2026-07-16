package org.flexlb.balance.strategy;

import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Cursor-based round-robin load balancer.
 *
 * <p>Intentionally load-unaware: each call advances a per-role atomic cursor and returns
 * the next alive worker. No queue time, cache hit, or resource-availability gating.
 *
 * <p>Trade-offs vs {@link ShortestTTFTStrategy}:
 * <ul>
 *   <li>Far cheaper schedule cost: no resource scan, no scoring, no sort — selection is a
 *       cursor bump plus an O(alive) liveness filter over the role's worker map</li>
 *   <li>Will hit hot workers proportionally under load skew (no avoidance)</li>
 *   <li>Does not consult {@link org.flexlb.balance.resource.ResourceMeasure};
 *       resource-availability gating is intentionally skipped.</li>
 * </ul>
 *
 * <p>Batch behavior ({@link BatchLoadBalancer}):
 * <ul>
 *   <li>{@link #selectBatch(int, RoleType, String)} advances the per-role cursor {@code count} times.</li>
 *   <li>Cursors are keyed by {@code (role, group)}, so each group filter rotates evenly over
 *       its own subset instead of sampling a shared cursor.</li>
 *   <li>Cursor wraps naturally; when {@code count > alive.size()} one worker receives multiple
 *       assignments (e.g. 20 over 10 workers -&gt; each worker gets 2).</li>
 *   <li>The role cursor is shared with {@link #select}, so single-call and batch-call interleave.</li>
 *   <li>If no workers are alive, returns an empty list.</li>
 *   <li><strong>No bookkeeping</strong>: batch path skips {@code localTaskMap} entirely. Reconciliation
 *       and lost-task detection are not provided for batch dispatches.</li>
 * </ul>
 */
@Component("roundRobinStrategy")
public class RoundRobinLoadBalancer implements BatchLoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;
    /** Engine type is fixed at boot; resolved once instead of a config lookup per selection. */
    private final boolean embeddingEngine;
    private final Map<String, AtomicInteger> cursors = new ConcurrentHashMap<>();

    public RoundRobinLoadBalancer(EngineWorkerStatus engineWorkerStatus, ConfigService configService) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.embeddingEngine = configService.loadBalanceConfig().getEngineType() == EngineType.EMBEDDING;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.ROUND_ROBIN, this);
    }

    private AtomicInteger cursor(RoleType roleType, String group) {
        String key = group == null ? roleType.name() : roleType.name() + '|' + group;
        return cursors.computeIfAbsent(key, k -> new AtomicInteger(0));
    }

    @Override
    public ServerStatus select(BalanceContext context, RoleType roleType, String group) {
        List<WorkerStatus> alive = aliveWorkers(roleType, group);
        if (alive.isEmpty()) {
            return ServerStatus.code(roleType.getErrorType());
        }
        WorkerStatus selected = alive.get(
                Math.floorMod(cursor(roleType, group).getAndIncrement(), alive.size()));
        long requestId = context.getRequestId();
        recordTask(selected, requestId, context.getRequest().getSeqLen());
        return ServerStatus.ok(selected, roleType, requestId);
    }

    @Override
    public List<BatchScheduleTarget> selectBatch(int count, RoleType roleType, String group) {
        List<WorkerStatus> alive = aliveWorkers(roleType, group);
        if (alive.isEmpty()) {
            return new ArrayList<>();
        }
        int aliveSize = alive.size();
        int start = cursor(roleType, group).getAndAdd(count);
        List<BatchScheduleTarget> targets = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int idx = Math.floorMod(start + i, aliveSize);
            targets.add(BatchScheduleTarget.of(alive.get(idx), roleType, embeddingEngine));
        }
        return targets;
    }

    @Override
    public void rollBack(String ipPort, long requestId) {
        for (RoleType role : RoleType.values()) {
            Map<String, WorkerStatus> map = engineWorkerStatus.selectModelWorkerStatus(role, null);
            if (map == null) {
                continue;
            }
            WorkerStatus worker = map.get(ipPort);
            if (worker != null) {
                worker.removeLocalTask(requestId);
                return;
            }
        }
    }

    private List<WorkerStatus> aliveWorkers(RoleType roleType, String group) {
        Map<String, WorkerStatus> map = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        List<WorkerStatus> alive = new ArrayList<>();
        if (map == null) {
            return alive;
        }
        for (WorkerStatus w : map.values()) {
            if (w != null && w.isAlive()) {
                alive.add(w);
            }
        }
        return alive;
    }

    private void recordTask(WorkerStatus worker, long requestId, long seqLen) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(seqLen);
        task.setPrefixLength(0);
        worker.putLocalTask(requestId, task);
    }

}

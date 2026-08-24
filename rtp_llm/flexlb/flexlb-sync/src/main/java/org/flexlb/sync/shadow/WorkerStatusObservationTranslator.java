package org.flexlb.sync.shadow;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.state.StateLedger;
import org.flexlb.state.spi.EngineObservation;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateEndpointRef;
import org.flexlb.state.spi.StateRole;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 状态账本翻译器：WorkerStatusResponse（flexlb-sync 引擎状态轮询报文）→
 * {@link EngineObservation}（flexlb-state 引擎观察契约）。
 *
 * <h2>endpointId / generation（端点世代注册）</h2>
 * endpointId 取 {@code ipPort} 的 {@link String#hashCode()} 稳定映射（跨进程稳定，
 * 同 ipPort 恒等）。generation 经 {@link StateLedger#newGeneration(StateEndpointRef)}
 * 首次注册后按 {@code role:ipPort} 缓存——影子模式下每端点进程生命周期内单代
 * （master 重启时 ledger 重建、epoch 兜底防归零；endpoint 引擎重建的换代信号
 * 由后续里程碑的 EndpointRegistry 代际事件接入，当前保守单代）。
 *
 * <h2>version 字段选择（相位裁决矩阵的版本屏障输入）</h2>
 * 引擎侧无 per-request 单调版本号，running/finished 明细的 version 统一取
 * <b>报级 statusVersion</b>：跨报严格单调（versionAdvanced 分支保证），
 * 同报内并列（相位裁决矩阵对并列版本按迟到/重复语义丢弃）。
 *
 * <h2>上报完整性</h2>
 * detailCount 直传 {@code runningDetailCount}（引擎契约字段）：
 * {@code runningDetailCount == runningTaskInfo.size()} 即完整；旧引擎未填（0）
 * 而 running 非空时自然判为不完整（{@link EngineObservation#isComplete()}）。
 *
 * <h2>相位缺失保守倒推</h2>
 * TaskInfo.phase 为 null（旧引擎未报相位）时保守映射为 {@link EnginePhase#PENDING}，
 * 不静默丢弃明细。
 */
final class WorkerStatusObservationTranslator {

    /** endpointId 稳定来源：ipPort 哈希（String.hashCode 规范定义，跨 JVM 稳定）。 */
    static long endpointIdOf(String ipPort) {
        return ipPort.hashCode();
    }

    private final StateLedger ledger;
    /** role:ipPort → 已注册端点（首次 newGeneration 后缓存，进程内单代）。 */
    private final ConcurrentHashMap<String, EndpointRef> registeredEndpoints = new ConcurrentHashMap<>();

    WorkerStatusObservationTranslator(StateLedger ledger) {
        this.ledger = Objects.requireNonNull(ledger, "ledger");
    }

    /**
     * 翻译一条引擎状态报文。roleType 非 PREFILL/DECODE（如 PDFUSION/VIT）时返回
     * null（账本挂载只覆盖 P/D 分离两侧——融合模式同一引擎兼具 P/D 相位，
     * 单侧账本语义待后续里程碑定义）。
     */
    EngineObservation translate(WorkerStatusResponse response, RoleType roleType, String ipPort) {
        StateRole side = toSide(roleType);
        if (side == null) {
            return null;
        }
        long statusVersion = response.getStatusVersion() != null ? response.getStatusVersion() : 0L;
        long nowMs = System.currentTimeMillis();
        StateEndpointRef endpoint = resolveEndpoint(roleType, ipPort, side);

        List<EngineObservation.RunningObservation> running = new ArrayList<>();
        Map<String, TaskInfo> runningTaskInfo = response.getRunningTaskInfo();
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                if (task == null) {
                    continue;
                }
                running.add(new EngineObservation.RunningObservation(
                        task.getRequestId(),
                        side,
                        toEnginePhase(task.getPhase()),
                        task.getBatchId(),
                        task.getKvTokens(),
                        statusVersion));
            }
        }

        List<EngineObservation.FinishedObservation> finished = new ArrayList<>();
        Map<String, TaskInfo> finishedTaskInfo = response.getFinishedTaskInfo();
        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task == null) {
                    continue;
                }
                finished.add(new EngineObservation.FinishedObservation(
                        task.getRequestId(),
                        side,
                        (int) task.getErrorCode(),
                        task.getEndTimeMs(),
                        statusVersion));
            }
        }

        return new EngineObservation(
                endpoint,
                statusVersion,
                nowMs,
                (int) response.getRunningDetailCount(),
                running,
                finished);
    }

    /**
     * 影子侧 decode 预占的绑定三元组来源：按 role:ipPort 查已注册端点世代，
     * 未见过的端点惰性注册（newGeneration）——不依赖事件泵先到（master 冷启动
     * 后首个 submit 可能早于首次引擎报文）。
     */
    GenerationTripleLike bindingOf(RoleType roleType, String ipPort) {
        StateRole side = toSide(roleType);
        if (side == null) {
            return null;
        }
        EndpointRef ref = resolveEndpoint(roleType, ipPort, side);
        return new GenerationTripleLike(ref.endpointId(), ref.generation());
    }

    /** 影子端点引用（endpointId + role + generation），实现 state SPI 纯值契约。 */
    record EndpointRef(long endpointId, StateRole role, long generation) implements StateEndpointRef {
    }

    /** binding 视图（endpointId/generation 对），避免向 shadow 包外泄漏 state 侧类型。 */
    record GenerationTripleLike(long endpointId, long generation) {
    }

    private EndpointRef resolveEndpoint(RoleType roleType, String ipPort, StateRole side) {
        return registeredEndpoints.computeIfAbsent(key(roleType, ipPort), k -> {
            long endpointId = endpointIdOf(ipPort);
            long generation = ledger.newGeneration(new EndpointRef(endpointId, side, 0L));
            return new EndpointRef(endpointId, side, generation);
        });
    }

    private static String key(RoleType roleType, String ipPort) {
        return roleType + ":" + ipPort;
    }

    private static StateRole toSide(RoleType roleType) {
        if (roleType == RoleType.PREFILL) {
            return StateRole.PREFILL;
        }
        if (roleType == RoleType.DECODE) {
            return StateRole.DECODE;
        }
        return null;
    }

    private static EnginePhase toEnginePhase(TaskPhase phase) {
        // 无显式相位时保守倒推为 PENDING（格内最低已知观察相位），不丢弃明细。
        return phase == null ? EnginePhase.PENDING : EnginePhase.fromTaskPhase(phase);
    }
}

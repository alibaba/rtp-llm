package org.flexlb.sync.shadow;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.state.GenerationTriple;
import org.flexlb.state.RegisterResult;
import org.flexlb.state.SettleReason;
import org.flexlb.state.StateLedger;
import org.flexlb.state.TerminalOutcome;
import org.flexlb.state.TerminalReason;
import org.flexlb.state.TerminalState;
import org.flexlb.state.spi.EngineObservation;
import org.flexlb.state.spi.StateRole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.Optional;

/**
 * G1 影子门面：flexlb-state v2 账本以<b>影子模式</b>接入 flexlb-sync 的唯一桥。
 *
 * <h2>影子语义（铁律）</h2>
 * <ul>
 *   <li><b>旧路径零行为变化</b>：所有公开方法 catch-all 包裹（Throwable →
 *       shadow.error 计数 + 限频 WARN，绝不外抛影响主路径）。</li>
 *   <li><b>开关关 = 零执行</b>：每个入口第一行 {@code if (!enabled) return;}
 *       字节码级短路（装配时 {@link #DISABLED} 单例，开关启动时定、不热切）。</li>
 *   <li><b>同一事件流</b>：事件泵（{@link #observeWorkerStatus}，Runner 水位推进前）
 *       + 本地生命周期点（submit/register 终局/cancel/decode reserve）。</li>
 * </ul>
 *
 * <h2>接入点（M3/G1）</h2>
 * <ol>
 *   <li>{@code GrpcWorkerStatusRunner.handleStatusResponse} versionAdvanced 分支：
 *       旧 calibrate/handleFinishedTasks 之后、latestFinishedVersion 水位推进之前。</li>
 *   <li>{@code AbstractScheduler.register} whenComplete：旧终态 diff 记录
 *       （读 item.state()）+ CANCELLED 影子双清（settle 双侧 LOCAL_CANCEL）。</li>
 *   <li>{@code BatchScheduler.submit}：P 侧 register+onQueued、D 侧 reserve 影子。</li>
 *   <li>{@code RouteService.cancel}：双侧 markPendingCancel（CAS 前意图标记）。</li>
 * </ol>
 *
 * <h2>开账语义（与 M2 ledger 契约对齐）</h2>
 * 正常 observe 模式下，引擎 running/finished 明细对<b>未开账条目只计 unknown
 * 事件、不收养</b>（收养仅 rebuild 重放路径）——因此本地生命周期点是开账前置：
 * P 侧由 onPrefillSubmit（register+onQueued）开账，D 侧由 onDecodeReserve
 * 开账（binding 由 translator 惰性注册端点世代，不依赖事件泵先到）；
 * 引擎事件流随后推进相位与终局。
 */
public final class StateShadowBridge {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    /** 关态单例：所有方法短路返回（装配处开关关时注入本实例）。 */
    public static final StateShadowBridge DISABLED = new StateShadowBridge();

    private final boolean enabled;
    private final StateLedger ledger;
    private final WorkerStatusObservationTranslator translator;
    private final StateShadowDiffCollector diff;

    /** DISABLED 构造。 */
    private StateShadowBridge() {
        this.enabled = false;
        this.ledger = null;
        this.translator = null;
        this.diff = null;
    }

    private StateShadowBridge(StateLedger ledger,
                              WorkerStatusObservationTranslator translator,
                              StateShadowDiffCollector diff) {
        this.enabled = true;
        this.ledger = Objects.requireNonNull(ledger);
        this.translator = Objects.requireNonNull(translator);
        this.diff = Objects.requireNonNull(diff);
    }

    /**
     * 装配工厂：开关关返回 {@link #DISABLED}；开时构建 ledger/translator/diff
     * 并注册影子指标、打印启用回显（R2 补充：ConfigService dump 之外的显式标记）。
     */
    public static StateShadowBridge create(FlexlbConfig config, FlexMonitor monitor) {
        Objects.requireNonNull(config, "config");
        if (!config.isFlexlbStateV2ShadowEnabled()) {
            logger.info("[state-shadow] flexlb-state v2 shadow mode disabled (flexlbStateV2ShadowEnabled=false)");
            return DISABLED;
        }
        StateLedger ledger = new StateLedger();
        StateShadowDiffCollector diff = new StateShadowDiffCollector(monitor);
        diff.registerMetrics();
        WorkerStatusObservationTranslator translator = new WorkerStatusObservationTranslator(ledger);
        logger.warn("[state-shadow] flexlb-state v2 shadow mode ENABLED "
                + "(env FLEXLB_STATE_V2_SHADOW_ENABLED / flexlbStateV2ShadowEnabled=true): "
                + "StateLedger now consuming the same event stream in shadow; legacy path unchanged");
        return new StateShadowBridge(ledger, translator, diff);
    }

    public boolean isEnabled() {
        return enabled;
    }

    // ==================== 事件泵（Runner 挂点）====================

    /**
     * 引擎状态报文影子消费（versionAdvanced 分支、latestFinishedVersion 水位推进之前
     * 调用——保证 S4 事件顺序与旧路径同 tick 一致）。
     */
    public void observeWorkerStatus(WorkerStatusResponse response, RoleType roleType, String ipPort) {
        if (!enabled) {
            return;
        }
        try {
            EngineObservation observation = translator.translate(response, roleType, ipPort);
            if (observation == null) {
                return; // 非 P/D 分离角色（PDFUSION/VIT）：影子 G1 不覆盖
            }
            ledger.observe(observation);
            diff.onEvent();
            // finished 明细 → 影子终态对账（observe 同步完成后墓碑即可见）
            for (EngineObservation.FinishedObservation finished : observation.finished()) {
                recordNewTerminalIfSettled(finished.requestId());
            }
        } catch (Throwable t) {
            diff.onError(t);
        }
    }

    // ==================== 本地生命周期点（Scheduler 挂点）====================

    /** BatchScheduler.submit：P 侧影子入账（register + onQueued，散请求 batchId=-1）。 */
    public void onPrefillSubmit(long requestId) {
        if (!enabled) {
            return;
        }
        try {
            RegisterResult result = ledger.prefill().register(requestId, -1L);
            if (result == RegisterResult.OK) {
                ledger.prefill().onQueued(requestId);
            }
        } catch (Throwable t) {
            diff.onError(t);
        }
    }

    /**
     * BatchScheduler.submit 路由成功后：D 侧影子预占（D① 影子双轨的预占侧，
     * 开账前置——正常 observe 不收养未开条目；binding 由 translator 惰性注册）。
     */
    public void onDecodeReserve(long requestId, long seqLen, long expectedKv,
                                RoleType roleType, String ipPort) {
        if (!enabled) {
            return;
        }
        try {
            WorkerStatusObservationTranslator.GenerationTripleLike binding =
                    translator.bindingOf(roleType, ipPort);
            if (binding == null) {
                return;
            }
            ledger.decode().reserve(requestId, seqLen, expectedKv,
                    new GenerationTriple((int) binding.endpointId(), binding.generation(), -1L));
        } catch (Throwable t) {
            diff.onError(t);
        }
    }

    /** RouteService.cancel：双侧 pendingCancel 意图标记（CAS 前调用；终局由 onOldTerminal 双清）。 */
    public void onLocalCancelRequested(long requestId) {
        if (!enabled) {
            return;
        }
        try {
            ledger.prefill().markPendingCancel(requestId);
            ledger.decode().markPendingCancel(requestId);
        } catch (Throwable t) {
            diff.onError(t);
        }
    }

    /**
     * AbstractScheduler.register whenComplete：旧路径终态（item.state() 名称）。
     * CANCELLED 时同 tick 影子双清（两侧 settle LOCAL_CANCEL，cancel 双清语义），
     * 随后按墓碑记录新侧终态进入 diff 对账。
     */
    public void onOldTerminal(long requestId, String oldStateName) {
        if (!enabled) {
            return;
        }
        try {
            if ("CANCELLED".equals(oldStateName)) {
                // 本地取消：无引擎 ack 证据，推定取消成立（与 diff 等价集 CANCELLED 族对齐）。
                TerminalOutcome outcome = new TerminalOutcome(
                        TerminalState.CANCELLED, TerminalReason.CANCELLED_IMPLICIT, "shadow:old-path-cancel");
                ledger.prefill().settle(requestId, outcome, SettleReason.LOCAL_CANCEL);
                ledger.decode().settle(requestId, outcome, SettleReason.LOCAL_CANCEL);
                recordNewTerminalIfSettled(requestId);
            }
            diff.recordOldTerminal(requestId, oldStateName);
        } catch (Throwable t) {
            diff.onError(t);
        }
    }

    // ==================== 内部 ====================

    /**
     * 新侧主终态：D 侧墓碑优先（decode 完成即请求完成）、P 侧兜底（prefill 阶段
     * 失败/取消，D 侧无条目时 P 终局即终局）。
     */
    private void recordNewTerminalIfSettled(long requestId) {
        Optional<TerminalOutcome> decodeOutcome = ledger.terminalOutcomeOf(requestId, StateRole.DECODE);
        if (decodeOutcome.isPresent()) {
            diff.recordNewTerminal(requestId, decodeOutcome.get().state(), decodeOutcome.get().reason());
            return;
        }
        Optional<TerminalOutcome> prefillOutcome = ledger.terminalOutcomeOf(requestId, StateRole.PREFILL);
        prefillOutcome.ifPresent(o -> diff.recordNewTerminal(requestId, o.state(), o.reason()));
    }

    /** 影子诊断（日志/测试用；不参与主路径）。 */
    public StateShadowDiffCollector diffCollector() {
        return diff;
    }

    public StateLedger ledger() {
        return ledger;
    }
}

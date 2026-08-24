package org.flexlb.sync.shadow;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.state.GenerationTriple;
import org.flexlb.state.LedgerJanitorConfig;
import org.flexlb.state.RegisterResult;
import org.flexlb.state.SettleReason;
import org.flexlb.state.StateLedger;
import org.flexlb.state.TerminalOutcome;
import org.flexlb.state.TerminalReason;
import org.flexlb.state.TerminalState;
import org.flexlb.state.internal.LedgerJanitor;
import org.flexlb.state.spi.EngineObservation;
import org.flexlb.state.spi.StateRole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

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
    /** M4 清理层（影子开时挂载；关态 null）。 */
    private final LedgerJanitor janitor;
    /** janitor 维护 tick 调度（autoStart=false 时 null；close 时停）。 */
    private final ScheduledExecutorService janitorScheduler;

    /** DISABLED 构造。 */
    private StateShadowBridge() {
        this.enabled = false;
        this.ledger = null;
        this.translator = null;
        this.diff = null;
        this.janitor = null;
        this.janitorScheduler = null;
    }

    private StateShadowBridge(StateLedger ledger,
                              WorkerStatusObservationTranslator translator,
                              StateShadowDiffCollector diff,
                              LedgerJanitor janitor,
                              ScheduledExecutorService janitorScheduler) {
        this.enabled = true;
        this.ledger = Objects.requireNonNull(ledger);
        this.translator = Objects.requireNonNull(translator);
        this.diff = Objects.requireNonNull(diff);
        this.janitor = janitor;
        this.janitorScheduler = janitorScheduler;
    }

    /**
     * 装配工厂（生产）：开关关返回 {@link #DISABLED}；开时构建 ledger/translator/diff/
     * janitor 并注册影子指标、启动 janitor 维护调度、打印启用回显（R2）。
     */
    public static StateShadowBridge create(FlexlbConfig config, FlexMonitor monitor) {
        return create(config, monitor, true);
    }

    /**
     * 装配工厂（测试钩子）：{@code autoStartJanitor=false} 时不创建调度线程
     * （janitor 仍挂载，由 {@link #runJanitorOnce()} 手动驱动——确定性测试）。
     */
    public static StateShadowBridge create(FlexlbConfig config, FlexMonitor monitor, boolean autoStartJanitor) {
        Objects.requireNonNull(config, "config");
        if (!config.isFlexlbStateV2ShadowEnabled()) {
            logger.info("[state-shadow] flexlb-state v2 shadow mode disabled (flexlbStateV2ShadowEnabled=false)");
            return DISABLED;
        }
        StateLedger ledger = new StateLedger();
        StateShadowDiffCollector diff = new StateShadowDiffCollector(monitor);
        diff.registerMetrics();
        WorkerStatusObservationTranslator translator = new WorkerStatusObservationTranslator(ledger);

        // M4：清理层四通道（F1-F4 + 三护栏）挂载；janitor 胜者结算同步进 diff 对账窗口
        LedgerJanitorConfig janitorConfig = new LedgerJanitorConfig(
                config.getFlexlbStateV2StaleRounds(),
                config.getFlexlbStateV2TtlMs(),
                config.getFlexlbStateV2HardCapMs(),
                LedgerJanitorConfig.DEFAULT_SCAN_BUDGET_PER_TICK);
        LedgerJanitor janitor = ledger.createJanitor(janitorConfig);
        janitor.setSettleListener((requestId, side) -> recordNewTerminalForJanitor(ledger, diff, requestId));

        ScheduledExecutorService scheduler = null;
        if (autoStartJanitor) {
            scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "flexlb-state-janitor");
                t.setDaemon(true);
                t.setUncaughtExceptionHandler((thread, e) ->
                        logger.warn("[state-shadow] janitor scheduler thread died: {}", e.getMessage(), e));
                return t;
            });
            long intervalMs = config.getFlexlbStateV2JanitorIntervalMs();
            scheduler.scheduleAtFixedRate(janitor::runMaintenanceTick,
                    intervalMs, intervalMs, TimeUnit.MILLISECONDS);
        }

        logger.warn("[state-shadow] flexlb-state v2 shadow mode ENABLED "
                + "(env FLEXLB_STATE_V2_SHADOW_ENABLED / flexlbStateV2ShadowEnabled=true): "
                + "StateLedger now consuming the same event stream in shadow; legacy path unchanged; "
                + "LedgerJanitor " + (autoStartJanitor ? "scheduled" : "mounted (manual tick)")
                + " interval=" + config.getFlexlbStateV2JanitorIntervalMs() + "ms"
                + " staleRounds=" + config.getFlexlbStateV2StaleRounds()
                + " ttlMs=" + config.getFlexlbStateV2TtlMs()
                + " hardCapMs=" + config.getFlexlbStateV2HardCapMs());
        return new StateShadowBridge(ledger, translator, diff, janitor, scheduler);
    }

    /**
     * 生命周期收尾（Spring @Bean destroyMethod 自动推断）：停 janitor 维护调度。
     * 幂等；仅影子链路资源——旧路径不受影响。
     */
    public void close() {
        ScheduledExecutorService scheduler = janitorScheduler;
        if (scheduler != null) {
            scheduler.shutdownNow();
        }
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

    /** M4 清理层（关态 null；观察/装配测试用）。 */
    public LedgerJanitor janitor() {
        return janitor;
    }

    /** 手动驱动一 janitor 维护 tick（测试钩子；生产走 flexlb-state-janitor 调度线程）。 */
    public void runJanitorOnce() {
        LedgerJanitor j = janitor;
        if (j != null) {
            j.runMaintenanceTick();
        }
    }

    /**
     * janitor 胜者结算的新侧终态进 diff 对账（静态：create 时的 settleListener 引用，
     * 与 recordNewTerminalIfSettled 同语义——D 侧墓碑优先、P 侧兑底）。
     */
    private static void recordNewTerminalForJanitor(StateLedger ledger, StateShadowDiffCollector diff,
                                                    long requestId) {
        Optional<TerminalOutcome> decodeOutcome = ledger.terminalOutcomeOf(requestId, StateRole.DECODE);
        if (decodeOutcome.isPresent()) {
            diff.recordNewTerminal(requestId, decodeOutcome.get().state(), decodeOutcome.get().reason());
            return;
        }
        Optional<TerminalOutcome> prefillOutcome = ledger.terminalOutcomeOf(requestId, StateRole.PREFILL);
        prefillOutcome.ifPresent(o -> diff.recordNewTerminal(requestId, o.state(), o.reason()));
    }
}

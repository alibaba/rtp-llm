package org.flexlb.sync.shadow;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.state.DecodeEndpointCounters;
import org.flexlb.state.GenerationTriple;
import org.flexlb.state.LedgerJanitorConfig;
import org.flexlb.state.PrefillEndpointCounters;
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
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * flexlb-state v2 账本接入 flexlb-sync 的唯一桥（账本启用后为全权模式：
 * 终态结算单出口 + 调度读点唯一数据源）。
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
 * <h2>接入点</h2>
 * <ol>
 *   <li>{@code GrpcWorkerStatusRunner.handleStatusResponse} versionAdvanced 分支：
 *       旧 calibrate/handleFinishedTasks 之后、latestFinishedVersion 水位推进之前。</li>
 *   <li>{@code AbstractScheduler.register} whenComplete：旧终态 diff 记录
 *       （读 item.state()）+ CANCELLED 影子双清（settle 双侧 LOCAL_CANCEL）。</li>
 *   <li>{@code BatchScheduler.submit}：P 侧 register+onQueued、D 侧 reserve 影子。</li>
 *   <li>{@code RouteService.cancel}：双侧 markPendingCancel（CAS 前意图标记）。</li>
 * </ol>
 *
 * <h2>开账语义（与 ledger 契约对齐）</h2>
 * 正常 observe 模式下，引擎 running/finished 明细对<b>未开账条目只计 unknown
 * 事件、不收养</b>（收养仅 rebuild 重放路径）——因此本地生命周期点是开账前置：
 * P 侧由 onPrefillSubmit（register+onQueued）开账，D 侧由 onDecodeReserve
 * 开账（binding 由 translator 惰性注册端点世代，不依赖事件泵先到）；
 * 引擎事件流随后推进相位与终局。
 *
 * <h2>结算换权（终态结算单出口，恒开）</h2>
 * 账本启用时 BATCH 路径的终态结算收敛到 ledger 权威单出口（
 * {@link #onOldTerminalAuthority}）：调度链只负责客户端 future（客户端可见行为
 * 不变）；终态 metric 生产点在 ledger settle 出口（每请求恰好一次）。
 * 旧全局 inflight 路径已移除——flexlbStateV2SettleEnabled 开关退化为 no-op
 * （启动时 WARN 提示，恒视为开）。
 *
 * <h2>读取换权（调度读点切账本，恒开）</h2>
 * 账本启用时（flexlbStateV2ReadEnabled 开关同样已废弃、恒视为开）：
 * <ul>
 *   <li>DecodeEndpoint 的 reserve/release 真实记账走 {@link #decodeReserveAuthority}
 *       / {@link #decodeReleaseAuthority}（ledger.decode() 单入账）；</li>
 *   <li>调度读点（总负载 / 真实 KV 用量与可用量 / 容量 gate / reporter）走
 *       {@link #decodeEndpointCounters} / {@link #prefillEndpointCounters}
 *       （端点级增量计数簿的无锁 O(1) 快照——读侧零条目遍历，调用方在引擎状态
 *       tick 时缓存刷新）；BATCH 准入 gate 与调度指标走
 *       {@link #prefillActiveCount()}（全账本活跃计数）；</li>
 *   <li>P 侧条目在派发提交时经 {@link #onPrefillDispatched} 绑定端点世代。</li>
 * </ul>
 */
public final class StateShadowBridge {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    /** 关态单例：所有方法短路返回（装配处开关关时注入本实例）。 */
    public static final StateShadowBridge DISABLED = new StateShadowBridge();

    private final boolean enabled;
    private final StateLedger ledger;
    private final WorkerStatusObservationTranslator translator;
    private final StateShadowDiffCollector diff;
    /** 清理层（账本开时挂载；关态 null）。 */
    private final LedgerJanitor janitor;
    /** janitor 维护 tick 调度（autoStart=false 时 null；close 时停）。 */
    private final ScheduledExecutorService janitorScheduler;

    // ---- 终态结算与调度读点（旧路径移除后恒开；开关已废弃） ----

    /** 终态 metric 统一出口 helper（与 BATCH 调度器同 path tag；monitor null 时 NullSafe）。 */
    private final FlexlbMetricHelper terminalMetricHelper;

    /**
     * COMPLETED（ACK）终态的挂起 metric 表：requestId → metric 上下文。ledger
     * 终局（引擎 finished / janitor 胜者结算）时消费——每请求恰好一次的单点生产。
     */
    private final ConcurrentHashMap<Long, TerminalMetricContext> pendingTerminalMetrics;

    /**
     * 已见端点键（role:ipPort）：master 重启后每端点首份报文走收养观察
     * （{@link StateLedger#observeAdopting}）——引擎上报即当前 inflight 事实，
     * 未开账 running 收养入账，恢复重启前丢失的账本计数。首报资格以
     * translate 成功为准（翻译异常的报文不消耗）。
     */
    private final Set<String> observedEndpoints;

    /** DISABLED 构造。 */
    private StateShadowBridge() {
        this.enabled = false;
        this.ledger = null;
        this.translator = null;
        this.diff = null;
        this.janitor = null;
        this.janitorScheduler = null;
        this.terminalMetricHelper = null;
        this.pendingTerminalMetrics = null;
        this.observedEndpoints = null;
    }

    private StateShadowBridge(StateLedger ledger,
                              WorkerStatusObservationTranslator translator,
                              StateShadowDiffCollector diff,
                              LedgerJanitor janitor,
                              ScheduledExecutorService janitorScheduler,
                              FlexlbMetricHelper terminalMetricHelper) {
        this.enabled = true;
        this.ledger = Objects.requireNonNull(ledger);
        this.translator = Objects.requireNonNull(translator);
        this.diff = Objects.requireNonNull(diff);
        this.janitor = janitor;
        this.janitorScheduler = janitorScheduler;
        this.terminalMetricHelper = terminalMetricHelper;
        this.pendingTerminalMetrics = new ConcurrentHashMap<>();
        this.observedEndpoints = ConcurrentHashMap.newKeySet();
    }

    /**
     * 装配工厂（生产）：开关关返回 {@link #DISABLED}；开时构建 ledger/translator/diff/
     * janitor 并注册影子指标、启动 janitor 维护调度、打印启用回显。
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
            logger.info("[state-shadow] flexlb-state v2 ledger disabled (flexlbStateV2ShadowEnabled=false): "
                    + "degraded mode — duplicate admission gate, global cancel and ledger metrics unavailable");
            return DISABLED;
        }
        // 开关收束：旧全局 inflight 路径已移除，结算/读取换权没有退路——
        // flexlbStateV2SettleEnabled / flexlbStateV2ReadEnabled 配置已废弃，
        // 账本启用即全权模式（终态结算与调度读点均以账本为唯一源）。
        if (!config.isFlexlbStateV2SettleEnabled() || !config.isFlexlbStateV2ReadEnabled()) {
            logger.warn("[state-shadow] flexlbStateV2SettleEnabled / flexlbStateV2ReadEnabled "
                    + "are deprecated (legacy inflight path removed): both are permanently ON "
                    + "while the state ledger is enabled; explicit false values are ignored");
        }
        StateLedger ledger = new StateLedger();
        StateShadowDiffCollector diff = new StateShadowDiffCollector(monitor);
        diff.registerMetrics();
        WorkerStatusObservationTranslator translator = new WorkerStatusObservationTranslator(ledger);

        // 清理层四通道挂载；janitor 胜者结算同步进 diff 对账窗口（listener 在
        // bridge 实例化后挂——见下方结算换权段，与调度线程启动前完成，无竞态）。
        LedgerJanitorConfig janitorConfig = new LedgerJanitorConfig(
                config.getFlexlbStateV2StaleRounds(),
                config.getFlexlbStateV2TtlMs(),
                config.getFlexlbStateV2HardCapMs(),
                LedgerJanitorConfig.DEFAULT_SCAN_BUDGET_PER_TICK);
        LedgerJanitor janitor = ledger.createJanitor(janitorConfig);

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

        // 终态 metric 统一出口 helper（与 BATCH 调度器同 path tag，指标口径连续；
        // monitor null（测试）时 helper 全部 NullSafe no-op）。
        FlexlbMetricHelper terminalMetricHelper = new FlexlbMetricHelper(monitor, MetricConstant.PATH_BATCH);
        terminalMetricHelper.register();

        StateShadowBridge bridge = new StateShadowBridge(ledger, translator, diff, janitor, scheduler,
                terminalMetricHelper);
        // janitor 胜者结算 → diff 对账 + 挂起 metric 消费（与引擎 finished 终局同语义）。
        // 首个维护 tick 至少在 intervalMs 之后，listener 同步挂载在前——无竞态。
        janitor.setSettleListener(bridge::onJanitorSettled);

        logger.warn("[state-shadow] flexlb-state v2 ledger ENABLED "
                + "(env FLEXLB_STATE_V2_SHADOW_ENABLED / flexlbStateV2ShadowEnabled=true): "
                + "StateLedger is the single source of truth for terminal settlement, "
                + "scheduling reads and admission gating; legacy inflight path removed; "
                + "LedgerJanitor " + (autoStartJanitor ? "scheduled" : "mounted (manual tick)")
                + " interval=" + config.getFlexlbStateV2JanitorIntervalMs() + "ms"
                + " staleRounds=" + config.getFlexlbStateV2StaleRounds()
                + " ttlMs=" + config.getFlexlbStateV2TtlMs()
                + " hardCapMs=" + config.getFlexlbStateV2HardCapMs() + ");");
        return bridge;
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
        // shutdown summary：全量对账统计单行出口（验收 gate 数据源——
        // 指标通道未部署时（如压测环境无 pushgateway）日志即权威证据）。
        if (enabled && diff != null) {
            logger.warn("[state-shadow] shutdown summary: {}", diff.summaryLine());
        }
    }

    public boolean isEnabled() {
        return enabled;
    }

    // ==================== 事件泵（Runner 挂点）====================

    /**
     * 引擎状态报文影子消费（versionAdvanced 分支、latestFinishedVersion 水位推进之前
     * 调用——保证相位事件顺序与旧路径同 tick 一致）。
     *
     * <p>每端点首份报文走收养观察（master 重启重建：未开账 running 按引擎事实
     * 收养入账，恢复 inflight 计数——引擎上报是唯一事实源）；后续报文正常
     * observe。首报带 running 时打 WARN（重启重建的运行证据，验收日志佐证）。</p>
     */
    public void observeWorkerStatus(WorkerStatusResponse response, RoleType roleType, String ipPort) {
        if (!enabled) {
            return;
        }
        try {
            EngineObservation observation = translator.translate(response, roleType, ipPort);
            if (observation == null) {
                return; // 非 P/D 分离角色（PDFUSION/VIT）：事件泵不覆盖
            }
            boolean firstReport = ipPort != null
                    && observedEndpoints.add(roleType + ":" + ipPort);
            if (firstReport) {
                ledger.observeAdopting(observation);
                if (!observation.running().isEmpty()) {
                    logger.warn("[state-shadow] master restart rebuild: first report from {} endpoint {} "
                            + "adopted {} running entries (engine report is the source of truth)",
                            roleType, ipPort, observation.running().size());
                }
            } else {
                ledger.observe(observation);
            }
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
     * BatchScheduler.submit 路由成功后：D 侧影子预占（影子双轨记账的预占侧，
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

    // ==================== 终态结算（权威单出口） ====================

    /** 终态结算单出口（旧路径移除后：账本启用即 true；关态 false）。 */
    public boolean isSettleAuthority() {
        return enabled;
    }

    // ==================== 调度读点（账本为唯一数据源） ====================

    /** 调度读点切账本（旧路径移除后：账本启用即 true；关态 false）。 */
    public boolean isReadAuthority() {
        return enabled;
    }

    /**
     * P 侧全账本活跃条目数（BATCH 准入 gate 与调度指标的数据源）：刷新端点级
     * 计数簿快照后读 inflight 总量。未启用（退化模式）返回 -1——调用方据此
     * 跳过 gate（无账本即无计数源）。
     */
    public long prefillActiveCount() {
        if (!enabled) {
            return -1L;
        }
        try {
            ledger.prefill().refreshSnapshot();
            return ledger.prefill().snapshot().inflight();
        } catch (Throwable t) {
            diff.onError(t);
            return -1L;
        }
    }

    /** 同 {@link #prefillActiveCount()}，但未启用/异常时返回 0（指标读点不产负值）。 */
    public long prefillActiveCountOrZero() {
        long count = prefillActiveCount();
        return count < 0 ? 0L : count;
    }

    /**
     * D 侧预占权威记账（读取换权模式下 DecodeEndpoint.reserve 的真实入账）：
     * 绑定世代后经 ledger.decode().reserve 单入账（判重拒绝保持首账——重复
     * reserve 不覆盖既有条目）。异常吞入 shadow.error（读数与记账同为 ledger，
     * 不回写旧 map——避免双源分裂）。
     *
     * @param ipPort 端点 ip:port（与事件泵同键——translator 惰性注册世代）
     */
    public void decodeReserveAuthority(long requestId, long seqLen, long expectedKv, String ipPort) {
        if (!isReadAuthority()) {
            return;
        }
        if (ipPort == null) {
            // 端点身份缺失（WorkerStatus 未填 ip）：明确丢弃并告警，
            // 不让 NPE 流入 shadow.error 计数污染对账指标。
            logger.warn("[state-shadow] decode reserve authority: null endpoint, requestId={} dropped",
                    requestId);
            return;
        }
        try {
            WorkerStatusObservationTranslator.GenerationTripleLike binding =
                    translator.bindingOf(RoleType.DECODE, ipPort);
            if (binding == null) {
                logger.warn("[state-shadow] decode reserve authority: endpoint {} unresolved, requestId={} dropped",
                        ipPort, requestId);
                return;
            }
            ledger.decode().reserve(requestId, seqLen, expectedKv,
                    new GenerationTriple((int) binding.endpointId(), binding.generation(), -1L));
        } catch (Throwable t) {
            diff.onError(t);
        }
    }

    /**
     * D 侧预占释放权威记账（读取换权模式下 DecodeEndpoint.release 的真实入账）：
     * 撤预占账并移除条目（未终态主动放弃；终局归 settle 单出口）。幂等。
     */
    public void decodeReleaseAuthority(long requestId) {
        if (!isReadAuthority()) {
            return;
        }
        try {
            ledger.decode().release(requestId);
        } catch (Throwable t) {
            diff.onError(t);
        }
    }

    /**
     * D 侧端点级计数（读取换权模式下 DecodeEndpoint 读点的数据源）：
     * per-EP 按需聚合（含未确认预占双轨 KV 口径）。异常时返回全零视图
     * （读数退化方向为低估——不阻断调度）。
     */
    public DecodeEndpointCounters decodeEndpointCounters(int endpointId) {
        if (!isReadAuthority()) {
            return DecodeEndpointCounters.empty();
        }
        try {
            return ledger.decode().endpointCounters(endpointId);
        } catch (Throwable t) {
            diff.onError(t);
            return DecodeEndpointCounters.empty();
        }
    }

    /**
     * P 侧端点级计数（读取换权模式下 PrefillEndpoint pending 读点的数据源）：
     * per-EP 按需聚合（仅已派发绑定条目；排队/攒批窗口由 batcher 队列覆盖）。
     */
    public PrefillEndpointCounters prefillEndpointCounters(int endpointId) {
        if (!isReadAuthority()) {
            return PrefillEndpointCounters.empty();
        }
        try {
            return ledger.prefill().endpointCounters(endpointId);
        } catch (Throwable t) {
            diff.onError(t);
            return PrefillEndpointCounters.empty();
        }
    }

    /**
     * P 侧条目派发提交挂点（batch 提交/单发提交时）：onDispatching（批次外键）
     * + notePredictedBatchMs（分摊批次预测耗时——等待估算读点数据源）
     * + onDispatched（绑定端点世代）。幂等（已在 DISPATCHED 及以上静默）。
     *
     * @param predictedBatchMs 分摊到该请求的批次预测耗时（批次总预测 /
     *                         成员数；散请求即单请求预测。调用方计算）
     */
    public void onPrefillDispatched(long requestId, long batchId, String ipPort, long predictedBatchMs) {
        if (!enabled) {
            return;
        }
        try {
            WorkerStatusObservationTranslator.GenerationTripleLike binding =
                    translator.bindingOf(RoleType.PREFILL, ipPort);
            if (binding == null) {
                return;
            }
            ledger.prefill().onDispatching(requestId, batchId);
            if (predictedBatchMs > 0) {
                ledger.prefill().notePredictedBatchMs(requestId, predictedBatchMs);
            }
            ledger.prefill().onDispatched(requestId,
                    new GenerationTriple((int) binding.endpointId(), binding.generation(), -1L));
        } catch (Throwable t) {
            diff.onError(t);
        }
    }

    /** 挂起终态 metric 表当前大小（测试观测钩子）。 */
    public int pendingTerminalMetricCount() {
        ConcurrentHashMap<Long, TerminalMetricContext> pending = pendingTerminalMetrics;
        return pending == null ? 0 : pending.size();
    }

    /**
     * 终态权威结算入口（AbstractScheduler.register 的 whenComplete 分流调用，
     * 账本启用即生效；仅账本关时旧路径退化直报）：
     * <ul>
     *   <li>COMPLETED（引擎 ACK）不提前 settle——引擎执行相位与 KV 计费移交
     *       （引擎上报 KV 在确认分配后接管本地预占）保持完整；终态 metric 挂
     *       pending 表，由 ledger 终局（引擎 finished / janitor 胜者结算）消费。</li>
     *   <li>FAILED / TIMED_OUT / CANCELLED：master 已判死——双侧主动 settle
     *       （权威单出口），metric 在 settle 出口即时生产。</li>
     * </ul>
     * catch-all 包裹，绝不外抛影响主路径。
     *
     * @param requestId    请求 ID
     * @param oldStateName 旧路径终态名（item.state().name()）
     * @param metricCtx    旧路径视角的终态 metric 上下文（reason 保持旧四值口径，
     *                    监控值域连续）
     */
    public void onOldTerminalAuthority(long requestId, String oldStateName, TerminalMetricContext metricCtx) {
        if (!enabled) {
            return;
        }
        try {
            diff.recordOldTerminal(requestId, oldStateName);
            switch (oldStateName) {
                case "COMPLETED" -> {
                    if (ledgerCovers(requestId)) {
                        // ACK 只完成客户端 future；ledger 终局由引擎事件/janitor 驱动。
                        pendingTerminalMetrics.put(requestId, metricCtx);
                        // 乱序兜底：引擎 finished 早于 ACK 到达时墓碑已在——立即消费。
                        consumePendingMetricIfTerminal(requestId);
                    } else {
                        // ledger 未覆盖（开账异常被吞等罕见场景）：退回旧语义 ACK 即报，
                        // 不进 pending 生命周期（防泄漏：无条目则永远无人消费）。
                        reportTerminalMetric(metricCtx);
                    }
                }
                case "CANCELLED", "FAILED", "TIMED_OUT" -> {
                    // master 已判死：双侧主动 settle + metric 即时出口（每请求恰好一次）。
                    settleBothSidesAuthoritatively(requestId, oldStateName);
                    reportTerminalMetric(metricCtx);
                }
                default -> reportTerminalMetric(metricCtx); // 未知旧终态：保守直报，不吞 metric
            }
        } catch (Throwable t) {
            diff.onError(t);
        }
    }

    /** 旧路径终态 → ledger 终局结果（受控原因 + detail）。 */
    private static TerminalOutcome authoritativeOutcome(String oldStateName) {
        return switch (oldStateName) {
            case "CANCELLED" -> new TerminalOutcome(TerminalState.CANCELLED,
                    TerminalReason.CANCELLED_IMPLICIT, "settle-authority:old-path-cancel");
            case "FAILED" -> new TerminalOutcome(TerminalState.FAILED,
                    TerminalReason.ENGINE_FAILED, "settle-authority:old-path-failed");
            case "TIMED_OUT" -> new TerminalOutcome(TerminalState.SLO_TIMEOUT,
                    TerminalReason.SLO_BUDGET_EXHAUSTED, "settle-authority:old-path-timeout");
            default -> throw new IllegalArgumentException("unexpected old terminal state: " + oldStateName);
        };
    }

    /** 旧路径终态 → ledger 终局证据通道（本地取消 / master 判死强制 / 存活时间上限）。 */
    private static SettleReason authoritativeSettleReason(String oldStateName) {
        return switch (oldStateName) {
            case "CANCELLED" -> SettleReason.LOCAL_CANCEL;
            case "FAILED" -> SettleReason.FORCE_CHANNEL;
            case "TIMED_OUT" -> SettleReason.TTL_CHANNEL;
            default -> throw new IllegalArgumentException("unexpected old terminal state: " + oldStateName);
        };
    }

    /**
     * 双侧主动结算（幂等：先到者赢，后到侧 absorb no-op）。CANCELLED 经 P 侧
     * propagate 自动双清；FAILED / TIMED_OUT 不传播——显式双调保证双侧收敛。
     */
    private void settleBothSidesAuthoritatively(long requestId, String oldStateName) {
        TerminalOutcome outcome = authoritativeOutcome(oldStateName);
        SettleReason settleReason = authoritativeSettleReason(oldStateName);
        ledger.prefill().settle(requestId, outcome, settleReason);
        ledger.decode().settle(requestId, outcome, settleReason);
        recordNewTerminalIfSettled(requestId);
    }

    /**
     * 旧路径终态 metric 上下文（结算换权的载荷）：旧四值终态原因 + 路由解析出的
     * 角色与引擎地址。reason 保持旧口径——监控值域与开启前连续。
     */
    public record TerminalMetricContext(org.flexlb.balance.scheduler.TerminalReason reason,
                                        String role, String engineIp) {
    }

    /** 消费挂起 metric（无条件 remove；终局已由调用方证实）。 */
    private void consumePendingMetric(long requestId) {
        TerminalMetricContext ctx = pendingTerminalMetrics.remove(requestId);
        if (ctx != null) {
            reportTerminalMetric(ctx);
        }
    }

    /** ledger 是否覆盖该请求（任一侧条目或墓碑存在）。 */
    private boolean ledgerCovers(long requestId) {
        return ledger.prefill().get(requestId).isPresent()
                || ledger.decode().get(requestId).isPresent()
                || ledger.terminalOutcomeOf(requestId, StateRole.PREFILL).isPresent()
                || ledger.terminalOutcomeOf(requestId, StateRole.DECODE).isPresent();
    }

    /** 终局已发生才消费（挂 pending 后的自查路径：乱序兜底）。 */
    private void consumePendingMetricIfTerminal(long requestId) {
        if (pendingTerminalMetrics.containsKey(requestId)
                && (ledger.terminalOutcomeOf(requestId, StateRole.DECODE).isPresent()
                    || ledger.terminalOutcomeOf(requestId, StateRole.PREFILL).isPresent())) {
            consumePendingMetric(requestId);
        }
    }

    /** 终态 metric 单点生产（helper null（测试）/ 异常均不影响主路径）。 */
    private void reportTerminalMetric(TerminalMetricContext ctx) {
        FlexlbMetricHelper helper = this.terminalMetricHelper;
        if (helper == null || ctx == null) {
            return;
        }
        try {
            helper.reportTerminal(ctx.reason(), ctx.role(), ctx.engineIp(), null);
        } catch (Throwable t) {
            diff.onError(t);
        }
    }

    // ==================== 内部 ====================

    /**
     * 新侧主终态：D 侧墓碑优先（decode 完成即请求完成）、P 侧兜底（prefill 阶段
     * 失败/取消，D 侧无条目时 P 终局即终局）。结算换权模式下同步消费挂起 metric
     * （ledger 终局即单点生产出口）。
     *
     * <p>P 兜底仅在 D 侧无活跃条目时生效：P 完成 ≠ 请求完成（状态核心跨侧
     * 语义——D 条目活跃时请求级终态由 D 终局驱动，引擎 finished / janitor
     * 兜底均可达）。若不加守卫，P 引擎 finished 先终局即记一次新侧终态、
     * D 终局再记一次——第二次无法与旧侧终态配对而永久滞留 diff 窗口，
     * 高频终态下窗口满载后 putBounded 的全量淘汰扫描退化为热路径灾难
     * （真机轮错误率 63% 的根因，配对模型要求每请求恰好一次新侧终态）。
     */
    private void recordNewTerminalIfSettled(long requestId) {
        Optional<TerminalOutcome> decodeOutcome = ledger.terminalOutcomeOf(requestId, StateRole.DECODE);
        if (decodeOutcome.isPresent()) {
            diff.recordNewTerminal(requestId, decodeOutcome.get().state(), decodeOutcome.get().reason());
            consumePendingMetric(requestId);
            return;
        }
        if (ledger.decode().get(requestId).isPresent()) {
            // D 条目活跃：P 终局只是阶段完成——请求级终态等 D 终局（引擎
            // finished / janitor 兜底终局时再次进入本方法走 D 墓碑分支）。
            return;
        }
        Optional<TerminalOutcome> prefillOutcome = ledger.terminalOutcomeOf(requestId, StateRole.PREFILL);
        if (prefillOutcome.isPresent()) {
            diff.recordNewTerminal(requestId, prefillOutcome.get().state(), prefillOutcome.get().reason());
            consumePendingMetric(requestId);
        }
    }

    /** 影子诊断（日志/测试用；不参与主路径）。 */
    public StateShadowDiffCollector diffCollector() {
        return diff;
    }

    public StateLedger ledger() {
        return ledger;
    }

    /** 清理层（关态 null；观察/装配测试用）。 */
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
     * janitor 胜者结算回调（实例方法引用：结算换权后需要同时消费挂起 metric
     * ——与 recordNewTerminalIfSettled 同语义，D 侧墓碑优先、P 侧兜底）。
     */
    private void onJanitorSettled(long requestId, StateRole side) {
        recordNewTerminalIfSettled(requestId);
    }
}

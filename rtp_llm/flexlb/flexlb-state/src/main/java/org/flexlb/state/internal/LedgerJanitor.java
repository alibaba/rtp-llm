package org.flexlb.state.internal;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Consumer;
import java.util.function.LongSupplier;
import org.flexlb.state.CleanupReason;
import org.flexlb.state.InternalApi;
import org.flexlb.state.LedgerJanitorConfig;
import org.flexlb.state.SettleReason;
import org.flexlb.state.TerminalOutcome;
import org.flexlb.state.TerminalReason;
import org.flexlb.state.TerminalState;
import org.flexlb.state.internal.decode.DecodeRequestState;
import org.flexlb.state.internal.decode.DecodeSideStore;
import org.flexlb.state.internal.prefill.PrefillRequestState;
import org.flexlb.state.internal.prefill.PrefillSideStore;
import org.flexlb.state.spi.StateEndpointRef;
import org.flexlb.state.spi.StateRole;

/**
 * 账本清理层：条目从活跃态被移除的四条受控通道 + 三护栏。
 *
 * <h2>四通道（F1-F4）</h2>
 * <ol>
 *   <li><b>F1 因果通道</b>：事件驱动，状态核心的跨侧规则已覆盖（D finished(success) ⇒ P 收缩为
 *       COMPLETED / CAUSAL_CLOSURE）——janitor 不做额外事，只透传观测计数
 *       （{@link JanitorStats#causalClosureSettles()}），零常驻成本。</li>
 *   <li><b>F2 证据通道（核心）</b>：连续 N 轮（{@code staleRounds}）完整 tick 缺席推定死亡
 *       → settle(VANISHED / EVIDENCE_CHANNEL)。输入是条目引擎上报观察的 lastSeenRound；
 *       缺席追踪由本类维护（见护栏 2），触发即结算，被墓碑吸收迟到事件。</li>
 *   <li><b>F3 时间通道（TTL）</b>：createdAtMs 基准（创建时刻固定不可续命——任何 touch/observe
 *       不影响基准，createdAt 为 final 已类型级保证）→ 到期
 *       settle(TTL_EXPIRED / TTL_CHANNEL)。低频扫（调度方每 janitorIntervalMs 一 tick），
 *       per-endpoint 轮转分摊 + 单 tick 条目预算（预算内完成）。</li>
 *   <li><b>F4 强制通道（hard cap）</b>：createdAtMs + hardCapMs，到期<b>无条件</b>
 *       settle(HARD_CAP / FORCE_CHANNEL) + 告警计数。</li>
 * </ol>
 *
 * <h2>三护栏（F2 专属）</h2>
 * <ul>
 *   <li><b>护栏 1 防抖</b>：缺席跨度超过 N 轮才触发（缺席期间条目重新出现即清零重算，
 *       由 {@link #onRoundEnd} 的缺席追踪保证）。</li>
 *   <li><b>护栏 2 上报完整性</b>：仅完整 tick（EngineObservation.isComplete()，
 *       detailCount == running.size()）才参与缺席判定——不完整 tick 只可能刷新条目
 *       lastSeenRound（截断上报中条目出现仍是活着的证据），<b>绝不推进缺席计数</b>：
 *       onRoundEnd 对不完整 tick 直接丢弃（不产生判定机会），缺席期间的（不完整 tick）
 *       目击同样重置缺席起算。</li>
 *   <li><b>护栏 3 fence 豁免</b>：缺席判定成立但条目被 fence 冻结（跨侧协调
 *       进行中）→ 跳过并计 fence_hold；settle 前再过 {@link FenceRegistry#canEvict}
 *       断言防线（isFenced 预检与 settle 之间 race 窗口内登记的 fence 兜底拒绝）。</li>
 * </ul>
 *
 * <h2>hard cap vs fence 决策记录</h2>
 * 设计存在张力：F1"createdAt hard cap 不可续命" vs F4"fence 永生（fence 驱逐断言）"。
 * <b>决策：hard cap 对 fenced 条目也执行</b>（跳过 fence 豁免与 canEvict 断言），
 * reason 标 HARD_CAP 且告警计数翻倍（{@code hardCapSettles} 与
 * {@code hardCapFenceViolations} 双计）——fence 超过硬上限说明 fence 自身泄漏
 * （owner 未解除），宁清勿留。证据/TTL 通道维持 fence 豁免不变。
 *
 * <h2>终态映射</h2>
 * <ul>
 *   <li>VANISHED → {@link TerminalState#SLO_TIMEOUT} + {@link TerminalReason#VANISHED}
 *       （缺席判死是"等待超时"的轮次版；与旧路径 TTL-evict 的 TIMED_OUT 终局对齐，
 *       对账 diff 等价类内——缺席几秒触发 vs 旧路径 300s TTL 兜底是同一请求的
 *       两种速度兜底）。</li>
 *   <li>TTL_EXPIRED → {@link TerminalState#SLO_TIMEOUT} + TTL_EXPIRED（对齐旧 TIMED_OUT）。</li>
 *   <li>HARD_CAP → {@link TerminalState#FAILED} + HARD_CAP（强制通道非正常终局）。</li>
 * </ul>
 *
 * <h2>并发与生命周期约定</h2>
 * <ul>
 *   <li>{@link #onRoundEnd} 在 StateLedger.observe 尾部（Runner 线程）调用——
 *       多端点并发；{@link #runMaintenanceTick()} 由调度方单线程定时驱动
 *       （scheduleAtFixedRate 不重入）。两者共享的缺席表/计数器均为并发结构。</li>
 *   <li>janitor 的 settle 与外部 settle 走<b>同一 CAS 单出口</b>
 *       （StateLedger.settlePrefill/settleDecode 委托回调）：janitor 败者 =
 *       快路径胜 = 正常（计 lostToFastPath 超车）；janitor 胜者 = 兜底触发
 *       （计对应通道计数 + 通知 {@link SettleListener}）。</li>
 *   <li>所有公开入口 catch-all（Throwable → errors 计数，绝不外抛——
 *       onRoundEnd 在 observe 主路径上）。</li>
 *   <li>构造仅经 StateLedger.createJanitor（包外不可直构；字段为 internal 协作句柄）。</li>
 * </ul>
 */
@InternalApi
public final class LedgerJanitor {

    /**
     * settle 通道回调：StateLedger 的 CAS 单出口委托（单侧 settle，不传播 cancel 双清
     * ——janitor 三通道终态均非 CANCELLED）。返回本调用是否终局胜者。
     *
     * @param reason 该通道的受控证据通道归类（F2→EVIDENCE_CHANNEL /
     *               F3→TTL_CHANNEL / F4→FORCE_CHANNEL——settle reason 记账维度）
     */
    @FunctionalInterface
    public interface SettleChannel {

        boolean settle(long requestId, TerminalOutcome outcome, SettleReason reason);
    }

    /**
     * janitor 胜者结算监听（对账窗口补全：janitor settle 产生的新侧终态
     * 也需进入对账窗口，否则旧侧终态后到会误报 missing_on_new）。
     */
    @FunctionalInterface
    public interface SettleListener {

        void onJanitorSettled(long requestId, StateRole side);
    }

    /**
     * 单条目缺席追踪记录：sinceRound/sinceCount 分别为缺席起始的 round 值与
     * 完整轮序数（entry 用于换代检测）。触发判定用<b>完整轮序数差</b>
     * （count - sinceCount ≥ staleRounds）——不完整轮穿插不放大跨度（上报完整性护栏）。
     */
    private record Absent(long sinceRound, long sinceCount, Object entry) {
    }

    /** 通道观测快照（LongAdder 汇总；F1 为 ledger 侧计数透传）。 */
    public record JanitorStats(
            long roundEndTicks,
            long incompleteTicksSkipped,
            long vanishedSettles,
            long ttlSettles,
            long hardCapSettles,
            long hardCapFenceViolations,
            long fenceHoldSkips,
            long lostToFastPath,
            long causalClosureSettles,
            long maintenanceTicks,
            long errors) {
    }

    private final LedgerJanitorConfig config;
    private final PrefillSideStore pStore;
    private final DecodeSideStore dStore;
    private final FenceRegistry fences;
    private final LongSupplier causalClosureCount;
    private final SettleChannel pSettle;
    private final SettleChannel dSettle;
    /** 清理通道记账回调（StateLedger 统一持有 CleanupReason 计数账；null = 无记账）。 */
    private final Consumer<CleanupReason> cleanupCountRecorder;

    /** 缺席追踪（精确护栏 2：requestId → 缺席起始完整轮；恢复即清，条目换代即重置）。 */
    private final ConcurrentHashMap<Long, Absent> pAbsentSince = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Long, Absent> dAbsentSince = new ConcurrentHashMap<>();

    /** per-endpoint 完整轮序数（每次完整 tick +1；缺席跨度按此计数——不完整轮不推进，上报完整性护栏）。 */
    private final ConcurrentHashMap<Integer, AtomicLong> completeRoundCounters = new ConcurrentHashMap<>();

    /** TTL/硬上限轮转游标（trackedEndpoints 快照的下标；单调度线程访问）。 */
    private int ttlCursor;

    private volatile SettleListener settleListener;

    // ---- 通道计数（LongAdder：多 Runner 线程 + 调度线程并发累加）----

    private final LongAdder roundEndTicks = new LongAdder();
    private final LongAdder incompleteTicksSkipped = new LongAdder();
    private final LongAdder vanishedSettles = new LongAdder();
    private final LongAdder ttlSettles = new LongAdder();
    private final LongAdder hardCapSettles = new LongAdder();
    private final LongAdder hardCapFenceViolations = new LongAdder();
    private final LongAdder fenceHoldSkips = new LongAdder();
    private final LongAdder lostToFastPath = new LongAdder();
    private final LongAdder maintenanceTicks = new LongAdder();
    private final LongAdder errors = new LongAdder();

    /**
     * 构造（包外仅经 StateLedger.createJanitor 调用——参数为 internal 协作句柄，
     * 不构成对外 API）。跨包可见性（org.flexlb.state 门面协作）要求 public。
     */
    public LedgerJanitor(LedgerJanitorConfig config,
                         PrefillSideStore pStore,
                         DecodeSideStore dStore,
                         FenceRegistry fences,
                         LongSupplier causalClosureCount,
                         SettleChannel pSettle,
                         SettleChannel dSettle) {
        this(config, pStore, dStore, fences, causalClosureCount, pSettle, dSettle, null);
    }

    /** 全参构造（StateLedger.createJanitor 调用：附带 CleanupReason 记账回调）。 */
    public LedgerJanitor(LedgerJanitorConfig config,
                         PrefillSideStore pStore,
                         DecodeSideStore dStore,
                         FenceRegistry fences,
                         LongSupplier causalClosureCount,
                         SettleChannel pSettle,
                         SettleChannel dSettle,
                         Consumer<CleanupReason> cleanupCountRecorder) {
        this.config = config;
        this.pStore = pStore;
        this.dStore = dStore;
        this.fences = fences;
        this.causalClosureCount = causalClosureCount;
        this.pSettle = pSettle;
        this.dSettle = dSettle;
        this.cleanupCountRecorder = cleanupCountRecorder;
    }

    /**
     * 注册 janitor 胜者结算监听（对账窗口补全；监听异常被吞掉不影响 janitor）。
     */
    public void setSettleListener(SettleListener listener) {
        this.settleListener = listener;
    }

    // ==================== F2 证据通道（observe 尾部回调）====================

    /**
     * 完整 tick 尾部回调（StateLedger.observe 尾部；rebuild 重放不触发）：
     * 对该端点名下条目做缺席检测。
     *
     * <p>护栏 2：不完整 tick（截断上报）直接丢弃——缺席判定不发生，
     * 缺席计数不推进（截断上报中的"缺席"不是死亡证据）。</p>
     */
    public void onRoundEnd(StateEndpointRef endpointRef, long round, boolean completeTick) {
        try {
            if (!completeTick) {
                incompleteTicksSkipped.increment();
                countCleanup(CleanupReason.TRUNCATED_REPORT_EXCLUDED);
                return;
            }
            roundEndTicks.increment();
            scanEvidence(endpointRef, round);
        } catch (Throwable t) {
            errors.increment();
        }
    }

    private void scanEvidence(StateEndpointRef endpointRef, long round) {
        int endpointId = (int) endpointRef.endpointId();
        // 完整轮序数（单端点 observe 由单 Runner 线程串行；跨端点并发隔离）
        long roundCount = completeRoundCounters
                .computeIfAbsent(endpointId, k -> new AtomicLong())
                .incrementAndGet();
        for (DecodeRequestState e : dStore.entriesByEndpoint(endpointId)) {
            trackDecodeAbsence(e, round, roundCount);
        }
        for (PrefillRequestState e : pStore.entriesByEndpoint(endpointId)) {
            trackPrefillAbsence(e, round, roundCount);
        }
    }

    /**
     * D 侧缺席追踪（护栏 1/2 语义）——缺席跨度按<b>完整轮序数差</b>计：
     * <ul>
     *   <li>本轮出现（lastSeenRound ≥ round，完整 tick 的 noteEngineObserved 已刷新）
     *       → 清零（缺席中断，护栏 1）。</li>
     *   <li>从未引擎确认（lastSeenRound &lt; 0，引擎上报观察轮次未建立）→ 不适用缺席判定。</li>
     *   <li>缺席期间被（不完整 tick）目击（lastSeenRound ≥ sinceRound）→ 重置起算
     *       ——目击即活着，截断与否不影响这一证据方向。</li>
     *   <li>条目换代（同 requestId 换了实例）→ 重置起算。</li>
     *   <li>完整轮跨度（roundCount − sinceCount）≥ staleRounds → 触发 VANISHED
     *       ——连续完整场景下等价于任务判定式 {@code lastSeenRound < round - N}，
     *       不完整轮穿插时严格不放大（上报完整性护栏）。</li>
     * </ul>
     */
    private void trackDecodeAbsence(DecodeRequestState entry, long round, long roundCount) {
        long id = entry.requestId();
        long lastSeen = entry.lastSeenRound();
        if (lastSeen >= round) {
            dAbsentSince.remove(id); // 本轮出现 → 清零（护栏 1）
            return;
        }
        if (lastSeen < 0) {
            return; // 从未引擎确认（引擎上报观察轮次未建立）
        }
        Absent absent = dAbsentSince.compute(id, (k, cur) -> {
            if (cur == null || cur.entry() != entry || lastSeen >= cur.sinceRound()) {
                return new Absent(round, roundCount, entry); // 首次缺席 / 换代 / 缺席期间目击 → （重）起算
            }
            return cur;
        });
        if (roundCount - absent.sinceCount() >= config.staleRounds()) {
            dAbsentSince.remove(id, absent);
            settleViaEvidence(id, roundCount - absent.sinceCount(), dSettle, StateRole.DECODE);
        }
    }

    /** P 侧缺席追踪（与 {@link #trackDecodeAbsence} 对称）。 */
    private void trackPrefillAbsence(PrefillRequestState entry, long round, long roundCount) {
        long id = entry.requestId();
        long lastSeen = entry.lastSeenRound();
        if (lastSeen >= round) {
            pAbsentSince.remove(id);
            return;
        }
        if (lastSeen < 0) {
            return;
        }
        Absent absent = pAbsentSince.compute(id, (k, cur) -> {
            if (cur == null || cur.entry() != entry || lastSeen >= cur.sinceRound()) {
                return new Absent(round, roundCount, entry);
            }
            return cur;
        });
        if (roundCount - absent.sinceCount() >= config.staleRounds()) {
            pAbsentSince.remove(id, absent);
            settleViaEvidence(id, roundCount - absent.sinceCount(), pSettle, StateRole.PREFILL);
        }
    }

    /** F2 触发：护栏 3 fence 豁免 + fence 驱逐断言防线 + CAS 单出口 settle(VANISHED)。 */
    private void settleViaEvidence(long requestId, long absentRounds, SettleChannel settle, StateRole side) {
        if (skipForFence(requestId)) {
            return;
        }
        boolean won = settle.settle(requestId, new TerminalOutcome(
                TerminalState.SLO_TIMEOUT, TerminalReason.VANISHED,
                "janitor:absent-" + absentRounds + "rounds"), SettleReason.EVIDENCE_CHANNEL);
        if (won) {
            vanishedSettles.increment();
            countCleanup(CleanupReason.ABSENT_N_ROUNDS);
            notifySettled(requestId, side);
        } else {
            lostToFastPath.increment(); // 超车（TTL 通道）：快路径已终局——janitor 败者属正常
        }
    }

    // ==================== F3/F4 TTL + hard cap（低频调度 tick）====================

    /** 维护 tick（调度方每 janitorIntervalMs 驱动；单线程不重入）。 */
    public void runMaintenanceTick() {
        runMaintenanceTick(System.currentTimeMillis());
    }

    /** 可注入时刻的重载（确定性测试用）。 */
    public void runMaintenanceTick(long nowMs) {
        try {
            maintenanceTicks.increment();
            scanTtlAndHardCap(nowMs);
            cleanOrphanAbsence();
            // 墓碑/fence TTL 过期清理（早期占位 Runnable 职责并入）
            pStore.tombstones().evictExpired(nowMs);
            dStore.tombstones().evictExpired(nowMs);
            fences.evictExpired(nowMs);
        } catch (Throwable t) {
            errors.increment();
        }
    }

    /**
     * TTL + hard cap 扫描（F3/F4 同轮）：
     * <ol>
     *   <li>未绑定条目（P 侧排队中，量级小、TTL 高危区）优先全扫。</li>
     *   <li>endpoint 名下条目按轮转游标分摊——单 tick 至多扫描约 scanBudgetPerTick
     *       条（预算内完成），超出部分延后到后续 tick（TTL 判定延迟至多一个
     *       轮转圈周期，10s tick 下可忽略）。</li>
     * </ol>
     */
    private void scanTtlAndHardCap(long nowMs) {
        int budget = config.scanBudgetPerTick();
        int scanned = 0;
        for (PrefillRequestState e : pStore.unboundEntries()) {
            checkTtlAndHardCap(e.requestId(), e.createdAtMs(), nowMs, pSettle, StateRole.PREFILL);
            if (++scanned >= budget) {
                return;
            }
        }
        for (DecodeRequestState e : dStore.unboundEntries()) {
            checkTtlAndHardCap(e.requestId(), e.createdAtMs(), nowMs, dSettle, StateRole.DECODE);
            if (++scanned >= budget) {
                return;
            }
        }
        List<Integer> endpoints = trackedEndpoints();
        if (endpoints.isEmpty()) {
            return;
        }
        if (ttlCursor >= endpoints.size()) {
            ttlCursor = 0;
        }
        int idx = ttlCursor;
        int visited = 0;
        while (visited++ < endpoints.size()) {
            int endpointId = endpoints.get(idx);
            for (PrefillRequestState e : pStore.entriesByEndpoint(endpointId)) {
                checkTtlAndHardCap(e.requestId(), e.createdAtMs(), nowMs, pSettle, StateRole.PREFILL);
                scanned++;
            }
            for (DecodeRequestState e : dStore.entriesByEndpoint(endpointId)) {
                checkTtlAndHardCap(e.requestId(), e.createdAtMs(), nowMs, dSettle, StateRole.DECODE);
                scanned++;
            }
            idx = (idx + 1) % endpoints.size();
            if (idx == ttlCursor || scanned >= budget) {
                break; // 整圈完成（游标归位）或预算用尽
            }
        }
        ttlCursor = idx;
    }

    /**
     * 单条目 TTL/hard cap 判定（F3/F4）：hard cap 先于 TTL（更严的兜底优先结算）。
     */
    private void checkTtlAndHardCap(long requestId, long createdAtMs, long nowMs,
                                    SettleChannel settle, StateRole side) {
        long age = nowMs - createdAtMs;
        if (age >= config.hardCapMs()) {
            // F4 强制通道：fence 不豁免（宁清勿留决策，见类 javadoc）——
            // fence 超硬上限 = fence 自身泄漏，告警计数翻倍（双计）
            boolean fenced = fences.isFenced(requestId);
            boolean won = settle.settle(requestId, new TerminalOutcome(
                    TerminalState.FAILED, TerminalReason.HARD_CAP,
                    "janitor:hard-cap@" + age + "ms" + (fenced ? " (fence-leak)" : "")),
                    SettleReason.FORCE_CHANNEL);
            if (won) {
                hardCapSettles.increment();
                countCleanup(CleanupReason.HARD_CAP);
                if (fenced) {
                    hardCapFenceViolations.increment();
                }
                notifySettled(requestId, side);
            } else {
                lostToFastPath.increment();
            }
            return;
        }
        if (age >= config.ttlMs()) {
            if (skipForFence(requestId)) {
                return; // F3 TTL 通道维持 fence 豁免（fence 驱逐断言）
            }
            boolean won = settle.settle(requestId, new TerminalOutcome(
                    TerminalState.SLO_TIMEOUT, TerminalReason.TTL_EXPIRED,
                    "janitor:ttl@" + age + "ms"), SettleReason.TTL_CHANNEL);
            if (won) {
                ttlSettles.increment();
                countCleanup(CleanupReason.TTL);
                notifySettled(requestId, side);
            } else {
                lostToFastPath.increment();
            }
        }
    }

    /** 护栏 3：fence 预检跳过（+计数）与 fence 驱逐断言防线（race 窗口兜底）。 */
    private boolean skipForFence(long requestId) {
        if (fences.isFenced(requestId)) {
            fenceHoldSkips.increment();
            countCleanup(CleanupReason.FENCE_HOLD);
            return true;
        }
        try {
            fences.canEvict(requestId);
        } catch (IllegalStateException fenceHold) {
            fenceHoldSkips.increment();
            countCleanup(CleanupReason.FENCE_HOLD);
            return true;
        }
        return false;
    }

    /** 缺席追踪孤儿清理（条目已终局/移除/换代：低频 tick 兜底，防 map 泄漏）。 */
    private void cleanOrphanAbsence() {
        pAbsentSince.entrySet().removeIf(en -> {
            PrefillRequestState cur = pStore.get(en.getKey());
            return cur == null || cur != en.getValue().entry() || cur.isFinished();
        });
        dAbsentSince.entrySet().removeIf(en -> {
            DecodeRequestState cur = dStore.get(en.getKey());
            return cur == null || cur != en.getValue().entry() || cur.isFinished();
        });
    }

    /** 两侧 byEndpoint 键并集（TreeSet 稳定轮转序）。 */
    private List<Integer> trackedEndpoints() {
        TreeSet<Integer> ids = new TreeSet<>();
        ids.addAll(pStore.trackedEndpointIds());
        ids.addAll(dStore.trackedEndpointIds());
        return new ArrayList<>(ids);
    }

    // ==================== 观测 ====================

    /** 通道观测快照（F1 causalClosureSettles 为 ledger 侧计数透传）。 */
    public JanitorStats stats() {
        return new JanitorStats(
                roundEndTicks.sum(),
                incompleteTicksSkipped.sum(),
                vanishedSettles.sum(),
                ttlSettles.sum(),
                hardCapSettles.sum(),
                hardCapFenceViolations.sum(),
                fenceHoldSkips.sum(),
                lostToFastPath.sum(),
                causalClosureCount.getAsLong(),
                maintenanceTicks.sum(),
                errors.sum());
    }

    private void notifySettled(long requestId, StateRole side) {
        SettleListener listener = settleListener;
        if (listener == null) {
            return;
        }
        try {
            listener.onJanitorSettled(requestId, side);
        } catch (Throwable t) {
            errors.increment(); // 监听异常不外抛（janitor 铁律）
        }
    }

    /** 清理通道记账（回调 null 时 no-op；异常吞入 errors——记账不外抛）。 */
    private void countCleanup(CleanupReason reason) {
        Consumer<CleanupReason> recorder = cleanupCountRecorder;
        if (recorder == null) {
            return;
        }
        try {
            recorder.accept(reason);
        } catch (Throwable t) {
            errors.increment();
        }
    }

    /** 缺席追踪存量（测试/诊断）。 */
    public int absentTracked() {
        return pAbsentSince.size() + dAbsentSince.size();
    }

    /** 轮转游标当前位置（测试/诊断）。 */
    int ttlCursor() {
        return ttlCursor;
    }

    /** 配置只读视图（测试/诊断）。 */
    public LedgerJanitorConfig config() {
        return config;
    }
}

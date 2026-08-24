package org.flexlb.balance.endpoint;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.state.DecodeEndpointCounters;
import org.flexlb.sync.shadow.StateShadowBridge;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Decode worker endpoint: KV reservation accounting and all scheduling read
 * points are served by the state ledger ({@link StateShadowBridge}) per-EP
 * view — reserve/release entries the ledger directly, and the per-EP
 * counter cache is refreshed on each engine status tick
 * ({@link #onWorkerStatusUpdate}).
 *
 * <p>Legacy two-layer inflight maps (local layer-1 reservations +
 * engine-accepted layer-2 tasks) have been removed: the ledger's decode
 * side is the single accounting source, with the LedgerJanitor providing
 * the stale/TTL safety net (engine-reported state is the single source of
 * truth). In ledger-disabled (degraded) mode all read points return zero
 * — no accounting source without the ledger.
 */
public class DecodeEndpoint extends WorkerEndpoint {

    private final AtomicLong reportedKvAvailable = new AtomicLong();
    private final FlexlbConfig config;

    /** 状态账本门面（装配点注入；null / 关态时读点全零——退化模式）。 */
    private final StateShadowBridge shadowBridge;

    /** 端点稳定 ID（ipPort 哈希——与账本 translator 的 endpointId 同映射）。 */
    private final int endpointId;

    /**
     * per-EP 账本计数缓存：引擎状态报文 tick 刷新，读点零锁 volatile
     * 读——策略 select 热路径不触发按需聚合。关态恒 null（读数全零）。
     */
    private volatile DecodeEndpointCounters ledgerCounters;

    public DecodeEndpoint(WorkerStatus status, FlexlbConfig config) {
        this(status, config, null);
    }

    /**
     * 账本装配构造：KV 预占记账与全部调度读点走 StateLedger per-EP 视图。
     */
    public DecodeEndpoint(WorkerStatus status, FlexlbConfig config,
                          StateShadowBridge shadowBridge) {
        super(status);
        this.config = config;
        this.shadowBridge = shadowBridge;
        this.endpointId = ipPort() != null ? ipPort().hashCode() : 0;
    }

    /** per-EP 账本计数（未刷新/退化模式时全零——读数退化方向为低估，不阻断调度）。 */
    private DecodeEndpointCounters ledgerCountersOrZero() {
        DecodeEndpointCounters c = ledgerCounters;
        return c != null ? c : DecodeEndpointCounters.empty();
    }

    /**
     * KV 预占记账入口（DecodeEndpoint 特有 API，非基类多态）：真实记账切
     * ledger.decode().reserve 单入账（含世代绑定与判重）；异常由门面吞入
     * shadow.error，不产生本地影子计数（读数与记账同源）。
     */
    public void reserve(long requestId, long kvTokens, long expectedKvTokens) {
        if (shadowBridge != null) {
            shadowBridge.decodeReserveAuthority(requestId, kvTokens, expectedKvTokens, ipPort());
        }
    }

    /**
     * Release the reservation for {@code requestId} (pre-terminal abandonment;
     * terminal settlement stays with the ledger's single settle exit).
     * Idempotent.
     */
    @Override
    public void release(long requestId) {
        if (shadowBridge != null) {
            shadowBridge.decodeReleaseAuthority(requestId);
        }
    }

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        this.reportedKvAvailable.set(status.getAvailableKvCacheTokens().get());
        // per-EP 账本计数缓存随引擎状态 tick 刷新（引擎上报 = 唯一事实源，
        // 本端点不再维护镜像条目——终局与陈旧清理由 ledger 事件泵与 janitor 承担）。
        if (shadowBridge != null) {
            this.ledgerCounters = shadowBridge.decodeEndpointCounters(endpointId);
        }
    }

    // ==================== 调度读点（账本 per-EP 视图） ====================

    /** 未确认预占条目数：reserve 起未被引擎上报接管的请求。 */
    public int decodeInflightCount() {
        return ledgerCountersOrZero().unconfirmedCount();
    }

    /**
     * 未确认条目 Σ kvTokens (hard demand) — seqLen only, the minimum KV
     * needed for the prompt itself. Used for hard-capacity filtering.
     */
    public long decodeInflightHardKvReserved() {
        return ledgerCountersOrZero().unconfirmedSeqKv();
    }

    /**
     * 未确认条目 Σ expectedKvTokens (conservative estimate) — seqLen +
     * maxNewTokens, accounting for generation-phase KV growth. Used for
     * scoring / load balancing.
     */
    public long decodeInflightExpectedKvReserved() {
        return ledgerCountersOrZero().unconfirmedExpectedKv();
    }

    /** 引擎已接管（引擎上报观察）条目数。 */
    public int decodeEngineWorkCount() {
        return ledgerCountersOrZero().engineOwnedCount();
    }

    /** 引擎侧排队窗口：已派发未被引擎上报接管的条目数。 */
    public int decodeEngineWaitingCount() {
        return (int) phaseCountOrZero(DecodePhaseOrdinal.DISPATCHED);
    }

    /** 引擎侧 KV 装载中条目数。 */
    public int decodeEngineLoadingCount() {
        return (int) phaseCountOrZero(DecodePhaseOrdinal.D_LOADING);
    }

    /** 引擎侧执行中条目数。 */
    public int decodeEngineRunningCount() {
        return (int) phaseCountOrZero(DecodePhaseOrdinal.D_RUNNING);
    }

    /** 账本相位人口下标（与 flexlb-state DecodePhase.ordinal 对齐）。 */
    private static final class DecodePhaseOrdinal {
        static final int RESERVED = 0;
        static final int DISPATCHED = 1;
        static final int D_LOADING = 2;
        static final int D_RUNNING = 3;
    }

    private long phaseCountOrZero(int ordinal) {
        DecodeEndpointCounters c = ledgerCountersOrZero();
        List<Long> phases = c.phaseCounts();
        return ordinal < phases.size() ? phases.get(ordinal) : 0L;
    }

    /**
     * Total active load: per-EP 活跃条目数（reserve 起至终局，全相位）。
     */
    public int decodeTotalLoad() {
        return ledgerCountersOrZero().activeTotal();
    }

    /**
     * Real KV used: engine-reported used (total - available) + unconfirmed
     * expected reservations（账本口径与旧 layer-1 expected 一致）。
     */
    public long decodeRealKvUsed() {
        long totalCap = status.getTotalKvCacheTokens().get();
        long avail = status.getAvailableKvCacheTokens().get();
        long reportedUsed = totalCap > 0 ? Math.max(0, totalCap - avail) : 0;
        return reportedUsed + ledgerCountersOrZero().unconfirmedExpectedKv();
    }

    /**
     * Real KV available: engine-reported available - unconfirmed hard
     * reservations (prompt-only KV — hard-capacity filter checks whether the
     * prompt itself fits, without being overly aggressive due to other
     * inflight requests' expected growth).
     *
     * <p><b>Approximate:</b> reads {@code reportedKvAvailable} and the
     * ledger counter cache non-atomically — the returned value may reflect
     * a slightly inconsistent snapshot. This is acceptable for scheduling
     * decisions.
     */
    public long decodeRealKvAvailable() {
        return Math.max(0, reportedKvAvailable.get() - ledgerCountersOrZero().unconfirmedSeqKv());
    }

    /** Real KV total capacity reported by the engine. */
    public long decodeKvTotal() {
        return status.getTotalKvCacheTokens().get();
    }

    // ==================== Metrics ====================

    /**
     * Report per-worker decode inflight metrics via the given reporter.
     * Called periodically by {@code RouteService#triggerSchedulerMetrics()}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        reporter.reportInflightRequestCount(RoleType.DECODE.name(), getIp(), decodeInflightCount());
        reporter.reportDecodeTotalLoad(getIp(), decodeTotalLoad());
        reporter.reportDecodeInflightKvReserved(getIp(), decodeInflightExpectedKvReserved());
        reporter.reportDecodeInflightKvReservedHard(getIp(), decodeInflightHardKvReserved());
        // Phase-split engine-side counts (WAITING / LOADING / RUNNING)
        reporter.reportDecodeEngineWaitingCount(getIp(), decodeEngineWaitingCount());
        reporter.reportDecodeEngineLoadingCount(getIp(), decodeEngineLoadingCount());
        reporter.reportDecodeEngineRunningCount(getIp(), decodeEngineRunningCount());
        // Two-layer breakdown (unconfirmed reservations vs engine-owned)
        reporter.reportDecodeInflightRequestsCount(getIp(), decodeInflightCount());
        reporter.reportDecodeEngineWorkCount(getIp(), decodeEngineWorkCount());
    }
}

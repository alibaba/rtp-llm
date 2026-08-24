package org.flexlb.service;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.AbstractScheduler;
import org.flexlb.balance.scheduler.BatchScheduler;
import org.flexlb.balance.scheduler.DiagnosticsProvider;
import org.flexlb.balance.scheduler.DirectScheduler;
import org.flexlb.balance.scheduler.QueueScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.shadow.StateShadowBridge;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.List;
import java.util.concurrent.CompletableFuture;

@Component
public class RouteService {

    private final ConfigService configService;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;
    private final EndpointRegistry endpointRegistry;
    private final BatchSchedulerReporter reporter;

    // --- the three scheduling paths ---

    private final BatchScheduler batchScheduler;
    private final QueueScheduler queueScheduler;
    private final DirectScheduler directScheduler;

    /** Last execution timestamp for metrics report throttle. */
    private volatile long lastMetricsReportTime = 0;

    /**
     * All schedulers, used for unified lifecycle management (start/shutdown)
     * and metrics reporting. Order: BATCH → QUEUE → DIRECT.
     */
    private final List<AbstractScheduler> schedulers;

    /**
     * All diagnostics providers (schedulers + endpointRegistry), used by
     * {@code HttpLoadBalanceServer} to aggregate diagnostics without
     * hard-coded QUEUE-specific method calls.
     */
    private final List<DiagnosticsProvider> diagnosticsProviders;

    /** 状态账本门面：开关关时为 no-op 单例（退化模式）。 */
    private final StateShadowBridge shadowBridge;

    public RouteService(ConfigService configService,
                        RecentCacheKeyTraceReporter recentCacheKeyTraceReporter,
                        EndpointRegistry endpointRegistry,
                        BatchSchedulerReporter reporter,
                        BatchScheduler batchScheduler,
                        QueueScheduler queueScheduler,
                        DirectScheduler directScheduler) {
        this(configService, recentCacheKeyTraceReporter, endpointRegistry,
                reporter, batchScheduler, queueScheduler, directScheduler, StateShadowBridge.DISABLED);
    }

    @Autowired
    public RouteService(ConfigService configService,
                        RecentCacheKeyTraceReporter recentCacheKeyTraceReporter,
                        EndpointRegistry endpointRegistry,
                        BatchSchedulerReporter reporter,
                        BatchScheduler batchScheduler,
                        QueueScheduler queueScheduler,
                        DirectScheduler directScheduler,
                        StateShadowBridge shadowBridge) {
        this.configService = configService;
        this.recentCacheKeyTraceReporter = recentCacheKeyTraceReporter;
        this.endpointRegistry = endpointRegistry;
        this.reporter = reporter;
        this.batchScheduler = batchScheduler;
        this.queueScheduler = queueScheduler;
        this.directScheduler = directScheduler;
        this.shadowBridge = shadowBridge == null ? StateShadowBridge.DISABLED : shadowBridge;
        this.schedulers = List.of(batchScheduler, queueScheduler, directScheduler);
        this.diagnosticsProviders = List.of(batchScheduler, queueScheduler, directScheduler,
                endpointRegistry);
    }

    /**
     * Start all schedulers' background resources (e.g. QUEUE worker pool).
     * Each scheduler's {@link AbstractScheduler#start()} is a no-op unless
     * overridden.
     */
    @PostConstruct
    public void start() {
        schedulers.forEach(AbstractScheduler::start);
    }

    /**
     * Route request to the appropriate worker based on the deployment-level
     * schedule mode. Pure dispatch — no mode-specific validation logic here;
     * each scheduler owns its own admission checks (e.g. BatchScheduler
     * validates generate input).
     *
     * @param balanceContext Load balancing context
     * @return Routing result
     */
    public CompletableFuture<Response> route(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(flexlbConfig);

        ScheduleModeEnum mode = flexlbConfig.getDefaultScheduleModeEnum();
        balanceContext.setScheduleMode(mode);

        AbstractScheduler scheduler = switch (mode) {
            case BATCH -> batchScheduler;
            case QUEUE -> queueScheduler;
            default -> directScheduler;
        };

        CompletableFuture<Response> resultFuture = scheduler.submit(balanceContext);

        return resultFuture.whenComplete((result, throwable) -> {
            if (throwable != null) {
                return;
            }
            balanceContext.setResponse(result);
            if (result != null && result.isSuccess()) {
                recentCacheKeyTraceReporter.report(balanceContext);
            }
        });
    }

    /**
     * Cancel an inflight request by its string-form request ID.
     *
     * <p>Resolves the pending submission across all three scheduling paths
     * (BATCH/QUEUE/DIRECT — each scheduler owns its pending registry),
     * completes its future with a CANCELLED error (first completion wins),
     * and — when the cancel wins — triggers the owning scheduler's
     * {@link AbstractScheduler#onLocalTerminal} hook to release
     * path-specific resources (e.g. a queue slot).
     *
     * <p>Returns {@code false} if the request was not found (never submitted
     * or already terminal).
     *
     * @param requestId string-form request ID
     * @return {@code true} if the request was found and cancelled
     */
    public boolean cancel(String requestId) {
        long id = shadowRequestId(requestId);
        // 账本双侧 pendingCancel 意图标记（终局由终态结算单出口收敛）。
        shadowBridge.onLocalCancelRequested(id);
        for (AbstractScheduler scheduler : schedulers) {
            if (scheduler.cancelIfPending(id)) {
                return true;
            }
        }
        return false;
    }

    /** 影子侧 requestId 解析（防御性）：非数字 ID 返回 -1（账本无对应条目，no-op）。 */
    private static long shadowRequestId(String requestId) {
        try {
            return Long.parseLong(requestId);
        } catch (NumberFormatException e) {
            return -1L;
        }
    }

    /**
     * Return the number of pending (not yet terminal) submissions across
     * all three scheduling paths — external monitors treat this as the
     * live inflight count.
     *
     * @return pending submission count
     */
    public int globalInflightSize() {
        int total = 0;
        for (AbstractScheduler scheduler : schedulers) {
            total += scheduler.pendingCount();
        }
        return total;
    }

    /**
     * Expose all {@link DiagnosticsProvider} components for HTTP diagnostic
     * endpoints. {@code HttpLoadBalanceServer} iterates this list to
     * aggregate diagnostics (queue length, inflight count, EP counts, etc.)
     * without hard-coded method calls on individual schedulers.
     */
    public List<DiagnosticsProvider> getDiagnosticsProviders() {
        return diagnosticsProviders;
    }

    /**
     * Periodically trigger scheduler-level and per-worker batch metrics
     * reporting.
     *
     * <p>Polls every 1s and throttles to {@code metricsReportIntervalMs}
     * (default 2000ms, configurable via {@code METRICS_REPORT_INTERVAL_MS}
     * env var). This pattern is used because {@code @Scheduled(fixedRate)}
     * cannot directly reference a runtime config value.
     */
    @Scheduled(fixedRate = 1000)
    public void triggerSchedulerMetrics() {
        long intervalMs = configService.loadBalanceConfig().getMetricsReportIntervalMs();
        long now = System.currentTimeMillis();
        if (now - lastMetricsReportTime < intervalMs) {
            return;
        }
        lastMetricsReportTime = now;

        schedulers.forEach(AbstractScheduler::reportMetrics);

        for (PrefillEndpoint ep : endpointRegistry.getPrefillEndpoints().values()) {
            ep.reportBatchMetrics(reporter);
        }
        for (DecodeEndpoint ep : endpointRegistry.getDecodeEndpoints().values()) {
            ep.reportBatchMetrics(reporter);
        }
    }

    @PreDestroy
    public void shutdown() {
        schedulers.forEach(AbstractScheduler::shutdown);
        endpointRegistry.close();
    }
}

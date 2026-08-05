package org.flexlb.service;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.balance.scheduler.AbstractScheduler;
import org.flexlb.balance.scheduler.BatchScheduler;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.DirectScheduler;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.balance.scheduler.QueueScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.QueueSnapshotResponse;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

import java.util.concurrent.CompletableFuture;

@Component
public class RouteService {

    private final ConfigService configService;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;
    private final EndpointRegistry endpointRegistry;
    private final BatchSchedulerReporter reporter;

    // --- the three scheduling paths + global inflight store ---

    private final BatchScheduler batchScheduler;
    private final QueueScheduler queueScheduler;
    private final DirectScheduler directScheduler;
    private final InflightStore globalInflightStore;

    public RouteService(ConfigService configService,
                        DefaultRouter router,
                        RecentCacheKeyTraceReporter recentCacheKeyTraceReporter,
                        FlexMonitor flexMonitor,
                        InflightStore globalInflightStore,
                        EndpointRegistry endpointRegistry,
                        BatchSchedulerReporter reporter,
                        RoutingQueueReporter routingQueueReporter,
                        DynamicWorkerManager dynamicWorkerManager) {
        this.configService = configService;
        this.recentCacheKeyTraceReporter = recentCacheKeyTraceReporter;
        this.globalInflightStore = globalInflightStore;
        this.endpointRegistry = endpointRegistry;
        this.reporter = reporter;

        FlexlbMetricHelper batchHelper = new FlexlbMetricHelper(flexMonitor, MetricConstant.PATH_BATCH);
        batchHelper.register();
        FlexlbMetricHelper queueHelper = new FlexlbMetricHelper(flexMonitor, MetricConstant.PATH_QUEUE);
        queueHelper.register();
        FlexlbMetricHelper directHelper = new FlexlbMetricHelper(flexMonitor, MetricConstant.PATH_DIRECT);
        directHelper.register();

        this.batchScheduler = new BatchScheduler(configService, router, endpointRegistry,
                reporter, globalInflightStore, batchHelper);
        this.queueScheduler = new QueueScheduler(router, configService, routingQueueReporter,
                dynamicWorkerManager, globalInflightStore, queueHelper);
        this.directScheduler = new DirectScheduler(router, globalInflightStore, directHelper);
    }

    /** Start the queue consumer worker pool. */
    @PostConstruct
    public void start() {
        queueScheduler.start();
    }

    /**
     * Route request to appropriate workers based on the deployment-level schedule mode.
     * @param balanceContext Load balancing context
     * @return Routing result
     */
    public CompletableFuture<Response> route(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(flexlbConfig);

        ScheduleModeEnum mode = flexlbConfig.getDefaultScheduleModeEnum();
        balanceContext.setScheduleMode(mode);

        AbstractScheduler scheduler;
        if (mode == ScheduleModeEnum.BATCH && !hasValidGenerateInput(balanceContext)) {
            Logger.warn("BATCH mode cannot process this request, falling back to DIRECT");
            balanceContext.setScheduleMode(ScheduleModeEnum.DIRECT);
            scheduler = directScheduler;
        } else {
            scheduler = switch (mode) {
                case BATCH -> batchScheduler;
                case QUEUE -> queueScheduler;
                default -> directScheduler;
            };
        }

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

    private boolean hasValidGenerateInput(BalanceContext ctx) {
        byte[] bytes = ctx.getGenerateInputPbBytes();
        return bytes != null && bytes.length > 0;
    }

    /**
     * Cancel an inflight request by its string-form request ID.
     *
     * <p>Delegates to {@link AbstractScheduler#cancel(String)} on the batch
     * scheduler — the underlying logic is scheduler-agnostic because all three
     * schedulers share the same global {@link InflightStore}, and the owning
     * scheduler is resolved via {@link InflightItem#scheduler()} inside the
     * delegate. This eliminates the previously duplicated cancel implementation
     * (review F9).
     *
     * @param requestId string-form request ID
     * @return {@code true} if the request was found and cancelled
     */
    public boolean cancel(String requestId) {
        return batchScheduler.cancel(requestId);
    }

    /**
     * Return the number of active (non-terminal) inflight items in the
     * global store. Tombstones within TTL are excluded — external monitors
     * treat this as the live inflight count.
     *
     * @return active inflight count
     */
    public int globalInflightSize() {
        return globalInflightStore.activeCount();
    }

    /**
     * Return the total number of entries in the global store, including
     * terminal tombstones within TTL. Diagnostic view only.
     *
     * @return inflight store size (active + tombstones)
     */
    public int globalInflightTotalSize() {
        return globalInflightStore.size();
    }

    /**
     * Current routing queue length (QUEUE path). Exposed for the HTTP
     * master-info endpoint.
     */
    public int queueLength() {
        return queueScheduler.queueSize();
    }

    /**
     * Dump the routing queue to a JSON snapshot file (QUEUE path). Exposed
     * for the HTTP queue-snapshot diagnostic endpoint.
     */
    public QueueSnapshotResponse snapshotQueue() {
        return queueScheduler.snapshotQueue();
    }

    /**
     * Periodically trigger scheduler-level and per-worker batch metrics
     * reporting.
     *
     * <p>Runs every {@code report.interval.ms} (default 2000ms).
     * <ul>
     *   <li>Path-specific metrics via each scheduler's {@link AbstractScheduler#reportMetrics()}</li>
     *   <li>Per-prefill-worker batch metrics (inflight batch count, queue depth, etc.)</li>
     *   <li>Per-decode-worker batch metrics (inflight request count, KV reserved, etc.)</li>
     * </ul>
     */
    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    public void triggerSchedulerMetrics() {
        batchScheduler.reportMetrics();
        queueScheduler.reportMetrics();
        directScheduler.reportMetrics();

        for (PrefillEndpoint ep : endpointRegistry.getPrefillEndpoints().values()) {
            ep.reportBatchMetrics(reporter);
        }
        for (DecodeEndpoint ep : endpointRegistry.getDecodeEndpoints().values()) {
            ep.reportBatchMetrics(reporter);
        }
    }

    @PreDestroy
    public void shutdown() {
        queueScheduler.shutdown();
        globalInflightStore.shutdown();
        endpointRegistry.close();
    }
}

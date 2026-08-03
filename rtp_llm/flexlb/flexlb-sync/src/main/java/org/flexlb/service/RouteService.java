package org.flexlb.service;

import org.flexlb.balance.scheduler.AbstractScheduler;
import org.flexlb.balance.scheduler.BatchScheduler;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.DirectScheduler;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.QueueScheduler;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;
    private final FlexlbBatchScheduler flexlbBatchScheduler;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    // --- Phase 3: thin-wrapper schedulers + global inflight store ---

    private final BatchScheduler batchScheduler;
    private final QueueScheduler queueScheduler;
    private final DirectScheduler directScheduler;
    private final InflightStore globalInflightStore;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        QueueManager queueManager,
                        FlexlbBatchScheduler flexlbBatchScheduler,
                        RecentCacheKeyTraceReporter recentCacheKeyTraceReporter) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
        this.flexlbBatchScheduler = flexlbBatchScheduler;
        this.recentCacheKeyTraceReporter = recentCacheKeyTraceReporter;

        this.globalInflightStore = new InflightStore();
        this.batchScheduler = new BatchScheduler(flexlbBatchScheduler, globalInflightStore);
        this.queueScheduler = new QueueScheduler(queueManager, globalInflightStore);
        this.directScheduler = new DirectScheduler(defaultScheduler);
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
        if (mode == ScheduleModeEnum.BATCH
                && (flexlbBatchScheduler == null || !hasValidGenerateInput(balanceContext))) {
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
     * <p>Looks up the {@link InflightItem} in the global inflight store and
     * atomically cancels it via CAS. Returns {@code false} if the request was
     * not found (already completed or never tracked, e.g. DIRECT mode).
     *
     * @param requestId string-form request ID
     * @return {@code true} if the request was found and cancelled
     */
    public boolean cancel(String requestId) {
        InflightItem item = globalInflightStore.get(requestId);
        if (item == null) {
            return false;
        }
        return item.cancel();
    }

    public RequestLifecycleSnapshot getRequestState(long requestId,
                                                    long expectedBatchId) {
        return flexlbBatchScheduler == null ? null
                : flexlbBatchScheduler.getRequestState(requestId, expectedBatchId);
    }
}

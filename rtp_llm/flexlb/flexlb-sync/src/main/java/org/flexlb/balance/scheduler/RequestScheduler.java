package org.flexlb.balance.scheduler;

import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_AVAILABLE_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_DECODE_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_PDFUSION_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_PREFILL_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_VIT_WORKER;

/**
 * Request scheduler - manages worker thread pool, consumes request queue, and executes routing
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class RequestScheduler {

    private final Router router;
    private final ConfigService configService;
    private final QueueManager queueManager;
    private final DynamicWorkerManager dynamicWorkerManager;
    private final RoutingQueueReporter metrics;

    // Worker thread pool
    private ExecutorService workerExecutor;
    private volatile boolean running = true;

    public RequestScheduler(Router router,
                            ConfigService configService,
                            QueueManager queueManager,
                            DynamicWorkerManager dynamicWorkerManager,
                            RoutingQueueReporter metrics) {
        this.router = router;
        this.configService = configService;
        this.queueManager = queueManager;
        this.dynamicWorkerManager = dynamicWorkerManager;
        this.metrics = metrics;
    }

    @PostConstruct
    public void start() {
        WhaleMasterConfig config = configService.loadBalanceConfig();

        // Start worker thread pool
        this.workerExecutor = Executors.newFixedThreadPool(config.getScheduleWorkerSize(), r -> {
            Thread t = new Thread(r, "routing-queue-worker");
            t.setDaemon(true);
            return t;
        });

        // Submit worker tasks
        for (int i = 0; i < config.getScheduleWorkerSize(); i++) {
            workerExecutor.submit(this::workerLoop);
        }

        Logger.info("RequestScheduler Worker thread pool started, worker count: {}", config.getScheduleWorkerSize());
    }

    /**
     * Worker thread main loop
     * <p>
     * Workflow:
     *   1. Wait for resource availability
     *   2. Take request from queue
     *   3. Process request
     */
    private void workerLoop() {
        Logger.info("Worker thread started, ready to process requests...");

        while (running && !Thread.currentThread().isInterrupted()) {
            try {
                dynamicWorkerManager.acquirePermit();
                try {
                    BalanceContext ctx = queueManager.takeRequest(true, 50);
                    if (ctx != null) {
                        Logger.debug("Worker processing request id: {}", ctx.getRequestId());
                        processRequest(ctx);
                    }
                } finally {
                    dynamicWorkerManager.releasePermit();
                }
            } catch (Exception e) {
                Logger.error("Worker thread encountered error", e);
            }
        }

        Logger.info("Worker thread stopped");
    }

    private void processRequest(BalanceContext ctx) {
        try {
            Response response = router.route(ctx);
            handleRoutingResult(ctx, response);
        } catch (Exception e) {
            Logger.error("Worker thread failed to route ctx id:{}", ctx.getRequestId(), e);
            ctx.getFuture().completeExceptionally(e);
        }
    }

    private void handleRoutingResult(BalanceContext ctx, Response response) {
        if (!response.isSuccess() && shouldRetry(response)) {
            ctx.incrementRetryCount();
            Logger.warn("Route failed for request id:{}, error: {}, attempting retry {}",
                    ctx.getRequestId(),
                    response.getCode(),
                    ctx.getRetryCount());
            metrics.reportRoutingFailureQps(response.getCode());

            queueManager.offerToHead(ctx);
        } else {
            ctx.getFuture().complete(response);
            metrics.reportRoutingSuccessQps(ctx.getRetryCount());
        }
    }

    /**
     * Determine if request should be retried
     * <p>
     * Only resource-unavailable errors should trigger retry to avoid ineffective retry attempts
     *
     * @param response Routing response
     * @return true if request should be retried, false otherwise
     */
    private boolean shouldRetry(Response response) {
        int code = response.getCode();
        return code == NO_AVAILABLE_WORKER.getErrorCode()
                || code == NO_PREFILL_WORKER.getErrorCode()
                || code == NO_DECODE_WORKER.getErrorCode()
                || code == NO_PDFUSION_WORKER.getErrorCode()
                || code == NO_VIT_WORKER.getErrorCode();
    }

    @PreDestroy
    public void shutdown() {
        running = false;
        if (workerExecutor != null && !workerExecutor.isShutdown()) {
            workerExecutor.shutdown();
            try {
                if (!workerExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                    workerExecutor.shutdownNow();
                }
                Logger.info("RequestScheduler Worker thread pool stopped");
            } catch (InterruptedException e) {
                workerExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
}
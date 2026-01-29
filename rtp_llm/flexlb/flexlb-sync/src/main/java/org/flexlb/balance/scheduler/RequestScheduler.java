package org.flexlb.balance.scheduler;

import org.flexlb.balance.resource.ResourceAvailableListener;
import org.flexlb.balance.resource.ResourceMonitor;
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
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_AVAILABLE_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_DECODE_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_PDFUSION_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_PREFILL_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_VIT_WORKER;

/**
 * 请求调度器: 管理Worker线程池,消费队列并执行路由
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class RequestScheduler implements ResourceAvailableListener {

    private final Router router;
    private final ConfigService configService;
    private final QueueManager queueManager;
    private final ResourceMonitor resourceMonitor;
    private final RoutingQueueReporter metrics;

    // Worker线程池
    private ExecutorService workerExecutor;

    // 资源可用信号
    private final ReentrantLock resourceLock = new ReentrantLock();
    private final Condition resourceAvailableCondition = resourceLock.newCondition();

    private volatile boolean running = true;

    public RequestScheduler(Router router,
                            ConfigService configService,
                            QueueManager queueManager,
                            ResourceMonitor resourceMonitor,
                            RoutingQueueReporter metrics) {
        this.router = router;
        this.configService = configService;
        this.queueManager = queueManager;
        this.resourceMonitor = resourceMonitor;
        this.metrics = metrics;
    }

    @PostConstruct
    public void start() {
        WhaleMasterConfig config = configService.loadBalanceConfig();

        // 注册资源可用事件监听器
        resourceMonitor.setResourceAvailableListener(this);

        // 启动 Worker 线程池
        this.workerExecutor = Executors.newFixedThreadPool(config.getWorkerThreadPoolCoreSize(), r -> {
            Thread t = new Thread(r, "routing-queue-worker");
            t.setDaemon(true);
            return t;
        });

        // 提交worker任务
        for (int i = 0; i < config.getWorkerThreadPoolCoreSize(); i++) {
            workerExecutor.submit(this::workerLoop);
        }

        Logger.info("RequestScheduler Worker thread pool started, worker count: {}", config.getWorkerThreadPoolCoreSize());
    }

    @Override
    public void onResourceAvailable() {
        resourceLock.lock();
        try {
            resourceAvailableCondition.signalAll();
        } finally {
            resourceLock.unlock();
        }
    }

    /**
     * Worker线程主循环
     * <p>
     * 工作流程:
     *   1. 等待资源可用
     *   2. 优先从重试队列取请求
     *   3. 如果重试队列空，从主队列取请求
     *   4. 处理请求
     */
    private void workerLoop() {
        Logger.info("Worker thread started, ready to process requests...");

        while (running && !Thread.currentThread().isInterrupted()) {
            try {
                // 1. 等待资源可用
                waitForResourceAvailable();

                // 2. 优先从重试队列取请求
                BalanceContext ctx = queueManager.takeFromRetryQueue();
                if (ctx != null) {
                    Logger.debug("Worker processing retry request id: {}", ctx.getRequestId());
                    processRequest(ctx);
                    continue;
                }

                // 3. 重试队列空，从主队列取请求
                ctx = queueManager.takeFromMainQueue();
                if (ctx != null) {
                    Logger.debug("Worker processing main queue request id: {}", ctx.getRequestId());
                    processRequest(ctx);
                }

            } catch (InterruptedException e) {
                Logger.info("Worker thread interrupted, exiting...");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Logger.error("Worker thread encountered error", e);
            }
        }

        Logger.info("Worker thread stopped");
    }

    /**
     * 等待资源可用
     * <p>
     * 循环检查资源是否可用,若不可用则阻塞等待资源信号
     *
     * @throws InterruptedException 如果等待被中断
     */
    private void waitForResourceAvailable() throws InterruptedException {
        while (!resourceMonitor.hasAvailableResource()) {
            Logger.debug("No available resources, Worker thread waiting for signal...");
            resourceLock.lock();
            try {
                resourceAvailableCondition.await();
            } finally {
                resourceLock.unlock();
            }
            Logger.debug("Worker thread woke up by resource signal");
        }
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

            queueManager.getRetryQueue().offer(ctx);
        } else {
            ctx.getFuture().complete(response);
            metrics.reportRoutingSuccessQps(ctx.getRetryCount());
        }
    }

    /**
     * 判断是否应该重试
     * <p>
     * 只有资源不足相关的错误才应该重试,避免无效重试
     *
     * @param response 路由响应
     * @return 是否应该重试
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
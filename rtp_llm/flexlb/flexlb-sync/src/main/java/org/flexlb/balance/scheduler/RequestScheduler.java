package org.flexlb.balance.scheduler;

import org.flexlb.balance.resource.ResourceAvailableListener;
import org.flexlb.balance.resource.ResourceMonitor;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.dao.loadbalance.QueuedRequest;
import org.flexlb.util.LoggingUtils;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_AVAILABLE_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_DECODE_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_PDFUSION_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_PREFILL_WORKER;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_VIT_WORKER;

/**
 * 请求调度器: 管理Boss/Worker线程池,消费队列并执行路由
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class RequestScheduler implements ResourceAvailableListener {

    private final Router router;
    private final ConfigService configService;
    private final RoutingQueueManager queueManager;
    private final ResourceMonitor resourceMonitor;

    // Boss线程
    private ExecutorService bossExecutor;

    // Worker线程池
    private ThreadPoolExecutor workerExecutor;

    // 资源可用信号
    private final ReentrantLock resourceLock = new ReentrantLock();
    private final Condition resourceAvailableCondition = resourceLock.newCondition();

    public RequestScheduler(Router router, ConfigService configService,
                            RoutingQueueManager queueManager, ResourceMonitor resourceMonitor) {
        this.router = router;
        this.configService = configService;
        this.queueManager = queueManager;
        this.resourceMonitor = resourceMonitor;
    }

    @PostConstruct
    public void start() {
        WhaleMasterConfig config = configService.loadBalanceConfig();

        // 注册资源可用事件监听器
        resourceMonitor.setResourceAvailableListener(this);

        // 1. 启动 Boss 线程
        this.bossExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "routing-queue-boss");
            t.setDaemon(true);
            return t;
        });
        bossExecutor.submit(this::bossLoop);
        LoggingUtils.info("RequestScheduler Boss thread started");

        // 2. 启动 Worker 线程池
        this.workerExecutor = new ThreadPoolExecutor(
                config.getWorkerThreadPoolCoreSize(),
                config.getWorkerThreadPoolMaxSize(),
                60L,
                TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(config.getWorkerThreadPoolQueueSize()),
                r -> {
                    Thread t = new Thread(r, "routing-queue-worker");
                    t.setDaemon(true);
                    return t;
                },
                new ThreadPoolExecutor.CallerRunsPolicy()
        );
        LoggingUtils.info("RequestScheduler Worker thread pool started, core: {}, max: {}, queue: {}",
                config.getWorkerThreadPoolCoreSize(),
                config.getWorkerThreadPoolMaxSize(),
                config.getWorkerThreadPoolQueueSize());
    }

    @Override
    public void onResourceAvailable() {
        resourceLock.lock();
        try {
            resourceAvailableCondition.signal();
        } finally {
            resourceLock.unlock();
        }
    }

    /**
     * Boss 线程主循环
     * <p>
     * 工作流程:
     *   1. dequeue() 阻塞等待请求           ← 新请求入队时唤醒
     *   2. waitForResourceAvailable()     ← 资源可用信号唤醒
     *      └─ 循环检查 hasAvailableResource()
     *      └─ 不可用则 await() 等待信号
     *   3. 提交到Worker线程池执行
     * <p>
     */
    private void bossLoop() {
        LoggingUtils.info("Boss thread started, ready to consume queued requests...");

        while (!Thread.currentThread().isInterrupted()) {
            try {
                // 1. 阻塞等待下一个可用请求
                QueuedRequest request = queueManager.dequeue();

                // 2. 检查资源是否可用,若不可用则等待资源信号
                waitForResourceAvailable();

                // 3. 异步提交请求路由
                processRequest(request);

            } catch (InterruptedException e) {
                LoggingUtils.info("Boss thread interrupted, exiting...");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                LoggingUtils.error("Boss thread encountered error", e);
            }
        }

        LoggingUtils.info("Boss thread stopped");
    }

    /**
     * 等待资源可用
     * <p>
     * 循环检查资源是否可用,若不可用则阻塞等待资源信号
     *
     * @throws InterruptedException 如果等待被中断
     */
    private void waitForResourceAvailable() throws InterruptedException {
        // 如果资源可用，则不会进入循环并等待
        while (!resourceMonitor.hasAvailableResource()) {
            LoggingUtils.debug("No available resources, Boss thread waiting for signal...");
            resourceLock.lock();
            try {
                resourceAvailableCondition.await();
            } finally {
                resourceLock.unlock();
            }
            LoggingUtils.debug("Boss thread woke up by resource signal");
        }
    }

    /**
     * 异步提交请求路由
     *
     * @param request   排队请求
     */
    private void processRequest(QueuedRequest request) {

        workerExecutor.submit(() -> {
            try {
                MasterResponse response = router.route(request.getBalanceContext());

                // 检查路由是否失败且可以重试
                if (!response.isSuccess()
                        && shouldRetry(response)
                        && request.canRetry(configService.loadBalanceConfig().getMaxRetryCount())) {

                    // 增加重试计数
                    request.incrementRetryCount();
                    LoggingUtils.warn("Route failed for request id:{}, error: {}, attempting retry {}/{}",
                            request.getBalanceContext().getInterRequestId(),
                            response.getCode(),
                            request.getRetryCount(),
                            configService.loadBalanceConfig().getMaxRetryCount());

                    // 重新入队到队列头部
                    boolean requeued = queueManager.reEnqueueAtFront(request);
                    if (!requeued) {
                        // 重新入队失败,直接返回失败响应
                        LoggingUtils.error("Failed to requeue request id:{}, returning error", request.getBalanceContext().getInterRequestId());
                        request.getFuture().complete(response);
                    }
                    // 重新入队成功,不完成 future,等待重新调度
                } else {
                    // 成功或不可重试的失败,完成 future
                    request.getFuture().complete(response);
                }
            } catch (Exception e) {
                LoggingUtils.error("Worker thread failed to route request id:{}",
                        request.getBalanceContext().getInterRequestId(), e);
                request.getFuture().completeExceptionally(e);
            }
        });
    }

    /**
     * 判断是否应该重试
     * <p>
     * 只有资源不足相关的错误才应该重试,避免无效重试
     *
     * @param response 路由响应
     * @return 是否应该重试
     */
    private boolean shouldRetry(MasterResponse response) {
        int code = response.getCode();
        // 只对以下错误类型进行重试
        return code == NO_AVAILABLE_WORKER.getErrorCode()
                || code == NO_PREFILL_WORKER.getErrorCode()
                || code == NO_DECODE_WORKER.getErrorCode()
                || code == NO_PDFUSION_WORKER.getErrorCode()
                || code == NO_VIT_WORKER.getErrorCode();
    }

    @PreDestroy
    public void shutdown() {
        // 关闭 Boss 线程
        if (bossExecutor != null && !bossExecutor.isShutdown()) {
            bossExecutor.shutdown();
            try {
                if (!bossExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                    bossExecutor.shutdownNow();
                }
                LoggingUtils.info("RequestScheduler Boss thread stopped");
            } catch (InterruptedException e) {
                bossExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }

        // 关闭 Worker 线程池
        if (workerExecutor != null && !workerExecutor.isShutdown()) {
            workerExecutor.shutdown();
            try {
                if (!workerExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                    workerExecutor.shutdownNow();
                }
                LoggingUtils.info("RequestScheduler Worker thread pool stopped");
            } catch (InterruptedException e) {
                workerExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
}

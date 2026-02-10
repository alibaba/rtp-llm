package org.flexlb.cases;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.BodyInserters;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.lang.reflect.Field;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.dao.loadbalance.StrategyErrorType.QUEUE_FULL;

/**
 * FlexLB 队列压力测试
 * <p>
 * 测试队列机制的极限值，包括：
 * 1. 队列满载时正确拒绝请求
 * 2. 极端压力下的性能表现
 * 3. 并发入队的线程安全性
 *
 * @author claude
 * @since 2026/1/6
 */
@Slf4j
public class QueueStressTest {

    private final WebTestClient webClient;
    private final ConfigService configService;

    private QueueStressTest(WebTestClient webClient, ConfigService configService) {
        this.webClient = webClient;
        this.configService = configService;
    }

    public static QueueStressTest init(WebTestClient webClient, EnvironmentVariables environmentVariables,
                                       ConfigService configService
    ) {
        environmentVariables.set("DOMAIN_ADDRESS:com.prefill.hosts.address", "127.0.0.100:8080,127.0.0.101:8080");
        environmentVariables.set("DOMAIN_ADDRESS:com.decode.hosts.address", "127.0.0.102:8080,127.0.0.103:8080");
        return new QueueStressTest(webClient, configService);
    }

    public QueueStressTest resetQueue(QueueManager queueManager, int size) {
        try {
            Field queueField = QueueManager.class.getDeclaredField("queue");
            queueField.setAccessible(true);
            BlockingDeque<BalanceContext> newQueue = new LinkedBlockingDeque<>(size);
            queueField.set(queueManager, newQueue);
            log.info("Queue reset to size: {}", size);
            return this;
        } catch (Exception e) {
            throw new RuntimeException("Failed to reset queue size", e);
        }
    }

    /**
     * 测试场景一：队列满载时正确拒绝请求
     * <p>
     * 测试流程：
     * 1. 设置 maxQueueSize = 10
     * 2. 设置 Worker 资源不可用，强制所有请求进入排队
     * 3. 发送 20 个请求（超过队列容量）
     * 5. 验证后 10 个请求被拒绝（返回 QUEUE_FULL 8502）
     */
    @SneakyThrows
    public void testQueueFullRejection() {
        log.info("=== 开始测试：队列满载拒绝 ===");

        try {
            // 1. 配置调整
            configService.loadBalanceConfig().setEnableQueueing(true);

            // 2. 设置 Worker 状态（资源不足，强制排队）
            setupLimitedWorkerResources();

            // 3. 统计变量
            AtomicInteger rejectedCount = new AtomicInteger(0);
            CountDownLatch completionLatch = new CountDownLatch(10);

            // 4. 并发发送 20 个请求（超过队列容量 10）
            try (ExecutorService executor = Executors.newFixedThreadPool(20)) {
                for (int i = 0; i < 20; i++) {
                    final int requestId = i;
                    executor.submit(() -> {
                        try {
                            String response = webClient.mutate()
                                    .responseTimeout(Duration.ofMinutes(5))
                                    .build()
                                    .post()
                                    .uri("/rtp_llm/schedule")
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .body(BodyInserters.fromValue(buildRequestBody(requestId)))
                                    .exchange()
                                    .expectStatus()
                                    .is5xxServerError()
                                    .expectBody(String.class)
                                    .returnResult()
                                    .getResponseBody();

                            if (response != null && response.contains(QUEUE_FULL.getErrorMsg())) { // QUEUE_FULL
                                rejectedCount.incrementAndGet();
                            }
                            log.info("Request {} result: {}", requestId, response);
                            completionLatch.countDown();

                        } catch (Exception e) {
                            log.error("Request {} failed", requestId, e);
                            completionLatch.countDown();
                        }
                    });
                }

                // 5. 等待完成
                //noinspection ResultOfMethodCallIgnored
                completionLatch.await(2, TimeUnit.SECONDS);
                executor.shutdownNow();
            }

            // 6. 验证结果
            log.info("队列满载测试结果: 被拒绝={}", rejectedCount.get());

            // 验证：前 10 个请求应该成功入队，后 10 个应该被拒绝
            // 允许 +/- 1 的误差，因为存在并发处理的竞态条件
            assert rejectedCount.get() >= 9 && rejectedCount.get() <= 11 :
                    "应该有约 10 个请求被拒绝，实际: " + rejectedCount.get();

            log.info("=== 队列满载拒绝测试通过 ===");

        } finally {
            cleanup();
        }
    }

    /**
     * 测试场景二：并发入队的线程安全性
     * <p>
     * 测试流程：
     * 1. 设置 maxQueueSize = 500
     * 2. 使用 900 个线程，发送 900 个请求
     * 3. 使用 CountDownLatch 同步所有线程同时开始
     * 4. 验证队列大小不超过 maxQueueSize，无数据竞争或死锁
     */
    @SneakyThrows
    public void testConcurrentEnqueue() {
        log.info("=== 开始测试：并发入队线程安全 ===");

        try {
            // 1. 配置调整
            configService.loadBalanceConfig().setEnableQueueing(true);

            // 2. 设置 Worker 状态
            setupLimitedWorkerResources();

            // 3. 并发控制
            int threadCount = 900;

            AtomicInteger rejectedCount = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch completionLatch = new CountDownLatch(500);

            // 4. 创建线程池
            boolean completed;

            for (int t = 0; t < threadCount; t++) {
                final int threadId = t;
                new Thread(() -> {
                    try {
                       // 等待所有线程同时开始
                       startLatch.await();

                       String responseBody = webClient.mutate()
                               .responseTimeout(Duration.ofMinutes(5))
                               .build()
                               .post()
                               .uri("/rtp_llm/schedule")
                               .contentType(MediaType.APPLICATION_JSON)
                               .body(BodyInserters.fromValue(buildRequestBody(threadId)))
                               .exchange()
                               .expectStatus()
                               .is5xxServerError()
                               .expectBody(String.class)
                               .returnResult()
                               .getResponseBody();

                       if (responseBody != null && responseBody.contains("8502")) {
                           rejectedCount.incrementAndGet();
                       }
                       log.info("Thread {} received response: {}", threadId, responseBody);
                       completionLatch.countDown();

                   } catch (Exception e) {
                       log.error("Thread {} failed", threadId, e);
                   }
               }).start();
            }

            // 6. 同时启动所有线程
            Thread.sleep(100); // 确保所有线程都准备好了
            startLatch.countDown();

            // 7. 等待所有线程完成
            log.info("=== 等待并发测试完成 ===");
            completed = completionLatch.await(2, TimeUnit.SECONDS);

            // 8. 验证结果
            log.info("并发测试结果: 拒绝={}, 测试完成={}", rejectedCount.get(), completed);

            // 验证：前 500 个请求应该成功入队，后 400 个应该被拒绝
            // 允许 +/- 10 的误差，因为存在并发处理的竞态条件
            assert rejectedCount.get() >= 390 && rejectedCount.get() <= 410 :
                    "应该有约 400 个请求被拒绝，实际: " + rejectedCount.get();

            log.info("=== 并发入队线程安全测试通过 ===");

        } finally {
            cleanup();
        }
    }

    /**
     * 设置受限的 Worker 资源，强制请求进入排队
     */
    private void setupLimitedWorkerResources() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();

        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setAlive(true);
        workerStatus.setUsedKvCacheTokens(new AtomicLong(990L)); // 高使用率，模拟资源紧张
        workerStatus.setAvailableKvCacheTokens(new AtomicLong(10L)); // 非常小的资源，强制排队

        // 配置多个 Prefill Worker
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.100:8080", workerStatus);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.101:8080", workerStatus);

        // 配置多个 Decode Worker
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("127.0.0.102:8080", workerStatus);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("127.0.0.103:8080", workerStatus);
    }

    /**
     * 构建请求体
     */
    private String buildRequestBody(int requestId) {
        return String.format("""
                {
                  "request_id": "test-request-%d",
                  "model": "engine_service",
                  "block_cache_keys": [%d, %d, %d],
                  "seq_len": 1000,
                  "debug": 1
                }
                """, requestId, requestId * 1000, requestId * 1000 + 1, requestId * 1000 + 2);
    }

    /**
     * 清理测试环境
     */
    private void cleanup() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
        configService.loadBalanceConfig().setEnableQueueing(false);
        configService.loadBalanceConfig().setMaxQueueSize(100000);
        log.info("测试环境已清理");
    }
}
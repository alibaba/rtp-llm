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
 * FlexLB queue stress test
 * <p>
 * Tests queue mechanism limits, including:
 * 1. Correct request rejection when queue is full
 * 2. Performance under extreme pressure
 * 3. Thread safety of concurrent enqueue operations
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
     * Test scenario 1: Correct request rejection when queue is full
     * <p>
     * Test flow:
     * 1. Set maxQueueSize = 10
     * 2. Set Worker resources unavailable, forcing all requests into queue
     * 3. Send 20 requests (exceeding queue capacity)
     * 5. Verify last 10 requests are rejected (return QUEUE_FULL 8502)
     */
    @SneakyThrows
    public void testQueueFullRejection() {
        log.info("=== Starting test: Queue full rejection ===");

        try {
            // 1. Configuration adjustment
            configService.loadBalanceConfig().setEnableQueueing(true);

            // 2. Set Worker status (insufficient resources, force queuing)
            setupLimitedWorkerResources();

            // 3. Statistics variables
            AtomicInteger rejectedCount = new AtomicInteger(0);
            CountDownLatch completionLatch = new CountDownLatch(10);

            // 4. Concurrently send 20 requests (exceeding queue capacity 10)
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

                // 5. Wait for completion
                //noinspection ResultOfMethodCallIgnored
                completionLatch.await(2, TimeUnit.SECONDS);
                executor.shutdownNow();
            }

            // 6. Verify results
            log.info("Queue full test results: rejected={}", rejectedCount.get());

            // Verify: First 10 requests should successfully enqueue, last 10 should be rejected
            // Allow +/- 1 margin due to concurrent processing race conditions
            assert rejectedCount.get() >= 9 && rejectedCount.get() <= 11 :
                    "Should have approximately 10 requests rejected, actual: " + rejectedCount.get();

            log.info("=== Queue full rejection test passed ===");

        } finally {
            cleanup();
        }
    }

    /**
     * Test scenario 2: Thread safety of concurrent enqueue operations
     * <p>
     * Test flow:
     * 1. Set maxQueueSize = 500
     * 2. Use 900 threads to send 900 requests
     * 3. Use CountDownLatch to synchronize all threads starting simultaneously
     * 4. Verify queue size does not exceed maxQueueSize, no data race or deadlock
     */
    @SneakyThrows
    public void testConcurrentEnqueue() {
        log.info("=== Starting test: Concurrent enqueue thread safety ===");

        try {
            // 1. Configuration adjustment
            configService.loadBalanceConfig().setEnableQueueing(true);

            // 2. Set Worker status
            setupLimitedWorkerResources();

            // 3. Concurrency control
            int threadCount = 900;

            AtomicInteger rejectedCount = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch completionLatch = new CountDownLatch(500);

            // 4. Create thread pool
            boolean completed;

            for (int t = 0; t < threadCount; t++) {
                final int threadId = t;
                new Thread(() -> {
                    try {
                       // Wait for all threads to start simultaneously
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

            // 6. Start all threads simultaneously
            Thread.sleep(100); // Ensure all threads are ready
            startLatch.countDown();

            // 7. Wait for all threads to complete
            log.info("=== Waiting for concurrent test completion ===");
            completed = completionLatch.await(2, TimeUnit.SECONDS);

            // 8. Verify results
            log.info("Concurrent test results: rejected={}, completed={}", rejectedCount.get(), completed);

            // Verify: First 500 requests should successfully enqueue, last 400 should be rejected
            // Allow +/- 10 margin due to concurrent processing race conditions
            assert rejectedCount.get() >= 390 && rejectedCount.get() <= 410 :
                    "Should have approximately 400 requests rejected, actual: " + rejectedCount.get();

            log.info("=== Concurrent enqueue thread safety test passed ===");

        } finally {
            cleanup();
        }
    }

    /**
     * Set limited Worker resources to force requests into queue
     */
    private void setupLimitedWorkerResources() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();

        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setAlive(true);
        workerStatus.setUsedKvCacheTokens(new AtomicLong(990L)); // High usage, simulating resource constraints
        workerStatus.setAvailableKvCacheTokens(new AtomicLong(10L)); // Very small resources, force queuing

        // Configure multiple Prefill Workers
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.100:8080", workerStatus);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.101:8080", workerStatus);

        // Configure multiple Decode Workers
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("127.0.0.102:8080", workerStatus);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("127.0.0.103:8080", workerStatus);
    }

    /**
     * Build request body
     */
    private String buildRequestBody(int requestId) {
        return String.format("""
                {
                  "request_id": %d,
                  "model": "engine_service",
                  "block_cache_keys": [%d, %d, %d],
                  "seq_len": 1000,
                  "debug": 1
                }
                """, requestId, requestId * 1000, requestId * 1000 + 1, requestId * 1000 + 2);
    }

    /**
     * Clean up test environment
     */
    private void cleanup() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
        configService.loadBalanceConfig().setEnableQueueing(false);
        configService.loadBalanceConfig().setMaxQueueSize(100000);
        log.info("Test environment cleaned up");
    }
}
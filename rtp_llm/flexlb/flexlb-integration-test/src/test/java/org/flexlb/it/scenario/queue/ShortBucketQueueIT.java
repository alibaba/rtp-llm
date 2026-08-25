package org.flexlb.it.scenario.queue;

import org.flexlb.Application;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.web.reactive.server.WebTestClient;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Verifies the short-bucket queue path through the production scheduler.
 *
 * <p>A burst waits behind an active retry while no worker is available, then drains after one
 * worker recovers. The test asserts the intermediate queue size to prove that it is not merely
 * direct routing under concurrent HTTP load.
 */
@ActiveProfiles("test")
@AutoConfigureWebTestClient
@ExtendWith(SystemStubsExtension.class)
@ContextConfiguration(initializers = QueueingShortestTtftContextInitializer.class)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@SpringBootTest(
        classes = Application.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = "logging.config=classpath:logback-it.xml")
class ShortBucketQueueIT {

    private static final RoleType SCHEDULING_ROLE = RoleType.PDFUSION;
    private static final int BURST_SIZE = 3;

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @Autowired
    private WebTestClient webTestClient;

    @Autowired
    private QueueManager queueManager;

    @BeforeAll
    static void configureEnvironment() {
        environmentVariables.set("HIPPO_ROLE", "flexlb-it");
        environmentVariables.remove("FLEXLB_NACOS_SERVER_ADDR");
    }

    @BeforeEach
    void resetScriptedWorkers() {
        IntegrationTestFixtures.clearWorkerStatusFailures();
        IntegrationTestFixtures.resetWorkers();
    }

    @Test
    void should_drain_short_request_burst_when_worker_recovers() throws Exception {
        IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, 0, false, 0, 0, 1.0);
        IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, 1, false, 0, 0, 1.0);
        await("both worker statuses to become unavailable").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(5))
                .until(() -> !workerAlive(0) && !workerAlive(1));

        ExecutorService executor = Executors.newFixedThreadPool(BURST_SIZE);
        try {
            List<CompletableFuture<Integer>> responses = IntStream.range(0, BURST_SIZE)
                    .mapToObj(index -> CompletableFuture.supplyAsync(
                            () -> schedule("short-bucket-" + index), executor))
                    .toList();

            await("short request burst to wait behind the active scheduler retry")
                    .pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(5))
                    .until(() -> queueManager.getQueue().size() == BURST_SIZE - 1);

            IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, 1, true, 0, 0, 1.0);
            await("worker recovery status").pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10))
                    .atMost(Duration.ofSeconds(5)).until(() -> workerAlive(1));

            for (CompletableFuture<Integer> response : responses) {
                assertEquals(200, response.get(5, TimeUnit.SECONDS));
            }
            await("short-bucket queue drain").pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10))
                    .atMost(Duration.ofSeconds(5)).until(() -> queueManager.getQueue().isEmpty());
        } finally {
            executor.shutdownNow();
            executor.awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    private int schedule(String requestId) {
        return webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", requestId,
                        "block_cache_keys", new long[]{11L, 22L},
                        "block_size", 16,
                        "seq_len", 32,
                        "generate_timeout", 5_000))
                .exchange()
                .returnResult(String.class)
                .getStatus()
                .value();
    }

    private boolean workerAlive(int workerIndex) {
        var worker = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getRoleStatusMap(SCHEDULING_ROLE)
                .get(IntegrationTestFixtures.workerIpPort(SCHEDULING_ROLE, workerIndex));
        return worker != null && worker.isAlive();
    }
}

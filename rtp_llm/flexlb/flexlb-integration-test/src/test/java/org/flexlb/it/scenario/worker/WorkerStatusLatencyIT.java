package org.flexlb.it.scenario.worker;

import org.flexlb.Application;
import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.it.scenario.strategy.shortestttft.ShortestTtftContextInitializer;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
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
import java.util.Map;

import static org.awaitility.Awaitility.await;

/**
 * Exercises the status synchronizer when its 50 ms cadence encounters intermittent 200 ms worker
 * responses.
 */
@ActiveProfiles("test")
@AutoConfigureWebTestClient
@ExtendWith(SystemStubsExtension.class)
@ContextConfiguration(initializers = ShortestTtftContextInitializer.class)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@SpringBootTest(
        classes = Application.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = "logging.config=classpath:logback-it.xml")
class WorkerStatusLatencyIT {

    private static final RoleType SCHEDULING_ROLE = RoleType.PDFUSION;

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @Autowired
    private WebTestClient webTestClient;

    @BeforeAll
    static void configureEnvironment() {
        environmentVariables.set("HIPPO_ROLE", "flexlb-it");
        environmentVariables.remove("FLEXLB_NACOS_SERVER_ADDR");
    }

    @BeforeEach
    void resetScriptedWorkers() {
        IntegrationTestFixtures.clearWorkerStatusFailures();
        IntegrationTestFixtures.clearWorkerStatusLatencies();
        IntegrationTestFixtures.resetWorkers();
        IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, 1, true, 0, 0, 1.0);
        await("initial worker-status synchronization").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(5))
                .until(() -> workerAlive(0) && workerAlive(1));
    }

    @AfterEach
    void clearScriptedLatency() {
        IntegrationTestFixtures.clearWorkerStatusLatencies();
    }

    @Test
    void should_continue_scheduling_when_worker_status_responses_are_intermittently_slow() {
        IntegrationTestFixtures.delayEveryWorkerStatusResponse(
                SCHEDULING_ROLE, 1, Duration.ofMillis(200), 3);
        await("first delayed worker-status response").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(3))
                .until(() -> IntegrationTestFixtures.delayedWorkerStatusResponses(SCHEDULING_ROLE, 1) > 0);

        for (int requestIndex = 0; requestIndex < 30; requestIndex++) {
            webTestClient.post()
                    .uri("/rtp_llm/schedule")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(request("slow-status-" + requestIndex))
                    .exchange()
                    .expectStatus().isOk()
                    .expectBody()
                    .jsonPath("$.success").isEqualTo(true);
        }

        await("healthy state after intermittent delayed status responses").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(3)).until(() ->
                        IntegrationTestFixtures.delayedWorkerStatusResponses(SCHEDULING_ROLE, 1) >= 2
                        && workerAlive(0)
                        && workerAlive(1));
    }

    private Map<String, Object> request(String requestId) {
        return Map.of(
                "request_id", requestId,
                "block_cache_keys", new long[]{11L, 22L},
                "block_size", 16,
                "seq_len", 32,
                "generate_timeout", 500);
    }

    private boolean workerAlive(int workerIndex) {
        var worker = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getRoleStatusMap(SCHEDULING_ROLE)
                .get(IntegrationTestFixtures.workerIpPort(SCHEDULING_ROLE, workerIndex));
        return worker != null && worker.isAlive();
    }
}

package org.flexlb.it.scenario.strategy.shortestttft;

import org.flexlb.Application;
import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
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
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Baseline {@code SHORTEST_TTFT} integration coverage for a single-engine topology.
 *
 * <p>The scenarios verify healthy/all-down behavior and that a short request avoids a worker
 * reporting a long 1M-token prefill snapshot.
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
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class SingleEngineSchedulingIT {

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
        IntegrationTestFixtures.resetWorkers();
    }

    @Test
    @Order(1)
    void should_select_only_healthy_worker_when_other_worker_is_unavailable() {
        await("worker-status synchronization").pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10))
                .atMost(Duration.ofSeconds(5)).until(() -> IntegrationTestFixtures.workerStatusCalls() > 0
                        && workerAlive(0, true)
                        && workerAlive(1, false));

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "single-engine-healthy",
                        "block_cache_keys", new long[]{11L, 22L},
                        "block_size", 16,
                        "seq_len", 32,
                        "generate_timeout", 500))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.server_status[0].server_ip").value(value ->
                        assertEquals(IntegrationTestFixtures.WORKER_IP, value))
                .jsonPath("$.server_status[0].http_port")
                .isEqualTo(IntegrationTestFixtures.workerHttpPort(SCHEDULING_ROLE, 0));
    }

    @Test
    @Order(2)
    void should_return_no_available_worker_when_all_workers_report_not_alive() {
        IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, 0, false, 0, 0, 1.0);
        IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, 1, false, 0, 0, 1.0);
        await("all worker statuses to become unavailable").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(5)).until(() ->
                        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(RoleType.PDFUSION).values()
                        .stream()
                        .noneMatch(worker -> worker.isAlive()));

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request("all-workers-down", 32))
                .exchange()
                .expectStatus().is5xxServerError()
                .expectBody()
                .jsonPath("$.success").isEqualTo(false);
    }

    @Test
    @Order(3)
    void should_select_lower_ttft_worker_when_short_bucket_has_long_prefill_competitor() {
        IntegrationTestFixtures.setWorkerLongPrefillStatus(SCHEDULING_ROLE, 0);
        IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, 1, true, 0, 0, 1.0);
        await("both worker statuses to become alive").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(5))
                .until(() -> workerAlive(0, true) && workerAlive(1, true));

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request("short-bucket-lower-ttft", 32))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.server_status[0].http_port")
                .isEqualTo(IntegrationTestFixtures.workerHttpPort(SCHEDULING_ROLE, 1));
    }

    private Map<String, Object> request(String requestId, int sequenceLength) {
        return Map.of(
                "request_id", requestId,
                "block_cache_keys", new long[]{11L, 22L},
                "block_size", 16,
                "seq_len", sequenceLength,
                "generate_timeout", 500);
    }

    private boolean workerAlive(int workerIndex, boolean expectedAlive) {
        var worker = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getRoleStatusMap(SCHEDULING_ROLE)
                .get(IntegrationTestFixtures.workerIpPort(SCHEDULING_ROLE, workerIndex));
        return worker != null && worker.isAlive() == expectedAlive;
    }
}

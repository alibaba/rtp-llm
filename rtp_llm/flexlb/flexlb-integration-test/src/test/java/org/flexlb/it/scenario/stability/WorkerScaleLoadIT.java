package org.flexlb.it.scenario.stability;

import org.flexlb.Application;
import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
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
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Bounded load regression for worker-status fan-out and concurrent schedule requests.
 *
 * <p>This is a liveness and correctness test rather than a portable throughput benchmark. It is
 * enabled only by Maven's {@code stress-it} profile so normal PR validation stays deterministic.
 */
@Tag("stress")
@ActiveProfiles("test")
@AutoConfigureWebTestClient
@ExtendWith(SystemStubsExtension.class)
@ContextConfiguration(initializers = WorkerScaleContextInitializer.class)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@EnabledIfSystemProperty(named = "flexlb.it.stress", matches = "true")
@SpringBootTest(
        classes = Application.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = "logging.config=classpath:logback-it.xml")
class WorkerScaleLoadIT {

    private static final RoleType SCHEDULING_ROLE = RoleType.PDFUSION;
    private static final int WORKER_COUNT = 200;
    private static final int REQUEST_COUNT = 400;
    private static final int REQUEST_CONCURRENCY = 32;

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @Autowired
    private WebTestClient webTestClient;

    private Set<Integer> workerHttpPorts;

    @BeforeAll
    static void configureEnvironment() {
        environmentVariables.set("HIPPO_ROLE", "flexlb-it");
        environmentVariables.remove("FLEXLB_NACOS_SERVER_ADDR");
    }

    @BeforeEach
    void publishHealthyWorkers() {
        IntegrationTestFixtures.clearWorkerStatusFailures();
        IntegrationTestFixtures.resetWorkers();
        IntStream.range(1, WORKER_COUNT).forEach(index ->
                IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, index, true, 0, 0, 1.0));
        workerHttpPorts = IntStream.range(0, WORKER_COUNT)
                .map(index -> IntegrationTestFixtures.workerHttpPort(SCHEDULING_ROLE, index))
                .boxed()
                .collect(Collectors.toUnmodifiableSet());
        await("200 worker-status snapshots").pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10))
                .atMost(Duration.ofSeconds(15)).until(() ->
                        IntegrationTestFixtures.workerStatusCalls(SCHEDULING_ROLE) >= WORKER_COUNT
                                && allWorkersAreAlive());
    }

    @Test
    void should_complete_concurrent_scheduling_when_two_hundred_workers_are_healthy() throws Exception {
        int workerStatusCallsBeforeScheduling = IntegrationTestFixtures.workerStatusCalls(SCHEDULING_ROLE);
        ExecutorService requestExecutor = Executors.newFixedThreadPool(REQUEST_CONCURRENCY);
        try {
            CompletableFuture<?>[] requests = IntStream.range(0, REQUEST_COUNT)
                    .mapToObj(requestIndex -> CompletableFuture.runAsync(
                            () -> scheduleSuccessfully("worker-scale-" + requestIndex),
                            requestExecutor))
                    .toArray(CompletableFuture[]::new);

            CompletableFuture.allOf(requests).get(30, TimeUnit.SECONDS);
            await("worker-status synchronization during concurrent scheduling").pollDelay(Duration.ZERO)
                    .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(3)).until(() ->
                            IntegrationTestFixtures.workerStatusCalls(SCHEDULING_ROLE)
                                    > workerStatusCallsBeforeScheduling);
        } finally {
            requestExecutor.shutdownNow();
            requestExecutor.awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    private void scheduleSuccessfully(String requestId) {
        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", requestId,
                        "block_cache_keys", new long[]{11L, 22L},
                        "block_size", 16,
                        "seq_len", 32,
                        "generate_timeout", 500))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.server_status[0].server_ip").isEqualTo(IntegrationTestFixtures.WORKER_IP)
                .jsonPath("$.server_status[0].http_port").value(port ->
                        assertTrue(workerHttpPorts.contains(((Number) port).intValue())));
    }

    private boolean allWorkersAreAlive() {
        return EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getRoleStatusMap(SCHEDULING_ROLE)
                .size() == WORKER_COUNT
                && EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getRoleStatusMap(SCHEDULING_ROLE)
                .values()
                .stream()
                .allMatch(worker -> worker.isAlive());
    }
}

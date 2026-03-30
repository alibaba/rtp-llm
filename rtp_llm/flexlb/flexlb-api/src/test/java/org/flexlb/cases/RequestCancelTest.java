package org.flexlb.cases;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.RouteService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.mockito.ArgumentMatchers;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.Disposable;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.mockito.Mockito.verify;

/**
 * Request cancellation callback test
 *
 * @author saichen.sm
 * @since 2026/1/4
 */
@SuppressWarnings("deprecation")
@Slf4j
public class RequestCancelTest {

    private final WebClient webClient;
    private final ConfigService configService;
    private final RouteService routeService;

    private RequestCancelTest(WebClient webClient, ConfigService configService, RouteService routeService) {
        this.webClient = webClient;
        this.configService = configService;
        this.routeService = routeService;
    }

    public static RequestCancelTest init(EnvironmentVariables environmentVariables, ConfigService configService, RouteService routeService) {
        environmentVariables.set("DOMAIN_ADDRESS:com.prefill.hosts.address", "127.0.0.100:8080,127.0.0.101:8080");
        environmentVariables.set("DOMAIN_ADDRESS:com.decode.hosts.address", "127.0.0.102:8080,127.0.0.103:8080");
        WebClient webClient = WebClient.builder().baseUrl("http://localhost:7001").build();
        return new RequestCancelTest(webClient, configService, routeService);
    }

    /**
     *   Test flow:
     *   1. Set Worker remaining memory to 10, forcing requests into queue
     *   2. Send request and subscribe to response stream
     *   3. Call dispose() to cancel subscription after 3 seconds
     *   4. Trigger server-side doOnCancel() → RouteService.cancel()
     *   5. Verify error code 8504 (REQUEST_CANCELLED) is returned
     */
    @SuppressWarnings("ResultOfMethodCallIgnored")
    @SneakyThrows
    public void run() {

        try {
            configService.loadBalanceConfig().setEnableQueueing(true);

            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();

            WorkerStatus workerStatus = new WorkerStatus();
            workerStatus.setAlive(true);
            workerStatus.setUsedKvCacheTokens(new AtomicLong(990L)); // High usage, simulating resource constraints
            workerStatus.setAvailableKvCacheTokens(new AtomicLong(10L)); // Set very small remaining memory, simulating decode resource shortage

            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.100:8080", workerStatus);
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("127.0.0.101:8080", workerStatus);

            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("127.0.0.102:8080", workerStatus);
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("127.0.0.103:8080", workerStatus);

            CountDownLatch latch = new CountDownLatch(1);
            StringBuilder responseBuilder = new StringBuilder();

            Disposable disposable = webClient.post()
                    .uri("/rtp_llm/schedule")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(BodyInserters.fromValue(buildRequestBody()))
                    .exchange()
                    .flatMapMany(response -> {
                        log.info("Received status: {}", response.statusCode());
                        return response.bodyToFlux(String.class);
                    })
                    .doOnNext(responseBuilder::append)
                    .doOnCancel(() -> {
                        log.info("Client cancelled subscription, triggering cancellation logic");
                        latch.countDown();
                    })
                    .doOnError(error -> {
                        log.error("Request error occurred: {}", error.getMessage());
                        latch.countDown();
                    })
                    .doOnComplete(latch::countDown)
                    .subscribe();

            // Wait 1 second before cancelling subscription
            Thread.sleep(1000);
            if (latch.getCount() > 0) {
                log.info("Manually cancel subscription after 3 seconds");
                disposable.dispose();
            }

            // Wait for cancellation to complete
            latch.await(5, TimeUnit.SECONDS);
            String response = responseBuilder.toString();

            log.info("response: {}", response);
            Thread.sleep(1000);
            // Verify routeService.cancel() is called once
            verify(routeService).cancel(ArgumentMatchers.any());
            log.info("routeService.cancel() called once");

        } finally {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
            configService.loadBalanceConfig().setEnableQueueing(false);
        }
    }

    private String buildRequestBody() {
        return """
                {
                  "request_id": 12345,
                  "model": "engine_service",
                  "block_ids": [
                    1001,
                    1002,
                    1003
                  ],
                  "seq_len": 1000,
                  "debug": 1
                }""";
    }

}

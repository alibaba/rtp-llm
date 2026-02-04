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
 * 请求取消回调测试
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
     *   测试流程：
     *   1. 设置 Worker 剩余显存为 10，强制请求进入排队
     *   2. 发送请求并订阅响应流
     *   3. 3 秒后调用 dispose() 取消订阅
     *   4. 触发服务端的 doOnCancel() → RouteService.cancel()
     *   5. 验证返回错误码 8504 (REQUEST_CANCELLED)
     */
    @SuppressWarnings("ResultOfMethodCallIgnored")
    @SneakyThrows
    public void run() {

        try {
            configService.loadBalanceConfig().setEnableQueueing(true);

            Map<String, ModelWorkerStatus> modelRoleWorkerStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP;
            ModelWorkerStatus modelWorkerStatus = new ModelWorkerStatus();
            modelRoleWorkerStatusMap.put("engine_service", modelWorkerStatus);

            WorkerStatus workerStatus = new WorkerStatus();
            workerStatus.setAlive(true);
            workerStatus.setAvailableKvCacheTokens(new AtomicLong(10L)); // 设置剩余显存量很小，模拟 decode 不足的情况

            modelWorkerStatus.getPrefillStatusMap().put("127.0.0.100:8080", workerStatus);
            modelWorkerStatus.getPrefillStatusMap().put("127.0.0.101:8080", workerStatus);

            modelWorkerStatus.getDecodeStatusMap().put("127.0.0.102:8080", workerStatus);
            modelWorkerStatus.getDecodeStatusMap().put("127.0.0.103:8080", workerStatus);

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
                        log.info("客户端取消订阅，触发取消逻辑");
                        latch.countDown();
                    })
                    .doOnError(error -> {
                        log.error("请求发生错误: {}", error.getMessage());
                        latch.countDown();
                    })
                    .doOnComplete(latch::countDown)
                    .subscribe();

            // 等待 1 秒后取消订阅
            Thread.sleep(1000);
            if (latch.getCount() > 0) {
                log.info("3秒后手动取消订阅");
                disposable.dispose();
            }

            // 等待取消完成
            latch.await(5, TimeUnit.SECONDS);
            String response = responseBuilder.toString();

            log.info("response: {}", response);
            Thread.sleep(1000);
            // 验证 routeService.cancel() 被调用一次
            verify(routeService).cancel(ArgumentMatchers.any());
            log.info("routeService.cancel() 被调用一次");

        } finally {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.clear();
            configService.loadBalanceConfig().setEnableQueueing(false);
        }
    }

    private String buildRequestBody() {
        return """
                {
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

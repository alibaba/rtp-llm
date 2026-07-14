package org.flexlb.httpserver;

import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.LogLevel;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.core.Disposable;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class HttpLoadBalanceServerTest {

    @Mock
    private GeneralHttpNettyService generalHttpNettyService;
    @Mock
    private RouteService routeService;
    @Mock
    private LBStatusConsistencyService lbStatusConsistencyService;
    @Mock
    private EngineHealthReporter engineHealthReporter;
    @Mock
    private QueueManager queueManager;
    @Mock
    private WorkerBlockSizeResolver blockSizeResolver;
    @Mock
    private FlexlbLogLevelManager logLevelManager;

    private WebTestClient webTestClient;
    private BlockHashExecutor blockHashExecutor;

    @BeforeEach
    void setUp() {
        blockHashExecutor = new BlockHashExecutor(NoOpFlexMonitor.getInstance(), 1, 1, 60, 1);
        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                generalHttpNettyService,
                routeService,
                lbStatusConsistencyService,
                engineHealthReporter,
                queueManager,
                new ActiveRequestCounter(),
                new ScheduleRequestPreprocessor(blockSizeResolver, blockHashExecutor),
                logLevelManager);
        webTestClient = WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();
    }

    @AfterEach
    void tearDown() {
        blockHashExecutor.shutdown();
    }

    @Test
    void calculatesBlockCacheKeysBeforeRouting() {
        Response response = new Response();
        response.setSuccess(true);
        when(routeService.route(any())).thenReturn(Mono.just(response));
        when(blockSizeResolver.resolve()).thenReturn(4L);

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", 1,
                        "seq_len", 5,
                        "input_ids", new long[]{1, 2, 3, 4, 5}))
                .exchange()
                .expectStatus().isOk();

        ArgumentCaptor<BalanceContext> contextCaptor = ArgumentCaptor.forClass(BalanceContext.class);
        verify(routeService).route(contextCaptor.capture());
        assertEquals(
                List.of(2164874634404590027L),
                contextCaptor.getValue().getRequest().getBlockCacheKeys());
        assertNull(contextCaptor.getValue().getRequest().getInputIds());
    }

    @Test
    void forwardsInputIdsUnchangedWhenRequestHitsSlave() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
        when(lbStatusConsistencyService.getMasterHostIpPort()).thenReturn("10.0.0.1:7001");
        Response response = new Response();
        response.setSuccess(true);
        when(generalHttpNettyService.request(
                any(Request.class),
                any(URI.class),
                eq("/rtp_llm/schedule"),
                eq(Response.class)))
                .thenReturn(Mono.just(response));

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", 1,
                        "seq_len", 4,
                        "input_ids", new long[]{1, 2, 3, 4}))
                .exchange()
                .expectStatus().isOk();

        ArgumentCaptor<Request> requestCaptor = ArgumentCaptor.forClass(Request.class);
        verify(generalHttpNettyService).request(
                requestCaptor.capture(),
                any(URI.class),
                eq("/rtp_llm/schedule"),
                eq(Response.class));
        assertEquals(List.of(1L, 2L, 3L, 4L), requestCaptor.getValue().getInputIds());
        assertNull(requestCaptor.getValue().getBlockCacheKeys());
        verify(routeService, never()).route(any());
        verify(blockSizeResolver, never()).resolve();
    }

    @Test
    void rejectsRequestWhenBlockCacheKeysAndInputIdsAreEmpty() {
        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("request_id", 1, "seq_len", 1))
                .exchange()
                .expectStatus().isBadRequest()
                .expectBody()
                .jsonPath("$.success").isEqualTo(false)
                .jsonPath("$.code").isEqualTo(8406)
                .jsonPath("$.error_message")
                .isEqualTo("block_cache_keys and input_ids must not both be empty");

        verify(routeService, never()).route(any());
    }

    @Test
    void rejectsRequestWhenBlockHashExecutorIsSaturated() throws Exception {
        CountDownLatch runningTaskStarted = new CountDownLatch(1);
        CountDownLatch releaseRunningTask = new CountDownLatch(1);
        Disposable runningTask = blockHashExecutor.submit(() -> {
                    runningTaskStarted.countDown();
                    releaseRunningTask.await(5, TimeUnit.SECONDS);
                    return "running";
                })
                .subscribe();
        assertTrue(runningTaskStarted.await(5, TimeUnit.SECONDS));
        Disposable queuedTask = blockHashExecutor.submit(() -> "queued").subscribe();

        try {
            webTestClient.post()
                    .uri("/rtp_llm/schedule")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(Map.of(
                            "request_id", 1,
                            "seq_len", 4,
                            "block_size", 4,
                            "input_ids", new long[]{1, 2, 3, 4}))
                    .exchange()
                    .expectStatus().isEqualTo(503)
                    .expectBody()
                    .jsonPath("$.success").isEqualTo(false)
                    .jsonPath("$.code").isEqualTo(8502)
                    .jsonPath("$.error_message").isEqualTo("block hash executor queue is full");

            verify(routeService, never()).route(any());
        } finally {
            releaseRunningTask.countDown();
            runningTask.dispose();
            queuedTask.dispose();
        }
    }

    @Test
    void updatesFlexlbLogGroupThroughLegacyEndpoint() {
        when(logLevelManager.setLogLevel(LogLevel.DEBUG)).thenReturn(LogLevel.DEBUG);

        webTestClient.post()
                .uri("/rtp_llm/update_log_level")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("log_level", "debug"))
                .exchange()
                .expectStatus().isOk()
                .expectBody(String.class).isEqualTo("Success! logLevel=DEBUG");

        verify(logLevelManager).setLogLevel(LogLevel.DEBUG);
    }
}

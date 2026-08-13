package org.flexlb.httpserver;

import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.cache.hash.RequestBlockHashService;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RouteService;
import org.flexlb.service.address.FlexlbInstanceAddressService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.optimizer.OptimizerClient;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InOrder;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.reset;
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
    private RequestBlockHashService requestBlockHashService;
    @Mock
    private CacheAwareService cacheAwareService;
    @Mock
    private FlexlbInstanceAddressService instanceAddressService;
    @Mock
    private OptimizerClient optimizerClient;
    @Mock
    private ExecutorService doFinallyExecutor;

    private WebTestClient webTestClient;

    @BeforeEach
    void setUp() {
        org.mockito.Mockito.lenient().doAnswer(invocation -> {
            invocation.getArgument(0, Runnable.class).run();
            return null;
        }).when(doFinallyExecutor).execute(any(Runnable.class));
        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                generalHttpNettyService,
                routeService,
                lbStatusConsistencyService,
                engineHealthReporter,
                queueManager,
                new ActiveRequestCounter(),
                requestBlockHashService,
                cacheAwareService,
                instanceAddressService,
                optimizerClient,
                doFinallyExecutor);
        webTestClient = WebTestClient.bindToRouterFunction(
                server.loadBalancePrefill()).build();
    }

    @Test
    void returnsPodAndInstanceAddressesInMasterInfo() {
        when(lbStatusConsistencyService.getMasterHostIpPort())
                .thenReturn("10.224.145.32:7001");
        when(instanceAddressService.getPodIp()).thenReturn("10.224.145.32");
        when(instanceAddressService.getInstanceIp()).thenReturn("10.101.105.30");
        when(queueManager.getQueue()).thenReturn(new LinkedBlockingDeque<>());

        webTestClient.post()
                .uri("/rtp_llm/master/info")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of())
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.code").isEqualTo(200)
                .jsonPath("$.real_master_host").isEqualTo("10.224.145.32:7001")
                .jsonPath("$.pod_ip").isEqualTo("10.224.145.32")
                .jsonPath("$.instance_ip").isEqualTo("10.101.105.30")
                .jsonPath("$.queue_length").isEqualTo(0);
    }

    @Test
    void preparesBlockCacheKeysBeforeRouting() {
        Response response = new Response();
        response.setSuccess(true);
        when(requestBlockHashService.prepareBlockCacheKeys(any()))
                .thenReturn(Mono.empty());
        when(routeService.route(any())).thenReturn(Mono.just(response));

        String body = "{\"request_id\":\"c68b72ff-982d-944f-9834-bc0e8bf2f43f\","
                + "\"seq_len\":5,\"request_time_ms\":1,\"input_ids\":[1,2,3,4,5]}";
        long requestBodyBytes = body.getBytes(StandardCharsets.UTF_8).length;

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .contentLength(requestBodyBytes)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk();

        ArgumentCaptor<BalanceContext> prepareContextCaptor =
                ArgumentCaptor.forClass(BalanceContext.class);
        ArgumentCaptor<BalanceContext> routeContextCaptor =
                ArgumentCaptor.forClass(BalanceContext.class);
        InOrder inOrder = inOrder(requestBlockHashService, routeService);
        inOrder.verify(requestBlockHashService)
                .prepareBlockCacheKeys(prepareContextCaptor.capture());
        inOrder.verify(routeService).route(routeContextCaptor.capture());
        assertSame(prepareContextCaptor.getValue(), routeContextCaptor.getValue());
        assertEquals(
                "c68b72ff-982d-944f-9834-bc0e8bf2f43f",
                routeContextCaptor.getValue().getRequestId());
        assertArrayEquals(
                new int[]{1, 2, 3, 4, 5},
                routeContextCaptor.getValue().getRequest().getInputIds());
        assertEquals(Long.valueOf(5), routeContextCaptor.getValue().getInputIdsCount());
        assertEquals(Long.valueOf(requestBodyBytes), routeContextCaptor.getValue().getRequestBodyBytes());
        assertTrue(routeContextCaptor.getValue().getRequestArrivalDelayMs() > 0);
        assertTrue(routeContextCaptor.getValue()
                .getRequestBodyReadAndDeserializeTimeUs() >= 0);
        verify(engineHealthReporter).reportRequestPayload(routeContextCaptor.getValue());
    }

    @Test
    void reportsStringRequestIdAndPreparedBlockKeysToOnlineOptimizer() {
        List<Long> blockCacheKeys = List.of(10L, 20L, 30L);
        when(requestBlockHashService.prepareBlockCacheKeys(any()))
                .thenAnswer(invocation -> {
                    BalanceContext ctx = invocation.getArgument(0);
                    ctx.getRequest().setBlockCacheKeys(blockCacheKeys);
                    return Mono.empty();
                });
        Response response = new Response();
        response.setSuccess(true);
        ServerStatus selectedWorker = new ServerStatus();
        selectedWorker.setRole(RoleType.PREFILL);
        selectedWorker.setGroup("default");
        response.setServerStatus(List.of(selectedWorker));
        when(routeService.route(any())).thenReturn(Mono.just(response));
        String requestId = "request-optimizer-1";
        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", requestId,
                        "seq_len", 4,
                        "input_ids", new int[]{1, 2, 3, 4}))
                .exchange()
                .expectStatus().isOk();

        ArgumentCaptor<Request> requestCaptor = ArgumentCaptor.forClass(Request.class);
        verify(optimizerClient).traceQuery(requestCaptor.capture(), eq(selectedWorker));
        assertEquals(requestId, requestCaptor.getValue().getRequestId());
        assertEquals(blockCacheKeys, requestCaptor.getValue().getBlockCacheKeys());
        assertEquals(4L, requestCaptor.getValue().getSeqLen());
    }

    @Test
    void defersFinalizationAndTraceQueryToDoFinallyExecutor() {
        AtomicReference<Runnable> submittedTask = new AtomicReference<>();
        reset(doFinallyExecutor);
        org.mockito.Mockito.doAnswer(invocation -> {
            submittedTask.set(invocation.getArgument(0, Runnable.class));
            return null;
        }).when(doFinallyExecutor).execute(any(Runnable.class));
        when(requestBlockHashService.prepareBlockCacheKeys(any())).thenReturn(Mono.empty());
        Response response = new Response();
        response.setSuccess(true);
        ServerStatus selectedWorker = new ServerStatus();
        selectedWorker.setRole(RoleType.PREFILL);
        selectedWorker.setGroup("default");
        response.setServerStatus(List.of(selectedWorker));
        when(routeService.route(any())).thenAnswer(invocation -> {
            BalanceContext ctx = invocation.getArgument(0);
            ctx.setResponse(response);
            return Mono.just(response);
        });

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "request-async-finally",
                        "seq_len", 4,
                        "input_ids", new int[]{1, 2, 3, 4}))
                .exchange()
                .expectStatus().isOk();

        verify(engineHealthReporter, never()).reportBalancingService(any());
        verify(optimizerClient, never()).traceQuery(any(), any());

        submittedTask.get().run();

        InOrder inOrder = inOrder(engineHealthReporter, optimizerClient);
        inOrder.verify(engineHealthReporter).reportBalancingService(any());
        inOrder.verify(optimizerClient).traceQuery(any(), eq(selectedWorker));
    }

    @Test
    void passesNullWorkerToTraceQueryWhenSuccessfulResponseHasNoServerStatus() {
        when(requestBlockHashService.prepareBlockCacheKeys(any())).thenReturn(Mono.empty());
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of());
        when(routeService.route(any())).thenReturn(Mono.just(response));

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "request-without-server-status",
                        "seq_len", 4,
                        "input_ids", new int[]{1, 2, 3, 4}))
                .exchange()
                .expectStatus().isOk();

        verify(optimizerClient).traceQuery(any(Request.class), isNull());
    }

    @Test
    void stillAttemptsTraceQueryWhenFinalizationFails() {
        when(requestBlockHashService.prepareBlockCacheKeys(any())).thenReturn(Mono.empty());
        Response response = new Response();
        response.setSuccess(true);
        ServerStatus selectedWorker = new ServerStatus();
        selectedWorker.setRole(RoleType.PREFILL);
        response.setServerStatus(List.of(selectedWorker));
        when(routeService.route(any())).thenReturn(Mono.just(response));
        doThrow(new IllegalStateException("metric unavailable"))
                .when(engineHealthReporter).reportRequestPayload(any());

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "request-finalization-failure",
                        "seq_len", 4,
                        "input_ids", new int[]{1, 2, 3, 4}))
                .exchange()
                .expectStatus().isOk();

        verify(optimizerClient).traceQuery(any(Request.class), eq(selectedWorker));
    }

    @Test
    void keepsScheduleSuccessfulWhenTraceQueryThrows() {
        when(requestBlockHashService.prepareBlockCacheKeys(any()))
                .thenReturn(Mono.empty());
        Response response = new Response();
        response.setSuccess(true);
        ServerStatus selectedWorker = new ServerStatus();
        selectedWorker.setRole(RoleType.PREFILL);
        selectedWorker.setGroup("default");
        response.setServerStatus(List.of(selectedWorker));
        when(routeService.route(any())).thenReturn(Mono.just(response));
        org.mockito.Mockito.doThrow(new IllegalStateException("optimizer unavailable"))
                .when(optimizerClient)
                .traceQuery(any(Request.class), any(ServerStatus.class));

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "request-optimizer-error",
                        "seq_len", 4,
                        "input_ids", new int[]{1, 2, 3, 4}))
                .exchange()
                .expectStatus().isOk();

        verify(optimizerClient).traceQuery(
                any(Request.class), eq(selectedWorker));
    }

    @Test
    void updatesRequestCacheMetadataAfterSuccessfulRoutingFinishes() {
        ServerStatus selectedWorker = new ServerStatus();
        selectedWorker.setSuccess(true);
        selectedWorker.setServerIp("10.0.0.1");
        selectedWorker.setHttpPort(8080);
        selectedWorker.setRole(RoleType.PREFILL);

        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(selectedWorker));
        when(requestBlockHashService.prepareBlockCacheKeys(any()))
                .thenReturn(Mono.empty());
        when(routeService.route(any())).thenAnswer(invocation -> {
            BalanceContext ctx = invocation.getArgument(0);
            ctx.setResponse(response);
            return Mono.just(response);
        });

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "request-1",
                        "seq_len", 4,
                        "input_ids", new int[]{1, 2, 3, 4}))
                .exchange()
                .expectStatus().isOk();

        ArgumentCaptor<BalanceContext> contextCaptor =
                ArgumentCaptor.forClass(BalanceContext.class);
        verify(routeService).route(contextCaptor.capture());
        verify(cacheAwareService).updateFromRoutedRequest(
                contextCaptor.getValue().getRequest(),
                response.getServerStatus());
    }

    @Test
    void doesNotUpdateRequestCacheMetadataAfterFailedRouting() {
        Response response = new Response();
        response.setSuccess(false);
        response.setErrorMessage("no worker");
        when(requestBlockHashService.prepareBlockCacheKeys(any()))
                .thenReturn(Mono.empty());
        when(routeService.route(any())).thenAnswer(invocation -> {
            BalanceContext ctx = invocation.getArgument(0);
            ctx.setResponse(response);
            return Mono.just(response);
        });

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "request-1",
                        "seq_len", 4,
                        "input_ids", new int[]{1, 2, 3, 4}))
                .exchange()
                .expectStatus().is5xxServerError();

        verify(cacheAwareService, never()).updateFromRoutedRequest(
                any(Request.class),
                any());
        verify(optimizerClient, never()).traceQuery(any(), any());
    }

    @Test
    void forwardsInputIdsUnchangedWhenRequestHitsSlave() {
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(true);
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);
        when(lbStatusConsistencyService.getMasterHostIpPort()).thenReturn("10.0.0.1:7001");
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(new ServerStatus()));
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
                        "request_id", "request-1",
                        "seq_len", 4,
                        "input_ids", new int[]{1, 2, 3, 4}))
                .exchange()
                .expectStatus().isOk();

        ArgumentCaptor<Request> requestCaptor = ArgumentCaptor.forClass(Request.class);
        verify(generalHttpNettyService).request(
                requestCaptor.capture(),
                any(URI.class),
                eq("/rtp_llm/schedule"),
                eq(Response.class));
        assertArrayEquals(new int[]{1, 2, 3, 4}, requestCaptor.getValue().getInputIds());
        assertNull(requestCaptor.getValue().getBlockCacheKeys());
        verify(routeService, never()).route(any());
        verify(requestBlockHashService, never()).prepareBlockCacheKeys(any());
        verify(cacheAwareService, never()).updateFromRoutedRequest(
                any(Request.class),
                any());
    }

    @Test
    void rejectsRequestWhenBlockCacheKeysAndInputIdsAreEmpty() {
        when(requestBlockHashService.prepareBlockCacheKeys(any()))
                .thenReturn(Mono.error(new IllegalArgumentException(
                        "block_cache_keys and input_ids must not both be empty")));

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("request_id", "request-1", "seq_len", 1))
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
    void rejectsBlankRequestId() {
        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", " ",
                        "seq_len", 1,
                        "input_ids", new int[]{1}))
                .exchange()
                .expectStatus().isBadRequest();

        verify(routeService, never()).route(any());
    }

    @Test
    void preservesPayloadMetadataWhenDecoderRejectsAnOversizedBody() {
        String body = "{\"request_id\":\"" + "x".repeat(300_000) + "\",\"input_ids\":[1]}";
        long requestBodyBytes = body.getBytes(StandardCharsets.UTF_8).length;

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .contentLength(requestBodyBytes)
                .bodyValue(body)
                .exchange()
                .expectStatus().is5xxServerError();

        ArgumentCaptor<BalanceContext> contextCaptor = ArgumentCaptor.forClass(BalanceContext.class);
        verify(engineHealthReporter).reportRequestPayload(contextCaptor.capture());
        BalanceContext context = contextCaptor.getValue();
        assertFalse(context.isSuccess());
        assertNull(context.getRequest());
        assertNull(context.getInputIdsCount());
        assertEquals(Long.valueOf(requestBodyBytes), context.getRequestBodyBytes());
        verify(routeService, never()).route(any());
    }

    @Test
    void rejectsRequestWhenBlockHashExecutorIsSaturated() {
        when(requestBlockHashService.prepareBlockCacheKeys(any()))
                .thenReturn(Mono.error(new RejectedExecutionException("queue full")));

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "request-1",
                        "seq_len", 4,
                        "block_size", 4,
                        "input_ids", new int[]{1, 2, 3, 4}))
                .exchange()
                .expectStatus().isEqualTo(503)
                .expectBody()
                .jsonPath("$.success").isEqualTo(false)
                .jsonPath("$.code").isEqualTo(8502)
                .jsonPath("$.error_message").isEqualTo("block hash executor queue is full");

        verify(routeService, never()).route(any());
    }

}

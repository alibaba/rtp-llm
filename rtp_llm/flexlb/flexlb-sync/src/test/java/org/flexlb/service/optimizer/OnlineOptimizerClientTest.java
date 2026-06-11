package org.flexlb.service.optimizer;

import org.flexlb.dao.optimizer.CommonResponseHeader;
import org.flexlb.dao.optimizer.OptimizerErrorCode;
import org.flexlb.dao.optimizer.OptimizerGetInstanceResponse;
import org.flexlb.dao.optimizer.OptimizerInstanceParams;
import org.flexlb.dao.optimizer.OptimizerRegisterRequest;
import org.flexlb.dao.optimizer.OptimizerRegisterResponse;
import org.flexlb.dao.optimizer.OptimizerRemoveInstanceResponse;
import org.flexlb.dao.optimizer.OptimizerStateInfo;
import org.flexlb.dao.optimizer.OptimizerTraceQueryRequest;
import org.flexlb.dao.optimizer.OptimizerTraceQueryResponse;
import org.flexlb.enums.StatusEnum;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.lang.reflect.Field;
import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.atLeast;
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class OnlineOptimizerClientTest {

    @Mock
    private GeneralHttpNettyService httpService;

    @Mock
    private OptimizerAddressResolver addressResolver;

    private OnlineOptimizerClient client;

    @BeforeEach
    void setUp() {
        client = new OnlineOptimizerClient(httpService, addressResolver, "test-group", "/api/optimizer", 5000);
        // Default resolver to "started". Tests that exercise discovery-start retry override this.
        lenient().when(addressResolver.start()).thenReturn(true);
    }

    @AfterEach
    void tearDown() {
        // Stop async retry thread; shutdown is idempotent.
        if (client != null) {
            client.shutdown();
        }
    }

    /** Poll until registered or timeout, instead of a fixed Thread.sleep. */
    private void awaitRegistered(long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (client.isRegistered()) {
                return;
            }
            Thread.sleep(20);
        }
        fail("client did not become registered within " + timeoutMs + "ms");
    }

    @Test
    void should_skip_traceQuery_when_not_registered() {
        client.traceQuery("123", List.of(1L, 2L, 3L), 64L);

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
    }

    @Test
    void should_skip_traceQuery_when_blockKeys_empty() {
        client.traceQuery("123", List.of(), 64L);

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
    }

    @Test
    void should_skip_traceQuery_when_blockKeys_null() {
        client.traceQuery("123", null, 64L);

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
    }

    @Test
    void should_register_successfully_when_instance_not_exists() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        CommonResponseHeader notFoundHeader = new CommonResponseHeader();
        CommonResponseHeader.Status notFoundStatus = new CommonResponseHeader.Status();
        notFoundStatus.setCode(OptimizerErrorCode.INSTANCE_NOT_EXIST);
        notFoundHeader.setStatus(notFoundStatus);
        getResp.setHeader(notFoundHeader);
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
        registerResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResp));

        OptimizerInstanceParams params = OptimizerInstanceParams.builder()
                .blockSize(64)
                .locationSpecInfos(List.of(new OptimizerRegisterRequest.LocationSpecInfo("full", 131072)))
                .optimizerStateInfo(new OptimizerStateInfo("full-group", "linear-group"))
                .build();
        client.startRegistrationAsync("test-instance", params);

        ArgumentCaptor<OptimizerRegisterRequest> requestCaptor =
                ArgumentCaptor.forClass(OptimizerRegisterRequest.class);
        verify(httpService, timeout(3000)).request(requestCaptor.capture(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        assertEquals("full-group",
                requestCaptor.getValue().getOptimizerStateInfo().getFullLocationSpecGroupName());
        assertEquals("linear-group",
                requestCaptor.getValue().getOptimizerStateInfo().getLinearLocationSpecGroupName());

        awaitRegistered(3000);
    }

    @Test
    void should_verify_existing_params_immediately_when_register_races_with_duplicate() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        OptimizerGetInstanceResponse existing = new OptimizerGetInstanceResponse();
        existing.setHeader(okHeader());
        existing.setInstanceGroup("test-group");
        existing.setInstanceId("test-instance");
        existing.setBlockSize(64);
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getInstanceResponse(OptimizerErrorCode.INSTANCE_NOT_EXIST)))
                .thenReturn(Mono.just(existing));
        OptimizerRegisterResponse duplicate = new OptimizerRegisterResponse();
        duplicate.setHeader(header(OptimizerErrorCode.DUPLICATE_ENTITY));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(duplicate));

        client.startRegistrationAsync(
                "test-instance",
                OptimizerInstanceParams.builder()
                        .instanceGroup("test-group")
                        .blockSize(64)
                        .build());

        verify(httpService, timeout(500).times(2)).request(any(), any(URI.class),
                eq("/api/optimizer/getInstance"), eq(OptimizerGetInstanceResponse.class));
        awaitRegistered(1000);
        verify(httpService, times(1)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
    }

    @Test
    void should_skip_registration_when_instance_exists_with_matching_params() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        getResp.setHeader(okHeader());
        getResp.setInstanceId("test-instance");
        getResp.setBlockSize(64);
        getResp.setLocationSpecInfos(List.of(createRemoteSpecInfo("full", 131072)));
        getResp.setLinearStep(0);
        getResp.setOptimizerStateInfo(new OptimizerStateInfo("full-group", ""));

        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        OptimizerInstanceParams params = OptimizerInstanceParams.builder()
                .blockSize(64)
                .locationSpecInfos(List.of(new OptimizerRegisterRequest.LocationSpecInfo("full", 131072)))
                .optimizerStateInfo(new OptimizerStateInfo("full-group", ""))
                .build();
        client.startRegistrationAsync("test-instance", params);

        awaitRegistered(3000);
        verify(httpService, never()).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
    }

    @Test
    void should_remove_and_reregister_when_params_differ() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        getResp.setHeader(okHeader());
        getResp.setInstanceId("test-instance");
        getResp.setBlockSize(64);
        getResp.setLocationSpecInfos(List.of(createRemoteSpecInfo("full", 131072)));
        getResp.setOptimizerStateInfo(new OptimizerStateInfo("full-group", "old-linear-group"));

        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        OptimizerRemoveInstanceResponse removeResp = new OptimizerRemoveInstanceResponse();
        removeResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/removeInstance"),
                eq(OptimizerRemoveInstanceResponse.class)))
                .thenReturn(Mono.just(removeResp));

        OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
        registerResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResp));

        OptimizerInstanceParams params = OptimizerInstanceParams.builder()
                .blockSize(64)
                .locationSpecInfos(List.of(new OptimizerRegisterRequest.LocationSpecInfo("full", 131072)))
                .optimizerStateInfo(new OptimizerStateInfo("full-group", "linear-group"))
                .build();
        client.startRegistrationAsync("test-instance", params);

        verify(httpService, timeout(3000)).request(any(), any(URI.class),
                eq("/api/optimizer/removeInstance"), eq(OptimizerRemoveInstanceResponse.class));
        verify(httpService, timeout(3000)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));

        awaitRegistered(3000);
    }

    @Test
    void should_retry_when_address_not_resolved() throws Exception {
        when(addressResolver.getAddresses())
                .thenReturn(List.of())
                .thenReturn(List.of())
                .thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        CommonResponseHeader notFoundHeader = new CommonResponseHeader();
        CommonResponseHeader.Status notFoundStatus = new CommonResponseHeader.Status();
        notFoundStatus.setCode(OptimizerErrorCode.INSTANCE_NOT_EXIST);
        notFoundHeader.setStatus(notFoundStatus);
        getResp.setHeader(notFoundHeader);
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
        registerResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResp));

        OptimizerInstanceParams params = OptimizerInstanceParams.builder().blockSize(64).build();
        client.startRegistrationAsync("test-instance", params);

        verify(httpService, timeout(10000)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));

        awaitRegistered(3000);
    }

    @Test
    void should_retry_when_registration_fails() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        CommonResponseHeader notFoundHeader = new CommonResponseHeader();
        CommonResponseHeader.Status notFoundStatus = new CommonResponseHeader.Status();
        notFoundStatus.setCode(OptimizerErrorCode.INSTANCE_NOT_EXIST);
        notFoundHeader.setStatus(notFoundStatus);
        getResp.setHeader(notFoundHeader);
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.error(StatusEnum.INTERNAL_ERROR.toException("connection refused")))
                .thenReturn(Mono.just(getResp));

        OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
        registerResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResp));

        OptimizerInstanceParams params = OptimizerInstanceParams.builder().blockSize(64).build();
        client.startRegistrationAsync("test-instance", params);

        verify(httpService, timeout(10000)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));

        awaitRegistered(3000);
    }

    @Test
    void should_retry_without_registering_when_get_returns_http_error() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.error(new RuntimeException("http error, httpStatusCode=404, body=not found")));

        OptimizerInstanceParams params = OptimizerInstanceParams.builder().blockSize(64).build();
        client.startRegistrationAsync("test-instance", params);

        verify(httpService, timeout(5000).atLeast(2)).request(any(), any(URI.class),
                eq("/api/optimizer/getInstance"), eq(OptimizerGetInstanceResponse.class));
        verify(httpService, never()).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        assertFalse(client.isRegistered());
    }

    @Test
    void should_retry_without_registering_when_get_returns_retryable_business_status() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getInstanceResponse(OptimizerErrorCode.SERVICE_NOT_READY)));

        client.startRegistrationAsync(
                "test-instance", OptimizerInstanceParams.builder().blockSize(64).build());

        verify(httpService, timeout(5000).atLeast(2)).request(any(), any(URI.class),
                eq("/api/optimizer/getInstance"), eq(OptimizerGetInstanceResponse.class));
        verify(httpService, never()).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        assertFalse(client.isRegistered());
    }

    @Test
    void should_fire_traceQuery_when_registered() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        CommonResponseHeader notFoundHeader = new CommonResponseHeader();
        CommonResponseHeader.Status notFoundStatus = new CommonResponseHeader.Status();
        notFoundStatus.setCode(OptimizerErrorCode.INSTANCE_NOT_EXIST);
        notFoundHeader.setStatus(notFoundStatus);
        getResp.setHeader(notFoundHeader);
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
        registerResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResp));

        OptimizerInstanceParams params = OptimizerInstanceParams.builder().blockSize(64).build();
        client.startRegistrationAsync("test-instance", params);
        awaitRegistered(3000);

        OptimizerTraceQueryResponse traceResp = new OptimizerTraceQueryResponse();
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.just(traceResp));

        String requestId = "request-999";
        client.traceQuery(requestId, List.of(10L, 20L, 30L), 128L);

        ArgumentCaptor<OptimizerTraceQueryRequest> requestCaptor =
                ArgumentCaptor.forClass(OptimizerTraceQueryRequest.class);
        verify(httpService, timeout(1000)).request(requestCaptor.capture(), any(URI.class),
                eq("/api/optimizer/traceQuery"), eq(OptimizerTraceQueryResponse.class));
        assertEquals(requestId, requestCaptor.getValue().getTraceId());
        assertEquals("test-instance", requestCaptor.getValue().getInstanceId());
        assertEquals(List.of(10L, 20L, 30L), requestCaptor.getValue().getBlockKeys());
        assertEquals(List.of(), requestCaptor.getValue().getTokenIds());
        assertEquals(128L, requestCaptor.getValue().getInputTokenLen());
    }

    @Test
    void traceQuery_should_not_wait_for_http_response() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getInstanceResponse(OptimizerErrorCode.INSTANCE_NOT_EXIST)));
        OptimizerRegisterResponse registerResponse = new OptimizerRegisterResponse();
        registerResponse.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResponse));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.never());
        client.startRegistrationAsync(
                "test-instance", OptimizerInstanceParams.builder().blockSize(64).build());
        awaitRegistered(3000);

        assertTimeoutPreemptively(
                Duration.ofMillis(250),
                () -> client.traceQuery("trace-async", List.of(10L), 64L));
    }

    @Test
    void should_reregister_when_traceQuery_reports_instance_not_exist() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getInstanceResponse(OptimizerErrorCode.INSTANCE_NOT_EXIST)));
        OptimizerRegisterResponse registerResponse = new OptimizerRegisterResponse();
        registerResponse.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResponse));
        OptimizerTraceQueryResponse traceResponse = new OptimizerTraceQueryResponse();
        traceResponse.setHeader(notFoundHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.just(traceResponse));
        client.startRegistrationAsync(
                "test-instance", OptimizerInstanceParams.builder().blockSize(64).build());
        awaitRegistered(3000);

        client.traceQuery("trace-missing-instance", List.of(10L), 64L);

        verify(httpService, timeout(3000).times(2)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
    }

    @Test
    void stale_trace_response_must_not_invalidate_new_registration() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getInstanceResponse(OptimizerErrorCode.INSTANCE_NOT_EXIST)));
        OptimizerRegisterResponse registerResponse = new OptimizerRegisterResponse();
        registerResponse.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResponse));
        Sinks.One<OptimizerTraceQueryResponse> firstTrace = Sinks.one();
        Sinks.One<OptimizerTraceQueryResponse> secondTrace = Sinks.one();
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(firstTrace.asMono())
                .thenReturn(secondTrace.asMono());

        client.startRegistrationAsync(
                "test-instance", OptimizerInstanceParams.builder().blockSize(64).build());
        awaitRegistered(3000);
        client.traceQuery("old-trace-1", List.of(10L), 64L);
        client.traceQuery("old-trace-2", List.of(20L), 64L);

        OptimizerTraceQueryResponse missing = new OptimizerTraceQueryResponse();
        missing.setHeader(notFoundHeader());
        assertEquals(Sinks.EmitResult.OK, firstTrace.tryEmitValue(missing));
        verify(httpService, timeout(3000).times(2)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        awaitRegistered(3000);

        assertEquals(Sinks.EmitResult.OK, secondTrace.tryEmitValue(missing));
        verify(httpService, after(500).times(2)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
    }

    @Test
    void target_change_should_register_new_target_after_inflight_attempt_finishes() throws Exception {
        AtomicReference<List<String>> addresses =
                new AtomicReference<>(List.of("10.0.0.1:8082"));
        Sinks.One<OptimizerGetInstanceResponse> firstGet = Sinks.one();
        when(addressResolver.getAddresses()).thenAnswer(invocation -> addresses.get());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenAnswer(invocation -> {
                    URI uri = invocation.getArgument(1);
                    if ("10.0.0.1".equals(uri.getHost())) {
                        return firstGet.asMono();
                    }
                    return Mono.just(getInstanceResponse(OptimizerErrorCode.INSTANCE_NOT_EXIST));
                });
        OptimizerRegisterResponse registerResponse = new OptimizerRegisterResponse();
        registerResponse.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResponse));

        client.startRegistrationAsync(
                "test-instance", OptimizerInstanceParams.builder().blockSize(64).build());
        verify(httpService, timeout(500)).request(any(),
                org.mockito.ArgumentMatchers.argThat(
                        uri -> "10.0.0.1".equals(((URI) uri).getHost())),
                eq("/api/optimizer/getInstance"), eq(OptimizerGetInstanceResponse.class));

        addresses.set(List.of("10.0.0.2:8082"));
        client.traceQuery("switch-target", List.of(10L), 64L);

        verify(httpService, after(200).never()).request(any(),
                org.mockito.ArgumentMatchers.argThat(
                        uri -> "10.0.0.2".equals(((URI) uri).getHost())),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));

        assertEquals(Sinks.EmitResult.OK,
                firstGet.tryEmitError(new RuntimeException("old target unavailable")));
        verify(httpService, timeout(1000)).request(any(),
                org.mockito.ArgumentMatchers.argThat(
                        uri -> "10.0.0.2".equals(((URI) uri).getHost())),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        awaitRegistered(1000);
    }

    @Test
    void address_reappearing_after_empty_membership_requires_reregistration() throws Exception {
        AtomicReference<List<String>> addresses =
                new AtomicReference<>(List.of("10.0.0.1:8082"));
        CountDownLatch emptyObservedTwice = new CountDownLatch(2);
        when(addressResolver.getAddresses()).thenAnswer(invocation -> {
            List<String> current = addresses.get();
            if (current.isEmpty()) {
                emptyObservedTwice.countDown();
            }
            return current;
        });
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getInstanceResponse(OptimizerErrorCode.INSTANCE_NOT_EXIST)));
        OptimizerRegisterResponse registerResponse = new OptimizerRegisterResponse();
        registerResponse.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResponse));

        client.startRegistrationAsync(
                "test-instance", OptimizerInstanceParams.builder().blockSize(64).build());
        awaitRegistered(3000);

        addresses.set(List.of());
        client.traceQuery("empty-membership", List.of(10L), 64L);
        if (!emptyObservedTwice.await(1, TimeUnit.SECONDS)) {
            fail("registration retry did not observe empty membership");
        }
        addresses.set(List.of("10.0.0.1:8082"));
        client.traceQuery("address-reappeared", List.of(10L), 64L);

        verify(httpService, timeout(700).times(2)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        awaitRegistered(1000);
    }

    @Test
    void should_keep_registered_address_when_discovery_only_reorders_hosts() throws Exception {
        when(addressResolver.getAddresses())
                .thenReturn(List.of("10.0.0.1:8082"))
                .thenReturn(List.of("10.0.0.2:8082", "10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getInstanceResponse(OptimizerErrorCode.INSTANCE_NOT_EXIST)));
        OptimizerRegisterResponse registerResponse = new OptimizerRegisterResponse();
        registerResponse.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResponse));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.just(new OptimizerTraceQueryResponse()));

        client.startRegistrationAsync(
                "test-instance", OptimizerInstanceParams.builder().blockSize(64).build());
        awaitRegistered(3000);
        client.traceQuery("trace-1", List.of(10L), 64L);

        ArgumentCaptor<URI> uriCaptor = ArgumentCaptor.forClass(URI.class);
        verify(httpService, timeout(1000)).request(any(), uriCaptor.capture(),
                eq("/api/optimizer/traceQuery"), eq(OptimizerTraceQueryResponse.class));
        assertEquals("10.0.0.1", uriCaptor.getValue().getHost());
    }

    @Test
    void should_reregister_before_traceQuery_when_registered_address_disappears() throws Exception {
        when(addressResolver.getAddresses())
                .thenReturn(List.of("10.0.0.1:8082"))
                .thenReturn(List.of("10.0.0.2:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getInstanceResponse(OptimizerErrorCode.INSTANCE_NOT_EXIST)));
        OptimizerRegisterResponse registerResponse = new OptimizerRegisterResponse();
        registerResponse.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResponse));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.just(new OptimizerTraceQueryResponse()));

        client.startRegistrationAsync(
                "test-instance", OptimizerInstanceParams.builder().blockSize(64).build());
        awaitRegistered(3000);

        client.traceQuery("trace-before-reregister", List.of(10L), 64L);
        verify(httpService, timeout(3000).times(2)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        awaitRegistered(3000);
        client.traceQuery("trace-after-reregister", List.of(10L), 64L);

        ArgumentCaptor<URI> uriCaptor = ArgumentCaptor.forClass(URI.class);
        verify(httpService, timeout(1000).times(1)).request(any(), uriCaptor.capture(),
                eq("/api/optimizer/traceQuery"), eq(OptimizerTraceQueryResponse.class));
        assertEquals("10.0.0.2", uriCaptor.getValue().getHost());
    }

    private static CommonResponseHeader okHeader() {
        return header(OptimizerErrorCode.OK);
    }

    private static CommonResponseHeader header(OptimizerErrorCode code) {
        CommonResponseHeader header = new CommonResponseHeader();
        CommonResponseHeader.Status status = new CommonResponseHeader.Status();
        status.setCode(code);
        header.setStatus(status);
        return header;
    }

    private static OptimizerGetInstanceResponse getInstanceResponse(OptimizerErrorCode code) {
        OptimizerGetInstanceResponse response = new OptimizerGetInstanceResponse();
        CommonResponseHeader header = new CommonResponseHeader();
        CommonResponseHeader.Status status = new CommonResponseHeader.Status();
        status.setCode(code);
        header.setStatus(status);
        response.setHeader(header);
        return response;
    }

    private static OptimizerGetInstanceResponse.LocationSpecInfo createRemoteSpecInfo(String name, long size) {
        OptimizerGetInstanceResponse.LocationSpecInfo info = new OptimizerGetInstanceResponse.LocationSpecInfo();
        info.setName(name);
        info.setSize(size);
        return info;
    }

    private static OptimizerGetInstanceResponse.LocationSpecGroup createRemoteSpecGroup(String name, List<String> specNames) {
        OptimizerGetInstanceResponse.LocationSpecGroup group = new OptimizerGetInstanceResponse.LocationSpecGroup();
        group.setName(name);
        group.setSpecNames(specNames);
        return group;
    }

    @Test
    void should_skip_duplicate_startRegistrationAsync() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        CommonResponseHeader notFoundHeader = new CommonResponseHeader();
        CommonResponseHeader.Status notFoundStatus = new CommonResponseHeader.Status();
        notFoundStatus.setCode(OptimizerErrorCode.INSTANCE_NOT_EXIST);
        notFoundHeader.setStatus(notFoundStatus);
        getResp.setHeader(notFoundHeader);
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
        registerResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResp));

        OptimizerInstanceParams params = OptimizerInstanceParams.builder().blockSize(64).build();
        client.startRegistrationAsync("test-instance", params);
        verify(httpService, timeout(3000)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        awaitRegistered(3000);

        // Duplicate call must be blocked by AtomicBoolean started; no extra register request.
        client.startRegistrationAsync("test-instance", params);
        // startRegistrationAsync rejects synchronously via AtomicBoolean; no async delay needed.
        verify(httpService, times(1)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
    }

    @Test
    void should_normalize_basePath_with_trailing_slash() {
        OnlineOptimizerClient c = new OnlineOptimizerClient(
                httpService, addressResolver, "g", "/api/optimizer///", 5000);
        try {
            when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

            OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
            CommonResponseHeader notFoundHeader = new CommonResponseHeader();
            CommonResponseHeader.Status notFoundStatus = new CommonResponseHeader.Status();
            notFoundStatus.setCode(OptimizerErrorCode.INSTANCE_NOT_EXIST);
            notFoundHeader.setStatus(notFoundStatus);
            getResp.setHeader(notFoundHeader);
            when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                    eq(OptimizerGetInstanceResponse.class)))
                    .thenReturn(Mono.just(getResp));

            OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
            registerResp.setHeader(okHeader());
            when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                    eq(OptimizerRegisterResponse.class)))
                    .thenReturn(Mono.just(registerResp));

            c.startRegistrationAsync("id", OptimizerInstanceParams.builder().blockSize(64).build());

            // Resolved path should be /api/optimizer/{getInstance,registerInstance}, not doubled slashes.
            verify(httpService, timeout(3000)).request(any(), any(URI.class),
                    eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        } finally {
            c.shutdown();
        }
    }

    @Test
    void should_swallow_exception_in_traceQuery() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        CommonResponseHeader notFoundHeader = new CommonResponseHeader();
        CommonResponseHeader.Status notFoundStatus = new CommonResponseHeader.Status();
        notFoundStatus.setCode(OptimizerErrorCode.INSTANCE_NOT_EXIST);
        notFoundHeader.setStatus(notFoundStatus);
        getResp.setHeader(notFoundHeader);
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
        registerResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResp));

        client.startRegistrationAsync("test-instance",
                OptimizerInstanceParams.builder().blockSize(64).build());
        awaitRegistered(3000);

        // httpService.request throws synchronously (not Mono.error); traceQuery must swallow it.
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenThrow(new RuntimeException("boom"));

        client.traceQuery("999", List.of(10L, 20L), 64L);
        // No exception thrown means the try/catch Throwable guard works.
    }

    @Test
    void should_shutdown_invoke_addressResolver() {
        client.shutdown();
        verify(addressResolver).shutdown();
        // Repeated shutdown must not throw.
        client.shutdown();
    }

    @Test
    void should_not_become_registered_when_inflight_registration_completes_during_shutdown() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        getResp.setHeader(notFoundHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        CountDownLatch registrationStarted = new CountDownLatch(1);
        CountDownLatch allowRegistrationToFinish = new CountDownLatch(1);
        OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
        registerResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.fromCallable(() -> {
                    registrationStarted.countDown();
                    allowRegistrationToFinish.await();
                    return registerResp;
                }));

        client.startRegistrationAsync(
                "test-instance", OptimizerInstanceParams.builder().blockSize(64).build());
        if (!registrationStarted.await(3, TimeUnit.SECONDS)) {
            fail("registration did not start");
        }

        CompletableFuture<Void> shutdown = CompletableFuture.runAsync(client::shutdown);
        ScheduledExecutorService retryScheduler = retryScheduler(client);
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!retryScheduler.isShutdown() && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
        if (!retryScheduler.isShutdown()) {
            fail("shutdown did not close the retry scheduler");
        }

        allowRegistrationToFinish.countDown();
        shutdown.get(3, TimeUnit.SECONDS);

        assertFalse(client.isRegistered());
    }

    @Test
    void should_not_register_after_shutdown() throws Exception {
        client.shutdown();

        OptimizerInstanceParams params = OptimizerInstanceParams.builder().blockSize(64).build();
        // RejectedExecutionException should be swallowed by safeSubmit and not propagate.
        client.startRegistrationAsync("test-instance", params);

        // RejectedExecutionException is swallowed synchronously by safeSubmit; no async delay.
        assertFalse(client.isRegistered());
    }

    @Test
    void should_match_params_regardless_of_order() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        getResp.setHeader(okHeader());
        getResp.setInstanceId("test-instance");
        getResp.setBlockSize(64);
        // Remote returns specs in REVERSED order compared to local
        getResp.setLocationSpecInfos(List.of(
                createRemoteSpecInfo("linear", 65536),
                createRemoteSpecInfo("full", 131072)));
        getResp.setLocationSpecGroups(List.of(createRemoteSpecGroup("group_b", List.of("linear")),
                createRemoteSpecGroup("group_a", List.of("full"))));
        getResp.setLinearStep(0);
        getResp.setOptimizerStateInfo(new OptimizerStateInfo("group_a", ""));

        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        // Local params have specs/groups in different order than remote
        OptimizerInstanceParams params = OptimizerInstanceParams.builder()
                .blockSize(64)
                .locationSpecInfos(List.of(
                        new OptimizerRegisterRequest.LocationSpecInfo("full", 131072),
                        new OptimizerRegisterRequest.LocationSpecInfo("linear", 65536)))
                .locationSpecGroups(List.of(
                        new OptimizerRegisterRequest.LocationSpecGroup("group_a", List.of("full")),
                        new OptimizerRegisterRequest.LocationSpecGroup("group_b", List.of("linear"))))
                .optimizerStateInfo(new OptimizerStateInfo("group_a", ""))
                .build();
        client.startRegistrationAsync("test-instance", params);

        awaitRegistered(3000);
        // Should NOT re-register since params match (order-independent)
        verify(httpService, never()).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
    }

    // ===== malformed response =====

    @Test
    void should_treat_missing_header_as_register_failure() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        getResp.setHeader(notFoundHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        // Malformed: response object exists but header is null
        OptimizerRegisterResponse malformed = new OptimizerRegisterResponse();
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(malformed));

        client.startRegistrationAsync("test-instance",
                OptimizerInstanceParams.builder().blockSize(64).build());

        verify(httpService, timeout(3000).atLeastOnce()).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));

        // Strict isOkHeader: missing header must NOT be treated as success
        assertFalse(client.isRegistered());
    }

    @Test
    void should_treat_missing_status_as_register_failure() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        getResp.setHeader(notFoundHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        // Malformed: header present but status is null
        OptimizerRegisterResponse malformed = new OptimizerRegisterResponse();
        malformed.setHeader(new CommonResponseHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(malformed));

        client.startRegistrationAsync("test-instance",
                OptimizerInstanceParams.builder().blockSize(64).build());

        verify(httpService, timeout(3000).atLeastOnce()).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));

        assertFalse(client.isRegistered());
    }

    // ===== empty addresses clears cached URI =====

    @Test
    void should_clear_cached_uri_when_addresses_become_empty() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        getResp.setHeader(notFoundHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
        registerResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResp));

        client.startRegistrationAsync("test-instance",
                OptimizerInstanceParams.builder().blockSize(64).build());
        // Wait for async registration to actually complete instead of a fixed sleep.
        verify(httpService, timeout(3000)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        awaitRegistered(3000);

        // Now resolver reports zero hosts (e.g. all instances down).
        when(addressResolver.getAddresses()).thenReturn(List.of());

        client.traceQuery("999", List.of(10L, 20L), 64L);

        verify(httpService, never()).request(any(), any(URI.class),
                eq("/api/optimizer/traceQuery"), eq(OptimizerTraceQueryResponse.class));
    }

    // ===== resolver start-failure async retry =====

    @Test
    void should_retry_when_resolver_start_fails_then_recovers() throws Exception {
        // Override default lenient stub: first two start() calls fail,
        // then recover. The retry chain inside attemptRegistration must keep calling start().
        when(addressResolver.start())
                .thenReturn(false)
                .thenReturn(false)
                .thenReturn(true);
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        getResp.setHeader(notFoundHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        OptimizerRegisterResponse registerResp = new OptimizerRegisterResponse();
        registerResp.setHeader(okHeader());
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class)))
                .thenReturn(Mono.just(registerResp));

        client.startRegistrationAsync("test-instance",
                OptimizerInstanceParams.builder().blockSize(64).build());

        // Backoff: 1s + jitter, then 2s + jitter — give it up to 15s
        verify(httpService, timeout(15000)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        awaitRegistered(3000);

        // start() must be called at least 3 times: first two return false to trigger retry,
        // third one finally succeeds.
        verify(addressResolver, atLeast(3)).start();
    }

    @Test
    void should_not_call_httpService_when_resolver_start_keeps_failing() throws Exception {
        // Override default: start() always returns false.
        when(addressResolver.start()).thenReturn(false);

        client.startRegistrationAsync("test-instance",
                OptimizerInstanceParams.builder().blockSize(64).build());

        // Wait until the retry chain ran at least twice (initial attempt + first retry)
        // to avoid jitter-boundary flakiness from a fixed sleep.
        verify(addressResolver, timeout(5000).atLeast(2)).start();

        // No HTTP traffic should have been issued because resolver never started.
        verify(httpService, never()).request(any(), any(URI.class), any(), any());
        assertFalse(client.isRegistered());
    }

    private static CommonResponseHeader notFoundHeader() {
        return header(OptimizerErrorCode.INSTANCE_NOT_EXIST);
    }

    private static ScheduledExecutorService retryScheduler(OnlineOptimizerClient client) throws Exception {
        Field field = OnlineOptimizerClient.class.getDeclaredField("retryScheduler");
        field.setAccessible(true);
        return (ScheduledExecutorService) field.get(client);
    }
}

package org.flexlb.service.optimizer;

import org.flexlb.dao.optimizer.CommonResponseHeader;
import org.flexlb.dao.optimizer.OptimizerGetInstanceResponse;
import org.flexlb.dao.optimizer.OptimizerInstanceParams;
import org.flexlb.dao.optimizer.OptimizerRegisterRequest;
import org.flexlb.dao.optimizer.OptimizerRegisterResponse;
import org.flexlb.dao.optimizer.OptimizerRemoveInstanceResponse;
import org.flexlb.dao.optimizer.OptimizerTraceQueryResponse;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.atLeast;
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
        // Default resolver to "started". Tests that exercise listen-fail retry override this.
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
        client.traceQuery(123L, List.of(1L, 2L, 3L));

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
    }

    @Test
    void should_skip_traceQuery_when_blockKeys_empty() {
        client.traceQuery(123L, List.of());

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
    }

    @Test
    void should_skip_traceQuery_when_blockKeys_null() {
        client.traceQuery(123L, null);

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
    }

    @Test
    void should_register_successfully_when_instance_not_exists() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        CommonResponseHeader notFoundHeader = new CommonResponseHeader();
        CommonResponseHeader.Status notFoundStatus = new CommonResponseHeader.Status();
        notFoundStatus.setCode(0);
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
                .build();
        client.startRegistrationAsync("test-instance", params);

        verify(httpService, timeout(3000)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));

        Thread.sleep(100);
        assertTrue(client.isRegistered());
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
        getResp.setFullGroupName("");

        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.just(getResp));

        OptimizerInstanceParams params = OptimizerInstanceParams.builder()
                .blockSize(64)
                .locationSpecInfos(List.of(new OptimizerRegisterRequest.LocationSpecInfo("full", 131072)))
                .build();
        client.startRegistrationAsync("test-instance", params);

        Thread.sleep(500);
        assertTrue(client.isRegistered());
        verify(httpService, never()).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
    }

    @Test
    void should_remove_and_reregister_when_params_differ() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        getResp.setHeader(okHeader());
        getResp.setInstanceId("test-instance");
        getResp.setBlockSize(32);
        getResp.setLocationSpecInfos(List.of(createRemoteSpecInfo("full", 65536)));

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
                .build();
        client.startRegistrationAsync("test-instance", params);

        verify(httpService, timeout(3000)).request(any(), any(URI.class),
                eq("/api/optimizer/removeInstance"), eq(OptimizerRemoveInstanceResponse.class));
        verify(httpService, timeout(3000)).request(any(), any(URI.class),
                eq("/api/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));

        Thread.sleep(100);
        assertTrue(client.isRegistered());
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
        notFoundStatus.setCode(0);
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

        Thread.sleep(100);
        assertTrue(client.isRegistered());
    }

    @Test
    void should_retry_when_registration_fails() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        CommonResponseHeader notFoundHeader = new CommonResponseHeader();
        CommonResponseHeader.Status notFoundStatus = new CommonResponseHeader.Status();
        notFoundStatus.setCode(0);
        notFoundHeader.setStatus(notFoundStatus);
        getResp.setHeader(notFoundHeader);
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class)))
                .thenReturn(Mono.error(new RuntimeException("connection refused")))
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

        Thread.sleep(100);
        assertTrue(client.isRegistered());
    }

    @Test
    void should_fire_traceQuery_when_registered() throws Exception {
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));

        OptimizerGetInstanceResponse getResp = new OptimizerGetInstanceResponse();
        CommonResponseHeader notFoundHeader = new CommonResponseHeader();
        CommonResponseHeader.Status notFoundStatus = new CommonResponseHeader.Status();
        notFoundStatus.setCode(0);
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
        Thread.sleep(500);
        assertTrue(client.isRegistered());

        OptimizerTraceQueryResponse traceResp = new OptimizerTraceQueryResponse();
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.just(traceResp));

        client.traceQuery(999L, List.of(10L, 20L, 30L));

        verify(httpService, timeout(1000)).request(any(), any(URI.class),
                eq("/api/optimizer/traceQuery"), eq(OptimizerTraceQueryResponse.class));
    }

    private static CommonResponseHeader okHeader() {
        CommonResponseHeader header = new CommonResponseHeader();
        CommonResponseHeader.Status status = new CommonResponseHeader.Status();
        status.setCode(1);
        header.setStatus(status);
        return header;
    }

    private static OptimizerGetInstanceResponse.LocationSpecInfo createRemoteSpecInfo(String name, int size) {
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
        notFoundStatus.setCode(0);
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
        Thread.sleep(100);

        // Duplicate call must be blocked by AtomicBoolean started; no extra register request.
        client.startRegistrationAsync("test-instance", params);
        Thread.sleep(200);
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
            notFoundStatus.setCode(0);
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
        notFoundStatus.setCode(0);
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
        Thread.sleep(500);
        assertTrue(client.isRegistered());

        // httpService.request throws synchronously (not Mono.error); traceQuery must swallow it.
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenThrow(new RuntimeException("boom"));

        client.traceQuery(999L, List.of(10L, 20L));
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
    void should_not_register_after_shutdown() throws Exception {
        client.shutdown();

        OptimizerInstanceParams params = OptimizerInstanceParams.builder().blockSize(64).build();
        // RejectedExecutionException should be swallowed by safeSubmit and not propagate.
        client.startRegistrationAsync("test-instance", params);
        Thread.sleep(200);
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
        getResp.setFullGroupName("");

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
                .build();
        client.startRegistrationAsync("test-instance", params);

        Thread.sleep(500);
        assertTrue(client.isRegistered());
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
        Thread.sleep(200);

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
        Thread.sleep(200);

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

        client.traceQuery(999L, List.of(10L, 20L));
        Thread.sleep(100);

        // refreshUri must clear optimizerUri so traceQuery short-circuits and never
        // hits the dead address.
        verify(httpService, never()).request(any(), any(URI.class),
                eq("/api/optimizer/traceQuery"), eq(OptimizerTraceQueryResponse.class));
    }

    // ===== resolver listen-fail async retry =====

    @Test
    void should_retry_when_resolver_start_fails_then_recovers() throws Exception {
        // Override default lenient stub: first two start() calls fail (listen broken),
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
        // Override default: start() always returns false (listen permanently broken).
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
        CommonResponseHeader header = new CommonResponseHeader();
        CommonResponseHeader.Status status = new CommonResponseHeader.Status();
        status.setCode(0);
        header.setStatus(status);
        return header;
    }
}

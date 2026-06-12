package org.flexlb.service.optimizer;

import org.flexlb.dao.optimizer.CommonResponseHeader;
import org.flexlb.dao.optimizer.OptimizerGetInstanceResponse;
import org.flexlb.dao.optimizer.OptimizerInstanceParams;
import org.flexlb.dao.optimizer.OptimizerRegisterRequest;
import org.flexlb.dao.optimizer.OptimizerRegisterResponse;
import org.flexlb.dao.optimizer.OptimizerRemoveInstanceResponse;
import org.flexlb.dao.optimizer.OptimizerTraceQueryResponse;
import org.flexlb.transport.GeneralHttpNettyService;
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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
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

        // 重复调用应被 AtomicBoolean started 拦截，不应产生额外 register 请求
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

            // 拼出的路径应为 /api/optimizer/getInstance 与 /api/optimizer/registerInstance，而非双斜杠
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

        // httpService.request 同步抛出（非 Mono.error），traceQuery 外层需咽住不要外传
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenThrow(new RuntimeException("boom"));

        client.traceQuery(999L, List.of(10L, 20L));
        // 未抛异常即表示 try/catch Throwable 生效
    }

    @Test
    void should_shutdown_invoke_addressResolver() {
        client.shutdown();
        verify(addressResolver).shutdown();
        // 重复 shutdown 不应抛异常
        client.shutdown();
    }

    @Test
    void should_not_register_after_shutdown() throws Exception {
        client.shutdown();

        OptimizerInstanceParams params = OptimizerInstanceParams.builder().blockSize(64).build();
        // RejectedExecutionException 应被 safeSubmit 咽住，不会传出
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
}

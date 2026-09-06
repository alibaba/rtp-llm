package org.flexlb.service.optimizer;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.optimizer.CommonResponseHeader;
import org.flexlb.dao.optimizer.OptimizerErrorCode;
import org.flexlb.dao.optimizer.OptimizerTraceQueryRequest;
import org.flexlb.dao.optimizer.OptimizerTraceQueryResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.engine.grpc.client.KvcmWorkerMetadataResolver;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;
import java.util.List;

import static org.flexlb.constant.MetricConstant.OPTIMIZER_TRACE_QUERY_FAILED_QPS;
import static org.flexlb.constant.MetricConstant.OPTIMIZER_TRACE_QUERY_SKIPPED_QPS;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class OptimizerClientTest {

    @Mock
    private GeneralHttpNettyService httpService;

    @Mock
    private OptimizerAddressResolver addressResolver;

    @Mock
    private FlexMonitor monitor;
    @Mock
    private KvcmWorkerMetadataResolver workerMetadataResolver;

    private OptimizerClient client;

    @BeforeEach
    void setUp() {
        client = new OptimizerClient(
                httpService, addressResolver, workerMetadataResolver, "/api/optimizer", monitor);
    }

    @AfterEach
    void tearDown() {
        client.shutdown();
    }

    @Test
    void dispatchesTraceQueryUsingResolvedKvcmNamespace() {
        resolvesTestInstanceId();
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.just(traceResponse(OptimizerErrorCode.OK)));

        client.traceQuery(traceRequest(List.of(1L, 2L, 3L)), selectedWorker());

        ArgumentCaptor<OptimizerTraceQueryRequest> requestCaptor =
                ArgumentCaptor.forClass(OptimizerTraceQueryRequest.class);
        verify(httpService).request(requestCaptor.capture(), any(URI.class),
                eq("/api/optimizer/traceQuery"), eq(OptimizerTraceQueryResponse.class));
        assertEquals("1", requestCaptor.getValue().getTraceId());
        assertEquals("test-instance", requestCaptor.getValue().getInstanceId());
        assertEquals(List.of(1L, 2L, 3L), requestCaptor.getValue().getBlockKeys());
        assertEquals(64L, requestCaptor.getValue().getInputTokenLen());
        verify(workerMetadataResolver).resolveNamespace(RoleType.PREFILL, "default", 16L);
    }

    @Test
    void startsAddressResolverDuringInitialization() {
        client.init();

        verify(addressResolver).start();
    }

    @Test
    void skipsTraceQueryWhenBlockKeysAreEmpty() {
        client.traceQuery(traceRequest(List.of()), selectedWorker());

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
        verify(monitor).report(
                OPTIMIZER_TRACE_QUERY_SKIPPED_QPS,
                FlexMetricTags.of("reason", "empty_block_keys"),
                1.0);
    }

    @Test
    void skipsTraceQueryWhenNoAddressIsAvailable() {
        resolvesTestInstanceId();
        when(addressResolver.getAddresses()).thenReturn(List.of());

        client.traceQuery(traceRequest(List.of(1L)), selectedWorker());

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
        verify(monitor).report(
                OPTIMIZER_TRACE_QUERY_SKIPPED_QPS,
                FlexMetricTags.of("reason", "no_available_address"),
                1.0);
    }

    @Test
    void skipsTraceQueryWhenOptimizerIsDisabled() {
        OptimizerClient disabledClient = new OptimizerClient(
                httpService, mock(ServiceDiscovery.class), new ModelMetaConfig(), workerMetadataResolver, monitor);
        disabledClient.shutdown();

        disabledClient.traceQuery(traceRequest(List.of(1L)), selectedWorker());

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
        verify(monitor, never()).report(eq(OPTIMIZER_TRACE_QUERY_SKIPPED_QPS), any(), eq(1.0));
    }

    @Test
    void skipsTraceQueryAfterShutdown() {
        client.shutdown();

        client.traceQuery(traceRequest(List.of(1L)), selectedWorker());

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
        verify(workerMetadataResolver, never()).resolveNamespace(any(), anyString(), anyLong());
        verify(monitor).report(
                OPTIMIZER_TRACE_QUERY_SKIPPED_QPS,
                FlexMetricTags.of("reason", "shutdown"),
                1.0);
    }

    @Test
    void reportsFailureWhenTraceQueryRequestFails() {
        resolvesTestInstanceId();
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.error(new IllegalStateException("optimizer unavailable")));

        assertDoesNotThrow(() -> client.traceQuery(traceRequest(List.of(1L)), selectedWorker()));

        verify(monitor).report(
                OPTIMIZER_TRACE_QUERY_FAILED_QPS,
                FlexMetricTags.of("reason", "http_error"),
                1.0);
    }

    @Test
    void reportsDispatchFailureWhenHttpRequestThrowsSynchronously() {
        resolvesTestInstanceId();
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenThrow(new IllegalStateException("request construction failed"));

        assertDoesNotThrow(() -> client.traceQuery(traceRequest(List.of(1L)), selectedWorker()));

        verify(monitor).report(
                OPTIMIZER_TRACE_QUERY_FAILED_QPS,
                FlexMetricTags.of("reason", "dispatch_error"),
                1.0);
    }

    @Test
    void doesNotWaitForTraceQueryResponse() {
        resolvesTestInstanceId();
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.never());

        assertTimeoutPreemptively(
                Duration.ofSeconds(1),
                () -> client.traceQuery(traceRequest(List.of(1L)), selectedWorker()));

        verify(httpService).request(any(), any(URI.class),
                eq("/api/optimizer/traceQuery"), eq(OptimizerTraceQueryResponse.class));
    }

    @Test
    void reportsFailureForNonOkTraceQueryResponse() {
        resolvesTestInstanceId();
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.just(traceResponse(OptimizerErrorCode.UNKNOWN_ERROR)));

        client.traceQuery(traceRequest(List.of(1L)), selectedWorker());

        verify(monitor).report(
                OPTIMIZER_TRACE_QUERY_FAILED_QPS,
                FlexMetricTags.of("reason", "status_UNKNOWN_ERROR"),
                1.0);
    }

    @Test
    void reportsMissingStatusWhenResponseHeaderIsMissing() {
        resolvesTestInstanceId();
        when(addressResolver.getAddresses()).thenReturn(List.of("10.0.0.1:8082"));
        when(httpService.request(any(), any(URI.class), eq("/api/optimizer/traceQuery"),
                eq(OptimizerTraceQueryResponse.class)))
                .thenReturn(Mono.just(new OptimizerTraceQueryResponse()));

        client.traceQuery(traceRequest(List.of(1L)), selectedWorker());

        verify(monitor).report(
                OPTIMIZER_TRACE_QUERY_FAILED_QPS,
                FlexMetricTags.of("reason", "status_MISSING"),
                1.0);
    }

    @Test
    void reportsMissingStatusWhenResponseIsNull() {
        client.handleTraceQueryResponse(null);

        verify(monitor).report(
                OPTIMIZER_TRACE_QUERY_FAILED_QPS,
                FlexMetricTags.of("reason", "status_MISSING"),
                1.0);
    }

    @Test
    void shutdownStopsAddressResolver() {
        client.shutdown();

        verify(addressResolver).shutdown();
    }

    @Test
    void skipsTraceQueryWhenKvcmNamespaceIsUnavailable() {
        when(workerMetadataResolver.resolveNamespace(RoleType.PREFILL, "default", 16L)).thenReturn(null);

        client.traceQuery(traceRequest(List.of(1L)), selectedWorker());

        verify(httpService, never()).request(any(), any(URI.class), any(), any());
        verify(monitor).report(
                OPTIMIZER_TRACE_QUERY_SKIPPED_QPS,
                FlexMetricTags.of("reason", "instance_id_unavailable"),
                1.0);
    }

    private static Request traceRequest(List<Long> blockCacheKeys) {
        Request request = new Request();
        request.setRequestId(1L);
        request.setBlockCacheKeys(blockCacheKeys);
        request.setBlockSize(16L);
        request.setSeqLen(64L);
        return request;
    }

    private void resolvesTestInstanceId() {
        when(workerMetadataResolver.resolveNamespace(any(), any(), anyLong())).thenReturn("test-instance");
    }

    private static ServerStatus selectedWorker() {
        ServerStatus worker = new ServerStatus();
        worker.setRole(RoleType.PREFILL);
        worker.setGroup("default");
        return worker;
    }

    private static OptimizerTraceQueryResponse traceResponse(OptimizerErrorCode code) {
        CommonResponseHeader.Status status = new CommonResponseHeader.Status();
        status.setCode(code);
        CommonResponseHeader header = new CommonResponseHeader();
        header.setStatus(status);
        OptimizerTraceQueryResponse response = new OptimizerTraceQueryResponse();
        response.setHeader(header);
        return response;
    }
}

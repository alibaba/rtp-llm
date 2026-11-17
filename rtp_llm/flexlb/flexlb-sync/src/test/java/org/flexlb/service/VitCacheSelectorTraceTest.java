package org.flexlb.service;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Context;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.trace.ReadWriteSpan;
import io.opentelemetry.sdk.trace.ReadableSpan;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.SpanProcessor;
import io.opentelemetry.sdk.trace.data.SpanData;
import org.flexlb.balance.endpoint.EndpointRegistry.VitRoutingEntry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.telemetry.FlexlbTrace;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyList;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies routing span parentage, decision diagnostics and fail-open telemetry.
 */
class VitCacheSelectorTraceTest {
    private final List<SpanData> spans = new ArrayList<>();
    private SdkTracerProvider provider;
    private VitCacheSelector selector;
    private WorkerDirectory workers;
    private VitCacheDirectory directory;
    private WorkerStatus worker;
    private BalanceContext ctx;
    private Span parent;

    @BeforeEach
    void setUp() {
        provider = SdkTracerProvider.builder().addSpanProcessor(new SpanProcessor() {
            public void onStart(Context context, ReadWriteSpan span) { }
            public boolean isStartRequired() { return false; }
            public void onEnd(ReadableSpan span) { spans.add(span.toSpanData()); }
            public boolean isEndRequired() { return true; }
        }).build();
        FlexlbTrace.configure(OpenTelemetrySdk.builder().setTracerProvider(provider).build(), "test");
        parent = FlexlbTrace.startServer("schedule", Context.root());
        ctx = new BalanceContext(new FlexlbConfig());
        ctx.getConfig().getRouter().getRoles().getVit().setCacheAffinityEnabled(false);
        Request request = new Request();
        request.setRequestId(42);
        request.setVitRouteOnly(true);
        request.setCacheAffinityKeys(List.of("private-url", "private-key"));
        ctx.setRequest(request);
        ctx.setTraceContext(FlexlbTrace.withSpan(parent, Context.root()));
        workers = mock(WorkerDirectory.class);
        worker = spy(WorkerStatus.createDiscovered(RoleType.VIT, "group", "10.0.0.1", 8000, 8001, ""));
        var routing = List.of(new VitRoutingEntry(worker.getIpPort(), new WorkerEndpoint(worker)));
        when(workers.vitRoutingSnapshot()).thenReturn(routing);
        when(workers.captureEndpoint(eq(RoleType.VIT), any())).thenAnswer(call -> new WorkerEndpoint(worker).tryPinGeneration());
        directory = mock(VitCacheDirectory.class);
        selector = new VitCacheSelector(directory, workers);
    }

    @AfterEach
    void tearDown() {
        selector.stop();
        parent.end();
        FlexlbTrace.configure(null, "");
        provider.close();
    }

    @Test
    void selectionAndStaleValidationAreChildrenOfScheduleWithoutMedia() {
        var result = selector.select(ctx, null);
        assertTrue(result.isSuccess());
        SpanData selection = spans.getFirst();
        assertEquals("rtp_llm.flexlb.vit.select", selection.getName());
        assertEquals(SpanKind.INTERNAL, selection.getKind());
        assertEquals(parent.getSpanContext().getSpanId(), selection.getParentSpanId());
        assertEquals(StatusCode.OK, selection.getStatus().getStatusCode());
        assertEquals("42", selection.getAttributes().get(FlexlbTrace.REQUEST_ID));
        assertEquals(2L, selection.getAttributes().get(FlexlbTrace.VIT_KEY_COUNT));
        assertEquals("10.0.0.1:8001", selection.getAttributes().get(FlexlbTrace.VIT_ADDRESS));
        assertFalse(selection.getAttributes().toString().contains("private"));
        ctx.getRequest().setSelectedVit(result);
        result.setWorkerGeneration(result.getWorkerGeneration() + 1);
        assertFalse(selector.validate(ctx, null).isSuccess());
        SpanData validation = spans.get(1);
        assertEquals("rtp_llm.flexlb.vit.validate", validation.getName());
        assertEquals(parent.getSpanContext().getSpanId(), validation.getParentSpanId());
        assertEquals(StatusCode.ERROR, validation.getStatus().getStatusCode());
        assertEquals("VIT_ROUTE_STALE", validation.getAttributes().get(FlexlbTrace.ERROR_TYPE));
        assertEquals("generation_mismatch", validation.getAttributes().get(FlexlbTrace.VIT_VALIDATION_REASON));
        assertEquals(result.getWorkerGeneration(), validation.getAttributes().get(FlexlbTrace.VIT_WORKER_GENERATION));
        assertEquals(result.getWorkerGeneration() - 1, validation.getAttributes().get(FlexlbTrace.VIT_CURRENT_GENERATION));
        assertNull(validation.getAttributes().get(FlexlbTrace.VIT_MM_INFLIGHT_INPUTS));
        assertNull(validation.getAttributes().get(FlexlbTrace.VIT_LOAD_STATE));
    }

    @Test
    void validationExplainsWhichIdentityCheckFailed() {
        assertFalse(selector.validate(ctx, null).isSuccess());
        assertEquals("missing_selection", spans.getLast().getAttributes().get(FlexlbTrace.VIT_VALIDATION_REASON));

        var selected = selector.select(ctx, null);
        ctx.getRequest().setSelectedVit(selected);
        assertFalse(selector.validate(ctx, "another-group").isSuccess());
        var policy = spans.getLast().getAttributes();
        assertEquals("policy_group_mismatch", policy.get(FlexlbTrace.VIT_VALIDATION_REASON));
        assertEquals("another-group", policy.get(FlexlbTrace.VIT_POLICY_GROUP));
        assertEquals("group", policy.get(FlexlbTrace.VIT_CURRENT_GROUP));
        assertNull(policy.get(FlexlbTrace.VIT_CURRENT_GRPC_PORT));
        assertNull(policy.get(FlexlbTrace.VIT_CURRENT_GENERATION));

        selected.setGrpcPort(9999);
        assertFalse(selector.validate(ctx, null).isSuccess());
        assertEquals("grpc_port_mismatch", spans.getLast().getAttributes().get(FlexlbTrace.VIT_VALIDATION_REASON));
        selected.setGrpcPort(8001);
        selected.setGroup("old-group");
        assertFalse(selector.validate(ctx, null).isSuccess());
        assertEquals("selected_group_mismatch", spans.getLast().getAttributes().get(FlexlbTrace.VIT_VALIDATION_REASON));
        selected.setGroup("group");
        selected.setRole(RoleType.PREFILL);
        assertFalse(selector.validate(ctx, null).isSuccess());
        assertEquals("role_mismatch", spans.getLast().getAttributes().get(FlexlbTrace.VIT_VALIDATION_REASON));
        selected.setRole(RoleType.VIT);
        when(workers.captureEndpoint(eq(RoleType.VIT), any())).thenReturn(null);
        assertFalse(selector.validate(ctx, null).isSuccess());
        assertEquals("endpoint_unavailable", spans.getLast().getAttributes().get(FlexlbTrace.VIT_VALIDATION_REASON));
    }

    @Test
    void selectedLoadIsDecisionSnapshotAndValidationReportsLatestSnapshot() {
        long now = System.nanoTime();
        ctx.getConfig().getRouter().getRoles().getVit().setLoadMaxAgeMs(60_000);
        doReturn(new WorkerStatus.VitLoad(3, now)).when(worker).vitLoad();
        WorkerStatus busy = spy(WorkerStatus.createDiscovered(RoleType.VIT, "group", "10.0.0.2", 8000, 8001, ""));
        doReturn(new WorkerStatus.VitLoad(8, now)).when(busy).vitLoad();
        var routing = List.of(
                new VitRoutingEntry(busy.getIpPort(), new WorkerEndpoint(busy)),
                new VitRoutingEntry(worker.getIpPort(), new WorkerEndpoint(worker)));
        when(workers.vitRoutingSnapshot()).thenReturn(routing);
        when(workers.captureEndpoint(eq(RoleType.VIT), any())).thenAnswer(call -> {
            // Simulate a new poll result after ranking but before endpoint validation.
            doReturn(new WorkerStatus.VitLoad(91, System.nanoTime())).when(worker).vitLoad();
            return new WorkerEndpoint(worker).tryPinGeneration();
        });
        var result = selector.select(ctx, null);
        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
        var selection = spans.getFirst().getAttributes();
        assertEquals(3L, selection.get(FlexlbTrace.VIT_MM_INFLIGHT_INPUTS));
        assertEquals("fresh", selection.get(FlexlbTrace.VIT_LOAD_STATE));
        assertTrue(selection.get(FlexlbTrace.VIT_LOAD_SAMPLE_AGE_MS) >= 0);
        verify(worker, times(1)).vitLoad();
        ctx.getRequest().setSelectedVit(result);
        assertTrue(selector.validate(ctx, null).isSuccess());
        assertEquals(91L, spans.get(1).getAttributes().get(FlexlbTrace.VIT_MM_INFLIGHT_INPUTS));
        assertEquals("fresh", spans.get(1).getAttributes().get(FlexlbTrace.VIT_LOAD_STATE));
        var validation = spans.get(1).getAttributes();
        assertNull(validation.get(FlexlbTrace.VIT_VALIDATION_REASON));
        assertNull(validation.get(FlexlbTrace.VIT_POLICY_GROUP));
        assertNull(validation.get(FlexlbTrace.VIT_SELECTED_GROUP));
        assertNull(validation.get(FlexlbTrace.VIT_CURRENT_GROUP));
        assertNull(validation.get(FlexlbTrace.VIT_CURRENT_GRPC_PORT));
        assertNull(validation.get(FlexlbTrace.VIT_CURRENT_GENERATION));
    }

    @Test
    void missingStaleAndZeroLoadsRemainDistinct() {
        assertTrue(selector.select(ctx, null).isSuccess());
        var missing = spans.getLast().getAttributes();
        assertEquals("missing", missing.get(FlexlbTrace.VIT_LOAD_STATE));
        assertNull(missing.get(FlexlbTrace.VIT_MM_INFLIGHT_INPUTS));
        assertNull(missing.get(FlexlbTrace.VIT_LOAD_SAMPLE_AGE_MS));

        doReturn(new WorkerStatus.VitLoad(7, System.nanoTime() - TimeUnit.SECONDS.toNanos(60)))
                .when(worker).vitLoad();
        assertTrue(selector.select(ctx, null).isSuccess());
        var stale = spans.getLast().getAttributes();
        assertEquals("stale", stale.get(FlexlbTrace.VIT_LOAD_STATE));
        assertEquals(7L, stale.get(FlexlbTrace.VIT_MM_INFLIGHT_INPUTS));
        assertTrue(stale.get(FlexlbTrace.VIT_LOAD_SAMPLE_AGE_MS) >= 60_000L);

        ctx.getConfig().getRouter().getRoles().getVit().setLoadMaxAgeMs(60_000);
        doReturn(new WorkerStatus.VitLoad(0, System.nanoTime())).when(worker).vitLoad();
        assertTrue(selector.select(ctx, null).isSuccess());
        var idle = spans.getLast().getAttributes();
        assertEquals("fresh", idle.get(FlexlbTrace.VIT_LOAD_STATE));
        assertEquals(0L, idle.get(FlexlbTrace.VIT_MM_INFLIGHT_INPUTS));
    }

    @Test
    void affinitySelectionReportsChosenWorkersOriginalLoad() {
        ctx.getConfig().getRouter().getRoles().getVit().setCacheAffinityEnabled(true);
        ctx.getConfig().getRouter().getRoles().getVit().setLoadMaxAgeMs(60_000);
        long now = System.nanoTime();
        doReturn(new WorkerStatus.VitLoad(41, now)).when(worker).vitLoad();
        WorkerStatus idle = spy(WorkerStatus.createDiscovered(RoleType.VIT, "group", "10.0.0.2", 8000, 8001, ""));
        doReturn(new WorkerStatus.VitLoad(0, now)).when(idle).vitLoad();
        var warm = new VitCacheDirectory.Candidate(worker, "instance-a", Set.of("private-key"), 1, 0, 0);
        var cold = new VitCacheDirectory.Candidate(idle, "instance-b", Set.of(), 0, 0, 0);
        var candidates = Map.of(worker.getIpPort(), warm, idle.getIpPort(), cold);
        when(directory.candidates(isNull(), anyList())).thenReturn(candidates);
        when(workers.captureEndpoint(eq(RoleType.VIT), any())).thenAnswer(call -> {
            doReturn(new WorkerStatus.VitLoad(99, System.nanoTime())).when(worker).vitLoad();
            return new WorkerEndpoint(worker).tryPinGeneration();
        });
        assertTrue(selector.select(ctx, null).isSuccess());
        var attributes = spans.getFirst().getAttributes();
        assertEquals("10.0.0.1:8001", attributes.get(FlexlbTrace.VIT_ADDRESS));
        assertEquals(41L, attributes.get(FlexlbTrace.VIT_MM_INFLIGHT_INPUTS));
        assertEquals("fresh", attributes.get(FlexlbTrace.VIT_LOAD_STATE));
        assertEquals(true, attributes.get(FlexlbTrace.VIT_CACHE_AFFINITY));
        assertFalse(attributes.toString().contains("private"));
        verify(worker, times(1)).vitLoad();
    }

    @Test
    void noWorkerAndExceptionsEndExactlyOnceWithoutExceptionPayload() {
        when(workers.vitRoutingSnapshot()).thenReturn(List.of());
        assertFalse(selector.select(ctx, null).isSuccess());
        assertEquals(StatusCode.ERROR, spans.getFirst().getStatus().getStatusCode());
        assertNull(spans.getFirst().getAttributes().get(FlexlbTrace.VIT_LOAD_STATE));
        when(workers.vitRoutingSnapshot()).thenThrow(new IllegalStateException("private-url"));
        assertThrows(IllegalStateException.class, () -> selector.select(ctx, null));
        assertEquals(2, spans.size());
        assertEquals("VIT_ROUTING_ERROR", spans.get(1).getAttributes().get(FlexlbTrace.ERROR_TYPE));
        assertTrue(spans.get(1).getEvents().isEmpty());
    }

    @Test
    void disabledAndFailingTelemetryDoNotChangeSelection() {
        FlexlbTrace.configure(null, "");
        assertTrue(selector.select(ctx, null).isSuccess());
        OpenTelemetry sdk = mock(OpenTelemetry.class);
        Tracer tracer = mock(Tracer.class);
        when(sdk.getTracer(anyString(), anyString())).thenReturn(tracer);
        when(tracer.spanBuilder(anyString())).thenThrow(new IllegalStateException("exporter failure"));
        FlexlbTrace.configure(sdk, "test");
        var selected = selector.select(ctx, null);
        assertTrue(selected.isSuccess());
        ctx.getRequest().setSelectedVit(selected);
        assertTrue(selector.validate(ctx, "group").isSuccess());
        assertTrue(spans.isEmpty());
    }
}

package org.flexlb.balance.scheduler;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;

import com.google.protobuf.DescriptorProtos;
import com.google.protobuf.Descriptors;
import com.google.protobuf.DynamicMessage;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher.PreparedSubmission;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DefaultBatchDispatcherTest {

    private ConfigService configService;
    private EngineGrpcClient grpcClient;
    private FlexlbConfig config;
    private DefaultBatchDispatcher dispatcher;
    private TestCallback callback;

    @BeforeEach
    void setUp() {
        org.flexlb.telemetry.FlexlbTrace.configure(io.opentelemetry.api.OpenTelemetry.noop(), "");
        configService = mock(ConfigService.class);
        grpcClient = mock(EngineGrpcClient.class);
        config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useBatchDispatcher(config);
        when(configService.loadBalanceConfig()).thenReturn(config);

        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        callback = new TestCallback();
    }

    @AfterEach
    void tearDown() {
        org.flexlb.telemetry.FlexlbTrace.configure(null, "");
        dispatcher.shutdown();
    }

    @Test
    void dispatchPreservesDistinctPerRequestTraceContextsInOneBatch() throws Exception {
        assertDispatchedTraceContexts(true, false);
    }

    @Test
    void dispatchUsesEachScheduleParentAndClearsStaleTracestate() throws Exception {
        assertDispatchedTraceContexts(true, true);
    }

    @Test
    void disabledTracingPreservesOriginalCarriersDespiteValidScheduleContexts() throws Exception {
        assertDispatchedTraceContexts(false, true);
    }

    private void assertDispatchedTraceContexts(boolean enabled, boolean validScheduleContext) throws Exception {
        org.flexlb.telemetry.FlexlbTrace.configure(enabled ? io.opentelemetry.api.OpenTelemetry.noop() : null, "");
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute first = createRequestRoute(501L, 500, 200, prefillEp);
        RequestRoute second = createRequestRoute(502L, 500, 200, prefillEp);
        first.ctx().setGenerateInputPb(generateInputWithTraceContext(
                501L,
                "00-11111111111111111111111111111111-1111111111111111-01",
                "vendor=one").toBuilder().setUnknownFields(com.google.protobuf.UnknownFieldSet.newBuilder()
                        .addField(999, com.google.protobuf.UnknownFieldSet.Field.newBuilder().addVarint(123).build())
                        .build()).build().toByteString());
        second.ctx().setGenerateInputPb(generateInputWithTraceContext(
                502L,
                "00-22222222222222222222222222222222-2222222222222222-01",
                "vendor=two").toByteString());
        if (validScheduleContext) {
            first.ctx().setTraceContext(scheduleContext(
                    "11111111111111111111111111111111", "aaaaaaaaaaaaaaaa", "schedule"));
            second.ctx().setTraceContext(scheduleContext(
                    "22222222222222222222222222222222", "bbbbbbbbbbbbbbbb", ""));
        } else {
            first.ctx().setTraceContext(io.opentelemetry.context.Context.root());
            second.ctx().setTraceContext(null);
        }

        List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenAnswer(inv -> {
                    sent.add(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(92L, List.of(501L, 502L)));
                });

        submit(List.of(first, second), 92L, 100, "trace_context", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, sent.size());
        List<EngineRpcService.EnqueueBatchExternalInputPB> requests =
                sent.getFirst().getDpSlots(0).getRequestsList();
        assertEquals(2, requests.size());
        EngineRpcService.GenerateInputPB firstSent = requests.stream()
                .map(EngineRpcService.EnqueueBatchExternalInputPB::getInput)
                .filter(input -> input.getRequestId() == 501L)
                .findFirst().orElseThrow();
        EngineRpcService.GenerateInputPB secondSent = requests.stream()
                .map(EngineRpcService.EnqueueBatchExternalInputPB::getInput)
                .filter(input -> input.getRequestId() == 502L)
                .findFirst().orElseThrow();
        assertEquals(first.ctx().getGenerateInput().getUnknownFields(), firstSent.getUnknownFields());
        boolean replaced = enabled && validScheduleContext;
        assertEquals("00-11111111111111111111111111111111-"
                        + (replaced ? "aaaaaaaaaaaaaaaa" : "1111111111111111") + "-01",
                firstSent.getRequestInfo().getTraceContext().getTraceparent());
        assertEquals(replaced ? "vendor=schedule" : "vendor=one",
                firstSent.getRequestInfo().getTraceContext().getTracestate());
        assertEquals("00-22222222222222222222222222222222-"
                        + (replaced ? "bbbbbbbbbbbbbbbb" : "2222222222222222") + "-01",
                secondSent.getRequestInfo().getTraceContext().getTraceparent());
        assertEquals(replaced ? "" : "vendor=two", secondSent.getRequestInfo().getTraceContext().getTracestate());
        assertEquals("vendor=one", first.ctx().getGenerateInput()
                .getRequestInfo().getTraceContext().getTracestate(), "dispatch must not mutate the queued payload");
    }

    private static io.opentelemetry.context.Context scheduleContext(String traceId, String spanId, String vendor) {
        var state = io.opentelemetry.api.trace.TraceState.builder();
        if (!vendor.isEmpty()) {
            state.put("vendor", vendor);
        }
        return io.opentelemetry.context.Context.root().with(io.opentelemetry.api.trace.Span.wrap(
                io.opentelemetry.api.trace.SpanContext.create(traceId, spanId,
                        io.opentelemetry.api.trace.TraceFlags.getSampled(), state.build())));
    }

    @Test
    void oldDescriptorRoundTripPreservesNestedTraceContextUnknownField() throws Exception {
        EngineRpcService.GenerateInputPB payload = generateInputWithTraceContext(
                503L,
                "00-33333333333333333333333333333333-3333333333333333-01",
                "vendor=legacy");
        Descriptors.Descriptor legacy = legacyGenerateInputDescriptor();

        DynamicMessage oldReader = DynamicMessage.parseFrom(legacy, payload.toByteArray());
        Descriptors.FieldDescriptor requestInfoField = legacy.findFieldByNumber(9);
        DynamicMessage oldRequestInfo = (DynamicMessage) oldReader.getField(requestInfoField);
        assertTrue(!oldRequestInfo.getUnknownFields().getField(6).getLengthDelimitedList().isEmpty());

        DynamicMessage oldWriter = oldReader.toBuilder()
                .setField(legacy.findFieldByNumber(10), 73)
                .build();
        EngineRpcService.GenerateInputPB reparsed =
                EngineRpcService.GenerateInputPB.parseFrom(oldWriter.toByteArray());

        assertEquals(73, reparsed.getPriority());
        assertEquals(payload.getRequestInfo().getTraceContext(),
                reparsed.getRequestInfo().getTraceContext());
    }

    private static EngineRpcService.GenerateInputPB generateInputWithTraceContext(
            long requestId, String traceparent, String tracestate) {
        return EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().build())
                .setRequestInfo(EngineRpcService.RequestInfoPB.newBuilder()
                        .setTraceContext(EngineRpcService.TraceContextPB.newBuilder()
                                .setTraceparent(traceparent)
                                .setTracestate(tracestate)
                                .build())
                        .build())
                .build();
    }

    private static Descriptors.Descriptor legacyGenerateInputDescriptor() throws Exception {
        DescriptorProtos.DescriptorProto requestInfo = DescriptorProtos.DescriptorProto.newBuilder()
                .setName("RequestInfoPB")
                .addField(optionalField("frontend_ip", 1,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_STRING, null))
                .addField(optionalField("dash_ip", 2,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_STRING, null))
                .addField(optionalField("trace_id", 3,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_STRING, null))
                .addField(optionalField("request_id", 4,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_STRING, null))
                .addField(optionalField("source_role", 5,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_STRING, null))
                .build();
        DescriptorProtos.DescriptorProto generateConfig = DescriptorProtos.DescriptorProto.newBuilder()
                .setName("GenerateConfigPB")
                .build();
        DescriptorProtos.DescriptorProto generateInput = DescriptorProtos.DescriptorProto.newBuilder()
                .setName("GenerateInputPB")
                .addField(optionalField("request_id", 1,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64, null))
                .addField(optionalField("generate_config", 4,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_MESSAGE,
                        ".legacy_trace.GenerateConfigPB"))
                .addField(optionalField("request_info", 9,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_MESSAGE,
                        ".legacy_trace.RequestInfoPB"))
                .addField(optionalField("priority", 10,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT32, null))
                .build();
        DescriptorProtos.FileDescriptorProto file = DescriptorProtos.FileDescriptorProto.newBuilder()
                .setName("legacy_trace_generate_input.proto")
                .setPackage("legacy_trace")
                .setSyntax("proto3")
                .addMessageType(requestInfo)
                .addMessageType(generateConfig)
                .addMessageType(generateInput)
                .build();
        return Descriptors.FileDescriptor.buildFrom(file, new Descriptors.FileDescriptor[0])
                .findMessageTypeByName("GenerateInputPB");
    }

    private static DescriptorProtos.FieldDescriptorProto optionalField(
            String name,
            int number,
            DescriptorProtos.FieldDescriptorProto.Type type,
            String typeName) {
        DescriptorProtos.FieldDescriptorProto.Builder builder =
                DescriptorProtos.FieldDescriptorProto.newBuilder()
                        .setName(name)
                        .setNumber(number)
                        .setLabel(DescriptorProtos.FieldDescriptorProto.Label.LABEL_OPTIONAL)
                        .setType(type);
        if (typeName != null) {
            builder.setTypeName(typeName);
        }
        return builder.build();
    }

    @Test
    void dispatchSendsItemsToGrpcAndReceivesAck() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp);

        EngineRpcService.EnqueueBatchResponsePB response = ackResponse(1L, List.of(1L));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenReturn(CompletableFuture.completedFuture(response));

        submit(List.of(item), 1L, 100, "test_reason", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS), "onSuccess should be called");
        assertEquals(1, callback.successCount.get());
        assertEquals(0, callback.failureCount.get());
        ArgumentCaptor<EngineRpcService.EnqueueBatchRequestPB> request =
                ArgumentCaptor.forClass(EngineRpcService.EnqueueBatchRequestPB.class);
        verify(grpcClient).batchEnqueueAsync(anyString(), anyInt(), request.capture());
        assertEquals(3000, request.getValue().getFetchAttachTimeoutMs());
    }

    @Test
    void dispatchPassesConfiguredFetchAttachTimeoutToEngine() throws Exception {
        config.getDispatcher().setFetchAttachTimeoutMs(1500);
        RequestRoute item = createRequestRoute(1L, 500, 200, createPrefillEndpoint());
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenReturn(CompletableFuture.completedFuture(ackResponse(1L, List.of(1L))));

        submit(List.of(item), 1L, 100, "test_reason", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS), "onSuccess should be called");
        ArgumentCaptor<EngineRpcService.EnqueueBatchRequestPB> request =
                ArgumentCaptor.forClass(EngineRpcService.EnqueueBatchRequestPB.class);
        verify(grpcClient).batchEnqueueAsync(anyString(), anyInt(), request.capture());
        assertEquals(1500, request.getValue().getFetchAttachTimeoutMs());
    }

    @Test
    void batchAttributesAndValidatedResponseTimePrecedeEveryItemCallback() throws Exception {
        PrefillEndpoint endpoint = createPrefillEndpoint();
        RequestRoute first = createRequestRoute(601L, 20, 0, endpoint);
        RequestRoute second = createRequestRoute(602L, 20, 0, endpoint);
        var firstSpan = mock(io.opentelemetry.api.trace.Span.class);
        var secondSpan = mock(io.opentelemetry.api.trace.Span.class);
        when(firstSpan.storeInContext(any(io.opentelemetry.context.Context.class))).thenCallRealMethod();
        when(secondSpan.storeInContext(any(io.opentelemetry.context.Context.class))).thenCallRealMethod();
        first.ctx().setTraceContext(io.opentelemetry.context.Context.root().with(firstSpan));
        second.ctx().setTraceContext(io.opentelemetry.context.Context.root().with(secondSpan));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenReturn(CompletableFuture.completedFuture(ackResponse(93L, List.of(601L, 602L))));
        CompletableFuture<Void> verified = new CompletableFuture<>();
        AtomicInteger remaining = new AtomicInteger(2);
        submit(List.of(first, second), 93L, 100, "trace_timing", (item, result) -> {
            try {
                var span = item == first ? firstSpan : secondSpan;
                verify(span).setAttribute(org.flexlb.telemetry.FlexlbTrace.BATCH_ID, 93L);
                verify(span).setAttribute(org.flexlb.telemetry.FlexlbTrace.BATCH_SIZE, 2L);
                verify(span).setAttribute(org.flexlb.telemetry.FlexlbTrace.DISPATCH_REASON, "trace_timing");
                verify(span).setAttribute(org.mockito.ArgumentMatchers.eq(
                        org.flexlb.telemetry.FlexlbTrace.ENQUEUE_BATCH_MS), anyLong());
                assertEquals(DeliveryResult.Status.DELIVERED, result.status());
                if (remaining.decrementAndGet() == 0) {
                    verified.complete(null);
                }
            } catch (Throwable error) {
                verified.completeExceptionally(error);
            }
        });
        verified.get(5, TimeUnit.SECONDS);
    }

    @Test
    void malformedAckDoesNotPublishValidatedResponseTime() throws Exception {
        RequestRoute item = createRequestRoute(603L, 20, 0, createPrefillEndpoint());
        var span = mock(io.opentelemetry.api.trace.Span.class);
        when(span.storeInContext(any(io.opentelemetry.context.Context.class))).thenCallRealMethod();
        item.ctx().setTraceContext(io.opentelemetry.context.Context.root().with(span));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenReturn(CompletableFuture.completedFuture(ackResponse(94L, List.of(999L))));
        submit(List.of(item), 94L, 100, "malformed", callback);
        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        verify(span, never()).setAttribute(org.mockito.ArgumentMatchers.eq(
                org.flexlb.telemetry.FlexlbTrace.ENQUEUE_BATCH_MS), anyLong());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void asynchronousTimeoutKeepsEveryMemberUncertain(boolean grpcDeadline) throws Exception {
        PrefillEndpoint endpoint = createPrefillEndpoint();
        RequestRoute first = createRequestRoute(611L, 20, 0, endpoint);
        RequestRoute second = createRequestRoute(612L, 20, 0, endpoint);
        var rpc = new CompletableFuture<EngineRpcService.EnqueueBatchResponsePB>();
        CountDownLatch invoked = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenAnswer(call -> { invoked.countDown(); return rpc; });
        var results = new CopyOnWriteArrayList<java.util.Map.Entry<RequestRoute, DeliveryResult>>();
        submit(List.of(first, second), 95L, 100, "timeout", (item, result) ->
                results.add(java.util.Map.entry(item, result)));
        assertTrue(invoked.await(5, TimeUnit.SECONDS));
        Throwable timeout = grpcDeadline ? io.grpc.Status.DEADLINE_EXCEEDED.asRuntimeException()
                : new java.util.concurrent.TimeoutException("RPC deadline elapsed");
        rpc.completeExceptionally(timeout);
        dispatcher.shutdownAndAwait();
        assertEquals(List.of(first, second), results.stream().map(java.util.Map.Entry::getKey).toList());
        for (var result : results) {
            assertEquals(DeliveryResult.Status.UNCERTAIN, result.getValue().status());
            org.junit.jupiter.api.Assertions.assertSame(timeout, result.getValue().cause());
            assertFalse(result.getValue().failed(), "timeout cannot prove that the engine did not accept work");
        }
        verify(grpcClient).batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class));
    }

    @Test
    void dispatchHandlesGrpcError() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp);

        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("gRPC connection refused")));

        submit(List.of(item), 1L, 100, "test_reason", callback);

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS),
                "post-send transport error must be reconciled");
        assertEquals(1, callback.uncertainCount.get());
        assertEquals(0, callback.failureCount.get());
        assertEquals(0, callback.successCount.get());
    }

    @Test
    void dispatchHandlesNullGrpcResponse() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp);

        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenReturn(CompletableFuture.completedFuture(null));

        submit(List.of(item), 1L, 100, "test_reason", callback);

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.uncertainCount.get());
    }

    @Test
    void dispatchHandlesNullGrpcFutureAsUncertain() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenReturn(null);

        submit(List.of(item), 1L, 100, "test", callback);

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.uncertainCount.get());
        assertEquals(0, callback.failureCount.get());
    }

    @Test
    void dispatchRejectsAckWithDifferentBatchId() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(8L, 500, 200, prefillEp);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any())).thenReturn(
                CompletableFuture.completedFuture(EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(87L)
                        .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(8L))
                        .build()));

        submit(List.of(item), 88L,
                100, "batch_id_mismatch", callback);

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(0, callback.successCount.get());
        assertEquals(1, callback.uncertainCount.get());
    }

    @Test
    void shutdownRejectsNewReservationsWithoutInvokingRequestCallbacks() {
        dispatcher.shutdown();

        CapacityBoundary.Attempt<?> rejected = dispatcher.tryPrepareSubmission();
        assertFalse(rejected.accepted());
        assertEquals(CapacityBoundary.Status.FAILED,
                rejected.boundary().status());
        assertEquals(0, callback.failureCount.get());
        assertEquals(0, callback.successCount.get());
        assertEquals(0, callback.uncertainCount.get());
    }

    @Test
    void shutdownWakesCapacityWaiterAndNextReservationReturnsAdmissionFailure()
            throws Exception {
        dispatcher.shutdown();
        dispatcher = SchedulerTestSupport.createDispatcher(grpcClient, configService, 1, 1);
        PreparedSubmission running = reservePermit();
        PreparedSubmission queued = reservePermit();
        CapacityBoundary unavailable = unavailableBoundary();
        assertFalse(unavailable.availability().isAvailable());
        CountDownLatch capacityChanged = new CountDownLatch(1);
        unavailable.availability().addListener(() -> {
            if (unavailable.availability().isAvailable()) {
                capacityChanged.countDown();
            }
        });

        dispatcher.shutdown();

        assertTrue(capacityChanged.await(5, TimeUnit.SECONDS));
        assertTrue(unavailable.availability().isAvailable(),
                "shutdown is a state transition which wakes admission waiters");
        CapacityBoundary.Attempt<?> rejected = dispatcher.tryPrepareSubmission();
        assertFalse(rejected.accepted());
        assertEquals(CapacityBoundary.Status.FAILED,
                rejected.boundary().status());
        running.close();
        queued.close();
    }

    @Test
    void unexpectedPreSendFailureIsDefiniteAndIsolatesCallbacks() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute first = createRequestRoute(1L, 500, 200, prefillEp);
        RequestRoute second = createRequestRoute(2L, 500, 200, prefillEp);
        when(configService.loadBalanceConfig())
                .thenThrow(new IllegalStateException("config unavailable before send"));
        CountDownLatch attempted = new CountDownLatch(2);
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger uncertain = new AtomicInteger();
        BiConsumer<RequestRoute, DeliveryResult> throwingCallback =
                (exactItem, completion) -> {
                RequestRoute item = assertInstanceOf(RequestRoute.class, exactItem);
                if (completion.status() == DeliveryResult.Status.NOT_SENT) {
                    failures.incrementAndGet();
                    attempted.countDown();
                    if (item.requestId() == 1L) {
                        throw new IllegalStateException("first callback failed");
                    }
                } else if (completion.status() == DeliveryResult.Status.UNCERTAIN) {
                    uncertain.incrementAndGet();
                }
            };

        submit(List.of(first, second),
                2L, 100, "pre_send_failure", throwingCallback);

        assertTrue(attempted.await(5, TimeUnit.SECONDS));
        assertEquals(2, failures.get());
        assertEquals(0, uncertain.get());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any());
    }

    @Test
    void synchronousRpcInvocationFailureIsUncertainAndIsolatesCallbacks() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute first = createRequestRoute(1L, 500, 200, prefillEp);
        RequestRoute second = createRequestRoute(2L, 500, 200, prefillEp);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenThrow(new IllegalStateException("client threw after invocation began"));
        CountDownLatch attempted = new CountDownLatch(2);
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger uncertain = new AtomicInteger();
        BiConsumer<RequestRoute, DeliveryResult> throwingCallback =
                (exactItem, completion) -> {
                RequestRoute item = assertInstanceOf(RequestRoute.class, exactItem);
                if (completion.status() == DeliveryResult.Status.NOT_SENT) {
                    failures.incrementAndGet();
                } else if (completion.status() == DeliveryResult.Status.UNCERTAIN) {
                    uncertain.incrementAndGet();
                    attempted.countDown();
                    if (item.requestId() == 1L) {
                        throw new IllegalStateException("first callback failed");
                    }
                }
            };

        submit(List.of(first, second),
                3L, 100, "post_boundary_throw", throwingCallback);

        assertTrue(attempted.await(5, TimeUnit.SECONDS));
        assertEquals(0, failures.get());
        assertEquals(2, uncertain.get());
    }

    @Test
    void acceptedRpcCompletesNormallyAfterShutdown() throws Exception {
        dispatcher.shutdown();
        dispatcher = SchedulerTestSupport.createDispatcher(grpcClient, configService, 1, 0);
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp);
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> rpcFuture = new CompletableFuture<>();
        CountDownLatch invoked = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenAnswer(invocation -> {
                    invoked.countDown();
                    return rpcFuture;
                });

        PreparedSubmission reservation = reservePermit();
        CapacityBoundary unavailable = unavailableBoundary();
        CountDownLatch capacityChanged = new CountDownLatch(1);
        unavailable.availability().addListener(() -> {
            if (unavailable.availability().isAvailable()) {
                capacityChanged.countDown();
            }
        });
        submit(reservation, List.of(item), 4L, 100,
                "shutdown_drain", callback);
        assertTrue(invoked.await(5, TimeUnit.SECONDS));
        assertTrue(capacityChanged.await(5, TimeUnit.SECONDS),
                "dispatch capacity must be released after the RPC handoff");
        assertFalse(rpcFuture.isDone());

        dispatcher.shutdown();
        rpcFuture.complete(ackResponse(4L, List.of(1L)));

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.successCount.get());
        assertEquals(0, callback.failureCount.get());
        assertEquals(0, callback.uncertainCount.get());
    }

    @Test
    void dispatchHandlesResponseWithErrors() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp);

        EngineRpcService.EnqueueBatchResponsePB response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(1L)
                        .addErrors(EngineRpcService.EnqueueBatchErrorPB.newBuilder()
                                .setRequestId(1L)
                                .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                        .setErrorCode(500)
                                        .setErrorMessage("engine busy")
                                        .build())
                                .build())
                        .build();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenReturn(CompletableFuture.completedFuture(response));

        submit(List.of(item), 1L, 100, "test", (exact, result) -> {
            assertEquals(DeliveryResult.Status.PREFILL_REJECTED, result.status());
            callback.accept(exact, result);
        });

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.failureCount.get());
        assertTrue(callback.lastError.getMessage().contains("error_code=500"));
    }

    @ParameterizedTest
    @ValueSource(strings = {"duplicate success", "duplicate error", "both success and error",
            "success references unknown", "error references unknown", "missing request_id", "multiple violations"})
    void malformedMemberAcknowledgementKeepsEveryMemberUncertain(String violation) throws Exception {
        PrefillEndpoint prefill = createPrefillEndpoint();
        var first = createRequestRoute(1L, 500, 200, prefill);
        var second = createRequestRoute(2L, 500, 200, prefill);
        var response = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(95L)
                .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(2L));
        var success = EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(1L).build();
        var error = EngineRpcService.EnqueueBatchErrorPB.newBuilder().setRequestId(1L).build();
        switch (violation) {
            case "duplicate success" -> response.addSuccesses(success).addSuccesses(success);
            case "duplicate error" -> response.addErrors(error).addErrors(error);
            case "both success and error" -> response.addSuccesses(success).addErrors(error);
            case "success references unknown" -> response.addSuccesses(success)
                    .addSuccesses(success.toBuilder().setRequestId(99L));
            case "error references unknown" -> response.addSuccesses(success)
                    .addErrors(error.toBuilder().setRequestId(99L));
            case "missing request_id" -> { }
            case "multiple violations" -> response.clearSuccesses().addSuccesses(success).addErrors(error)
                    .addErrors(error.toBuilder().setRequestId(99L))
                    .addErrors(error.toBuilder().setRequestId(99L));
            default -> throw new AssertionError(violation);
        }
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenReturn(CompletableFuture.completedFuture(response.build()));
        var results = new CopyOnWriteArrayList<DeliveryResult>();
        var identities = new CopyOnWriteArrayList<RequestRoute>();
        CountDownLatch observed = new CountDownLatch(2);
        submit(List.of(first, second), 95L, 100L, "malformed_members", (item, result) -> {
            identities.add(item);
            results.add(result);
            observed.countDown();
        });
        assertTrue(observed.await(5L, TimeUnit.SECONDS));
        dispatcher.shutdownAndAwait();
        assertEquals(List.of(first, second), identities);
        assertEquals(2, results.size());
        for (DeliveryResult result : results) {
            assertEquals(DeliveryResult.Status.UNCERTAIN, result.status());
            List<String> expectedViolations = violation.equals("multiple violations")
                    ? List.of("error references unknown request_id=99", "duplicate error for request_id=99",
                            "request_id appears in both success and error: 1", "response is missing request_id=2")
                    : List.of(violation);
            for (String expected : expectedViolations) {
                assertTrue(result.cause().getMessage().contains(expected), result.cause().getMessage());
            }
        }
    }

    @Test
    void dispatchHandlesMissingAck() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp);

        EngineRpcService.EnqueueBatchResponsePB response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(1L)
                        .build(); // no success, no error
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenReturn(CompletableFuture.completedFuture(response));

        submit(List.of(item), 1L, 100, "test", callback);

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.uncertainCount.get());
    }

    @Test
    void responseCallbackFailureIsIsolatedAndNeverReclassifiesOtherItemsAsUncertain()
            throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute succeeded = createRequestRoute(11L, 500, 200, prefillEp);
        RequestRoute rejected = createRequestRoute(12L, 500, 200, prefillEp);
        EngineRpcService.EnqueueBatchResponsePB response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(91L)
                        .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                .setRequestId(11L).build())
                        .addErrors(EngineRpcService.EnqueueBatchErrorPB.newBuilder()
                                .setRequestId(12L)
                                .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                        .setErrorCode(500L)
                                        .setErrorMessage("rejected")
                                        .build())
                                .build())
                        .build();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));

        CountDownLatch callbacksAttempted = new CountDownLatch(2);
        AtomicInteger successes = new AtomicInteger();
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger uncertain = new AtomicInteger();
        BiConsumer<RequestRoute, DeliveryResult> throwingCallback =
                (exactItem, completion) -> {
                if (completion.status() == DeliveryResult.Status.DELIVERED) {
                    successes.incrementAndGet();
                    callbacksAttempted.countDown();
                } else if (completion.status() == DeliveryResult.Status.PREFILL_REJECTED) {
                    failures.incrementAndGet();
                    callbacksAttempted.countDown();
                    throw new IllegalStateException(
                            "callback failed after committing failure");
                } else if (completion.status() == DeliveryResult.Status.UNCERTAIN) {
                    uncertain.incrementAndGet();
                }
            };

        submit(List.of(succeeded, rejected),
                91L, 100, "callback_isolation", throwingCallback);

        assertTrue(callbacksAttempted.await(5, TimeUnit.SECONDS));
        assertEquals(1, successes.get());
        assertEquals(1, failures.get());
        assertEquals(0, uncertain.get(),
                "a later callback exception must not reclassify any item");
    }

    @Test
    void permitReservedBeforeShutdownCanStillBeSubmitted() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        CountDownLatch rpcInvoked = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenAnswer(invocation -> {
                    rpcInvoked.countDown();
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp);
        PreparedSubmission permit = reservePermit();

        dispatcher.shutdown();
        assertDoesNotThrow(() -> submit(
                permit, List.of(item), 1L, 100,
                "accepted_before_shutdown", callback));

        assertTrue(rpcInvoked.await(5, TimeUnit.SECONDS),
                "the task accepted before shutdown must still invoke the RPC");
        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS),
                "shutdown must drain the accepted task and its completion callback");
        assertEquals(0, callback.failureCount.get());
        CapacityBoundary.Attempt<?> rejected = dispatcher.tryPrepareSubmission();
        assertFalse(rejected.accepted());
        assertEquals(CapacityBoundary.Status.FAILED,
                rejected.boundary().status());
    }

    @Test
    void shutdownAndAwaitDrainsAnInvokedRpcAndItsObserver() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> rpc = new CompletableFuture<>();
        CountDownLatch invoked = new CountDownLatch(1);
        CountDownLatch observerEntered = new CountDownLatch(1);
        CountDownLatch releaseObserver = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            invoked.countDown();
            return rpc;
        });
        submit(List.of(createRequestRoute(1L, 500, 200, prefillEp)),
                1L, 100, "drain_rpc", (item, result) -> {
                    observerEntered.countDown();
                    try { assertTrue(releaseObserver.await(5, TimeUnit.SECONDS)); }
                    catch (InterruptedException interrupted) { throw new AssertionError(interrupted); }
                    callback.accept(item, result);
                });
        assertTrue(invoked.await(5, TimeUnit.SECONDS));

        Thread closer = new Thread(dispatcher::shutdownAndAwait);
        closer.start();
        try {
            assertFalse(callback.successLatch.await(100, TimeUnit.MILLISECONDS));
            assertTrue(closer.isAlive(), "shutdown must wait for the invoked RPC");
            rpc.complete(ackResponse(1L, List.of(1L)));
            assertTrue(observerEntered.await(5, TimeUnit.SECONDS));
            assertTrue(closer.isAlive(), "shutdown must wait for the delivery observer");
            releaseObserver.countDown();
            closer.join(5_000);
            assertFalse(closer.isAlive());
            assertEquals(1, callback.successCount.get());
        } finally {
            releaseObserver.countDown();
            rpc.complete(ackResponse(1L, List.of(1L)));
            closer.join(5_000);
        }
    }

    @Test
    void completionExecutorRejectionPublishesUncertainOnceAndDrains() throws Exception {
        var executor = (java.util.concurrent.ThreadPoolExecutor)
                org.springframework.test.util.ReflectionTestUtils.getField(dispatcher, "completionExecutor");
        executor.shutdown();
        PrefillEndpoint endpoint = createPrefillEndpoint();
        RequestRoute first = createRequestRoute(1L, 500, 200, endpoint);
        RequestRoute second = createRequestRoute(2L, 500, 200, endpoint);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenReturn(CompletableFuture.completedFuture(ackResponse(91L, List.of(1L, 2L))));
        var results = new CopyOnWriteArrayList<DeliveryResult>();
        var identities = new CopyOnWriteArrayList<RequestRoute>();
        CountDownLatch observed = new CountDownLatch(2);
        submit(List.of(first, second), 91L, 100L, "observer_rejected", (item, result) -> {
            identities.add(item);
            results.add(result);
            observed.countDown();
        });
        assertTrue(observed.await(5, TimeUnit.SECONDS));
        org.junit.jupiter.api.Assertions.assertTimeoutPreemptively(java.time.Duration.ofSeconds(5), dispatcher::shutdownAndAwait);
        assertEquals(List.of(first, second), identities);
        assertEquals(2, results.size());
        for (DeliveryResult result : results) {
            assertEquals(DeliveryResult.Status.UNCERTAIN, result.status());
            assertInstanceOf(java.util.concurrent.RejectedExecutionException.class, result.cause());
        }
        verify(grpcClient).batchEnqueueAsync(anyString(), anyInt(), any());
    }

    @Test
    void logicalCapacityRejectsAndUnusedReservationRestoresCapacity()
            throws Exception {
        dispatcher.shutdown();
        dispatcher = SchedulerTestSupport.createDispatcher(grpcClient, configService, 1, 1);
        PreparedSubmission running = reservePermit();
        PreparedSubmission queued = reservePermit();
        CapacityBoundary unavailable = unavailableBoundary();
        assertFalse(unavailable.availability().isAvailable());
        CountDownLatch capacityChanged = new CountDownLatch(1);
        unavailable.availability().addListener(() -> {
            if (unavailable.availability().isAvailable()) {
                capacityChanged.countDown();
            }
        });

        queued.close();

        assertTrue(capacityChanged.await(5, TimeUnit.SECONDS));
        assertTrue(unavailable.availability().isAvailable());
        PreparedSubmission replacement = reservePermit();
        running.close();
        replacement.close();
    }

    @Test
    void closingAnyUnusedReservationSignalsAndRestoresCapacity() throws Exception {
        dispatcher.shutdown();
        dispatcher = SchedulerTestSupport.createDispatcher(grpcClient, configService, 1, 1);
        PreparedSubmission running = reservePermit();
        PreparedSubmission queued = reservePermit();
        CapacityBoundary unavailable = unavailableBoundary();
        assertFalse(unavailable.availability().isAvailable());
        Object capacityMonitor = new Object();
        unavailable.availability().addListener(() -> {
            synchronized (capacityMonitor) {
                capacityMonitor.notifyAll();
            }
        });

        running.close();

        awaitAvailable(unavailable.availability(), capacityMonitor);
        assertTrue(unavailable.availability().isAvailable());
        PreparedSubmission replacement = reservePermit();
        queued.close();
        replacement.close();
    }

    @Test
    void acceptedReservationsSubmitWithoutSecondCapacityCheck() {
        dispatcher.shutdown();
        dispatcher = SchedulerTestSupport.createDispatcher(grpcClient, configService, 1, 1);
        PrefillEndpoint endpoint = createPrefillEndpoint();
        RequestRoute firstItem = createRequestRoute(1L, 500, 200, endpoint);
        RequestRoute secondItem = createRequestRoute(2L, 500, 200, endpoint);
        PreparedSubmission running = reservePermit();
        PreparedSubmission queued = reservePermit();
        CapacityBoundary.Attempt<?> rejected = dispatcher.tryPrepareSubmission();
        assertFalse(rejected.accepted());
        assertEquals(CapacityBoundary.Status.UNAVAILABLE,
                rejected.boundary().status());

        assertDoesNotThrow(() -> submit(
                running, List.of(firstItem), 1L, 100,
                "already_accepted", callback));
        assertDoesNotThrow(() -> submit(
                queued, List.of(secondItem), 2L, 100,
                "already_accepted", callback));
    }

    @Test
    void preparedSubmissionRejectsMissingDeliveryWithoutConsumingPermit() {
        PreparedSubmission submission = reservePermit();
        assertThrows(NullPointerException.class, () -> submission.submit(null));
        submission.close();
        reservePermit().close();
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void replacementAfterPreparationIsUsedAtSendTime(boolean malformed) throws Exception {
        RequestRoute item = createRequestRoute(1L, 500, 200, createPrefillEndpoint());
        item.ctx().prepareGenerateInput();
        item.ctx().setGenerateInputPb(malformed
                ? com.google.protobuf.ByteString.copyFrom(new byte[] {(byte) 0xff})
                : com.google.protobuf.ByteString.EMPTY);
        CompletableFuture<DeliveryResult> result = new CompletableFuture<>();
        submit(List.of(item), 1L, 100, "replacement", (request, completion) -> result.complete(completion));
        assertEquals(DeliveryResult.Status.NOT_SENT, result.get(5, TimeUnit.SECONDS).status());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any());
    }

    @Test
    void invalidFinalBatchFailsBeforeRpcOnTheDeliveryThread() throws Exception {
        RequestRoute item = createRequestRoute(1L, 500, 200, createPrefillEndpoint());
        CompletableFuture<Void> checked = new CompletableFuture<>();
        reservePermit().submit(sender -> {
            try {
                assertThrows(IllegalArgumentException.class,
                        () -> sender.sendBatch(List.of(), 1L, 100, "empty", callback));
                assertThrows(IllegalArgumentException.class,
                        () -> sender.sendBatch(List.of(item), 0L, 100, "id", callback));
                assertThrows(IllegalArgumentException.class,
                        () -> sender.sendBatch(List.of(item), 1L, -1, "prediction", callback));
                assertThrows(NullPointerException.class,
                        () -> sender.sendBatch(List.of(item), 1L, 100, null, callback));
                checked.complete(null);
            } catch (Throwable failure) {
                checked.completeExceptionally(failure);
            }
        });
        checked.get(5, TimeUnit.SECONDS);
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any());
    }

    @Test
    void submittedReservationReleasesCapacityAfterRpcHandoff() throws Exception {
        dispatcher.shutdown();
        dispatcher = SchedulerTestSupport.createDispatcher(grpcClient, configService, 1, 0);
        PrefillEndpoint endpoint = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, endpoint);
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> rpcFuture =
                new CompletableFuture<>();
        CountDownLatch rpcInvoked = new CountDownLatch(1);
        CountDownLatch allowHandoff = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenAnswer(invocation -> {
                    rpcInvoked.countDown();
                    assertTrue(allowHandoff.await(5, TimeUnit.SECONDS));
                    return rpcFuture;
                });

        PreparedSubmission reservation = reservePermit();
        CapacityBoundary unavailable = unavailableBoundary();
        CountDownLatch capacityChanged = new CountDownLatch(1);
        unavailable.availability().addListener(() -> {
            if (unavailable.availability().isAvailable()) {
                capacityChanged.countDown();
            }
        });
        List<RequestRoute> exactItems = List.of(item);
        submit(reservation, exactItems, 1L, 100,
                "dispatch_handoff_capacity", callback);
        assertTrue(rpcInvoked.await(5, TimeUnit.SECONDS));

        reservation.close();
        reservation.close();
        assertFalse(unavailable.availability().isAvailable(),
                "close after submit must not release a dispatch still handing off");
        assertThrows(IllegalStateException.class,
                () -> submit(reservation, exactItems, 1L, 100,
                        "dispatch_handoff_capacity", callback));

        allowHandoff.countDown();
        assertTrue(capacityChanged.await(5, TimeUnit.SECONDS));
        assertTrue(unavailable.availability().isAvailable());
        assertFalse(rpcFuture.isDone());
        assertEquals(0, callback.successCount.get());
        PreparedSubmission replacement = reservePermit();

        replacement.close();

        rpcFuture.complete(ackResponse(1L, List.of(1L)));

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.successCount.get());
    }

    // ---- task40: priority passthrough to GenerateInputPB ----

    @Test
    void dispatchUsesAdmissionPriorityEvenIfOriginalRequestChanges() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp, 60);
        item.ctx().getRequest().setPriority(7);

        List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenAnswer(inv -> {
                    sent.add(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        submit(List.of(item), 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        assertEquals(60, sentInput(sent.getFirst()).getPriority());
    }

    @Test
    void dispatchLeavesPriorityUnsetForNoPriorityRequests() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp);
        // default Request priority is the no-priority sentinel (0)

        List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class)))
                .thenAnswer(inv -> {
                    sent.add(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        submit(List.of(item), 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        assertEquals(0, sentInput(sent.getFirst()).getPriority());
    }

    @Test
    void dispatchDualWritesCompatibleRoleAddress() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        RequestRoute item = createRequestRoute(1L, 500, 200, prefillEp);

        List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenAnswer(inv -> {
                    sent.add(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        submit(List.of(item), 1L, 100, "role_compat", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        EngineRpcService.RoleAddrPB addr = sentInput(sent.getFirst())
                .getGenerateConfig().getRoleAddrs(0);
        assertEquals(EngineRpcService.RoleAddrPB.RoleType.PREFILL, addr.getRole());
        assertEquals("PREFILL", addr.getRoleStr());
        assertEquals(RoleType.PREFILL, RoleTypeProtoConverter.fromRoleAddr(addr));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void dispatchPreservesPerRequestRolesAcrossMissingDecodeAndDpRanks(boolean multipleRanks) throws Exception {
        PrefillEndpoint endpoint = createPrefillEndpoint();
        var items = new java.util.ArrayList<RequestRoute>();
        for (int index = 0; index < 6; index++) {
            RequestRoute original = createRequestRoute(index + 1L, 500L, 200L, endpoint);
            original.prefill().setDpRank(multipleRanks && index % 2 != 0 ? 1L : 2L);
            if (index == 4) {
                original.prefill().setServerIp("10.0.1.2");
                original.prefill().setHttpPort(8301);
                original.prefill().setGrpcPort(8401);
            }
            ServerStatus decode = null;
            if (index != 1) {
                decode = new ServerStatus();
                decode.setRole(RoleType.DECODE);
                decode.setServerIp(index == 5 ? "10.0.0.2" : "10.0.0.1");
                decode.setHttpPort(index >= 3 ? 8101 : 8100);
                decode.setGrpcPort(index >= 4 ? 8201 : 8200);
            }
            items.add(org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(original.ctx()), null,
                    original.prefill(), decode, endpoint, null, null, original.enqueuedAtMs()));
        }
        CompletableFuture<EngineRpcService.EnqueueBatchRequestPB> sent = new CompletableFuture<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            sent.complete(call.getArgument(2));
            return CompletableFuture.completedFuture(ackResponse(8L,
                    items.stream().map(RequestRoute::requestId).toList()));
        });
        submit(items, 8L, 100L, "role-cache", callback);
        var batch = sent.get(5, TimeUnit.SECONDS);
        assertEquals(multipleRanks ? List.of(1, 2) : List.of(2),
                batch.getDpSlotsList().stream().map(EngineRpcService.EnqueueBatchDpSlotPB::getDpRank).toList());
        var inputs = new java.util.HashMap<Long, EngineRpcService.GenerateInputPB>();
        for (var slot : batch.getDpSlotsList()) {
            assertEquals(items.stream().filter(item -> item.prefill().getDpRank() == slot.getDpRank())
                            .map(RequestRoute::requestId).toList(),
                    slot.getRequestsList().stream().map(input -> input.getInput().getRequestId()).toList());
            for (var external : slot.getRequestsList()) {
                inputs.put(external.getInput().getRequestId(), external.getInput());
            }
        }
        assertEquals(items.size(), inputs.size());
        for (RequestRoute item : items) {
            var roles = inputs.get(item.requestId()).getGenerateConfig().getRoleAddrsList();
            assertEquals(item.decode() == null ? 1 : 2, roles.size());
            assertEquals(RoleType.PREFILL, RoleTypeProtoConverter.fromRoleAddr(roles.getFirst()));
            assertEquals(item.prefill().getServerIp(), roles.getFirst().getIp());
            assertEquals(item.prefill().getHttpPort(), roles.getFirst().getHttpPort());
            assertEquals(item.prefill().getGrpcPort(), roles.getFirst().getGrpcPort());
            if (item.decode() != null) {
                var decode = roles.get(1);
                assertEquals(RoleType.DECODE, RoleTypeProtoConverter.fromRoleAddr(decode));
                assertEquals(item.decode().getServerIp(), decode.getIp());
                assertEquals(item.decode().getHttpPort(), decode.getHttpPort());
                assertEquals(item.decode().getGrpcPort(), decode.getGrpcPort());
            }
        }
        org.junit.jupiter.api.Assertions.assertSame(
                inputs.get(1L).getGenerateConfig().getRoleAddrs(1),
                inputs.get(3L).getGenerateConfig().getRoleAddrs(1),
                "a missing Decode address must neither leak a stale address nor discard the reusable proto");
    }

    private static EngineRpcService.GenerateInputPB sentInput(EngineRpcService.EnqueueBatchRequestPB request) {
        return request.getDpSlotsList().getFirst().getRequestsList().getFirst().getInput();
    }

    // ---- helpers ----

    private PreparedSubmission reservePermit() {
        CapacityBoundary.Attempt<?> accepted = dispatcher.tryPrepareSubmission();
        assertTrue(accepted.accepted());
        return assertInstanceOf(
                PreparedSubmission.class, accepted.value());
    }

    private CapacityBoundary unavailableBoundary() {
        CapacityBoundary.Attempt<?> rejected = dispatcher.tryPrepareSubmission();
        assertFalse(rejected.accepted());
        assertEquals(CapacityBoundary.Status.UNAVAILABLE,
                rejected.boundary().status());
        return rejected.boundary();
    }

    private void submit(List<RequestRoute> items,
                        long batchId,
                        long predictedMs,
                        String reason,
                        BiConsumer<RequestRoute,
                                DeliveryResult> observer) {
        reservePermit().submit(sender -> sender.sendBatch(
                items, batchId, predictedMs, reason, observer));
    }

    private static void submit(
            PreparedSubmission submission,
            List<RequestRoute> items,
            long batchId,
            long predictedMs,
            String reason,
            BiConsumer<RequestRoute, DeliveryResult> observer) {
        submission.submit(sender -> sender.sendBatch(
                items, batchId, predictedMs, reason, observer));
    }

    private static void awaitAvailable(
            CapacityBoundary.Availability availability,
            Object capacityMonitor) throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        synchronized (capacityMonitor) {
            while (!availability.isAvailable()) {
                long remainingNanos = deadlineNanos - System.nanoTime();
                if (remainingNanos <= 0) {
                    throw new AssertionError("dispatcher capacity did not become available");
                }
                TimeUnit.NANOSECONDS.timedWait(capacityMonitor, remainingNanos);
            }
        }
    }

    private PrefillEndpoint createPrefillEndpoint() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.getHttpPort()).thenReturn(8080);
        when(endpoint.getGrpcPort()).thenReturn(8090);
        return endpoint;
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void rpcCompletionUsesItsOwnPoolForImmediateAndNetworkReplies(boolean immediate) throws Exception {
        var item = createRequestRoute(721L, 20L, 0L, createPrefillEndpoint());
        var rpc = new CompletableFuture<EngineRpcService.EnqueueBatchResponsePB>();
        var invoked = new CountDownLatch(1);
        var completed = new CompletableFuture<Thread>();
        var sendingThread = new java.util.concurrent.atomic.AtomicReference<Thread>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            sendingThread.set(Thread.currentThread());
            invoked.countDown();
            return immediate ? CompletableFuture.completedFuture(ackResponse(74L, List.of(721L))) : rpc;
        });
        submit(List.of(item), 74L, 10L, "thread-contract", (member, result) -> {
            assertFalse(Thread.holdsLock(member.ctx()));
            assertEquals(DeliveryResult.Status.DELIVERED, result.status());
            completed.complete(Thread.currentThread());
        });
        assertTrue(invoked.await(5, TimeUnit.SECONDS));
        if (!immediate) {
            Thread network = new Thread(() -> rpc.complete(ackResponse(74L, List.of(721L))), "test-engine-network");
            network.start();
            network.join(5_000);
            assertFalse(network.isAlive());
        }
        Thread completion = completed.get(5, TimeUnit.SECONDS);
        assertTrue(sendingThread.get().getName().startsWith("flexlb-dispatch-executor"));
        assertTrue(completion.getName().startsWith("flexlb-dispatch-completion"));
        org.junit.jupiter.api.Assertions.assertNotSame(sendingThread.get(), completion);
        dispatcher.shutdownAndAwait();
        verify(grpcClient, org.mockito.Mockito.times(1)).batchEnqueueAsync(anyString(), anyInt(), any());
    }

    @Test
    void sendBoundaryRemovesCancelledMembersFromTheActualPayload() throws Exception {
        var endpoint = createPrefillEndpoint();
        var cancelled = createRequestRoute(701L, 20L, 0L, endpoint);
        var admitted = createRequestRoute(702L, 20L, 0L, endpoint);
        when(cancelled.ctx().scheduler().tryStartSend(cancelled.ctx().delivery())).thenReturn(false);
        var admittedSpan = mock(io.opentelemetry.api.trace.Span.class);
        var cancelledSpan = mock(io.opentelemetry.api.trace.Span.class);
        when(admittedSpan.storeInContext(any(io.opentelemetry.context.Context.class))).thenCallRealMethod();
        when(cancelledSpan.storeInContext(any(io.opentelemetry.context.Context.class))).thenCallRealMethod();
        admitted.ctx().setTraceContext(io.opentelemetry.context.Context.root().with(admittedSpan));
        cancelled.ctx().setTraceContext(io.opentelemetry.context.Context.root().with(cancelledSpan));
        var sent = new CompletableFuture<EngineRpcService.EnqueueBatchRequestPB>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            sent.complete(call.getArgument(2));
            return CompletableFuture.completedFuture(ackResponse(71L, List.of(702L)));
        });
        var results = new java.util.concurrent.ConcurrentHashMap<Long, DeliveryResult.Status>();
        var completed = new CountDownLatch(2);
        submit(List.of(cancelled, admitted), 71L, 10L, "partial-cancel", (item, result) -> {
            results.put(item.requestId(), result.status());
            completed.countDown();
        });
        var batch = sent.get(2, TimeUnit.SECONDS);
        assertEquals(List.of(702L), batch.getDpSlotsList().stream().flatMap(slot -> slot.getRequestsList().stream())
                .map(request -> request.getInput().getRequestId()).toList());
        assertTrue(completed.await(2, TimeUnit.SECONDS));
        assertEquals(DeliveryResult.Status.NOT_SENT, results.get(701L));
        assertEquals(DeliveryResult.Status.DELIVERED, results.get(702L));
        verify(admittedSpan).setAttribute(org.flexlb.telemetry.FlexlbTrace.BATCH_SIZE, 1L);
        verify(cancelledSpan, org.mockito.Mockito.never()).setAttribute(
                org.mockito.ArgumentMatchers.eq(org.flexlb.telemetry.FlexlbTrace.BATCH_SIZE), anyLong());
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 2, 3, 4, 5, 6, 7})
    void sendBoundaryPreservesTheAcceptedSubsetAtEveryPosition(int refusedMask) throws Exception {
        var endpoint = createPrefillEndpoint();
        var items = List.of(createRequestRoute(711L, 20L, 0L, endpoint),
                createRequestRoute(712L, 20L, 0L, endpoint),
                createRequestRoute(713L, 20L, 0L, endpoint));
        var acceptedIds = new java.util.ArrayList<Long>();
        for (int index = 0; index < items.size(); index++) {
            boolean accepted = (refusedMask & (1 << index)) == 0;
            when(items.get(index).ctx().scheduler().tryStartSend(items.get(index).ctx().delivery())).thenReturn(accepted);
            if (accepted) { acceptedIds.add(items.get(index).requestId()); }
        }
        var sent = new CompletableFuture<EngineRpcService.EnqueueBatchRequestPB>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            sent.complete(call.getArgument(2));
            return CompletableFuture.completedFuture(ackResponse(73L, acceptedIds));
        });
        var results = new java.util.concurrent.ConcurrentHashMap<Long, DeliveryResult.Status>();
        var completed = new CountDownLatch(items.size());
        submit(items, 73L, 10L, "subset-cancel", (item, result) -> {
            assertEquals(null, results.putIfAbsent(item.requestId(), result.status()), "one result per member");
            completed.countDown();
        });
        assertTrue(completed.await(2, TimeUnit.SECONDS));
        for (int index = 0; index < items.size(); index++) {
            var expected = (refusedMask & (1 << index)) == 0
                    ? DeliveryResult.Status.DELIVERED : DeliveryResult.Status.NOT_SENT;
            assertEquals(expected, results.get(items.get(index).requestId()));
            verify(items.get(index).ctx().scheduler()).tryStartSend(items.get(index).ctx().delivery());
        }
        if (acceptedIds.isEmpty()) {
            verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any());
        } else {
            assertEquals(acceptedIds, sent.get(2, TimeUnit.SECONDS).getDpSlotsList().stream()
                    .flatMap(slot -> slot.getRequestsList().stream())
                    .map(request -> request.getInput().getRequestId()).toList());
        }
    }

    @Test
    void sendBoundarySkipsRpcWhenEveryMemberWasCancelled() throws Exception {
        var item = createRequestRoute(703L, 20L, 0L, createPrefillEndpoint());
        when(item.ctx().scheduler().tryStartSend(item.ctx().delivery())).thenReturn(false);
        var completed = new CompletableFuture<DeliveryResult>();
        submit(List.of(item), 72L, 10L, "cancelled", (ignored, result) -> completed.complete(result));
        assertEquals(DeliveryResult.Status.NOT_SENT, completed.get(2, TimeUnit.SECONDS).status());
        org.mockito.Mockito.verify(grpcClient, org.mockito.Mockito.never()).batchEnqueueAsync(anyString(), anyInt(), any());
    }

    private RequestRoute createRequestRoute(long requestId, long seqLen, long hitCacheLen, PrefillEndpoint prefillEp) {
        return createRequestRoute(requestId, seqLen, hitCacheLen, prefillEp, 0);
    }

    private RequestRoute createRequestRoute(long requestId, long seqLen, long hitCacheLen,
            PrefillEndpoint prefillEp, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);

        RequestContext ctx = new RequestContext(config);
        ctx.setRequest(request);

        // Provide a valid GenerateInputPB bytes (minimum: requestId + empty config)
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().build())
                .build();
        ctx.setGenerateInputPb(input.toByteString());

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("127.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8090);
        prefill.setDpRank(0L);
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(hitCacheLen);
        prefill.setDebugInfo(debugInfo);

        ctx.setFuture(new CompletableFuture<>());
        var delivery = mock(RequestContext.DeliveryClaim.class);
        ctx.bindScheduler(mock(AbstractRequestScheduler.class));
        when(ctx.scheduler().tryStartSend(delivery)).thenReturn(true);
        org.springframework.test.util.ReflectionTestUtils.setField(ctx, "delivery", delivery);
        return org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(ctx), null, prefill, null,
                prefillEp, null, null, System.currentTimeMillis());
    }

    private EngineRpcService.EnqueueBatchResponsePB ackResponse(long batchId, List<Long> successIds) {
        EngineRpcService.EnqueueBatchResponsePB.Builder builder =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(batchId);
        for (long id : successIds) {
            builder.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                    .setRequestId(id)
                    .build());
        }
        return builder.build();
    }

    // ---- Test callback ----

    private static class TestCallback
            implements BiConsumer<RequestRoute,
                    DeliveryResult> {
        final AtomicInteger successCount = new AtomicInteger(0);
        final AtomicInteger failureCount = new AtomicInteger(0);
        final AtomicInteger uncertainCount = new AtomicInteger(0);
        final CountDownLatch successLatch = new CountDownLatch(1);
        final CountDownLatch failureLatch = new CountDownLatch(1);
        final CountDownLatch uncertainLatch = new CountDownLatch(1);
        volatile Throwable lastError;

        @Override
        public void accept(
                RequestRoute exactItem,
                DeliveryResult completion) {
            if (completion.status() == DeliveryResult.Status.DELIVERED) {
                successCount.incrementAndGet();
                successLatch.countDown();
            } else if (completion.status() == DeliveryResult.Status.NOT_SENT
                    || completion.status() == DeliveryResult.Status.PREFILL_REJECTED) {
                lastError = completion.cause();
                failureCount.incrementAndGet();
                failureLatch.countDown();
            } else if (completion.status() == DeliveryResult.Status.UNCERTAIN) {
                lastError = completion.cause();
                uncertainCount.incrementAndGet();
                uncertainLatch.countDown();
            }
        }
    }
}

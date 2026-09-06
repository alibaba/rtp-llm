package org.flexlb.balance.scheduler;

import com.google.protobuf.DescriptorProtos;
import com.google.protobuf.Descriptors;
import com.google.protobuf.DynamicMessage;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatchDeliveryStrategy.PreparedSubmission;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
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
        org.flexlb.telemetry.FlexlbTrace.configureEnabled(true);
        configService = mock(ConfigService.class);
        grpcClient = mock(EngineGrpcClient.class);
        config = new FlexlbConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setEnqueueRpcTimeoutMs(5000);
        when(configService.loadBalanceConfig()).thenReturn(config);

        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        callback = new TestCallback();
    }

    @AfterEach
    void tearDown() {
        org.flexlb.telemetry.FlexlbTrace.configureEnabled(false);
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
        org.flexlb.telemetry.FlexlbTrace.configureEnabled(enabled);
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        ScheduledRequest first = createScheduledRequest(501L, 500, 200, prefillEp);
        ScheduledRequest second = createScheduledRequest(502L, 500, 200, prefillEp);
        first.ctx().setGenerateInputPb(generateInputWithTraceContext(
                501L,
                "00-11111111111111111111111111111111-1111111111111111-01",
                "vendor=one").toByteString());
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
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
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
        assertEquals("vendor=one", EngineRpcService.GenerateInputPB.parseFrom(first.ctx().getGenerateInputPb())
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
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);

        EngineRpcService.EnqueueBatchResponsePB response = ackResponse(1L, List.of(1L));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(response));

        submit(List.of(item), 1L, 100, "test_reason", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS), "onSuccess should be called");
        assertEquals(1, callback.successCount.get());
        assertEquals(0, callback.failureCount.get());
    }

    @Test
    void batchAttributesAndValidatedResponseTimePrecedeEveryItemCallback() throws Exception {
        PrefillEndpoint endpoint = createPrefillEndpoint();
        ScheduledRequest first = createScheduledRequest(601L, 20, 0, endpoint);
        ScheduledRequest second = createScheduledRequest(602L, 20, 0, endpoint);
        var firstSpan = mock(io.opentelemetry.api.trace.Span.class);
        var secondSpan = mock(io.opentelemetry.api.trace.Span.class);
        when(firstSpan.storeInContext(any(io.opentelemetry.context.Context.class))).thenCallRealMethod();
        when(secondSpan.storeInContext(any(io.opentelemetry.context.Context.class))).thenCallRealMethod();
        first.ctx().setTraceContext(io.opentelemetry.context.Context.root().with(firstSpan));
        second.ctx().setTraceContext(io.opentelemetry.context.Context.root().with(secondSpan));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
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
        ScheduledRequest item = createScheduledRequest(603L, 20, 0, createPrefillEndpoint());
        var span = mock(io.opentelemetry.api.trace.Span.class);
        when(span.storeInContext(any(io.opentelemetry.context.Context.class))).thenCallRealMethod();
        item.ctx().setTraceContext(io.opentelemetry.context.Context.root().with(span));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(ackResponse(94L, List.of(999L))));
        submit(List.of(item), 94L, 100, "malformed", callback);
        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        verify(span, never()).setAttribute(org.mockito.ArgumentMatchers.eq(
                org.flexlb.telemetry.FlexlbTrace.ENQUEUE_BATCH_MS), anyLong());
    }

    @Test
    void dispatchHandlesGrpcError() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);

        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
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
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);

        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(null));

        submit(List.of(item), 1L, 100, "test_reason", callback);

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.uncertainCount.get());
    }

    @Test
    void dispatchHandlesNullGrpcFutureAsUncertain() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenReturn(null);

        submit(List.of(item), 1L, 100, "test", callback);

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.uncertainCount.get());
        assertEquals(0, callback.failureCount.get());
    }

    @Test
    void dispatchRejectsAckWithDifferentBatchId() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        ScheduledRequest item = createScheduledRequest(8L, 500, 200, prefillEp);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong())).thenReturn(
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
        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null, 1, 1);
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
        ScheduledRequest first = createScheduledRequest(1L, 500, 200, prefillEp);
        ScheduledRequest second = createScheduledRequest(2L, 500, 200, prefillEp);
        when(configService.loadBalanceConfig())
                .thenThrow(new IllegalStateException("config unavailable before send"));
        CountDownLatch attempted = new CountDownLatch(2);
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger uncertain = new AtomicInteger();
        BiConsumer<ScheduledRequest, DeliveryResult> throwingCallback =
                (exactItem, completion) -> {
                ScheduledRequest item = assertInstanceOf(ScheduledRequest.class, exactItem);
                if (completion.status() == DeliveryResult.Status.FAILED) {
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
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void synchronousRpcInvocationFailureIsUncertainAndIsolatesCallbacks() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        ScheduledRequest first = createScheduledRequest(1L, 500, 200, prefillEp);
        ScheduledRequest second = createScheduledRequest(2L, 500, 200, prefillEp);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenThrow(new IllegalStateException("client threw after invocation began"));
        CountDownLatch attempted = new CountDownLatch(2);
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger uncertain = new AtomicInteger();
        BiConsumer<ScheduledRequest, DeliveryResult> throwingCallback =
                (exactItem, completion) -> {
                ScheduledRequest item = assertInstanceOf(ScheduledRequest.class, exactItem);
                if (completion.status() == DeliveryResult.Status.FAILED) {
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
        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null, 1, 0);
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> rpcFuture = new CompletableFuture<>();
        CountDownLatch invoked = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
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
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);

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
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(response));

        submit(List.of(item), 1L, 100, "test", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.failureCount.get());
        assertTrue(callback.lastError.getMessage().contains("error_code=500"));
    }

    @Test
    void dispatchHandlesMissingAck() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);

        EngineRpcService.EnqueueBatchResponsePB response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(1L)
                        .build(); // no success, no error
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(response));

        submit(List.of(item), 1L, 100, "test", callback);

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.uncertainCount.get());
    }

    @Test
    void responseCallbackFailureIsIsolatedAndNeverReclassifiesOtherItemsAsUncertain()
            throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        ScheduledRequest succeeded = createScheduledRequest(11L, 500, 200, prefillEp);
        ScheduledRequest rejected = createScheduledRequest(12L, 500, 200, prefillEp);
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
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(response));

        CountDownLatch callbacksAttempted = new CountDownLatch(2);
        AtomicInteger successes = new AtomicInteger();
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger uncertain = new AtomicInteger();
        BiConsumer<ScheduledRequest, DeliveryResult> throwingCallback =
                (exactItem, completion) -> {
                if (completion.status() == DeliveryResult.Status.DELIVERED) {
                    successes.incrementAndGet();
                    callbacksAttempted.countDown();
                } else if (completion.status() == DeliveryResult.Status.FAILED) {
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
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(invocation -> {
                    rpcInvoked.countDown();
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);
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
    void logicalCapacityRejectsAndUnusedReservationRestoresCapacity()
            throws Exception {
        dispatcher.shutdown();
        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null, 1, 1);
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
        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null, 1, 1);
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
        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null, 1, 1);
        PrefillEndpoint endpoint = createPrefillEndpoint();
        ScheduledRequest firstItem = createScheduledRequest(1L, 500, 200, endpoint);
        ScheduledRequest secondItem = createScheduledRequest(2L, 500, 200, endpoint);
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
    void preparedSubmissionRejectsInvalidTaskWithoutConsumingPermit() {
        PrefillEndpoint endpoint = createPrefillEndpoint();
        ScheduledRequest item = createScheduledRequest(
                1L, 500, 200, endpoint);
        PreparedSubmission submission = reservePermit();

        assertThrows(IllegalArgumentException.class,
                () -> submit(submission, List.of(), 1L, 100, "empty", callback));
        assertThrows(IllegalArgumentException.class,
                () -> submit(submission, List.of(item), 0L, 100, "id", callback));
        assertThrows(IllegalArgumentException.class,
                () -> submit(submission, List.of(item), 1L, -1, "prediction", callback));
        assertThrows(NullPointerException.class,
                () -> submit(submission, List.of(item), 1L, 100, null, callback));

        submission.close();
        reservePermit().close();
    }

    @Test
    void submittedReservationReleasesCapacityAfterRpcHandoff() throws Exception {
        dispatcher.shutdown();
        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null, 1, 0);
        PrefillEndpoint endpoint = createPrefillEndpoint();
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, endpoint);
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> rpcFuture =
                new CompletableFuture<>();
        CountDownLatch rpcInvoked = new CountDownLatch(1);
        CountDownLatch allowHandoff = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
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
        List<ScheduledRequest> exactItems = List.of(item);
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
    void dispatchForwardsCarriedPriorityIntoGenerateInput() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);
        item.ctx().getRequest().setPriority(60);

        List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
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
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);
        // default Request priority is the no-priority sentinel (0)

        List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
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
        ScheduledRequest item = createScheduledRequest(1L, 500, 200, prefillEp);

        List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
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

    private void submit(List<ScheduledRequest> items,
                        long batchId,
                        long predictedMs,
                        String reason,
                        BiConsumer<ScheduledRequest,
                                DeliveryResult> observer) {
        reservePermit().submitBatch(
                items, batchId, predictedMs, reason, observer);
    }

    private static void submit(
            PreparedSubmission submission,
            List<ScheduledRequest> items,
            long batchId,
            long predictedMs,
            String reason,
            BiConsumer<ScheduledRequest, DeliveryResult> observer) {
        submission.submitBatch(
                items, batchId, predictedMs, reason, observer);
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

    private ScheduledRequest createScheduledRequest(long requestId, long seqLen, long hitCacheLen, PrefillEndpoint prefillEp) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);

        BalanceContext ctx = new BalanceContext();
        ctx.setConfig(config);
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

        return new ScheduledRequest(ctx, new CompletableFuture<>(), null, prefill, null,
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
            implements BiConsumer<ScheduledRequest,
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
                ScheduledRequest exactItem,
                DeliveryResult completion) {
            if (completion.status() == DeliveryResult.Status.DELIVERED) {
                successCount.incrementAndGet();
                successLatch.countDown();
            } else if (completion.status() == DeliveryResult.Status.FAILED) {
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

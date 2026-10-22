package org.flexlb.httpserver;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.google.protobuf.ByteString;
import com.google.protobuf.DescriptorProtos;
import com.google.protobuf.Descriptors;
import com.google.protobuf.DynamicMessage;
import io.grpc.CallOptions;
import io.grpc.ManagedChannel;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.netty.channel.EventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Executor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class FlexlbGrpcForwarderTest {

    @Test
    void missingMasterDoesNotAttemptRpc() {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbGrpcForwarder.MasterForwardResult result =
                await(forwarder.forwardScheduleToMaster(request(1)));

        assertFalse(result.masterFound());
        assertNull(result.response());
        assertEquals("MASTER_NULL", result.failure());
        verify(reporter).reportForwardToMasterResult("LOCAL", "MASTER_NULL");
    }

    @Test
    void missingMasterStateQueryReturnsNullWithoutGuardSideEffects() {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbScheduleProtocol.GetRequestStateResponsePB result =
                forwarder.forwardGetRequestStateToMaster(
                        FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                                .setRequestIdStr(String.valueOf(2L))
                                .build());

        assertNull(result);
        verify(consistency).getMasterHostIpPort();
        verify(consistency).getLocalHostIp();
        verifyNoInteractions(reporter);
    }

    @Test
    void grpcFailureDoesNotSetIndependentDeadlineAndKeepsChannelUntilShutdown() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);
        ManagedChannel channel = mock(ManagedChannel.class);
        ArgumentCaptor<CallOptions> callOptions = ArgumentCaptor.forClass(CallOptions.class);
        when(channel.newCall(any(MethodDescriptor.class), callOptions.capture()))
                .thenThrow(new StatusRuntimeException(Status.DEADLINE_EXCEEDED));
        channels(forwarder).put("10.0.0.2:7003", channel);

        FlexlbGrpcForwarder.MasterForwardResult result =
                await(forwarder.forwardScheduleToMaster(request(4)));

        assertTrue(result.masterFound());
        assertEquals("DEADLINE_EXCEEDED", result.failure());
        assertNull(
                callOptions.getValue().getDeadline(),
                "the inbound Context owns the request deadline");
        assertEquals(Status.Code.DEADLINE_EXCEEDED, Status.fromThrowable(result.error()).getCode());
        assertSame(channel, channels(forwarder).get("10.0.0.2:7003"));
        verify(channel, never()).shutdownNow();
        verify(reporter).reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");

        forwarder.shutdown();
        verify(channel).shutdownNow();
        assertTrue(channels(forwarder).isEmpty());
    }

    @Test
    void grpcFailureLogsStatusDescriptionAndThrowableForScheduleAndCancel() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);
        ManagedChannel channel = mock(ManagedChannel.class);
        String description = "stream reset\nfrom peer";
        int expectedDescriptionLimit = 512;
        description +=
                "x".repeat(expectedDescriptionLimit - 1 - description.length())
                        + "\uD83D\uDE00tail";
        String expectedDescription =
                description.replace('\n', ' ').substring(0, expectedDescriptionLimit - 1);
        StatusRuntimeException failure =
                Status.UNKNOWN
                        .withDescription(description)
                        .withCause(new IllegalStateException("transport closed"))
                        .asRuntimeException();
        when(channel.newCall(any(MethodDescriptor.class), any(CallOptions.class)))
                .thenThrow(failure);
        channels(forwarder).put("10.0.0.2:7003", channel);

        ch.qos.logback.classic.Logger logger =
                (ch.qos.logback.classic.Logger)
                        LoggerFactory.getLogger(org.flexlb.util.Logger.LOGGER_NAME);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
        try {
            assertEquals(
                    "UNKNOWN",
                    await(
                                    forwarder.forwardScheduleToMaster(
                                            FlexlbScheduleProtocol.FlexlbScheduleRequestPB
                                                    .newBuilder()
                                                    .setRequestIdStr("41\r\n{}")
                                                    .build()))
                            .failure());
            assertEquals(
                    "UNKNOWN",
                    forwarder
                            .forwardCancelToMaster(
                                    FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                                            .setRequestIdStr("42")
                                            .build())
                            .toCompletableFuture()
                            .join()
                            .failure());

            List<ILoggingEvent> failures =
                    appender.list.stream()
                            .filter(
                                    event ->
                                            event.getFormattedMessage()
                                                    .contains("event=flexlb_forward_failed"))
                            .toList();
            assertEquals(2, failures.size());
            assertTrue(failures.get(0).getFormattedMessage().contains("operation=schedule"));
            assertTrue(failures.get(0).getFormattedMessage().contains("request_id=41  {}"));
            assertTrue(failures.get(1).getFormattedMessage().contains("operation=cancel"));
            for (ILoggingEvent event : failures) {
                assertTrue(event.getFormattedMessage().contains("status=UNKNOWN"));
                assertEquals(
                        expectedDescription,
                        event.getFormattedMessage().split(" description=", 2)[1]);
                assertFalse(event.getFormattedMessage().contains("\n"));
                assertFalse(event.getFormattedMessage().contains("\r"));
                assertFalse(event.getFormattedMessage().contains("\uD83D"));
                assertFalse(event.getFormattedMessage().contains("\uDE00"));
                assertEquals(
                        StatusRuntimeException.class.getName(),
                        event.getThrowableProxy().getClassName());
                assertEquals(
                        IllegalStateException.class.getName(),
                        event.getThrowableProxy().getCause().getClassName());
            }
            verify(reporter, times(2)).reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");
        } finally {
            logger.detachAppender(appender);
            appender.stop();
            forwarder.shutdown();
        }
    }

    @Test
    void forwardedRequestCannotBeForwardedAgain() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbGrpcForwarder.MasterForwardResult result =
                await(
                        forwarder.forwardScheduleToMaster(
                                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                                        .setRequestIdStr(String.valueOf(5L))
                                        .setForwardHop(1)
                                        .build()));

        assertTrue(result.masterFound());
        assertEquals("FORWARD_HOP_LIMIT", result.failure());
        assertTrue(channels(forwarder).isEmpty());
        verify(reporter).reportForwardToMasterResult("10.0.0.2", "HOP_LIMIT");
    }

    @Test
    void cancellationReceivedBySecondFollowerIsNotRelayedAgain() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbGrpcForwarder.CancelForwardResult result =
                forwarder
                        .forwardCancelToMaster(
                                FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                                        .setRequestIdStr(String.valueOf(15L))
                                        .setForwardHop(1)
                                        .build())
                        .toCompletableFuture()
                        .join();

        assertTrue(result.masterFound());
        assertEquals("FORWARD_HOP_LIMIT", result.failure());
        assertTrue(channels(forwarder).isEmpty());
        verify(reporter).reportForwardToMasterResult("10.0.0.2", "HOP_LIMIT");
    }

    @Test
    void staleSelfLeaderIsRejectedWithoutOpeningChannel() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.3:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbGrpcForwarder.MasterForwardResult result =
                await(forwarder.forwardScheduleToMaster(request(6)));

        assertTrue(result.masterFound());
        assertEquals("SELF_FORWARD_BLOCKED", result.failure());
        assertTrue(channels(forwarder).isEmpty());
        verify(reporter).reportForwardToMasterResult("10.0.0.3", "SELF_TARGET");
    }

    @Test
    void staleSelfLeaderNeverBlocksOnSynchronousRefreshOrOpensChannel() throws Exception {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.3:7001");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbGrpcForwarder.MasterForwardResult result =
                await(forwarder.forwardScheduleToMaster(request(7)));

        assertEquals("SELF_FORWARD_BLOCKED", result.failure());
        verify(consistency, never()).refreshMasterHost(true);
        assertTrue(channels(forwarder).isEmpty());
    }

    @Test
    void shutdownRejectsNewForwardWithoutCreatingOrLeakingAChannel() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        forwarder.shutdown();
        FlexlbGrpcForwarder.MasterForwardResult result =
                await(forwarder.forwardScheduleToMaster(request(14)));

        assertTrue(result.masterFound());
        assertEquals("UNAVAILABLE", result.failure());
        assertTrue(channels(forwarder).isEmpty());
        verify(reporter, times(1)).reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");
    }

    @Test
    void forwardedStateQueryCannotBeForwardedAgain() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbScheduleProtocol.GetRequestStateResponsePB result =
                forwarder.forwardGetRequestStateToMaster(
                        FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                                .setRequestIdStr(String.valueOf(8L))
                                .setForwardHop(1)
                                .build());

        assertNull(result);
        assertTrue(channels(forwarder).isEmpty());
        verify(reporter).reportForwardToMasterResult("10.0.0.2", "HOP_LIMIT");
    }

    @Test
    void stateQueryNeverForwardsToStaleSelfTarget() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.3:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbScheduleProtocol.GetRequestStateResponsePB result =
                forwarder.forwardGetRequestStateToMaster(
                        FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                                .setRequestIdStr(String.valueOf(9L))
                                .build());

        assertNull(result);
        assertTrue(channels(forwarder).isEmpty());
        verify(reporter).reportForwardToMasterResult("10.0.0.3", "SELF_TARGET");
    }

    @Test
    void unifiedGuardGivesHopLimitPrecedenceForBothOperations() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.3:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbGrpcForwarder.MasterForwardResult scheduleResult =
                await(
                        forwarder.forwardScheduleToMaster(
                                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                                        .setRequestIdStr(String.valueOf(12L))
                                        .setForwardHop(1)
                                        .build()));
        FlexlbScheduleProtocol.GetRequestStateResponsePB stateResult =
                forwarder.forwardGetRequestStateToMaster(
                        FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                                .setRequestIdStr(String.valueOf(13L))
                                .setForwardHop(1)
                                .build());

        assertEquals("FORWARD_HOP_LIMIT", scheduleResult.failure());
        assertNull(stateResult);
        assertTrue(channels(forwarder).isEmpty());
        verify(consistency, times(2)).getMasterHostIpPort();
        verify(consistency, times(2)).getLocalHostIp();
        verify(reporter, times(2)).reportForwardToMasterResult("10.0.0.3", "HOP_LIMIT");
        verify(reporter, never()).reportForwardToMasterResult(anyString(), eq("SELF_TARGET"));
    }

    @Test
    void forwardHopSurvivesRelayByAnOlderProtobufSchema() throws Exception {
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB newRequest =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestIdStr(String.valueOf(10L))
                        .setSeqLen(4096)
                        .setForwardHop(1)
                        .build();

        // Model an older FlexLB binary whose descriptor only knows fields 1
        // and 4. Protobuf must retain field 15 in UnknownFieldSet when that
        // process parses and reserializes the request during a rolling upgrade.
        DescriptorProtos.DescriptorProto oldMessage =
                DescriptorProtos.DescriptorProto.newBuilder()
                        .setName("OldFlexlbScheduleRequestPB")
                        .addField(
                                DescriptorProtos.FieldDescriptorProto.newBuilder()
                                        .setName("request_id")
                                        .setNumber(1)
                                        .setType(
                                                DescriptorProtos.FieldDescriptorProto.Type
                                                        .TYPE_INT64)
                                        .setLabel(
                                                DescriptorProtos.FieldDescriptorProto.Label
                                                        .LABEL_OPTIONAL))
                        .addField(
                                DescriptorProtos.FieldDescriptorProto.newBuilder()
                                        .setName("seq_len")
                                        .setNumber(4)
                                        .setType(
                                                DescriptorProtos.FieldDescriptorProto.Type
                                                        .TYPE_INT64)
                                        .setLabel(
                                                DescriptorProtos.FieldDescriptorProto.Label
                                                        .LABEL_OPTIONAL))
                        .build();
        Descriptors.FileDescriptor oldFile =
                Descriptors.FileDescriptor.buildFrom(
                        DescriptorProtos.FileDescriptorProto.newBuilder()
                                .setName("old_flexlb_schedule.proto")
                                .setSyntax("proto3")
                                .addMessageType(oldMessage)
                                .build(),
                        new Descriptors.FileDescriptor[0]);
        DynamicMessage oldRelay =
                DynamicMessage.parseFrom(
                        oldFile.findMessageTypeByName("OldFlexlbScheduleRequestPB"),
                        newRequest.toByteArray());

        assertTrue(oldRelay.getUnknownFields().hasField(15));
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB reparsed =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(oldRelay.toByteArray());
        assertEquals("10", reparsed.getRequestIdStr());
        assertEquals(4096L, reparsed.getSeqLen());
        assertEquals(1, reparsed.getForwardHop());
    }

    @Test
    void stateQueryForwardHopSurvivesOlderProtobufRelay() throws Exception {
        FlexlbScheduleProtocol.GetRequestStateRequestPB newRequest =
                FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                        .setRequestIdStr(String.valueOf(11L))
                        .setBatchId(12L)
                        .setForwardHop(1)
                        .build();
        DescriptorProtos.DescriptorProto oldMessage =
                DescriptorProtos.DescriptorProto.newBuilder()
                        .setName("OldGetRequestStateRequestPB")
                        .addField(
                                DescriptorProtos.FieldDescriptorProto.newBuilder()
                                        .setName("request_id")
                                        .setNumber(1)
                                        .setType(
                                                DescriptorProtos.FieldDescriptorProto.Type
                                                        .TYPE_INT64)
                                        .setLabel(
                                                DescriptorProtos.FieldDescriptorProto.Label
                                                        .LABEL_OPTIONAL))
                        .addField(
                                DescriptorProtos.FieldDescriptorProto.newBuilder()
                                        .setName("batch_id")
                                        .setNumber(2)
                                        .setType(
                                                DescriptorProtos.FieldDescriptorProto.Type
                                                        .TYPE_INT64)
                                        .setLabel(
                                                DescriptorProtos.FieldDescriptorProto.Label
                                                        .LABEL_OPTIONAL))
                        .build();
        Descriptors.FileDescriptor oldFile =
                Descriptors.FileDescriptor.buildFrom(
                        DescriptorProtos.FileDescriptorProto.newBuilder()
                                .setName("old_get_request_state.proto")
                                .setSyntax("proto3")
                                .addMessageType(oldMessage)
                                .build(),
                        new Descriptors.FileDescriptor[0]);
        DynamicMessage oldRelay =
                DynamicMessage.parseFrom(
                        oldFile.findMessageTypeByName("OldGetRequestStateRequestPB"),
                        newRequest.toByteArray());

        assertTrue(oldRelay.getUnknownFields().hasField(3));
        FlexlbScheduleProtocol.GetRequestStateRequestPB reparsed =
                FlexlbScheduleProtocol.GetRequestStateRequestPB.parseFrom(oldRelay.toByteArray());
        assertEquals("11", reparsed.getRequestIdStr());
        assertEquals(12L, reparsed.getBatchId());
        assertEquals(1, reparsed.getForwardHop());
    }

    @Test
    void cancelForwardHopSurvivesOlderProtobufRelay() throws Exception {
        FlexlbScheduleProtocol.FlexlbCancelRequestPB newRequest =
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                        .setRequestIdStr(String.valueOf(16L))
                        .setBatchId(17L)
                        .setReason(
                                FlexlbScheduleProtocol.CancelReasonPB
                                        .CANCEL_REASON_CLIENT_CANCELLED)
                        .setForwardHop(1)
                        .build();
        DescriptorProtos.DescriptorProto oldMessage =
                DescriptorProtos.DescriptorProto.newBuilder()
                        .setName("OldFlexlbCancelRequestPB")
                        .addField(
                                DescriptorProtos.FieldDescriptorProto.newBuilder()
                                        .setName("request_id")
                                        .setNumber(1)
                                        .setType(
                                                DescriptorProtos.FieldDescriptorProto.Type
                                                        .TYPE_INT64)
                                        .setLabel(
                                                DescriptorProtos.FieldDescriptorProto.Label
                                                        .LABEL_OPTIONAL))
                        .addField(
                                DescriptorProtos.FieldDescriptorProto.newBuilder()
                                        .setName("batch_id")
                                        .setNumber(2)
                                        .setType(
                                                DescriptorProtos.FieldDescriptorProto.Type
                                                        .TYPE_INT64)
                                        .setLabel(
                                                DescriptorProtos.FieldDescriptorProto.Label
                                                        .LABEL_OPTIONAL))
                        .addField(
                                DescriptorProtos.FieldDescriptorProto.newBuilder()
                                        .setName("reason")
                                        .setNumber(3)
                                        .setType(
                                                DescriptorProtos.FieldDescriptorProto.Type
                                                        .TYPE_ENUM)
                                        .setTypeName(".old.CancelReasonPB")
                                        .setLabel(
                                                DescriptorProtos.FieldDescriptorProto.Label
                                                        .LABEL_OPTIONAL))
                        .build();
        DescriptorProtos.EnumDescriptorProto oldReason =
                DescriptorProtos.EnumDescriptorProto.newBuilder()
                        .setName("CancelReasonPB")
                        .addValue(
                                DescriptorProtos.EnumValueDescriptorProto.newBuilder()
                                        .setName("CANCEL_REASON_UNSPECIFIED")
                                        .setNumber(0))
                        .addValue(
                                DescriptorProtos.EnumValueDescriptorProto.newBuilder()
                                        .setName("CANCEL_REASON_CLIENT_CANCELLED")
                                        .setNumber(1))
                        .build();
        Descriptors.FileDescriptor oldFile =
                Descriptors.FileDescriptor.buildFrom(
                        DescriptorProtos.FileDescriptorProto.newBuilder()
                                .setName("old_flexlb_cancel.proto")
                                .setPackage("old")
                                .setSyntax("proto3")
                                .addEnumType(oldReason)
                                .addMessageType(oldMessage)
                                .build(),
                        new Descriptors.FileDescriptor[0]);
        DynamicMessage oldRelay =
                DynamicMessage.parseFrom(
                        oldFile.findMessageTypeByName("OldFlexlbCancelRequestPB"),
                        newRequest.toByteArray());

        assertTrue(oldRelay.getUnknownFields().hasField(4));
        FlexlbScheduleProtocol.FlexlbCancelRequestPB reparsed =
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.parseFrom(oldRelay.toByteArray());
        assertEquals("16", reparsed.getRequestIdStr());
        assertEquals(17L, reparsed.getBatchId());
        assertEquals(1, reparsed.getForwardHop());
    }

    /**
     * The follower's forwardScheduleToMaster re-serializes
     * the full inbound Schedule payload into Netty direct buffers with no bound —
     * risking direct-buffer OOM. The in-flight forward byte budget must
     * (a) reject an oversize forward BEFORE any RPC/serialization, (b) return a
     * typed failure the caller maps to a capacity response, and (c) leave no
     * budget residue (charge fully released on every terminal path).
     */
    @Test
    void forwardByteBudgetRejectsOversizeForwardBeforeRpcAndReleasesFully() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);
        // Shrink the budget below the serialized request size via reflection
        // (the production default is 4GiB; the constructor path is not injectable
        // without widening the API, and this test only needs the guard math).
        java.lang.reflect.Field budget =
                FlexlbGrpcForwarder.class.getDeclaredField("maxForwardInflightBytes");
        budget.setAccessible(true);
        budget.setLong(forwarder, 8L);

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB big =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(21L)
                        // any non-trivial payload larger than the 8-byte budget
                        .setGenerateInput(ByteString.copyFrom(new byte[32]))
                        .build();
        assertTrue(
                big.getSerializedSize() > 8,
                "request must exceed the shrunken budget; got " + big.getSerializedSize());

        FlexlbGrpcForwarder.MasterForwardResult rejected =
                await(forwarder.forwardScheduleToMaster(big));
        assertTrue(rejected.masterFound());
        assertEquals("FORWARD_BUDGET_EXHAUSTED", rejected.failure());
        assertTrue(
                channels(forwarder).isEmpty(),
                "no channel/RPC may be created for a budget-rejected forward");
        verify(reporter).reportForwardToMasterResult("10.0.0.2", "FORWARD_BUDGET_EXHAUSTED");

        // Charge release: a failed RPC (channel throws) must return the budget
        // so a later small forward is admitted.
        ManagedChannel channel = mock(ManagedChannel.class);
        when(channel.newCall(any(MethodDescriptor.class), any(CallOptions.class)))
                .thenThrow(new StatusRuntimeException(Status.UNAVAILABLE));
        channels(forwarder).put("10.0.0.2:7003", channel);
        FlexlbGrpcForwarder.MasterForwardResult failed =
                await(forwarder.forwardScheduleToMaster(request(23L)));
        assertTrue(failed.masterFound());
        assertEquals("UNAVAILABLE", failed.failure());

        java.lang.reflect.Field inflight =
                FlexlbGrpcForwarder.class.getDeclaredField("forwardInflightBytes");
        inflight.setAccessible(true);
        assertEquals(
                0L,
                ((java.util.concurrent.atomic.AtomicLong) inflight.get(forwarder)).get(),
                "budget must be fully released after terminal forward outcomes");

        forwarder.shutdown();
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void telemetryErrorCannotOrphanForwardResultOrByteCharge(boolean success) throws Exception {
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        var fatal = new AssertionError("telemetry failed");
        org.mockito.Mockito.doThrow(fatal).when(reporter).reportForwardToMasterResult(anyString(), anyString());
        FlexlbGrpcForwarder forwarder = forwarder(masterAt("10.0.0.2:7001"), reporter);
        ManagedChannel channel = mock(ManagedChannel.class);
        ControllableScheduleCall call = new ControllableScheduleCall();
        when(channel.newCall(any(MethodDescriptor.class), any(CallOptions.class))).thenReturn(call);
        channels(forwarder).put("10.0.0.2:7003", channel);
        var stage = forwarder.forwardScheduleToMaster(request(88001L)).toCompletableFuture();
        var response = FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setSuccess(true).setCode(200).build();
        if (success) { call.listener.onMessage(response); }
        org.junit.jupiter.api.Assertions.assertThrows(AssertionError.class,
                () -> call.listener.onClose(success ? Status.OK : Status.UNAVAILABLE, new io.grpc.Metadata()));
        assertTrue(stage.isDone(), "telemetry Error must not strand the result");
        if (success) { assertSame(response, stage.join().response()); }
        else { assertEquals("UNAVAILABLE", stage.join().failure()); }
        assertEquals(0L, forwardCharge(forwarder));
    }

    @Test
    void preparationErrorReturnsChargedBytesBeforePropagating() throws Exception {
        FlexlbGrpcForwarder forwarder = forwarder(masterAt("10.0.0.2:7001"), mock(EngineHealthReporter.class));
        ManagedChannel channel = mock(ManagedChannel.class);
        channels(forwarder).put("10.0.0.2:7003", channel);
        var request = mock(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.class);
        when(request.getRequestId()).thenReturn(88002L);
        when(request.getSerializedSize()).thenReturn(1024);
        var fatal = new AssertionError("protobuf preparation failed");
        when(request.toBuilder()).thenThrow(fatal);
        assertSame(fatal, org.junit.jupiter.api.Assertions.assertThrows(AssertionError.class,
                () -> forwarder.forwardScheduleToMaster(request)));
        assertEquals(0L, forwardCharge(forwarder));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void telemetryErrorCannotOrphanCancelResult(boolean success) throws Exception {
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        org.mockito.Mockito.doThrow(new AssertionError("cancel telemetry failed"))
                .when(reporter).reportForwardToMasterResult(anyString(), anyString());
        var forwarder = forwarder(masterAt("10.0.0.2:7001"), reporter);
        var channel = mock(ManagedChannel.class);
        var call = new ControllableCancelCall();
        when(channel.newCall(any(MethodDescriptor.class), any(CallOptions.class))).thenReturn(call);
        channels(forwarder).put("10.0.0.2:7003", channel);
        var stage = forwarder.forwardCancelToMaster(FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setRequestId(88003L).build()).toCompletableFuture();
        var response = FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder().setFound(true).build();
        if (success) { call.listener.onMessage(response); }
        org.junit.jupiter.api.Assertions.assertThrows(AssertionError.class,
                () -> call.listener.onClose(success ? Status.OK : Status.UNAVAILABLE, new io.grpc.Metadata()));
        assertTrue(stage.isDone());
        if (success) { assertSame(response, stage.join().response()); }
        else { assertEquals("UNAVAILABLE", stage.join().failure()); }
    }

    private static final class ControllableCancelCall extends io.grpc.ClientCall<
            FlexlbScheduleProtocol.FlexlbCancelRequestPB, FlexlbScheduleProtocol.FlexlbCancelResponsePB> {
        Listener<FlexlbScheduleProtocol.FlexlbCancelResponsePB> listener;
        int cancelCalls;
        @Override public void start(Listener<FlexlbScheduleProtocol.FlexlbCancelResponsePB> responseListener,
                io.grpc.Metadata headers) { listener = responseListener; }
        @Override public void request(int count) { }
        @Override public void cancel(String message, Throwable cause) { cancelCalls++; }
        @Override public void halfClose() { }
        @Override public void sendMessage(FlexlbScheduleProtocol.FlexlbCancelRequestPB request) { }
    }

    @Test
    void asyncStateQueryPreservesHopDeadlineAndExplicitCancellation() throws Exception {
        var config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.mock.TestFlexlbConfigs.create());
        var forwarder = new FlexlbGrpcForwarder(masterAt("10.0.0.2:7001"), config,
                mock(EngineHealthReporter.class), mock(EventLoopGroup.class), mock(Executor.class));
        var channel = mock(ManagedChannel.class);
        var call = new ControllableStateCall();
        var options = ArgumentCaptor.forClass(CallOptions.class);
        when(channel.newCall(any(MethodDescriptor.class), options.capture())).thenReturn(call);
        channels(forwarder).put("10.0.0.2:7003", channel);
        var pending = forwarder.forwardGetRequestStateToMasterAsync(
                FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder().setRequestId(88004L).setBatchId(901L).build()).toCompletableFuture();
        assertFalse(pending.isDone());
        assertEquals(1, call.sent.getForwardHop());
        assertEquals(901L, call.sent.getBatchId());
        assertTrue(options.getValue().getDeadline() != null);
        pending.cancel(true);
        assertEquals(1, call.cancelCalls);
    }

    private static final class ControllableStateCall extends io.grpc.ClientCall<
            FlexlbScheduleProtocol.GetRequestStateRequestPB, FlexlbScheduleProtocol.GetRequestStateResponsePB> {
        FlexlbScheduleProtocol.GetRequestStateRequestPB sent;
        int cancelCalls;
        @Override public void start(Listener<FlexlbScheduleProtocol.GetRequestStateResponsePB> listener,
                io.grpc.Metadata headers) { }
        @Override public void request(int count) { }
        @Override public void cancel(String message, Throwable cause) { cancelCalls++; }
        @Override public void halfClose() { }
        @Override public void sendMessage(FlexlbScheduleProtocol.GetRequestStateRequestPB request) { sent = request; }
    }

    @Test
    void wrapperTimeoutStopsLiveScheduleRpcAndReleasesBudget() throws Exception {
        var forwarder = forwarder(masterAt("10.0.0.2:7001"), mock(EngineHealthReporter.class));
        var channel = mock(ManagedChannel.class);
        var call = new ControllableScheduleCall();
        when(channel.newCall(any(MethodDescriptor.class), any(CallOptions.class))).thenReturn(call);
        channels(forwarder).put("10.0.0.2:7003", channel);
        var result = forwarder.forwardScheduleToMaster(request(88005L)).toCompletableFuture();
        result.completeExceptionally(new java.util.concurrent.TimeoutException("wrapper timeout"));
        assertEquals(1, call.cancelCalls);
        assertEquals(0L, forwardCharge(forwarder));
    }

    @Test
    void externallyCompletedCancelAndStateWrappersStopTheirLiveRpcs() throws Exception {
        var config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.mock.TestFlexlbConfigs.create());
        var forwarder = new FlexlbGrpcForwarder(masterAt("10.0.0.2:7001"), config,
                mock(EngineHealthReporter.class), mock(EventLoopGroup.class), mock(Executor.class));
        var channel = mock(ManagedChannel.class);
        var cancelCall = new ControllableCancelCall();
        var stateCall = new ControllableStateCall();
        when(channel.newCall(any(MethodDescriptor.class), any(CallOptions.class))).thenReturn(cancelCall, stateCall);
        channels(forwarder).put("10.0.0.2:7003", channel);
        var cancel = forwarder.forwardCancelToMaster(FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setRequestId(88006L).build()).toCompletableFuture();
        cancel.completeExceptionally(new java.util.concurrent.TimeoutException("wrapper timeout"));
        var state = forwarder.forwardGetRequestStateToMasterAsync(FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                .setRequestId(88006L).build()).toCompletableFuture();
        state.complete(null);
        assertEquals(1, cancelCall.cancelCalls);
        assertEquals(1, stateCall.cancelCalls);
    }

    private static long forwardCharge(FlexlbGrpcForwarder forwarder) throws Exception {
        Field field = FlexlbGrpcForwarder.class.getDeclaredField("forwardInflightBytes");
        field.setAccessible(true);
        return ((java.util.concurrent.atomic.AtomicLong) field.get(forwarder)).get();
    }

    private static final class ControllableScheduleCall extends io.grpc.ClientCall<
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB,
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB> {
        Listener<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> listener;
        int cancelCalls;
        @Override public void start(Listener<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseListener,
                io.grpc.Metadata headers) { listener = responseListener; }
        @Override public void request(int count) { }
        @Override public void cancel(String message, Throwable cause) { cancelCalls++; }
        @Override public void halfClose() { }
        @Override public void sendMessage(FlexlbScheduleProtocol.FlexlbScheduleRequestPB request) { }
    }

    private static FlexlbGrpcForwarder forwarder(
            LBStatusConsistencyService consistency, EngineHealthReporter reporter) {
        return new FlexlbGrpcForwarder(
                consistency,
                mock(ConfigService.class),
                reporter,
                mock(EventLoopGroup.class),
                mock(Executor.class));
    }

    private static LBStatusConsistencyService masterAt(String address) {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        when(consistency.getMasterHostIpPort()).thenReturn(address);
        return consistency;
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleRequestPB request(long id) {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestIdStr(String.valueOf(id))
                .build();
    }

    private static FlexlbGrpcForwarder.MasterForwardResult await(
            CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> result) {
        return result.toCompletableFuture().join();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, ManagedChannel> channels(FlexlbGrpcForwarder forwarder)
            throws Exception {
        Field field = FlexlbGrpcForwarder.class.getDeclaredField("channels");
        field.setAccessible(true);
        return (Map<String, ManagedChannel>) field.get(forwarder);
    }
}

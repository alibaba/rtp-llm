package org.flexlb.httpserver;

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

import java.lang.reflect.Field;
import java.util.Map;
import java.util.concurrent.Executor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class FlexlbGrpcForwarderTest {

    @Test
    void missingMasterIsTheOnlyLocalFallbackCase() {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbGrpcForwarder.MasterForwardResult result =
                await(forwarder.forwardScheduleToMaster(request(1L)));

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
                                .setRequestId(2L)
                                .build());

        assertNull(result);
        verify(consistency).getMasterHostIpPort();
        verify(consistency).getLocalHostIp();
        verifyNoInteractions(reporter);
    }

    @Test
    void grpcFailureDoesNotSetIndependentDeadlineAndKeepsChannelUntilShutdown()
            throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);
        ManagedChannel channel = mock(ManagedChannel.class);
        ArgumentCaptor<CallOptions> callOptions = ArgumentCaptor.forClass(CallOptions.class);
        when(channel.newCall(any(MethodDescriptor.class), callOptions.capture()))
                .thenThrow(new StatusRuntimeException(Status.DEADLINE_EXCEEDED));
        channels(forwarder).put("10.0.0.2:7003", channel);

        FlexlbGrpcForwarder.MasterForwardResult result =
                await(forwarder.forwardScheduleToMaster(request(4L)));

        assertTrue(result.masterFound());
        assertEquals("DEADLINE_EXCEEDED", result.failure());
        assertNull(callOptions.getValue().getDeadline());
        assertSame(channel, channels(forwarder).get("10.0.0.2:7003"));
        verify(channel, never()).shutdownNow();
        verify(reporter).reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");

        forwarder.shutdown();
        verify(channel).shutdownNow();
        assertTrue(channels(forwarder).isEmpty());
    }

    @Test
    void forwardedRequestCannotBeForwardedAgain() throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbGrpcForwarder.MasterForwardResult result = await(
                forwarder.forwardScheduleToMaster(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(5L)
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
                forwarder.forwardCancelToMaster(
                                FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                                        .setRequestId(15L)
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
                await(forwarder.forwardScheduleToMaster(request(6L)));

        assertTrue(result.masterFound());
        assertEquals("SELF_FORWARD_BLOCKED", result.failure());
        assertTrue(channels(forwarder).isEmpty());
        verify(reporter).reportForwardToMasterResult("10.0.0.3", "SELF_TARGET");
    }

    @Test
    void staleSelfLeaderNeverBlocksOnSynchronousRefreshOrOpensChannel()
            throws Exception {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.3:7001");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbGrpcForwarder.MasterForwardResult result =
                await(forwarder.forwardScheduleToMaster(request(7L)));

        assertEquals("SELF_FORWARD_BLOCKED", result.failure());
        verify(consistency, never()).refreshMasterHost(true);
        assertTrue(channels(forwarder).isEmpty());
    }

    @Test
    void shutdownRejectsNewForwardWithoutCreatingOrLeakingAChannel()
            throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.3");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        forwarder.shutdown();
        FlexlbGrpcForwarder.MasterForwardResult result =
                await(forwarder.forwardScheduleToMaster(request(14L)));

        assertTrue(result.masterFound());
        assertEquals("UNAVAILABLE", result.failure());
        assertTrue(channels(forwarder).isEmpty());
        verify(reporter, times(1))
                .reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");
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
                                .setRequestId(8L)
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
                                .setRequestId(9L)
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
                await(forwarder.forwardScheduleToMaster(
                        FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                                .setRequestId(12L)
                                .setForwardHop(1)
                                .build()));
        FlexlbScheduleProtocol.GetRequestStateResponsePB stateResult =
                forwarder.forwardGetRequestStateToMaster(
                        FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                                .setRequestId(13L)
                                .setForwardHop(1)
                                .build());

        assertEquals("FORWARD_HOP_LIMIT", scheduleResult.failure());
        assertNull(stateResult);
        assertTrue(channels(forwarder).isEmpty());
        verify(consistency, times(2)).getMasterHostIpPort();
        verify(consistency, times(2)).getLocalHostIp();
        verify(reporter, times(2))
                .reportForwardToMasterResult("10.0.0.3", "HOP_LIMIT");
        verify(reporter, never())
                .reportForwardToMasterResult(anyString(), eq("SELF_TARGET"));
    }

    @Test
    void forwardHopSurvivesRelayByAnOlderProtobufSchema() throws Exception {
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB newRequest =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId(10L)
                        .setSeqLen(4096)
                        .setForwardHop(1)
                        .setSessionRoutingHint(FlexlbScheduleProtocol.SessionRoutingHintPB.newBuilder()
                                .setSchemaVersion(1)
                                .setSessionId("isess_v1_relay")
                                .setState(FlexlbScheduleProtocol.SessionStatePB.ESTABLISHED))
                        .build();

        // Model an older FlexLB binary whose descriptor only knows fields 1
        // and 4. Protobuf must retain field 15 in UnknownFieldSet when that
        // process parses and reserializes the request during a rolling upgrade.
        DescriptorProtos.DescriptorProto oldMessage =
                DescriptorProtos.DescriptorProto.newBuilder()
                        .setName("OldFlexlbScheduleRequestPB")
                        .addField(DescriptorProtos.FieldDescriptorProto.newBuilder()
                                .setName("request_id")
                                .setNumber(1)
                                .setType(DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64)
                                .setLabel(DescriptorProtos.FieldDescriptorProto.Label.LABEL_OPTIONAL))
                        .addField(DescriptorProtos.FieldDescriptorProto.newBuilder()
                                .setName("seq_len")
                                .setNumber(4)
                                .setType(DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64)
                                .setLabel(DescriptorProtos.FieldDescriptorProto.Label.LABEL_OPTIONAL))
                        .build();
        Descriptors.FileDescriptor oldFile = Descriptors.FileDescriptor.buildFrom(
                DescriptorProtos.FileDescriptorProto.newBuilder()
                        .setName("old_flexlb_schedule.proto")
                        .setSyntax("proto3")
                        .addMessageType(oldMessage)
                        .build(),
                new Descriptors.FileDescriptor[0]);
        DynamicMessage oldRelay = DynamicMessage.parseFrom(
                oldFile.findMessageTypeByName("OldFlexlbScheduleRequestPB"),
                newRequest.toByteArray());

        assertTrue(oldRelay.getUnknownFields().hasField(15));
        assertTrue(oldRelay.getUnknownFields().hasField(16));
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB reparsed =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(
                        oldRelay.toByteArray());
        assertEquals(10L, reparsed.getRequestId());
        assertEquals(4096L, reparsed.getSeqLen());
        assertEquals(1, reparsed.getForwardHop());
        assertEquals("isess_v1_relay", reparsed.getSessionRoutingHint().getSessionId());
    }

    @Test
    void stateQueryForwardHopSurvivesOlderProtobufRelay() throws Exception {
        FlexlbScheduleProtocol.GetRequestStateRequestPB newRequest =
                FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                        .setRequestId(11L)
                        .setBatchId(12L)
                        .setForwardHop(1)
                        .build();
        DescriptorProtos.DescriptorProto oldMessage =
                DescriptorProtos.DescriptorProto.newBuilder()
                        .setName("OldGetRequestStateRequestPB")
                        .addField(DescriptorProtos.FieldDescriptorProto.newBuilder()
                                .setName("request_id")
                                .setNumber(1)
                                .setType(DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64)
                                .setLabel(DescriptorProtos.FieldDescriptorProto.Label.LABEL_OPTIONAL))
                        .addField(DescriptorProtos.FieldDescriptorProto.newBuilder()
                                .setName("batch_id")
                                .setNumber(2)
                                .setType(DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64)
                                .setLabel(DescriptorProtos.FieldDescriptorProto.Label.LABEL_OPTIONAL))
                        .build();
        Descriptors.FileDescriptor oldFile = Descriptors.FileDescriptor.buildFrom(
                DescriptorProtos.FileDescriptorProto.newBuilder()
                        .setName("old_get_request_state.proto")
                        .setSyntax("proto3")
                        .addMessageType(oldMessage)
                        .build(),
                new Descriptors.FileDescriptor[0]);
        DynamicMessage oldRelay = DynamicMessage.parseFrom(
                oldFile.findMessageTypeByName("OldGetRequestStateRequestPB"),
                newRequest.toByteArray());

        assertTrue(oldRelay.getUnknownFields().hasField(3));
        FlexlbScheduleProtocol.GetRequestStateRequestPB reparsed =
                FlexlbScheduleProtocol.GetRequestStateRequestPB.parseFrom(
                        oldRelay.toByteArray());
        assertEquals(11L, reparsed.getRequestId());
        assertEquals(12L, reparsed.getBatchId());
        assertEquals(1, reparsed.getForwardHop());
    }

    @Test
    void cancelForwardHopSurvivesOlderProtobufRelay() throws Exception {
        FlexlbScheduleProtocol.FlexlbCancelRequestPB newRequest =
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                        .setRequestId(16L)
                        .setBatchId(17L)
                        .setReason(FlexlbScheduleProtocol.CancelReasonPB
                                .CANCEL_REASON_CLIENT_CANCELLED)
                        .setForwardHop(1)
                        .build();
        DescriptorProtos.DescriptorProto oldMessage =
                DescriptorProtos.DescriptorProto.newBuilder()
                        .setName("OldFlexlbCancelRequestPB")
                        .addField(DescriptorProtos.FieldDescriptorProto.newBuilder()
                                .setName("request_id")
                                .setNumber(1)
                                .setType(DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64)
                                .setLabel(DescriptorProtos.FieldDescriptorProto.Label.LABEL_OPTIONAL))
                        .addField(DescriptorProtos.FieldDescriptorProto.newBuilder()
                                .setName("batch_id")
                                .setNumber(2)
                                .setType(DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64)
                                .setLabel(DescriptorProtos.FieldDescriptorProto.Label.LABEL_OPTIONAL))
                        .addField(DescriptorProtos.FieldDescriptorProto.newBuilder()
                                .setName("reason")
                                .setNumber(3)
                                .setType(DescriptorProtos.FieldDescriptorProto.Type.TYPE_ENUM)
                                .setTypeName(".old.CancelReasonPB")
                                .setLabel(DescriptorProtos.FieldDescriptorProto.Label.LABEL_OPTIONAL))
                        .build();
        DescriptorProtos.EnumDescriptorProto oldReason =
                DescriptorProtos.EnumDescriptorProto.newBuilder()
                        .setName("CancelReasonPB")
                        .addValue(DescriptorProtos.EnumValueDescriptorProto.newBuilder()
                                .setName("CANCEL_REASON_UNSPECIFIED")
                                .setNumber(0))
                        .addValue(DescriptorProtos.EnumValueDescriptorProto.newBuilder()
                                .setName("CANCEL_REASON_CLIENT_CANCELLED")
                                .setNumber(1))
                        .build();
        Descriptors.FileDescriptor oldFile = Descriptors.FileDescriptor.buildFrom(
                DescriptorProtos.FileDescriptorProto.newBuilder()
                        .setName("old_flexlb_cancel.proto")
                        .setPackage("old")
                        .setSyntax("proto3")
                        .addEnumType(oldReason)
                        .addMessageType(oldMessage)
                        .build(),
                new Descriptors.FileDescriptor[0]);
        DynamicMessage oldRelay = DynamicMessage.parseFrom(
                oldFile.findMessageTypeByName("OldFlexlbCancelRequestPB"),
                newRequest.toByteArray());

        assertTrue(oldRelay.getUnknownFields().hasField(4));
        FlexlbScheduleProtocol.FlexlbCancelRequestPB reparsed =
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.parseFrom(
                        oldRelay.toByteArray());
        assertEquals(16L, reparsed.getRequestId());
        assertEquals(17L, reparsed.getBatchId());
        assertEquals(1, reparsed.getForwardHop());
    }

    private static FlexlbGrpcForwarder forwarder(
            LBStatusConsistencyService consistency,
            EngineHealthReporter reporter) {
        return new FlexlbGrpcForwarder(consistency, mock(ConfigService.class), reporter,
                mock(EventLoopGroup.class), mock(Executor.class));
    }

    private static LBStatusConsistencyService masterAt(String address) {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        when(consistency.getMasterHostIpPort()).thenReturn(address);
        return consistency;
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleRequestPB request(long id) {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(id)
                .build();
    }

    private static FlexlbGrpcForwarder.MasterForwardResult await(
            java.util.concurrent.CompletionStage<FlexlbGrpcForwarder.MasterForwardResult> result) {
        return result.toCompletableFuture().join();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, ManagedChannel> channels(
            FlexlbGrpcForwarder forwarder) throws Exception {
        Field field = FlexlbGrpcForwarder.class.getDeclaredField("channels");
        field.setAccessible(true);
        return (Map<String, ManagedChannel>) field.get(forwarder);
    }
}

package org.flexlb.engine.grpc;

import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.Descriptors;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlexlbScheduleProtocolTest {

    @Test
    void scheduleContractIsSeparatedButKeepsOriginalWireServiceName() {
        assertNull(EngineRpcService.getDescriptor().findMessageTypeByName("FlexlbScheduleRequestPB"));
        assertNull(EngineRpcService.getDescriptor().findServiceByName("FlexlbService"));

        var service = FlexlbScheduleProtocol.getDescriptor().findServiceByName("FlexlbService");
        assertEquals("FlexlbService", service.getFullName());
        assertEquals("Schedule", service.findMethodByName("Schedule").getName());
        assertEquals("GetRequestState", service.findMethodByName("GetRequestState").getName());
        // Task40 explicitly reverses the P0-2 guard: GenerateInputPB field 10
        // is now the per-request priority forwarded to the engine.
        Descriptors.FieldDescriptor priority =
                EngineRpcService.GenerateInputPB.getDescriptor().findFieldByNumber(10);
        assertEquals("priority", priority.getName());
        assertEquals(Descriptors.FieldDescriptor.Type.INT32, priority.getType());
        // AutoTPM Cancel: field 14 is the typed weak-ACK completion progress.
        Descriptors.FieldDescriptor preemptionProgress =
                EngineRpcService.TaskInfoPB.getDescriptor().findFieldByNumber(14);
        assertNotNull(preemptionProgress);
        assertEquals("priority_preemption_progress", preemptionProgress.getName());
        assertEquals(Descriptors.FieldDescriptor.Type.ENUM, preemptionProgress.getType());
        assertEquals(EngineRpcService.PriorityPreemptionProgressPB.PRIORITY_PREEMPTION_NONE,
                EngineRpcService.TaskInfoPB.getDefaultInstance()
                        .getPriorityPreemptionProgress());
        Descriptors.FieldDescriptor taskPriority =
                EngineRpcService.TaskInfoPB.getDescriptor().findFieldByNumber(15);
        assertEquals("priority", taskPriority.getName());
        assertEquals(Descriptors.FieldDescriptor.Type.INT32, taskPriority.getType());
        assertEquals(Descriptors.FieldDescriptor.Type.STRING,
                EngineRpcService.WorkerStatusPB.getDescriptor().findFieldByNumber(1).getType());
        assertNull(FlexlbScheduleProtocol.FlexlbServerStatusPB.getDescriptor().findFieldByNumber(5));
    }

    @Test
    void engineIndexDistinguishesAbsentFromExplicitZeroOnTheWire() throws Exception {
        var descriptor = FlexlbScheduleProtocol.FlexlbServerStatusPB.getDescriptor();
        var field = descriptor.findFieldByName("engine_index");
        assertNotNull(field);
        assertEquals(9, field.getNumber());
        assertEquals(Descriptors.FieldDescriptor.Type.INT32, field.getType());
        assertTrue(field.hasPresence());
        assertNull(descriptor.findFieldByNumber(5));

        var absent = FlexlbScheduleProtocol.FlexlbServerStatusPB.parseFrom(new byte[0]);
        assertFalse(absent.hasField(field));
        for (int index : new int[]{0, 1}) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            CodedOutputStream coded = CodedOutputStream.newInstance(output);
            coded.writeInt32(9, index);
            coded.flush();
            var parsed = FlexlbScheduleProtocol.FlexlbServerStatusPB.parseFrom(output.toByteArray());
            assertTrue(parsed.hasField(field));
            assertEquals(index, parsed.getField(field));
            assertArrayEquals(output.toByteArray(), parsed.toByteArray());
        }
    }

    @Test
    void historicalEmbeddedGenerateInputWireParsesAsOpaquePayload() throws Exception {
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(123L)
                .addTokenIds(1)
                .addTokenIds(2)
                .build();

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        CodedOutputStream coded = CodedOutputStream.newInstance(output);
        coded.writeInt64(1, 123L);
        coded.writeByteArray(2, input.toByteArray());
        coded.flush();

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB parsed =
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(output.toByteArray());

        assertEquals("123", RequestId.parse(parsed));
        assertArrayEquals(input.toByteArray(), parsed.getGenerateInput().toByteArray());
    }

    @Test
    void scheduleRolesAndRequestPhaseHaveStableWireFields() throws Exception {
        assertEquals(0, FlexlbScheduleProtocol.ScheduleRolePB.SCHEDULE_ROLE_UNSPECIFIED.getNumber());
        assertEquals(1, FlexlbScheduleProtocol.ScheduleRolePB.SCHEDULE_ROLE_ENCODER.getNumber());
        assertEquals(2, FlexlbScheduleProtocol.ScheduleRolePB.SCHEDULE_ROLE_PREFILL.getNumber());
        assertEquals(3, FlexlbScheduleProtocol.ScheduleRolePB.SCHEDULE_ROLE_DECODE.getNumber());
        assertEquals(4, FlexlbScheduleProtocol.ScheduleRolePB.SCHEDULE_ROLE_PDFUSION.getNumber());
        var schedule = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId("request-1")
                .addScheduleRoles(FlexlbScheduleProtocol.ScheduleRolePB.SCHEDULE_ROLE_ENCODER)
                .addScheduleRoles(FlexlbScheduleProtocol.ScheduleRolePB.SCHEDULE_ROLE_PREFILL)
                .build();
        assertEquals(17, schedule.getDescriptorForType()
                .findFieldByName("schedule_roles").getNumber());
        assertEquals(schedule.getScheduleRolesList(),
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(schedule.toByteArray())
                        .getScheduleRolesList());

        var cancel = FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setPhase(FlexlbScheduleProtocol.RequestPhasePB.REQUEST_PHASE_ENCODER)
                .build();
        assertEquals(5, cancel.getDescriptorForType().findFieldByName("phase").getNumber());
        assertEquals(cancel.getPhase(), FlexlbScheduleProtocol.FlexlbCancelRequestPB
                .parseFrom(cancel.toByteArray()).getPhase());
        assertEquals(FlexlbScheduleProtocol.RequestPhasePB.REQUEST_PHASE_UNSPECIFIED,
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.getDefaultInstance().getPhase());

        var query = FlexlbScheduleProtocol.GetRequestStateRequestPB.newBuilder()
                .setPhase(FlexlbScheduleProtocol.RequestPhasePB.REQUEST_PHASE_GENERATION)
                .build();
        assertEquals(4, query.getDescriptorForType().findFieldByName("phase").getNumber());
        assertEquals(query.getPhase(), FlexlbScheduleProtocol.GetRequestStateRequestPB
                .parseFrom(query.toByteArray()).getPhase());
    }

    @Test
    void encoderCacheHitLengthDistinguishesOmittedFromKnownZero() throws Exception {
        var omitted = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setSeqLen(1000)
                .build();
        var knownMiss = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setSeqLen(1000)
                .setEncoderCacheHitLen(0)
                .build();
        var partialHit = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setSeqLen(1000)
                .setEncoderCacheHitLen(200)
                .build();

        assertFalse(omitted.hasEncoderCacheHitLen());
        assertTrue(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(knownMiss.toByteArray())
                .hasEncoderCacheHitLen());
        assertEquals(200, FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(partialHit.toByteArray())
                .getEncoderCacheHitLen());
        assertEquals(18, partialHit.getDescriptorForType()
                .findFieldByName("encoder_cache_hit_len").getNumber());
    }

}

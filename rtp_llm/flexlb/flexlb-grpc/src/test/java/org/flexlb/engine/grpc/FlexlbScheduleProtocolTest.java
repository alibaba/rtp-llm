package org.flexlb.engine.grpc;

import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.Descriptors;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

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

        assertEquals(123L, parsed.getRequestId());
        assertArrayEquals(input.toByteArray(), parsed.getGenerateInput().toByteArray());
    }

}

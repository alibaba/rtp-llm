package org.flexlb.engine.grpc;

import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.Descriptors;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class FlexlbScheduleProtocolTest {

    @Test
    void scheduleContractIsSeparatedButKeepsOriginalWireServiceName() {
        assertNull(EngineRpcService.getDescriptor().findMessageTypeByName("FlexlbScheduleRequestPB"));
        assertNull(EngineRpcService.getDescriptor().findServiceByName("FlexlbService"));

        var service = FlexlbScheduleProtocol.getDescriptor().findServiceByName("FlexlbService");
        assertEquals("FlexlbService", service.getFullName());
        assertEquals("Schedule", service.getMethods().get(0).getName());
        assertEquals("Cancel", service.getMethods().get(1).getName());
        assertEquals("GetRequestState", service.getMethods().get(2).getName());
        assertNull(EngineRpcService.GenerateInputPB.getDescriptor().findFieldByNumber(10));
        assertNull(EngineRpcService.TaskInfoPB.getDescriptor().findFieldByNumber(14));
        assertNull(EngineRpcService.TaskInfoPB.getDescriptor().findFieldByNumber(15));
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

    @Test
    void cancelRemainsWireCompatibleWithOriginalRequestAndEmptyResponse() throws Exception {
        EngineRpcService.CancelRequestPB oldRequest = EngineRpcService.CancelRequestPB.newBuilder()
                .setRequestId(456L)
                .build();

        FlexlbScheduleProtocol.FlexlbCancelRequestPB parsedRequest =
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.parseFrom(oldRequest.toByteArray());
        assertEquals(456L, parsedRequest.getRequestId());

        FlexlbScheduleProtocol.FlexlbCancelResponsePB newResponse =
                FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder()
                        .setFound(true)
                        .build();
        EngineRpcService.EmptyPB.parseFrom(newResponse.toByteArray());
    }
}

package org.flexlb.engine.grpc;

import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.Descriptors;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RequestIdTest {
    @Test
    void scheduleContractsHaveOneStringIdAtOriginalTag() {
        for (var descriptor : List.of(
                FlexlbScheduleProtocol.FlexlbScheduleRequestPB.getDescriptor(),
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.getDescriptor(),
                FlexlbScheduleProtocol.GetRequestStateRequestPB.getDescriptor(),
                FlexlbScheduleProtocol.RequestLifecyclePB.getDescriptor())) {
            assertEquals("request_id", descriptor.findFieldByNumber(1).getName());
            assertEquals(Descriptors.FieldDescriptor.Type.STRING, descriptor.findFieldByNumber(1).getType());
            assertEquals(1, descriptor.getFields().stream().filter(field -> field.getName().startsWith("request_id")).count());
        }
        assertEquals(Descriptors.FieldDescriptor.Type.INT64,
                EngineRpcService.TaskInfoPB.getDescriptor()
                        .findFieldByNumber(1).getType());
    }

    @Test
    void preservesOriginalStrings() {
        for (String id : new String[]{"req-abc-001", "00123", "123", "0", "9223372036854775808"}) {
            assertEquals(id, RequestId.parse(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder().setRequestId(id)));
        }
    }

    @Test
    void readsOldIntegerEncodingForScheduleCancelAndState() throws Exception {
        for (long id : new long[]{123, Long.MAX_VALUE, Long.MIN_VALUE, 0}) {
            byte[] wire = oldIntegerId(id);
            assertEquals(Long.toString(id), RequestId.parse(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(wire)));
            assertEquals(Long.toString(id), RequestId.parse(FlexlbScheduleProtocol.FlexlbCancelRequestPB.parseFrom(wire)));
            assertEquals(Long.toString(id), RequestId.parse(FlexlbScheduleProtocol.GetRequestStateRequestPB.parseFrom(wire)));
        }
    }

    @Test
    void prefersStringOverOldIntegerAndPreservesForwardedId() throws Exception {
        var request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(oldIntegerId(123));
        var forwarded = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.parseFrom(request.toBuilder().setForwardHop(1).build().toByteArray());
        assertEquals("123", RequestId.parse(forwarded));
        assertEquals("req-abc", RequestId.parse(forwarded.toBuilder().setRequestId("req-abc")));
    }

    @Test
    void rejectsMissingAndBlankIdsWithoutDefaultingToZero() {
        assertThrows(IllegalArgumentException.class, () -> RequestId.parse(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.getDefaultInstance()));
    }

    @Test
    void preservesWorkerStatusFieldsWithoutRemappingTaskLayout() throws Exception {
        var oldTask = EngineRpcService.TaskInfoPB.newBuilder().setRequestId(123)
                .setBatchId(99).setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).setWaitingEnteredTimeMs(1700000000123L).build();
        var current = EngineRpcService.TaskInfoPB.newBuilder().setRequestId(456).setBatchId(42).build();
        var status = EngineRpcService.WorkerStatusPB.parseFrom(EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(oldTask).addFinishedTaskList(current).build().toByteArray());
        assertEquals("123", RequestId.parse(status.getRunningTaskInfo(0)));
        assertEquals("456", RequestId.parse(status.getFinishedTaskList(0)));
        assertEquals(oldTask, status.getRunningTaskInfo(0));
        assertEquals(current, status.getFinishedTaskList(0));
    }

    @Test
    void integerMessageParsersReturnLongStrings() throws Exception {
        for (long id : new long[]{123, Long.MAX_VALUE, Long.MIN_VALUE, 0}) {
            byte[] wire = oldIntegerId(id);
            var input = EngineRpcService.GenerateInputPB.parseFrom(wire);
            assertEquals(Long.toString(id), RequestId.parse(input));
            assertEquals(Long.toString(id), RequestId.parse(input.toBuilder()));
            assertEquals(Long.toString(id), RequestId.parse(EngineRpcService.EnqueueBatchSuccessPB.parseFrom(wire)));
            assertEquals(Long.toString(id), RequestId.parse(EngineRpcService.EnqueueBatchErrorPB.parseFrom(wire)));
        }
    }

    private static byte[] oldIntegerId(long id) throws Exception {
        var bytes = new ByteArrayOutputStream();
        var output = CodedOutputStream.newInstance(bytes);
        output.writeInt64(1, id);
        output.flush();
        return bytes.toByteArray();
    }
}

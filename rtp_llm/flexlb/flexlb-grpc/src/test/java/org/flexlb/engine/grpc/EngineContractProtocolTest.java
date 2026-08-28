package org.flexlb.engine.grpc;

import com.google.protobuf.DescriptorProtos;
import com.google.protobuf.Descriptors;
import com.google.protobuf.DynamicMessage;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stage-0 engine-contract wire compatibility: TaskInfoPB.kv_tokens (field 16)
 * and WorkerStatusPB.running_detail_truncated (field 23) are additive-only.
 *
 * <p>Verified properties:
 * <ul>
 *   <li>both fields exist on the current descriptor with the agreed types</li>
 *   <li>a current writer's payload is still fully readable by a legacy
 *       descriptor (new fields land in unknown fields, existing fields are
 *       untouched)</li>
 *   <li>a legacy writer's payload reads back with default values for the new
 *       fields (the mixed-fleet default path: a legacy engine that never sets
 *       them is indistinguishable from an engine reporting zero usage)</li>
 * </ul>
 */
class EngineContractProtocolTest {

    @Test
    void contractFieldsExistOnCurrentDescriptor() {
        Descriptors.Descriptor task = EngineRpcService.TaskInfoPB.getDescriptor();
        Descriptors.Descriptor worker = EngineRpcService.WorkerStatusPB.getDescriptor();

        // Guard first: without these the assertEquals below would die on an
        // NPE from a null FieldDescriptor and hide the actual intent (field
        // missing from the descriptor).
        assertNotNull(task.findFieldByNumber(16), "kv_tokens field 16 must exist on TaskInfoPB");
        assertNotNull(worker.findFieldByNumber(23), "running_detail_truncated field 23 must exist on WorkerStatusPB");
        assertEquals(Descriptors.FieldDescriptor.Type.INT64,
                task.findFieldByNumber(16).getType());
        assertEquals("kv_tokens", task.findFieldByNumber(16).getName());
        assertEquals(Descriptors.FieldDescriptor.Type.BOOL,
                worker.findFieldByNumber(23).getType());
        assertEquals("running_detail_truncated", worker.findFieldByNumber(23).getName());
    }

    @Test
    void contractPayloadIsReadableByLegacyDescriptor() throws Exception {
        Descriptors.Descriptor legacy = legacyDescriptor();
        Descriptors.Descriptor legacyTask = legacy.findFieldByNumber(3).getMessageType();

        EngineRpcService.TaskInfoPB running = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(42L)
                .setInputLength(1024L)
                .setIterateCount(3L)
                .setBatchId(7L)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                .setKvTokens(2048L)
                .build();
        EngineRpcService.WorkerStatusPB payload = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("RoleType.PREFILL")
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setLatestFinishedVersion(99L)
                .setRunningDetailTruncated(true)
                .addRunningTaskInfo(running)
                .build();

        DynamicMessage legacyReader = DynamicMessage.parseFrom(legacy, payload.toByteArray());
        assertEquals("RoleType.PREFILL", legacyReader.getField(legacy.findFieldByNumber(1)));
        assertEquals(99L, legacyReader.getField(legacy.findFieldByNumber(15)));
        DynamicMessage legacyRunning = (DynamicMessage) ((java.util.List<?>)
                legacyReader.getField(legacy.findFieldByNumber(3))).get(0);
        assertEquals(42L, legacyRunning.getField(legacyTask.findFieldByNumber(1)));
        assertEquals(1024L, legacyRunning.getField(legacyTask.findFieldByNumber(4)));
        assertEquals(3L, legacyRunning.getField(legacyTask.findFieldByNumber(6)));
        assertEquals(7L, legacyRunning.getField(legacyTask.findFieldByNumber(11)));
        assertEquals(Descriptors.EnumValueDescriptor.class,
                legacyRunning.getField(legacyTask.findFieldByNumber(12)).getClass());
        // Legacy reader silently preserves the unknown new fields on re-serialization.
        assertEquals(2048L, EngineRpcService.TaskInfoPB.parseFrom(
                legacyRunning.toByteArray()).getKvTokens());
    }

    @Test
    void legacyPayloadReadsContractDefaults() throws Exception {
        Descriptors.Descriptor legacy = legacyDescriptor();
        Descriptors.Descriptor legacyTask = legacy.findFieldByNumber(3).getMessageType();

        DynamicMessage legacyRunning = DynamicMessage.newBuilder(legacyTask)
                .setField(legacyTask.findFieldByNumber(1), 7L)
                .setField(legacyTask.findFieldByNumber(4), 512L)
                .setField(legacyTask.findFieldByNumber(6), 2L)
                .setField(legacyTask.findFieldByNumber(9), true)
                .setField(legacyTask.findFieldByNumber(11), 4L)
                .setField(legacyTask.findFieldByNumber(12),
                        legacyTask.findEnumTypeByName("TaskPhase").findValueByName("TASK_PHASE_PENDING"))
                .build();
        DynamicMessage legacyFinished = DynamicMessage.newBuilder(legacyTask)
                .setField(legacyTask.findFieldByNumber(1), 9L)
                .setField(legacyTask.findFieldByNumber(4), 256L)
                .build();
        DynamicMessage legacyWriter = DynamicMessage.newBuilder(legacy)
                .setField(legacy.findFieldByNumber(1), "RoleType.DECODE")
                .setField(legacy.findFieldByNumber(15), 40L)
                .addRepeatedField(legacy.findFieldByNumber(3), legacyRunning)
                .addRepeatedField(legacy.findFieldByNumber(4), legacyFinished)
                .build();

        EngineRpcService.WorkerStatusPB parsed =
                EngineRpcService.WorkerStatusPB.parseFrom(legacyWriter.toByteArray());

        // Mixed-fleet default path: absent contract fields read as zero usage
        // and a complete running detail.
        assertFalse(parsed.getRunningDetailTruncated());
        assertEquals(0L, parsed.getRunningTaskInfo(0).getKvTokens());
        assertEquals(0L, parsed.getFinishedTaskList(0).getKvTokens());
        // Existing fields keep their exact semantics.
        assertEquals("RoleType.DECODE", parsed.getRole());
        assertEquals(40L, parsed.getLatestFinishedVersion());
        assertEquals(7L, parsed.getRunningTaskInfo(0).getRequestId());
        assertEquals(512L, parsed.getRunningTaskInfo(0).getInputLength());
        assertEquals(2L, parsed.getRunningTaskInfo(0).getIterateCount());
        assertTrue(parsed.getRunningTaskInfo(0).getIsWaiting());
        assertEquals(4L, parsed.getRunningTaskInfo(0).getBatchId());
        assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_PENDING,
                parsed.getRunningTaskInfo(0).getPhase());
        assertEquals(9L, parsed.getFinishedTaskList(0).getRequestId());
    }

    @Test
    void zeroValuedContractFieldsStayOffTheWire() throws Exception {
        // A new-shape writer that reports no usage produces the exact same
        // bytes a legacy writer would: proto3 omits default-valued fields.
        EngineRpcService.TaskInfoPB task = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(11L)
                .setInputLength(64L)
                .setKvTokens(0L)
                .build();
        EngineRpcService.WorkerStatusPB worker = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("RoleType.PREFILL")
                .addRunningTaskInfo(task)
                .setRunningDetailTruncated(false)
                .build();
        Descriptors.Descriptor legacy = legacyDescriptor();
        DynamicMessage legacyReader = DynamicMessage.parseFrom(legacy, worker.toByteArray());
        assertTrue(legacyReader.getUnknownFields().asMap().isEmpty(),
                "zero-valued contract fields must not appear on the wire");
    }

    /**
     * Hand-built pre-contract descriptor: TaskInfoPB and WorkerStatusPB exactly
     * as they were before kv_tokens/running_detail_truncated existed.
     */
    private static Descriptors.Descriptor legacyDescriptor() throws Exception {
        DescriptorProtos.EnumDescriptorProto taskPhase = DescriptorProtos.EnumDescriptorProto.newBuilder()
                .setName("TaskPhase")
                .addValue(DescriptorProtos.EnumValueDescriptorProto.newBuilder()
                        .setName("TASK_PHASE_PENDING").setNumber(0))
                .addValue(DescriptorProtos.EnumValueDescriptorProto.newBuilder()
                        .setName("TASK_PHASE_RECEIVED").setNumber(1))
                .addValue(DescriptorProtos.EnumValueDescriptorProto.newBuilder()
                        .setName("TASK_PHASE_KV_ALLOCATED").setNumber(2))
                .addValue(DescriptorProtos.EnumValueDescriptorProto.newBuilder()
                        .setName("TASK_PHASE_RUNNING").setNumber(3))
                .build();
        DescriptorProtos.DescriptorProto task = DescriptorProtos.DescriptorProto.newBuilder()
                .setName("TaskInfoPB")
                .addEnumType(taskPhase)
                .addField(field("request_id", 1,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64))
                .addField(field("input_length", 4,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64))
                .addField(field("iterate_count", 6,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64))
                .addField(field("is_waiting", 9,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_BOOL))
                .addField(field("batch_id", 11,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64))
                .addField(field("phase", 12,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_ENUM,
                        ".legacy_contract.TaskInfoPB.TaskPhase"))
                .build();
        DescriptorProtos.DescriptorProto worker = DescriptorProtos.DescriptorProto.newBuilder()
                .setName("WorkerStatusPB")
                .addField(field("role", 1,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_STRING))
                .addField(repeatedField("running_task_info", 3,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_MESSAGE,
                        ".legacy_contract.TaskInfoPB"))
                .addField(repeatedField("finished_task_list", 4,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_MESSAGE,
                        ".legacy_contract.TaskInfoPB"))
                .addField(field("latest_finished_version", 15,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64))
                .build();
        DescriptorProtos.FileDescriptorProto file = DescriptorProtos.FileDescriptorProto.newBuilder()
                .setName("legacy_engine_contract.proto")
                .setPackage("legacy_contract")
                .setSyntax("proto3")
                .addMessageType(task)
                .addMessageType(worker)
                .build();
        return Descriptors.FileDescriptor.buildFrom(file, new Descriptors.FileDescriptor[0])
                .findMessageTypeByName("WorkerStatusPB");
    }

    private static DescriptorProtos.FieldDescriptorProto field(
            String name,
            int number,
            DescriptorProtos.FieldDescriptorProto.Type type) {
        return field(name, number, type, null);
    }

    private static DescriptorProtos.FieldDescriptorProto field(
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

    private static DescriptorProtos.FieldDescriptorProto repeatedField(
            String name,
            int number,
            DescriptorProtos.FieldDescriptorProto.Type type,
            String typeName) {
        return field(name, number, type, typeName).toBuilder()
                .setLabel(DescriptorProtos.FieldDescriptorProto.Label.LABEL_REPEATED)
                .build();
    }
}

package org.flexlb.engine.grpc;

import com.google.protobuf.DescriptorProtos;
import com.google.protobuf.Descriptors;
import com.google.protobuf.DynamicMessage;
import com.google.protobuf.Int32Value;
import com.google.protobuf.WrappersProto;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RoleAddrProtocolCompatibilityTest {

    @Test
    void descriptorPreservesDsv4FieldsAndAddsExtensionFields() {
        Descriptors.Descriptor roleAddr = EngineRpcService.RoleAddrPB.getDescriptor();
        Descriptors.Descriptor generateConfig = EngineRpcService.GenerateConfigPB.getDescriptor();
        Descriptors.Descriptor worker = EngineRpcService.WorkerStatusPB.getDescriptor();
        Descriptors.Descriptor task = EngineRpcService.TaskInfoPB.getDescriptor();

        assertEquals(Descriptors.FieldDescriptor.Type.ENUM,
                roleAddr.findFieldByNumber(1).getType());
        assertEquals(Descriptors.FieldDescriptor.Type.STRING,
                roleAddr.findFieldByNumber(5).getType());
        assertNull(generateConfig.findFieldByNumber(55));
        assertEquals(Descriptors.FieldDescriptor.Type.STRING,
                worker.findFieldByNumber(1).getType());
        assertEquals(Descriptors.FieldDescriptor.Type.ENUM,
                worker.findFieldByNumber(20).getType());
        assertEquals(Descriptors.FieldDescriptor.Type.BOOL,
                task.findFieldByNumber(9).getType());
        assertEquals(Descriptors.FieldDescriptor.Type.ENUM,
                task.findFieldByNumber(12).getType());
    }

    @Test
    void dualRoleAddrPayloadIsReadableByDsv4Descriptor() throws Exception {
        Descriptors.Descriptor legacy = legacyRoleAddrDescriptor();
        for (RoleType role : RoleType.values()) {
            EngineRpcService.RoleAddrPB payload = EngineRpcService.RoleAddrPB.newBuilder()
                    .setRole(RoleTypeProtoConverter.toLegacyProto(role))
                    .setRoleStr(role.getCode())
                    .setIp("127.0.0.1")
                    .setGrpcPort(9000)
                    .build();

            DynamicMessage oldReader = DynamicMessage.parseFrom(legacy, payload.toByteArray());
            assertEquals(role.ordinal(), ((Descriptors.EnumValueDescriptor) oldReader.getField(
                    legacy.findFieldByNumber(1))).getNumber());
        }
    }

    @Test
    void currentRoleAddrReaderAcceptsDsv4PayloadAndRejectsConflict() throws Exception {
        Descriptors.Descriptor legacy = legacyRoleAddrDescriptor();
        for (RoleType role : RoleType.values()) {
            DynamicMessage oldWriter = DynamicMessage.newBuilder(legacy)
                    .setField(legacy.findFieldByNumber(1),
                            legacy.findEnumTypeByName("RoleType")
                                    .findValueByNumber(role.ordinal()))
                    .build();
            EngineRpcService.RoleAddrPB parsed =
                    EngineRpcService.RoleAddrPB.parseFrom(oldWriter.toByteArray());
            assertEquals(role, RoleTypeProtoConverter.fromRoleAddr(parsed));
        }

        assertThrows(IllegalArgumentException.class,
                () -> RoleTypeProtoConverter.fromRoleAddr(
                        EngineRpcService.RoleAddrPB.newBuilder()
                                .setRole(EngineRpcService.RoleAddrPB.RoleType.PREFILL)
                                .setRoleStr("DECODE")
                                .build()));
    }

    @Test
    void dualWorkerStatusPayloadIsReadableByDsv4Descriptor() throws Exception {
        Descriptors.Descriptor legacy = legacyWorkerStatusDescriptor();
        EngineRpcService.TaskInfoPB running = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId("42")
                .setIsWaiting(false)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                .build();
        EngineRpcService.WorkerStatusPB payload = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("RoleType.PREFILL")
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .addRunningTaskInfo(running)
                .build();

        DynamicMessage oldReader = DynamicMessage.parseFrom(legacy, payload.toByteArray());
        assertEquals("RoleType.PREFILL", oldReader.getField(legacy.findFieldByNumber(1)));
        DynamicMessage oldTask = (DynamicMessage) ((List<?>) oldReader.getField(
                legacy.findFieldByNumber(3))).get(0);
        assertFalse((Boolean) oldTask.getField(
                oldTask.getDescriptorForType().findFieldByNumber(9)));
    }

    @Test
    void currentWorkerStatusReaderAcceptsDsv4Payload() throws Exception {
        Descriptors.Descriptor legacy = legacyWorkerStatusDescriptor();
        Descriptors.Descriptor legacyTask = legacy.findFieldByNumber(3).getMessageType();
        DynamicMessage waiting = DynamicMessage.newBuilder(legacyTask)
                .setField(legacyTask.findFieldByNumber(1), 7L)
                .setField(legacyTask.findFieldByNumber(9), true)
                .build();
        DynamicMessage oldWriter = DynamicMessage.newBuilder(legacy)
                .setField(legacy.findFieldByNumber(1), "RoleType.DECODE")
                .addRepeatedField(legacy.findFieldByNumber(3), waiting)
                .build();

        EngineRpcService.WorkerStatusPB parsed =
                EngineRpcService.WorkerStatusPB.parseFrom(oldWriter.toByteArray());

        assertEquals(RoleType.DECODE, RoleTypeProtoConverter.fromWorkerStatus(parsed));
        assertEquals(true, parsed.getRunningTaskInfo(0).getIsWaiting());
        assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_PENDING,
                parsed.getRunningTaskInfo(0).getPhase());
    }

    @Test
    void legacyForceBatchField55SurvivesCurrentUnknownFieldRoundTrip() throws Exception {
        Descriptors.Descriptor legacy = legacyGenerateConfigDescriptor();
        Descriptors.FieldDescriptor legacyForceBatch = legacy.findFieldByNumber(55);

        DynamicMessage oldWriter = DynamicMessage.newBuilder(legacy)
                .setField(legacyForceBatch, DynamicMessage.newBuilder(Int32Value.getDescriptor())
                        .setField(Int32Value.getDescriptor().findFieldByNumber(1), 1)
                        .build())
                .build();
        EngineRpcService.GenerateConfigPB parsed =
                EngineRpcService.GenerateConfigPB.parseFrom(oldWriter.toByteArray());
        assertFalse(parsed.getUnknownFields().getField(55).getLengthDelimitedList().isEmpty());

        DynamicMessage oldReader = DynamicMessage.parseFrom(legacy, parsed.toByteArray());
        assertEquals(1, ((DynamicMessage) oldReader.getField(legacyForceBatch))
                .getField(Int32Value.getDescriptor().findFieldByNumber(1)));
    }

    private static Descriptors.Descriptor legacyRoleAddrDescriptor() throws Exception {
        DescriptorProtos.DescriptorProto message = DescriptorProtos.DescriptorProto.newBuilder()
                .setName("RoleAddrPB")
                .addEnumType(roleEnum("RoleType", ""))
                .addField(field("role", 1, DescriptorProtos.FieldDescriptorProto.Type.TYPE_ENUM,
                        ".legacy.RoleAddrPB.RoleType"))
                .addField(field("ip", 2, DescriptorProtos.FieldDescriptorProto.Type.TYPE_STRING, null))
                .addField(field("http_port", 3, DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT32, null))
                .addField(field("grpc_port", 4, DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT32, null))
                .build();
        DescriptorProtos.FileDescriptorProto file = DescriptorProtos.FileDescriptorProto.newBuilder()
                .setName("legacy_role_addr.proto")
                .setPackage("legacy")
                .setSyntax("proto3")
                .addMessageType(message)
                .build();
        return Descriptors.FileDescriptor.buildFrom(file, new Descriptors.FileDescriptor[0])
                .findMessageTypeByName("RoleAddrPB");
    }

    private static Descriptors.Descriptor legacyWorkerStatusDescriptor() throws Exception {
        DescriptorProtos.DescriptorProto task = DescriptorProtos.DescriptorProto.newBuilder()
                .setName("TaskInfoPB")
                .addField(field("request_id", 1,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_INT64, null))
                .addField(field("is_waiting", 9,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_BOOL, null))
                .build();
        DescriptorProtos.DescriptorProto worker = DescriptorProtos.DescriptorProto.newBuilder()
                .setName("WorkerStatusPB")
                .addField(field("role", 1,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_STRING, null))
                .addField(repeatedField("running_task_info", 3,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_MESSAGE,
                        ".legacy_status.TaskInfoPB"))
                .build();
        DescriptorProtos.FileDescriptorProto file = DescriptorProtos.FileDescriptorProto.newBuilder()
                .setName("legacy_worker_status.proto")
                .setPackage("legacy_status")
                .setSyntax("proto3")
                .addMessageType(task)
                .addMessageType(worker)
                .build();
        return Descriptors.FileDescriptor.buildFrom(file, new Descriptors.FileDescriptor[0])
                .findMessageTypeByName("WorkerStatusPB");
    }

    private static Descriptors.Descriptor legacyGenerateConfigDescriptor() throws Exception {
        DescriptorProtos.DescriptorProto message = DescriptorProtos.DescriptorProto.newBuilder()
                .setName("GenerateConfigPB")
                .addField(field("force_batch", 55,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_MESSAGE,
                        ".google.protobuf.Int32Value"))
                .addField(field("batch_group_timeout", 56,
                        DescriptorProtos.FieldDescriptorProto.Type.TYPE_MESSAGE,
                        ".google.protobuf.Int32Value"))
                .build();
        DescriptorProtos.FileDescriptorProto file = DescriptorProtos.FileDescriptorProto.newBuilder()
                .setName("legacy_generate_config.proto")
                .setPackage("legacy_generate_config")
                .setSyntax("proto3")
                .addDependency("google/protobuf/wrappers.proto")
                .addMessageType(message)
                .build();
        return Descriptors.FileDescriptor.buildFrom(
                        file, new Descriptors.FileDescriptor[] {WrappersProto.getDescriptor()})
                .findMessageTypeByName("GenerateConfigPB");
    }

    private static DescriptorProtos.EnumDescriptorProto roleEnum(String name, String prefix) {
        DescriptorProtos.EnumDescriptorProto.Builder builder =
                DescriptorProtos.EnumDescriptorProto.newBuilder().setName(name);
        for (RoleType role : RoleType.values()) {
            builder.addValue(DescriptorProtos.EnumValueDescriptorProto.newBuilder()
                    .setName(prefix + role.name())
                    .setNumber(role.ordinal()));
        }
        return builder.build();
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

package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.UnknownFieldSet;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GenerateInputPayloadTest {
    @Test
    void packedTokensStayEncodedAcrossMetadataChangesAndWholeBatchSerialization() throws Exception {
        var original =
                GenerateInputPB.newBuilder()
                        .setRequestId(17)
                        .setPriority(9)
                        .addAllTokenIds(
                                Arrays.asList(
                                        0,
                                        1,
                                        127,
                                        128,
                                        65536,
                                        Integer.MAX_VALUE,
                                        -1,
                                        Integer.MIN_VALUE))
                        .setGenerateConfig(
                                EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(37))
                        .setRequestInfo(
                                EngineRpcService.RequestInfoPB.newBuilder()
                                        .setRequestId("client-id"))
                        .setUnknownFields(
                                UnknownFieldSet.newBuilder().mergeVarintField(999, 71).build())
                        .build();
        var wire = GenerateInputPayload.parse(original.toByteString()).toWireBuilder();
        assertEquals(0, wire.getTokenIdsCount(), "Master must not materialize the token list");
        wire.setPriority(60)
                .getGenerateConfigBuilder()
                .clearRoleAddrs()
                .addRoleAddrs(
                        EngineRpcService.RoleAddrPB.newBuilder()
                                .setRoleStr("prefill")
                                .setIp("127.0.0.1"));
        var batch =
                EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                        .setBatchId(31)
                        .addDpSlots(
                                EngineRpcService.EnqueueBatchDpSlotPB.newBuilder()
                                        .setDpRank(2)
                                        .addRequests(
                                                EngineRpcService.EnqueueBatchExternalInputPB
                                                        .newBuilder()
                                                        .setInput(wire)))
                        .build();
        var decoded =
                EngineRpcService.EnqueueBatchRequestPB.parseFrom(batch.toByteString())
                        .getDpSlots(0)
                        .getRequests(0)
                        .getInput();
        var expected = original.toBuilder().setPriority(60);
        expected.getGenerateConfigBuilder()
                .clearRoleAddrs()
                .addRoleAddrs(
                        EngineRpcService.RoleAddrPB.newBuilder()
                                .setRoleStr("prefill")
                                .setIp("127.0.0.1"));
        assertEquals(expected.build(), decoded);
    }

    @Test
    void mixedPackedUnpackedAndDuplicateMetadataRetainNormalProtobufSemantics() throws Exception {
        var bytes = ByteString.newOutput();
        var output = CodedOutputStream.newInstance(bytes);
        output.writeInt64(1, 17);
        output.writeInt32(2, -1);
        output.writeBytes(2, packed(0, 128, Integer.MIN_VALUE));
        output.writeMessage(
                4, EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(31).build());
        output.writeString(5, "first");
        output.writeInt32(2, 7);
        output.writeMessage(
                4, EngineRpcService.GenerateConfigPB.newBuilder().setNumBeams(3).build());
        output.writeBytes(2, packed(9, 10));
        output.writeString(5, "last");
        output.writeInt64(1, 18);
        output.writeFixed32(2, 0x1234); // Same field number, unknown wire type.
        output.writeTag(99, 3);
        output.writeInt32(2, 51);
        output.writeTag(99, 4);
        output.flush();
        assertEquivalent(bytes.toByteString());
    }

    @Test
    void segmentedBytesAndEveryTruncatedPrefixMatchTheStandardParser() throws Exception {
        var message =
                GenerateInputPB.newBuilder()
                        .setRequestId(71)
                        .setClientId("hello")
                        .addAllTokenIds(Arrays.asList(0, 127, 128, -1, Integer.MIN_VALUE))
                        .setGenerateConfig(
                                EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(91))
                        .build();
        byte[] bytes = message.toByteArray();
        for (int size = 0; size <= bytes.length; size++) {
            var pieces = new ArrayList<ByteString>();
            for (int start = 0; start < size; start += 3) {
                pieces.add(ByteString.copyFrom(bytes, start, Math.min(3, size - start)));
            }
            assertEquivalent(ByteString.copyFrom(pieces));
        }
    }

    @Test
    void randomizedWireRepresentationsMatchStandardParser() throws Exception {
        var random = new Random(17001L);
        for (int sample = 0; sample < 600; sample++) {
            var bytes = ByteString.newOutput();
            var output = CodedOutputStream.newInstance(bytes);
            for (int field = 0; field < 25; field++) {
                switch (random.nextInt(6)) {
                    case 0 -> output.writeInt64(1, random.nextLong());
                    case 1 -> output.writeInt32(2, random.nextInt());
                    case 2 -> output.writeBytes(2, packed(random.nextInt(), random.nextInt()));
                    case 3 ->
                            output.writeMessage(
                                    4,
                                    EngineRpcService.GenerateConfigPB.newBuilder()
                                            .setMaxNewTokens(random.nextInt(1000))
                                            .build());
                    case 4 -> output.writeString(5, "client-" + random.nextInt(100));
                    case 5 -> output.writeUInt64(99, random.nextLong());
                }
            }
            output.flush();
            var encoded = bytes.toByteString();
            assertEquivalent(encoded);
            assertEquivalent(encoded.substring(0, random.nextInt(encoded.size())));
        }
    }

    @Test
    void malformedTokensTagsAndNestedMessagesAreStillRejected() throws Exception {
        for (byte[] bytes :
                new byte[][] {
                    {18, 1, (byte) 0x80},
                    {18, (byte) 0xff, (byte) 0xff, (byte) 0xff, (byte) 0xff, 15},
                    {16, (byte) 0x80},
                    {0},
                    {20},
                    {34, 1, (byte) 0xff},
                    {42, 1, (byte) 0xff},
                    {
                        18,
                        11,
                        (byte) 0x80,
                        (byte) 0x80,
                        (byte) 0x80,
                        (byte) 0x80,
                        (byte) 0x80,
                        (byte) 0x80,
                        (byte) 0x80,
                        (byte) 0x80,
                        (byte) 0x80,
                        (byte) 0x80,
                        0
                    }
                }) {
            assertThrows(
                    InvalidProtocolBufferException.class, () -> GenerateInputPB.parseFrom(bytes));
            assertThrows(
                    InvalidProtocolBufferException.class,
                    () -> GenerateInputPayload.parse(ByteString.copyFrom(bytes)));
        }
    }

    @Test
    void parseFailureDoesNotRetainAnAlreadyDecodedLargeTensor() throws Exception {
        var tensor =
                EngineRpcService.TensorPB.newBuilder()
                        .setFp16Data(ByteString.copyFrom(new byte[1 << 20]));
        var partial =
                EngineRpcService.MultimodalInputPB.newBuilder()
                        .setMultimodalTensor(tensor)
                        .build()
                        .toByteString();
        var invalid =
                partial.concat(
                        ByteString.copyFrom(
                                new byte[] {0})); // Invalid tag after a valid large tensor.
        var bytes = ByteString.newOutput();
        var output = CodedOutputStream.newInstance(bytes);
        output.writeBytes(
                GenerateInputPB.getDescriptor().findFieldByName("multimodal_inputs").getNumber(),
                invalid);
        output.flush();
        var wire = bytes.toByteString();
        var standard =
                assertThrows(
                        InvalidProtocolBufferException.class,
                        () -> GenerateInputPB.parseFrom(wire));
        assertNotNull(
                standard.getUnfinishedMessage(), "fixture must exercise partial-message retention");
        var failure =
                assertThrows(
                        InvalidProtocolBufferException.class,
                        () -> GenerateInputPayload.parse(wire));
        assertNull(failure.getUnfinishedMessage());
        assertNull(
                failure.getCause(),
                "a diagnostic cause must not retain the discarded partial message either");
        assertEquals(standard.getMessage(), failure.getMessage());
        assertTrue(failure.getStackTrace().length > 0);
    }

    @Test
    void largeMultimodalBytesDoNotGetCopiedWhileTokensRetainTheOriginalBuffer() throws Exception {
        var tensor =
                EngineRpcService.TensorPB.newBuilder()
                        .setFp16Data(ByteString.copyFrom(new byte[4 * 1024 * 1024]));
        ByteString bytes =
                GenerateInputPB.newBuilder()
                        .setRequestId(17)
                        .addTokenIds(71)
                        .addMultimodalInputs(
                                EngineRpcService.MultimodalInputPB.newBuilder()
                                        .setMultimodalTensor(tensor))
                        .build()
                        .toByteString();
        var management = java.lang.management.ManagementFactory.getThreadMXBean();
        org.junit.jupiter.api.Assumptions.assumeTrue(
                management instanceof com.sun.management.ThreadMXBean);
        var allocations = (com.sun.management.ThreadMXBean) management;
        org.junit.jupiter.api.Assumptions.assumeTrue(
                allocations.isThreadAllocatedMemorySupported());
        allocations.setThreadAllocatedMemoryEnabled(true);
        for (int i = 0; i < 3; i++) {
            GenerateInputPayload.parse(bytes);
        }
        long thread = Thread.currentThread().threadId();
        long before = allocations.getThreadAllocatedBytes(thread);
        var prepared = GenerateInputPayload.parse(bytes);
        long allocated = allocations.getThreadAllocatedBytes(thread) - before;
        assertTrue(
                allocated < 256 * 1024L,
                "immutable multimodal bytes must be shared; allocated=" + allocated);
        assertEquals(
                GenerateInputPB.parseFrom(bytes),
                GenerateInputPB.parseFrom(prepared.toWireBuilder().build().toByteString()));
    }

    private static ByteString packed(int... tokens) throws IOException {
        var bytes = ByteString.newOutput();
        var output = CodedOutputStream.newInstance(bytes);
        for (int token : tokens) {
            output.writeInt32NoTag(token);
        }
        output.flush();
        return bytes.toByteString();
    }

    private static void assertEquivalent(ByteString bytes) throws Exception {
        GenerateInputPB expected;
        try {
            expected = GenerateInputPB.parseFrom(bytes);
        } catch (InvalidProtocolBufferException malformed) {
            assertThrows(
                    InvalidProtocolBufferException.class, () -> GenerateInputPayload.parse(bytes));
            return;
        }
        var wire = GenerateInputPayload.parse(bytes).toWireBuilder().build();
        assertEquals(0, wire.getTokenIdsCount());
        assertEquals(expected, GenerateInputPB.parseFrom(wire.toByteString()));
    }
}

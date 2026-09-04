package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.Artifact;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/** Compact little-endian wire format shared by FlexLB Master and Whale workers. */
public final class ConstraintTreeCsrCodec {

    static final byte[] MAGIC = "RTPCSR01".getBytes(StandardCharsets.US_ASCII);
    static final int FORMAT_VERSION = 1;
    static final int HEADER_SIZE = 48;

    private ConstraintTreeCsrCodec() {
    }

    public static byte[] encode(Artifact artifact) {
        if (artifact == null) {
            throw new IllegalArgumentException("artifact must not be null");
        }
        validateArtifact(artifact.startTokenId(), artifact.endTokenId(), artifact.sidCount(),
                artifact.rowPtr(), artifact.colIdx(), artifact.nextState());
        long elementCount = (long) artifact.rowPtr().length
                + artifact.colIdx().length
                + artifact.nextState().length;
        long encodedSize = HEADER_SIZE + elementCount * Integer.BYTES;
        if (encodedSize > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("serialized CSR artifact exceeds HTTP int32 body capacity");
        }

        ByteBuffer output = ByteBuffer.allocate((int) encodedSize).order(ByteOrder.LITTLE_ENDIAN);
        output.put(MAGIC);
        output.putInt(FORMAT_VERSION);
        output.putInt(HEADER_SIZE);
        output.putLong(artifact.version());
        output.putInt(artifact.startTokenId());
        output.putInt(artifact.endTokenId());
        output.putInt(artifact.rowPtr().length - 1);
        output.putInt(artifact.colIdx().length);
        output.putLong(artifact.sidCount());
        putInts(output, artifact.rowPtr());
        putInts(output, artifact.colIdx());
        putInts(output, artifact.nextState());
        return output.array();
    }

    public static DecodedArtifact decode(byte[] payload) {
        if (payload == null || payload.length < HEADER_SIZE) {
            throw new IllegalArgumentException("CSR artifact is shorter than its header");
        }
        ByteBuffer input = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        byte[] magic = new byte[MAGIC.length];
        input.get(magic);
        if (!Arrays.equals(MAGIC, magic)) {
            throw new IllegalArgumentException("CSR artifact has invalid magic");
        }
        int formatVersion = input.getInt();
        int headerSize = input.getInt();
        if (formatVersion != FORMAT_VERSION || headerSize != HEADER_SIZE) {
            throw new IllegalArgumentException("unsupported CSR artifact format");
        }
        long version = input.getLong();
        int startTokenId = input.getInt();
        int endTokenId = input.getInt();
        int stateCount = input.getInt();
        int edgeCount = input.getInt();
        long sidCount = input.getLong();
        if (version <= 0 || stateCount <= 0 || edgeCount <= 0 || sidCount <= 0) {
            throw new IllegalArgumentException("CSR artifact header contains invalid counts or version");
        }
        long expectedSize = HEADER_SIZE + Integer.BYTES * ((long) stateCount + 1L + 2L * edgeCount);
        if (expectedSize != payload.length) {
            throw new IllegalArgumentException("CSR artifact length does not match its header");
        }
        int[] rowPtr = readInts(input, stateCount + 1);
        int[] colIdx = readInts(input, edgeCount);
        int[] nextState = readInts(input, edgeCount);
        validateArtifact(startTokenId, endTokenId, sidCount, rowPtr, colIdx, nextState);
        return new DecodedArtifact(
                version, startTokenId, endTokenId, sidCount, rowPtr, colIdx, nextState);
    }

    private static void validateArtifact(int startTokenId,
                                         int endTokenId,
                                         long sidCount,
                                         int[] rowPtr,
                                         int[] colIdx,
                                         int[] nextState) {
        if (startTokenId < 0 || endTokenId < 0 || startTokenId == endTokenId || sidCount <= 0) {
            throw new IllegalArgumentException("CSR artifact has invalid tokens or sid_count");
        }
        if (rowPtr == null || rowPtr.length < 2 || colIdx == null || nextState == null) {
            throw new IllegalArgumentException("CSR arrays must not be null or empty");
        }
        if (colIdx.length == 0 || colIdx.length != nextState.length) {
            throw new IllegalArgumentException("CSR edge arrays have different lengths");
        }
        if (rowPtr[0] != 0 || rowPtr[rowPtr.length - 1] != colIdx.length) {
            throw new IllegalArgumentException("CSR row_ptr boundaries are invalid");
        }
        long terminalEdges = 0;
        int stateCount = rowPtr.length - 1;
        for (int state = 0; state < stateCount; state++) {
            int begin = rowPtr[state];
            int end = rowPtr[state + 1];
            if (begin < 0 || end <= begin || end > colIdx.length) {
                throw new IllegalArgumentException(
                        "CSR row_ptr must be increasing and every state must have an outgoing edge");
            }
            int previousToken = -1;
            for (int edge = begin; edge < end; edge++) {
                int token = colIdx[edge];
                int target = nextState[edge];
                if (token < 0 || token <= previousToken || token == startTokenId) {
                    throw new IllegalArgumentException(
                            "CSR candidate rows must contain sorted unique non-negative token ids");
                }
                if (token == endTokenId) {
                    if (target != ConstraintTreeBuilder.TERMINAL_STATE) {
                        throw new IllegalArgumentException("CSR end-token edge must point to the terminal state");
                    }
                    terminalEdges++;
                } else if (target < 0 || target >= stateCount) {
                    throw new IllegalArgumentException("CSR non-terminal edge points to an invalid state");
                }
                previousToken = token;
            }
        }
        if (terminalEdges != sidCount) {
            throw new IllegalArgumentException("CSR terminal edge count does not match sid_count");
        }
    }

    private static void putInts(ByteBuffer output, int[] values) {
        for (int value : values) {
            output.putInt(value);
        }
    }

    private static int[] readInts(ByteBuffer input, int count) {
        int[] values = new int[count];
        for (int index = 0; index < count; index++) {
            values[index] = input.getInt();
        }
        return values;
    }

    public record DecodedArtifact(
            long version,
            int startTokenId,
            int endTokenId,
            long sidCount,
            int[] rowPtr,
            int[] colIdx,
            int[] nextState) {
    }
}

package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.Artifact;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConstraintTreeBuilderTest {

    private final ConstraintTreeBuilder builder = new ConstraintTreeBuilder(
            Clock.fixed(Instant.ofEpochMilli(1234), ZoneOffset.UTC), 4);

    @AfterEach
    void tearDown() {
        builder.close();
    }

    @Test
    void mergesVariableLengthSidsAndDeduplicatesEdges() {
        BuildRequest request = stringRequest(
                7, 225, 2, "10_20", "10_20_31", "11", "10_20");

        Artifact artifact = builder.build(request);

        assertEquals(7, artifact.version());
        assertEquals(4, artifact.inputSidCount());
        assertEquals(3, artifact.sidCount());
        assertEquals(5, artifact.prefixCount());
        assertEquals(7, artifact.edgeCount());
        assertEquals(1234, artifact.createdAtEpochMs());
        assertArrayEquals(new int[]{10, 11}, candidates(artifact, 0));
        int state10 = nextState(artifact, 0, 10);
        int state1020 = nextState(artifact, state10, 20);
        assertArrayEquals(new int[]{20}, candidates(artifact, state10));
        assertArrayEquals(new int[]{2, 31}, candidates(artifact, state1020));
        assertEquals(ConstraintTreeBuilder.TERMINAL_STATE, nextState(artifact, state1020, 2));
        assertArrayEquals(new int[]{2}, candidates(artifact, nextState(artifact, state1020, 31)));
        assertArrayEquals(new int[]{2}, candidates(artifact, nextState(artifact, 0, 11)));
    }

    @Test
    void acceptsTokenArraysOfDifferentLengths() {
        BuildRequest request = new BuildRequest(
                8, "gul_item", 1699, 151645, "_",
                List.of(new int[]{169967, 216546}, new int[]{42, 43, 44}), null);

        Artifact artifact = builder.build(request);

        assertArrayEquals(new int[]{42, 169967}, candidates(artifact, 0));
        int state169967 = nextState(artifact, 0, 169967);
        int state216546 = nextState(artifact, state169967, 216546);
        assertArrayEquals(new int[]{216546}, candidates(artifact, state169967));
        assertArrayEquals(new int[]{151645}, candidates(artifact, state216546));
        int state42 = nextState(artifact, 0, 42);
        int state43 = nextState(artifact, state42, 43);
        assertArrayEquals(new int[]{44}, candidates(artifact, state43));
    }

    @Test
    void appliesExampleDefaults() {
        Artifact artifact = builder.build(new BuildRequest(
                1, "gul_item", null, null, null, null, List.of("169967_216546")));

        assertEquals(1699, artifact.startTokenId());
        assertEquals(151645, artifact.endTokenId());
    }

    @Test
    void binaryCodecRoundTripsCsrWithoutStringPrefixes() {
        Artifact artifact = builder.build(stringRequest(9, 225, 2, "10_20", "10_21", "11"));

        byte[] payload = ConstraintTreeCsrCodec.encode(artifact);
        ConstraintTreeCsrCodec.DecodedArtifact decoded = ConstraintTreeCsrCodec.decode(payload);

        assertEquals(9, decoded.version());
        assertEquals(225, decoded.startTokenId());
        assertEquals(2, decoded.endTokenId());
        assertEquals(3, decoded.sidCount());
        assertArrayEquals(artifact.rowPtr(), decoded.rowPtr());
        assertArrayEquals(artifact.colIdx(), decoded.colIdx());
        assertArrayEquals(artifact.nextState(), decoded.nextState());
    }

    @Test
    void rejectsMalformedSidWithItsInputIndex() {
        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> builder.build(stringRequest(1, 225, 2, "1_2_3", "4_")));

        assertTrue(error.getMessage().contains("index 1"));
        assertTrue(error.getMessage().contains("empty token"));
    }

    @Test
    void rejectsNegativeOverflowingAndReservedTokenIds() {
        assertThrows(IllegalArgumentException.class,
                () -> builder.build(stringRequest(1, 225, 2, "1_-2_3")));
        assertThrows(IllegalArgumentException.class,
                () -> builder.build(stringRequest(1, 225, 2, "1_2147483648_3")));
        assertThrows(IllegalArgumentException.class,
                () -> builder.build(stringRequest(1, 225, 2, "1_2_3")));
    }

    @Test
    void requiresExactlyOneInputRepresentationAndModel() {
        assertThrows(IllegalArgumentException.class,
                () -> builder.build(new BuildRequest(1, null, 225, 2, "_", null, List.of("1"))));
        assertThrows(IllegalArgumentException.class,
                () -> builder.build(new BuildRequest(
                        1, "gul_item", 225, 2, "_", List.of(new int[]{1}), List.of("1"))));
        assertThrows(IllegalArgumentException.class,
                () -> builder.build(new BuildRequest(
                        1, "gul_item", 225, 225, "_", null, List.of("1"))));
    }

    private BuildRequest stringRequest(long version,
                                       Integer startTokenId,
                                       Integer endTokenId,
                                       String... sids) {
        return new BuildRequest(version, "gul_item", startTokenId, endTokenId, "_", null, List.of(sids));
    }

    private int[] candidates(Artifact artifact, int state) {
        int begin = artifact.rowPtr()[state];
        int end = artifact.rowPtr()[state + 1];
        return java.util.Arrays.copyOfRange(artifact.colIdx(), begin, end);
    }

    private int nextState(Artifact artifact, int state, int token) {
        for (int edge = artifact.rowPtr()[state]; edge < artifact.rowPtr()[state + 1]; edge++) {
            if (artifact.colIdx()[edge] == token) {
                return artifact.nextState()[edge];
            }
        }
        throw new AssertionError("missing token " + token + " from state " + state);
    }
}

package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.Artifact;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;

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
        assertEquals(1234, artifact.createdAtEpochMs());
        assertEquals(List.of(10, 11), artifact.prefixDict().get("225"));
        assertEquals(List.of(20), artifact.prefixDict().get("225_10"));
        assertEquals(List.of(2, 31), artifact.prefixDict().get("225_10_20"));
        assertEquals(List.of(2), artifact.prefixDict().get("225_10_20_31"));
        assertEquals(List.of(2), artifact.prefixDict().get("225_11"));
    }

    @Test
    void acceptsTokenArraysOfDifferentLengths() {
        BuildRequest request = new BuildRequest(
                8, "gul_item", 1699, 151645, "_",
                List.of(new int[]{169967, 216546}, new int[]{42, 43, 44}), null);

        Artifact artifact = builder.build(request);

        assertEquals(List.of(42, 169967), artifact.prefixDict().get("1699"));
        assertEquals(List.of(216546), artifact.prefixDict().get("1699_169967"));
        assertEquals(List.of(151645), artifact.prefixDict().get("1699_169967_216546"));
        assertEquals(List.of(44), artifact.prefixDict().get("1699_42_43"));
    }

    @Test
    void appliesExampleDefaults() {
        Artifact artifact = builder.build(new BuildRequest(
                1, "gul_item", null, null, null, null, List.of("169967_216546")));

        assertEquals(1699, artifact.startTokenId());
        assertEquals(151645, artifact.endTokenId());
        assertEquals("_", artifact.separator());
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
}

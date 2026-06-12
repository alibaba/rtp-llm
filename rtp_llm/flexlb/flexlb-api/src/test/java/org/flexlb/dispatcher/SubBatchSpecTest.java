package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SubBatchSpecTest {

    @Test
    void parsesSizeColonValue() {
        SubBatchSpec spec = SubBatchSpec.parse("size:5");
        assertEquals(SubBatchSpec.Mode.SIZE, spec.mode());
        assertEquals(5, spec.value());
    }

    @Test
    void parsesCountColonValue() {
        SubBatchSpec spec = SubBatchSpec.parse("count:5");
        assertEquals(SubBatchSpec.Mode.COUNT, spec.mode());
        assertEquals(5, spec.value());
    }

    @Test
    void bareIntegerShorthandIsSizeMode() {
        SubBatchSpec spec = SubBatchSpec.parse("5");
        assertEquals(SubBatchSpec.Mode.SIZE, spec.mode());
        assertEquals(5, spec.value(),
                "bare integer is shorthand for size:N to keep the migration from the old subBatchSize field trivial");
    }

    @Test
    void modeIsCaseInsensitive() {
        SubBatchSpec spec = SubBatchSpec.parse("SIZE:10");
        assertEquals(SubBatchSpec.Mode.SIZE, spec.mode());
        assertEquals(10, spec.value());
    }

    @Test
    void whitespaceAroundValueIsTrimmed() {
        SubBatchSpec spec = SubBatchSpec.parse("  count:5  ");
        assertEquals(SubBatchSpec.Mode.COUNT, spec.mode());
        assertEquals(5, spec.value());
    }

    @Test
    void valueLessThanOneRejected() {
        assertThrows(IllegalArgumentException.class, () -> SubBatchSpec.parse("size:0"),
                "value must be >= 1 — 0 chunks/chunk-size is meaningless");
        assertThrows(IllegalArgumentException.class, () -> SubBatchSpec.parse("count:0"));
        assertThrows(IllegalArgumentException.class, () -> SubBatchSpec.parse("size:-3"));
    }

    @Test
    void unknownModeRejected() {
        assertThrows(IllegalArgumentException.class, () -> SubBatchSpec.parse("foo:5"),
                "only size and count are recognized modes");
    }

    @Test
    void missingValueAfterColonRejected() {
        assertThrows(IllegalArgumentException.class, () -> SubBatchSpec.parse("size:"));
    }

    @Test
    void nonIntegerValueRejected() {
        assertThrows(IllegalArgumentException.class, () -> SubBatchSpec.parse("size:abc"));
    }

    @Test
    void nullRejected() {
        assertThrows(IllegalArgumentException.class, () -> SubBatchSpec.parse(null));
    }

    @Test
    void blankRejected() {
        assertThrows(IllegalArgumentException.class, () -> SubBatchSpec.parse(""));
        assertThrows(IllegalArgumentException.class, () -> SubBatchSpec.parse("   "));
    }
}

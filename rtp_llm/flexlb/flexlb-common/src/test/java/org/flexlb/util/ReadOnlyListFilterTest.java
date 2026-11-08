package org.flexlb.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ReadOnlyListFilterTest {

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
    void preservesOrderAndReadOnlyResultsForEveryFilterCombination(int mask) {
        List<String> source = List.of("a", "b", "c", "d");
        List<String> expected = new ArrayList<>();
        ReadOnlyListFilter<String> filter = new ReadOnlyListFilter<>(source);
        for (int index = 0; index < source.size(); index++) {
            boolean keep = (mask & (1 << index)) != 0;
            filter.keepIf(index, keep);
            if (keep) {
                expected.add(source.get(index));
            }
        }

        List<String> result = filter.toList();

        assertEquals(expected, result);
        assertEquals(List.of("a", "b", "c", "d"), source);
        assertThrows(UnsupportedOperationException.class, () -> result.add("e"));
        assertThrows(UnsupportedOperationException.class, result::clear);
        if (mask == 15) {
            assertSame(source, result);
        }
    }

    @Test
    void reusesEmptySnapshot() {
        List<String> source = List.of();

        assertSame(source, new ReadOnlyListFilter<>(source).toList());
    }

    @Test
    void retainsElementIdentityWithoutCopyingElements() {
        Object first = new Object();
        Object second = new Object();
        List<Object> source = List.of(first, second);
        ReadOnlyListFilter<Object> filter = new ReadOnlyListFilter<>(source);
        filter.keepIf(0, false);
        filter.keepIf(1, true);

        assertSame(second, filter.toList().getFirst());
    }
}

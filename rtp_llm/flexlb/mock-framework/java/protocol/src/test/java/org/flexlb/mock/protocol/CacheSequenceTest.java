package org.flexlb.mock.protocol;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import org.junit.jupiter.api.Test;

class CacheSequenceTest {
    @Test void appendedKeysEqualHashingTheEntireComputedSequenceAcrossPartialBlocksAndMedia() {
        var random = new Random(20261106);
        for (int stride : List.of(1,2,4))
        for (int block : List.of(1, 2, 7, 128))
            for (int length : List.of(0, 1, block - 1, block, block + 1, 3 * block + 1, 4 * block + 1)) {
                var tokens = new ArrayList<Integer>();
                for (int i = 0; i < length; i++) tokens.add(random.nextInt());
                var spans = new ArrayList<CacheSequence.Span>();
                if (length > 1) {
                    var hashes = new ArrayList<Integer>();
                    for (int i = 1; i < length; i++) hashes.add(random.nextInt());
                    spans.add(new CacheSequence.Span(1, hashes));
                }
                long seed = random.nextLong();
                var sequence = new CacheSequence(block, tokens, spans, seed,stride);
                assertSame(sequence.keys(), sequence.keys());
                assertSame(sequence.keys(), sequence.keys(0, 42));
                for (int output : List.of(1, 2, block, block + 1, 2 * block + 5)) {
                    int token = random.nextInt();
                    var computed = new ArrayList<>(tokens); computed.addAll(Collections.nCopies(output, token));
                    assertEquals(new CacheSequence(block, computed, spans, seed,stride).keys(), sequence.keys(output, token));
                    assertEquals(length / (block*stride), sequence.keys().size());
                    assertEquals(block*stride,sequence.blockSize());
                }
            }
    }

    @Test void retainedPromptKeysAndSemanticInputAreImmutable() {
        var tokens = new ArrayList<>(List.of(1, 2, 3));
        var sequence = new CacheSequence(2, tokens, List.of(), Long.MIN_VALUE);
        var expected = sequence.keys(); tokens.set(0, 99);
        assertEquals(List.of(1, 2, 3), sequence.tokens()); assertSame(expected, sequence.keys());
        assertThrows(UnsupportedOperationException.class, () -> sequence.keys().add(1L));
        assertEquals(sequence, new CacheSequence(2, List.of(1, 2, 3), List.of(), Long.MIN_VALUE));
        assertThrows(IllegalArgumentException.class, () -> sequence.keys(-1, 0));
        assertThrows(ArithmeticException.class, () -> sequence.keys(Integer.MAX_VALUE, 0));
    }
}
